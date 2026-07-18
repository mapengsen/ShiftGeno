from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence
from urllib.parse import urlparse

import yaml


DEFAULT_SPLITS = ("train", "test")


@dataclass(frozen=True)
class ReferenceSpec:
    url: str
    ref_type: str


@dataclass(frozen=True)
class RawGbRecord:
    dataset_name: str
    split: str
    class_name: str
    label: int
    source_id: str
    region: str
    start_0based: int
    end_0based: int
    strand: str
    source_index_path: Path
    reference_path: Path

    @property
    def length(self) -> int:
        return self.end_0based - self.start_0based


class ProgressBar:
    def __init__(self, total: int, enabled: bool, label: str) -> None:
        self.total = max(total, 1)
        self.enabled = enabled
        self.label = label
        self.start_time = time.time()
        self.last_render_time = 0.0

    def update(self, current: int) -> None:
        if not self.enabled:
            return
        now = time.time()
        if current < self.total and now - self.last_render_time < 0.2:
            return
        self.last_render_time = now
        bounded = min(max(current, 0), self.total)
        ratio = bounded / self.total
        width = 30
        filled = int(width * ratio)
        bar = "#" * filled + "-" * (width - filled)
        elapsed = now - self.start_time
        print(
            f"\r{self.label}: [{bar}] {bounded}/{self.total} "
            f"({ratio * 100:5.1f}%) | elapsed {elapsed:6.1f}s",
            end="",
            file=sys.stderr,
            flush=True,
        )

    def finish(self) -> None:
        if self.enabled:
            self.update(self.total)
            print(file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Genomic Benchmarks indexed-csv dataset from the official raw interval "
            "index without extending intervals or deduplicating records. The output layout "
            "matches rank_merged_nt_sample_similarity.py --dataset genomic_benchmarks."
        )
    )
    parser.add_argument(
        "--datasets-root",
        type=Path,
        required=True,
        help="GB official index root, for example data/raw_download/GB_github_index_metayml/datasets.",
    )
    parser.add_argument(
        "--reference-cache-dir",
        type=Path,
        required=True,
        help="Directory containing downloaded reference FASTA files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Output root for per-dataset train.csv/test.csv indexed files.",
    )
    parser.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        default=None,
        help="Only process this dataset. Can be passed multiple times.",
    )
    parser.add_argument(
        "--split",
        dest="splits",
        action="append",
        choices=DEFAULT_SPLITS,
        default=None,
        help="Only process this split. Can be passed multiple times.",
    )
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=None,
        help="Manifest CSV path. Default: output-root/manifest.csv.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Summary JSON path. Default: output-root/summary.json.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove an existing output root before writing.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars.",
    )
    return parser.parse_args()


def load_metadata(metadata_path: Path) -> dict[str, object]:
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"metadata.yaml is not a mapping: {metadata_path}")
    classes = metadata.get("classes")
    if not isinstance(classes, dict) or not classes:
        raise ValueError(f"metadata.yaml has no classes mapping: {metadata_path}")
    return metadata


def reference_cache_name(url: str) -> str:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    basename = Path(urlparse(url).path).name
    if not basename:
        raise ValueError(f"Cannot derive reference filename from URL: {url}")
    return f"{digest}_{basename}"


def resolve_reference_path(reference_cache_dir: Path, reference_spec: ReferenceSpec) -> Path:
    if reference_spec.ref_type != "fa.gz":
        raise ValueError(f"Only fa.gz references are supported, got {reference_spec.ref_type}")

    hashed_path = reference_cache_dir / reference_cache_name(reference_spec.url)
    if hashed_path.exists():
        return hashed_path

    basename = Path(urlparse(reference_spec.url).path).name
    plain_path = reference_cache_dir / basename
    if plain_path.exists():
        return plain_path

    raise FileNotFoundError(
        "Missing reference FASTA for URL "
        f"{reference_spec.url}. Expected {hashed_path} or {plain_path}."
    )


def resolve_dataset_dirs(datasets_root: Path, selected_datasets: Sequence[str] | None) -> list[Path]:
    if not datasets_root.exists():
        raise FileNotFoundError(f"Missing datasets root: {datasets_root}")
    if not datasets_root.is_dir():
        raise NotADirectoryError(f"Datasets root is not a directory: {datasets_root}")

    dataset_dirs = [
        path
        for path in sorted(datasets_root.iterdir())
        if path.is_dir() and (path / "metadata.yaml").exists()
    ]
    if selected_datasets is None:
        return dataset_dirs

    wanted = set(selected_datasets)
    resolved = [path for path in dataset_dirs if path.name in wanted]
    missing = sorted(wanted - {path.name for path in resolved})
    if missing:
        available = [path.name for path in dataset_dirs]
        raise ValueError(f"Unknown datasets: {missing}. Available datasets: {available}")
    return resolved


def iter_index_rows(index_path: Path) -> Iterator[dict[str, str]]:
    with gzip.open(index_path, mode="rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "region", "start", "end", "strand"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(
                f"Index file is missing columns {missing}: {index_path}. "
                f"Columns: {reader.fieldnames}"
            )
        for row in reader:
            yield row


def collect_records(
    dataset_dirs: Sequence[Path],
    splits: Sequence[str],
    reference_cache_dir: Path,
) -> tuple[list[RawGbRecord], dict[str, dict[str, int]]]:
    records: list[RawGbRecord] = []
    label_mappings: dict[str, dict[str, int]] = {}

    for dataset_dir in dataset_dirs:
        metadata = load_metadata(dataset_dir / "metadata.yaml")
        classes = metadata["classes"]
        assert isinstance(classes, dict)
        label_mapping = {class_name: label for label, class_name in enumerate(classes)}
        label_mappings[dataset_dir.name] = label_mapping

        for split in splits:
            split_dir = dataset_dir / split
            if not split_dir.exists():
                raise FileNotFoundError(f"Missing split directory: {split_dir}")

            for class_name, class_config in classes.items():
                if not isinstance(class_config, dict):
                    raise ValueError(f"Invalid class config for {dataset_dir.name}/{class_name}")
                index_path = split_dir / f"{class_name}.csv.gz"
                if not index_path.exists():
                    raise FileNotFoundError(f"Missing class index: {index_path}")
                reference_spec = ReferenceSpec(
                    url=str(class_config.get("url", "")),
                    ref_type=str(class_config.get("type", "")),
                )
                reference_path = resolve_reference_path(reference_cache_dir, reference_spec)

                for row in iter_index_rows(index_path):
                    start = int(row["start"])
                    end = int(row["end"])
                    if end <= start:
                        continue
                    records.append(
                        RawGbRecord(
                            dataset_name=dataset_dir.name,
                            split=split,
                            class_name=class_name,
                            label=label_mapping[class_name],
                            source_id=str(row["id"]),
                            region=str(row["region"]),
                            start_0based=start,
                            end_0based=end,
                            strand=str(row.get("strand") or "+"),
                            source_index_path=index_path,
                            reference_path=reference_path,
                        )
                    )

    return records, label_mappings


def open_split_writer(
    output_root: Path,
    dataset_name: str,
    split: str,
    handles: dict[tuple[str, str], object],
    writers: dict[tuple[str, str], csv.DictWriter],
) -> csv.DictWriter:
    key = (dataset_name, split)
    writer = writers.get(key)
    if writer is not None:
        return writer

    dataset_dir = output_root / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    csv_path = dataset_dir / f"{split}.csv"
    handle = csv_path.open("w", encoding="utf-8", newline="")
    fieldnames = [
        "index",
        "original_index",
        "label",
        "class_name",
        "reference_path",
        "region",
        "strand",
        "expanded_start_0based",
        "expanded_end_0based",
        "expanded_length",
        "expanded_truncated",
        "note",
    ]
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    handles[key] = handle
    writers[key] = writer
    return writer


def write_outputs(
    records: Sequence[RawGbRecord],
    label_mappings: dict[str, dict[str, int]],
    output_root: Path,
    manifest_csv: Path,
    progress: bool,
) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)

    for dataset_name, label_mapping in label_mappings.items():
        dataset_dir = output_root / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        (dataset_dir / "label_mapping.json").write_text(
            json.dumps(label_mapping, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    handles: dict[tuple[str, str], object] = {}
    writers: dict[tuple[str, str], csv.DictWriter] = {}
    counts: dict[str, dict[str, int]] = {}
    progress_bar = ProgressBar(total=len(records), enabled=progress, label="Writing raw GB index")

    with manifest_csv.open("w", encoding="utf-8", newline="") as manifest_handle:
        manifest_writer = csv.DictWriter(
            manifest_handle,
            fieldnames=[
                "dataset",
                "split",
                "class_name",
                "label",
                "source_index_path",
                "original_index",
                "index",
                "reference_path",
                "region",
                "strand",
                "input_start_0based",
                "input_end_0based",
                "input_length",
                "wrote_output",
                "note",
            ],
        )
        manifest_writer.writeheader()

        try:
            for running_index, record in enumerate(records):
                data_key = f"{record.split}_{running_index:09d}"
                writer = open_split_writer(
                    output_root=output_root,
                    dataset_name=record.dataset_name,
                    split=record.split,
                    handles=handles,
                    writers=writers,
                )
                row = {
                    "index": data_key,
                    "original_index": record.source_id,
                    "label": record.label,
                    "class_name": record.class_name,
                    "reference_path": str(record.reference_path),
                    "region": record.region,
                    "strand": record.strand or "+",
                    "expanded_start_0based": record.start_0based,
                    "expanded_end_0based": record.end_0based,
                    "expanded_length": record.length,
                    "expanded_truncated": False,
                    "note": "original_interval_no_extension",
                }
                writer.writerow(row)
                manifest_writer.writerow(
                    {
                        "dataset": record.dataset_name,
                        "split": record.split,
                        "class_name": record.class_name,
                        "label": record.label,
                        "source_index_path": str(record.source_index_path),
                        "original_index": record.source_id,
                        "index": data_key,
                        "reference_path": str(record.reference_path),
                        "region": record.region,
                        "strand": record.strand or "+",
                        "input_start_0based": record.start_0based,
                        "input_end_0based": record.end_0based,
                        "input_length": record.length,
                        "wrote_output": True,
                        "note": "original_interval_no_extension",
                    }
                )
                dataset_counts = counts.setdefault(record.dataset_name, {})
                dataset_counts[record.split] = dataset_counts.get(record.split, 0) + 1
                progress_bar.update(running_index + 1)
        finally:
            for handle in handles.values():
                handle.close()
            progress_bar.finish()

    return {
        "output_root": str(output_root),
        "manifest_csv": str(manifest_csv),
        "total_records": len(records),
        "dataset_split_counts": counts,
    }


def main() -> None:
    args = parse_args()
    splits = tuple(args.splits or DEFAULT_SPLITS)
    progress = not args.no_progress

    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output root already exists: {args.output_root}")
        shutil.rmtree(args.output_root)

    dataset_dirs = resolve_dataset_dirs(args.datasets_root, args.datasets)
    records, label_mappings = collect_records(
        dataset_dirs=dataset_dirs,
        splits=splits,
        reference_cache_dir=args.reference_cache_dir,
    )
    manifest_csv = args.manifest_csv or (args.output_root / "manifest.csv")
    summary_json = args.summary_json or (args.output_root / "summary.json")
    summary = write_outputs(
        records=records,
        label_mappings=label_mappings,
        output_root=args.output_root,
        manifest_csv=manifest_csv,
        progress=progress,
    )
    summary["datasets"] = [path.name for path in dataset_dirs]
    summary["splits"] = list(splits)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
