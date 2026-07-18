from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Iterable, Iterator, Sequence

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

try:
    from fastparquet import ParquetFile
except ImportError:  # pragma: no cover
    ParquetFile = None


SPLIT_ORDER = ("train", "valid", "test")
RAW_SOURCE_SPLITS = ("train", "test")
OUTPUT_FORMAT_CHOICES = ("csv", "parquet", "both")
PARQUET_COMPRESSION_CHOICES = ("zstd", "snappy", "gzip", "brotli", "none")


@dataclass(frozen=True)
class RequestedSample:
    group: str
    sample_id: str
    source_split: str
    assigned_split: str
    label: str
    order: int


@dataclass(frozen=True)
class RawNtRecord:
    sample_id: str
    source_split: str
    sequence: str
    label: str


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
            "Materialize low-similarity NT splits produced from raw NT parquet/fna files "
            "into VisualDNA raw CSV/Parquet files: index,seq,label,split,name."
        )
    )
    parser.add_argument(
        "--nt-root",
        type=Path,
        required=True,
        help="Raw NT root containing one directory per task with train/test parquet or fna files.",
    )
    parser.add_argument(
        "--nt-low-split-root",
        type=Path,
        required=True,
        help="NT low_similarity_split root produced by split_groups_by_low_similarity.py.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Output root. Each group is written to output-root/<group>/raw/<group>.*.",
    )
    parser.add_argument(
        "--output-format",
        choices=OUTPUT_FORMAT_CHOICES,
        default="both",
        help="Output format for raw sample tables.",
    )
    parser.add_argument(
        "--parquet-compression",
        choices=PARQUET_COMPRESSION_CHOICES,
        default="zstd",
        help="Parquet compression. Used only for parquet or both.",
    )
    parser.add_argument(
        "--groups",
        default="",
        help="Optional comma-separated group list. Empty means all groups.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output folders.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars.",
    )
    return parser.parse_args()


def parse_csv_list(value: str) -> set[str] | None:
    items = [item.strip() for item in value.split(",") if item.strip()]
    return set(items) if items else None


def progress_iter(iterable: Iterable, *, total: int | None, desc: str, enabled: bool):
    if enabled and tqdm is not None:
        return tqdm(iterable, total=total, desc=desc)
    return iterable


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [{str(key): str(value) for key, value in row.items()} for row in reader]


def normalize_label_text(value: object) -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    try:
        numeric = float(text)
    except ValueError:
        return text
    if numeric.is_integer():
        return str(int(numeric))
    return text


def list_group_dirs(low_split_root: Path, selected_groups: set[str] | None) -> list[Path]:
    if not low_split_root.exists():
        raise FileNotFoundError(f"NT low split root does not exist: {low_split_root}")
    if not low_split_root.is_dir():
        raise NotADirectoryError(f"NT low split root is not a directory: {low_split_root}")

    group_dirs = [
        path
        for path in sorted(low_split_root.iterdir())
        if path.is_dir() and any((path / f"{split}.csv").exists() for split in SPLIT_ORDER)
    ]
    if selected_groups is None:
        return group_dirs

    resolved = [path for path in group_dirs if path.name in selected_groups]
    missing = sorted(selected_groups - {path.name for path in resolved})
    if missing:
        available = [path.name for path in group_dirs]
        raise ValueError(f"Missing groups under low split root: {missing}. Available: {available}")
    return resolved


def infer_source_split_from_sample_id(sample_id: str) -> str:
    if ":" not in sample_id:
        return ""
    sample_name = sample_id.split(":", 1)[1]
    for split in RAW_SOURCE_SPLITS:
        if sample_name == split or sample_name.startswith(f"{split}_") or sample_name.startswith(f"{split}:"):
            return split
    return ""


def load_requested_samples(group_dir: Path) -> list[RequestedSample]:
    samples: list[RequestedSample] = []
    for assigned_split_name in SPLIT_ORDER:
        csv_path = group_dir / f"{assigned_split_name}.csv"
        if not csv_path.exists():
            continue
        for row in read_csv_rows(csv_path):
            sample_id = str(row.get("sample_id", "")).strip()
            if not sample_id:
                raise ValueError(f"Missing sample_id in low split CSV: {csv_path}")
            group_name = sample_id.split(":", 1)[0] if ":" in sample_id else ""
            if group_name and group_name != group_dir.name:
                raise ValueError(
                    f"sample_id group does not match folder name: sample_id={sample_id}, "
                    f"group_dir={group_dir.name}"
                )
            source_split = str(row.get("sample_split", "")).strip()
            if source_split not in RAW_SOURCE_SPLITS:
                source_split = infer_source_split_from_sample_id(sample_id)
            if source_split not in RAW_SOURCE_SPLITS:
                raise ValueError(f"Cannot infer raw source split for sample_id={sample_id}")

            assigned_split = str(row.get("assigned_split", "")).strip() or assigned_split_name
            if assigned_split not in SPLIT_ORDER:
                raise ValueError(f"Invalid assigned_split={assigned_split!r} in {csv_path}")

            label = normalize_label_text(row.get("sample_label_raw", ""))
            if not label:
                label = normalize_label_text(row.get("sample_label_numeric", ""))

            samples.append(
                RequestedSample(
                    group=group_dir.name,
                    sample_id=sample_id,
                    source_split=source_split,
                    assigned_split=assigned_split,
                    label=label,
                    order=len(samples),
                )
            )
    return samples


def iter_fasta_records(path: Path) -> Iterator[tuple[str, str]]:
    header: str | None = None
    sequence_parts: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(sequence_parts).upper()
                header = line[1:]
                sequence_parts = []
            else:
                sequence_parts.append(line)
    if header is not None:
        yield header, "".join(sequence_parts).upper()


def parse_fasta_header(header: str) -> tuple[str, str]:
    header_text = header.strip()
    if "|" not in header_text:
        return header_text, ""
    _, label_text = header_text.rsplit("|", 1)
    return header_text, normalize_label_text(label_text)


def build_raw_nt_sample_id(group_name: str, split: str, row_index: int, sample_name: str) -> str:
    return f"{group_name}:{split}:{row_index:09d}:{sample_name}"


def load_raw_nt_fasta_records(group_name: str, split: str, fasta_path: Path) -> list[RawNtRecord]:
    records: list[RawNtRecord] = []
    for index, (header, sequence) in enumerate(iter_fasta_records(fasta_path), start=1):
        sample_name, label = parse_fasta_header(header)
        if not sample_name:
            sample_name = f"{split}_{index:09d}"
        records.append(
            RawNtRecord(
                sample_id=build_raw_nt_sample_id(group_name, split, index, sample_name),
                source_split=split,
                sequence=sequence,
                label=label,
            )
        )
    return records


def iter_parquet_rows(parquet_path: Path, columns: Sequence[str]) -> Iterator[object]:
    if ParquetFile is not None:
        parquet_file = ParquetFile(parquet_path)
        selected_columns = [column for column in columns if column in parquet_file.columns]
        for frame in parquet_file.iter_row_groups(columns=selected_columns):
            if not frame.empty:
                yield from frame.itertuples(index=False)
        return

    if pd is not None:
        frame = pd.read_parquet(parquet_path, columns=list(columns))
        yield from frame.itertuples(index=False)
        return

    raise ImportError(
        "Reading NT raw parquet requires fastparquet or pandas. "
        f"Parquet path: {parquet_path}"
    )


def load_raw_nt_parquet_records(group_name: str, split: str, parquet_path: Path) -> list[RawNtRecord]:
    records: list[RawNtRecord] = []

    if ParquetFile is not None:
        parquet_columns = set(ParquetFile(parquet_path).columns)
        sequence_column = "sequence" if "sequence" in parquet_columns else "seq" if "seq" in parquet_columns else None
        if sequence_column is None:
            raise ValueError(f"NT raw parquet has no sequence/seq column: {parquet_path}")
        columns = [column for column in (sequence_column, "label", "name") if column in parquet_columns]
        row_iterator = iter_parquet_rows(parquet_path, columns)
        for index, row in enumerate(row_iterator, start=1):
            sample_name = str(getattr(row, "name", "")).strip() or f"{split}_{index:09d}"
            label = normalize_label_text(getattr(row, "label", ""))
            sequence = str(getattr(row, sequence_column)).upper()
            records.append(
                RawNtRecord(
                    sample_id=build_raw_nt_sample_id(group_name, split, index, sample_name),
                    source_split=split,
                    sequence=sequence,
                    label=label,
                )
            )
        return records

    if pd is not None:
        frame = pd.read_parquet(parquet_path)
        parquet_columns = set(frame.columns)
        sequence_column = "sequence" if "sequence" in parquet_columns else "seq" if "seq" in parquet_columns else None
        if sequence_column is None:
            raise ValueError(f"NT raw parquet has no sequence/seq column: {parquet_path}")
        columns = [column for column in (sequence_column, "label", "name") if column in parquet_columns]
        for index, row in enumerate(frame[columns].itertuples(index=False), start=1):
            sample_name = str(getattr(row, "name", "")).strip() or f"{split}_{index:09d}"
            label = normalize_label_text(getattr(row, "label", ""))
            sequence = str(getattr(row, sequence_column)).upper()
            records.append(
                RawNtRecord(
                    sample_id=build_raw_nt_sample_id(group_name, split, index, sample_name),
                    source_split=split,
                    sequence=sequence,
                    label=label,
                )
            )
        return records

    raise ImportError(
        "Reading NT raw parquet requires fastparquet or pandas. "
        f"Parquet path: {parquet_path}"
    )
    return records


def load_raw_nt_split_records(nt_group_dir: Path, split: str) -> list[RawNtRecord]:
    parquet_path = nt_group_dir / f"{split}.parquet"
    fasta_path = nt_group_dir / f"{split}.fna"
    if parquet_path.exists():
        return load_raw_nt_parquet_records(nt_group_dir.name, split, parquet_path)
    if fasta_path.exists():
        return load_raw_nt_fasta_records(nt_group_dir.name, split, fasta_path)
    raise FileNotFoundError(
        f"Raw NT group has no {split}.parquet or {split}.fna: {nt_group_dir}"
    )


def build_raw_lookup(
    nt_root: Path,
    group_name: str,
    source_splits: set[str],
    progress: bool,
) -> dict[tuple[str, str], RawNtRecord]:
    nt_group_dir = nt_root / group_name
    if not nt_group_dir.exists():
        raise FileNotFoundError(f"Raw NT group directory does not exist: {nt_group_dir}")

    lookup: dict[tuple[str, str], RawNtRecord] = {}
    duplicate_keys = 0
    iterator = progress_iter(
        sorted(source_splits),
        total=len(source_splits),
        desc=f"Load raw {group_name}",
        enabled=progress,
    )
    for split in iterator:
        for record in load_raw_nt_split_records(nt_group_dir, split):
            key = (split, record.sample_id)
            if key in lookup:
                duplicate_keys += 1
                continue
            lookup[key] = record
    if duplicate_keys:
        print(
            f"[warn] group={group_name}: found {duplicate_keys} duplicate raw sample_id rows; "
            "using the first row for sequence lookup because low split CSVs only store sample_id.",
            file=sys.stderr,
            flush=True,
        )
    return lookup


def parquet_compression_name(value: str) -> str | None:
    return None if value == "none" else value


def write_csv_rows(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def write_parquet_rows(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[dict[str, object]],
    compression: str,
    progress: bool,
) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Writing parquet requires pyarrow.") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for name in fieldnames:
        if name == "seq":
            fields.append((name, pa.large_string()))
        elif name in {"index", "sequence_length"}:
            fields.append((name, pa.int64()))
        else:
            fields.append((name, pa.string()))
    schema = pa.schema(fields)

    writer: pq.ParquetWriter | None = None
    max_rows_per_batch = 20_000
    max_seq_bytes_per_batch = 512 * 1024 * 1024
    buffer: dict[str, list[object]] = {name: [] for name in fieldnames}
    current_seq_bytes = 0

    def flush() -> None:
        nonlocal writer, buffer, current_seq_bytes
        if not buffer[fieldnames[0]]:
            return
        table = pa.Table.from_pydict(buffer, schema=schema)
        if writer is None:
            writer = pq.ParquetWriter(
                path,
                schema,
                compression=parquet_compression_name(compression),
            )
        writer.write_table(table)
        for name in buffer:
            buffer[name] = []
        current_seq_bytes = 0

    iterator = progress_iter(
        rows,
        total=len(rows),
        desc=f"Parquet write {path.stem}",
        enabled=progress,
    )
    try:
        for row in iterator:
            for name in fieldnames:
                value = row.get(name)
                if name in {"index", "sequence_length"}:
                    value = int(value) if str(value).strip() else None
                elif value is not None:
                    value = str(value)
                buffer[name].append(value)
                if name == "seq" and value is not None:
                    current_seq_bytes += len(str(value).encode("utf-8"))
            if len(buffer[fieldnames[0]]) >= max_rows_per_batch or current_seq_bytes >= max_seq_bytes_per_batch:
                flush()
        flush()
    finally:
        if writer is not None:
            writer.close()


def write_statistic(path: Path, group: str, rows: Sequence[dict[str, object]]) -> None:
    split_counts = Counter(str(row["split"]) for row in rows)
    label_counts = Counter(str(row["label"]) for row in rows)
    split_label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    lengths = [int(row["sequence_length"]) for row in rows]
    for row in rows:
        split_label_counts[str(row["split"])][str(row["label"])] += 1

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"dataset={group}\n")
        for split in SPLIT_ORDER:
            handle.write(f"n_{split}={split_counts.get(split, 0)}\n")
        handle.write(f"total={len(rows)}\n")
        handle.write(f"n_classes={len(label_counts)}\n")
        handle.write(f"classes={sorted(label_counts)}\n\n")
        if lengths:
            handle.write("sequence_length_statistics:\n")
            handle.write(f"  min={min(lengths)}\n")
            handle.write(f"  max={max(lengths)}\n")
            handle.write(f"  mean={mean(lengths):.2f}\n")
            handle.write(f"  median={median(lengths)}\n\n")
        handle.write("class_distribution:\n")
        header = ["label", *SPLIT_ORDER]
        handle.write("".join(f"{item:>12}" for item in header) + "\n")
        for label in sorted(label_counts):
            values = [label, *[split_label_counts[split].get(label, 0) for split in SPLIT_ORDER]]
            handle.write("".join(f"{str(item):>12}" for item in values) + "\n")


def length_summary(rows: Sequence[dict[str, object]]) -> dict[str, float | int]:
    lengths = [int(row["sequence_length"]) for row in rows]
    if not lengths:
        return {
            "count": 0,
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "median": 0.0,
        }
    return {
        "count": len(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "mean": float(mean(lengths)),
        "median": float(median(lengths)),
    }


def build_materialized_meta(
    *,
    group: str,
    rows: Sequence[dict[str, object]],
    output_root: Path,
    output_format: str,
    nt_root: Path | None = None,
    low_group_dir: Path | None = None,
) -> dict[str, object]:
    labels = sorted({str(row["label"]) for row in rows})
    label_mapping = {label: index for index, label in enumerate(labels)}
    split_rows = {
        split: [row for row in rows if str(row["split"]) == split]
        for split in SPLIT_ORDER
    }

    group_output_dir = output_root / group
    raw_dir = group_output_dir / "raw"
    raw_files: dict[str, str] = {}
    if output_format in {"csv", "both"}:
        raw_files["csv"] = str(raw_dir / f"{group}.csv")
    if output_format in {"parquet", "both"}:
        raw_files["parquet"] = str(raw_dir / f"{group}.parquet")

    meta: dict[str, object] = {
        "task_name": group,
        "task_family": "NT",
        "task_type": "classification",
        "input_schema": "single_sequence",
        "num_labels": len(labels),
        "label_mapping": label_mapping,
        "inverse_label_mapping": {str(index): label for label, index in label_mapping.items()},
        "raw_root": str(nt_root) if nt_root is not None else None,
        "low_similarity_split_dir": str(low_group_dir) if low_group_dir is not None else None,
        "raw_files": raw_files,
        "source_format": "materialized_raw_with_split_column",
        "splits": {
            split: length_summary(split_rows[split])
            for split in SPLIT_ORDER
        },
    }
    return meta


def write_materialized_meta(
    *,
    group_output_dir: Path,
    raw_dir: Path,
    group: str,
    rows: Sequence[dict[str, object]],
    output_root: Path,
    output_format: str,
    nt_root: Path | None = None,
    low_group_dir: Path | None = None,
) -> None:
    meta = build_materialized_meta(
        group=group,
        rows=rows,
        output_root=output_root,
        output_format=output_format,
        nt_root=nt_root,
        low_group_dir=low_group_dir,
    )
    meta_text = json.dumps(meta, ensure_ascii=False, indent=2)
    group_output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    (group_output_dir / "meta.json").write_text(meta_text + "\n", encoding="utf-8")
    (raw_dir / "meta.json").write_text(meta_text + "\n", encoding="utf-8")


def materialize_group(
    *,
    nt_root: Path,
    low_group_dir: Path,
    output_root: Path,
    output_format: str,
    parquet_compression: str,
    overwrite: bool,
    progress: bool,
) -> dict[str, object]:
    requests = load_requested_samples(low_group_dir)
    source_splits = {request.source_split for request in requests}
    raw_lookup = build_raw_lookup(
        nt_root=nt_root,
        group_name=low_group_dir.name,
        source_splits=source_splits,
        progress=progress,
    )

    rows: list[dict[str, object]] = []
    iterator = progress_iter(
        requests,
        total=len(requests),
        desc=f"Materialize {low_group_dir.name}",
        enabled=progress,
    )
    for raw_index, request in enumerate(iterator):
        raw_record = raw_lookup.get((request.source_split, request.sample_id))
        if raw_record is None:
            raise KeyError(
                f"Cannot find raw NT sample for group={request.group}, "
                f"source_split={request.source_split}, sample_id={request.sample_id}"
            )
        label = request.label or raw_record.label
        rows.append(
            {
                "index": raw_index,
                "seq": raw_record.sequence,
                "label": label,
                "split": request.assigned_split,
                "name": request.sample_id,
                "source_split": request.source_split,
                "sample_id": request.sample_id,
                "sequence_length": len(raw_record.sequence),
            }
        )

    group_output_dir = output_root / low_group_dir.name
    raw_dir = group_output_dir / "raw"
    if group_output_dir.exists() and overwrite:
        shutil.rmtree(group_output_dir)
    elif group_output_dir.exists():
        raise FileExistsError(f"Output group directory already exists: {group_output_dir}")
    raw_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "index",
        "seq",
        "label",
        "split",
        "name",
        "source_split",
        "sample_id",
        "sequence_length",
    ]
    if output_format in {"csv", "both"}:
        write_csv_rows(raw_dir / f"{low_group_dir.name}.csv", fieldnames, rows)
    if output_format in {"parquet", "both"}:
        write_parquet_rows(
            raw_dir / f"{low_group_dir.name}.parquet",
            fieldnames,
            rows,
            compression=parquet_compression,
            progress=progress,
        )
    write_statistic(group_output_dir / "statistic.txt", low_group_dir.name, rows)
    write_materialized_meta(
        group_output_dir=group_output_dir,
        raw_dir=raw_dir,
        group=low_group_dir.name,
        rows=rows,
        output_root=output_root,
        output_format=output_format,
        nt_root=nt_root,
        low_group_dir=low_group_dir,
    )

    split_counts = Counter(str(row["split"]) for row in rows)
    return {
        "group": low_group_dir.name,
        "records": len(rows),
        "train": split_counts.get("train", 0),
        "valid": split_counts.get("valid", 0),
        "test": split_counts.get("test", 0),
        "output_dir": str(group_output_dir),
    }


def write_manifest(output_root: Path, rows: Sequence[dict[str, object]]) -> None:
    manifest_path = output_root / "materialized_groups.csv"
    fieldnames = ["group", "records", "train", "valid", "test", "output_dir"]
    write_csv_rows(manifest_path, fieldnames, rows)
    summary = {
        "groups": len(rows),
        "records": sum(int(row["records"]) for row in rows),
        "train": sum(int(row["train"]) for row in rows),
        "valid": sum(int(row["valid"]) for row in rows),
        "test": sum(int(row["test"]) for row in rows),
        "group_summaries": list(rows),
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    selected_groups = parse_csv_list(args.groups)
    progress = not args.no_progress
    group_dirs = list_group_dirs(args.nt_low_split_root, selected_groups)
    args.output_root.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, object]] = []
    progress_bar = ProgressBar(total=len(group_dirs), enabled=progress, label="Materialize NT groups")
    for index, group_dir in enumerate(group_dirs, start=1):
        summaries.append(
            materialize_group(
                nt_root=args.nt_root,
                low_group_dir=group_dir,
                output_root=args.output_root,
                output_format=args.output_format,
                parquet_compression=args.parquet_compression,
                overwrite=args.overwrite,
                progress=progress,
            )
        )
        progress_bar.update(index)
    progress_bar.finish()
    write_manifest(args.output_root, summaries)
    print(
        json.dumps(
            {
                "output_root": str(args.output_root),
                "groups": len(summaries),
                "records": sum(int(row["records"]) for row in summaries),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
