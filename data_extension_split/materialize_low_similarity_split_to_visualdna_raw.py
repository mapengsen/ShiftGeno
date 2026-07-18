from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Iterable, Sequence

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_extension_split.DNA_similarity.dataset_records import (  # noqa: E402
    DEFAULT_REFERENCE_MATERIALIZATION_DIR,
    ReferenceContigInfo,
    ReferenceContigLoader,
    build_gb_reference_aliases,
    get_nt_reference_loader,
    materialize_gzip_fasta,
    resolve_repo_path,
    reverse_complement,
)


SPLIT_ORDER = ("train", "valid", "test")
DATASET_CHOICES = ("nt", "genomic_benchmarks", "all")
OUTPUT_FORMAT_CHOICES = ("csv", "parquet", "both")
PARQUET_COMPRESSION_CHOICES = ("zstd", "snappy", "gzip", "brotli", "none")
DEFAULT_NT_INDEXED_ROOT = REPO_ROOT / "data/processed_download/deduplicated_indexed_sequences/nt"
DEFAULT_GB_INDEXED_ROOT = REPO_ROOT / "data/processed_download/deduplicated_indexed_sequences/genomic_benchmarks"
DEFAULT_NT_LOW_SPLIT_ROOT = (
    REPO_ROOT / "data_extension_split/DNA_similarity/outputs/all_nt_k8_from_deduplicated/low_similarity_split"
)
DEFAULT_GB_LOW_SPLIT_ROOT = (
    REPO_ROOT / "data_extension_split/DNA_similarity/outputs/all_gb_k8_from_deduplicated/low_similarity_split"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data/processed_download/low_similarity_sequence_csv"
GB_REFERENCE_INDEX_CACHE: dict[Path, tuple[dict[str, ReferenceContigInfo], dict[str, str]]] = {}


@dataclass(frozen=True)
class RequestedSample:
    group: str
    sample_id: str
    source_split: str
    assigned_split: str
    sample_index: str
    class_name: str
    label: str
    order: int


@dataclass(frozen=True)
class PendingRecord:
    request: RequestedSample
    indexed_row: dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "把 low_similarity_split 划分结果和去重后的 indexed-csv 坐标回查合并，"
            "物化为 VisualDNA 可直接读取的 raw CSV/Parquet：index,seq,label,split,name。"
        )
    )
    parser.add_argument(
        "--dataset",
        choices=DATASET_CHOICES,
        required=True,
        help="要物化的数据集：nt、genomic_benchmarks 或 all。",
    )
    parser.add_argument(
        "--nt-indexed-root",
        type=Path,
        default=DEFAULT_NT_INDEXED_ROOT,
        help="去重后的 NT indexed-csv 根目录。",
    )
    parser.add_argument(
        "--gb-indexed-root",
        type=Path,
        default=DEFAULT_GB_INDEXED_ROOT,
        help="去重后的 Genomic Benchmarks indexed-csv 根目录。",
    )
    parser.add_argument(
        "--nt-low-split-root",
        type=Path,
        default=DEFAULT_NT_LOW_SPLIT_ROOT,
        help="NT low_similarity_split 根目录。",
    )
    parser.add_argument(
        "--gb-low-split-root",
        type=Path,
        default=DEFAULT_GB_LOW_SPLIT_ROOT,
        help="Genomic Benchmarks low_similarity_split 根目录。",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=(
            "输出根目录。dataset=all 时写入 output-root/nt 和 output-root/genomic_benchmarks；"
            "单数据集时写入 output-root/<task>/raw/<task>.csv|parquet。"
        ),
    )
    parser.add_argument(
        "--output-format",
        choices=OUTPUT_FORMAT_CHOICES,
        default="both",
        help="raw 样本表输出格式。默认 both，同时写 CSV 和 Parquet。",
    )
    parser.add_argument(
        "--parquet-compression",
        choices=PARQUET_COMPRESSION_CHOICES,
        default="zstd",
        help="Parquet 压缩算法。仅在 --output-format 为 parquet 或 both 时生效。",
    )
    parser.add_argument(
        "--groups",
        default="",
        help="可选，逗号分隔的 group/task 名称。为空表示处理全部。",
    )
    parser.add_argument(
        "--reference-materialization-dir",
        type=Path,
        default=DEFAULT_REFERENCE_MATERIALIZATION_DIR,
        help="用于物化 .gz reference FASTA 的缓存目录。",
    )
    parser.add_argument(
        "--group-workers",
        type=int,
        default=1,
        help="并行处理 group/task 的进程数。1 表示顺序执行，默认 1。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已有输出。",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="关闭进度条。",
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


def write_csv_rows(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def parquet_compression_name(value: str) -> str | None:
    return None if value == "none" else value


def parquet_schema_for_fieldnames(fieldnames: Sequence[str]):
    try:
        import pyarrow as pa
    except ImportError as exc:  # pragma: no cover
        raise ImportError("写出 Parquet 需要安装 pyarrow。") from exc

    int64_fields = {"index", "sequence_length", "expanded_start_0based", "expanded_end_0based"}
    fields = []
    for name in fieldnames:
        if name == "seq":
            fields.append((name, pa.large_string()))
        elif name in int64_fields:
            fields.append((name, pa.int64()))
        else:
            fields.append((name, pa.string()))
    return pa.schema(fields)


def coerce_parquet_value(field_name: str, value: object) -> object:
    if value is None:
        return None
    if field_name in {"index", "sequence_length", "expanded_start_0based", "expanded_end_0based"}:
        text = str(value).strip()
        return int(text) if text else None
    return str(value)


def write_parquet_rows(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[dict[str, object]],
    *,
    compression: str,
    progress: bool,
) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise ImportError("写出 Parquet 需要安装 pyarrow。") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    schema = parquet_schema_for_fieldnames(fieldnames)
    writer: pq.ParquetWriter | None = None
    max_rows_per_batch = 20_000
    max_seq_bytes_per_batch = 512 * 1024 * 1024
    buffer: dict[str, list[object]] = {name: [] for name in fieldnames}
    current_seq_bytes = 0

    if not rows:
        empty_table = pa.Table.from_pydict(buffer, schema=schema)
        pq.write_table(
            empty_table,
            path,
            compression=parquet_compression_name(compression),
        )
        return

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
                value = coerce_parquet_value(name, row.get(name))
                buffer[name].append(value)
                if name == "seq" and value is not None:
                    current_seq_bytes += len(str(value).encode("utf-8"))
            if len(buffer[fieldnames[0]]) >= max_rows_per_batch or current_seq_bytes >= max_seq_bytes_per_batch:
                flush()
        flush()
    finally:
        if writer is not None:
            writer.close()


def list_group_dirs(root: Path, selected_groups: set[str] | None, label: str) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"{label} 根目录不存在：{root}")
    if not root.is_dir():
        raise NotADirectoryError(f"{label} 不是目录：{root}")

    group_dirs = [
        path
        for path in sorted(root.iterdir())
        if path.is_dir() and any((path / f"{split}.csv").exists() for split in SPLIT_ORDER)
    ]
    if selected_groups is None:
        return group_dirs

    resolved = [path for path in group_dirs if path.name in selected_groups]
    missing = sorted(selected_groups - {path.name for path in resolved})
    if missing:
        raise ValueError(
            f"{label} 下找不到指定 group：{missing}。"
            f"可选 group：{[path.name for path in group_dirs]}"
        )
    return resolved


def split_sample_id(sample_id: str) -> tuple[str, str, str]:
    if ":" not in sample_id:
        return "", "", sample_id
    group, remainder = sample_id.split(":", 1)
    if ":" not in remainder:
        return group, "", remainder
    class_name, sample_index = remainder.split(":", 1)
    return group, class_name, sample_index


def infer_source_split(sample_index: str) -> str:
    for split in SPLIT_ORDER:
        if sample_index == split or sample_index.startswith(f"{split}_"):
            return split
    return ""


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


def load_requested_samples(group_dir: Path) -> list[RequestedSample]:
    samples: list[RequestedSample] = []
    for split in SPLIT_ORDER:
        csv_path = group_dir / f"{split}.csv"
        if not csv_path.exists():
            continue
        for row in read_csv_rows(csv_path):
            sample_id = str(row.get("sample_id", "")).strip()
            if not sample_id:
                raise ValueError(f"低相似度划分缺少 sample_id：{csv_path}")
            group, class_name, sample_index = split_sample_id(sample_id)
            if group and group != group_dir.name:
                raise ValueError(
                    f"sample_id group 与目录名不一致：sample_id={sample_id}, group_dir={group_dir.name}"
                )
            source_split = str(row.get("sample_split", "")).strip() or infer_source_split(sample_index)
            if source_split not in SPLIT_ORDER:
                raise ValueError(f"无法确定 sample_id 的原始 split：{sample_id}")
            assigned_split = str(row.get("assigned_split", "")).strip() or split
            if assigned_split not in SPLIT_ORDER:
                raise ValueError(f"无效 assigned_split={assigned_split!r}：{csv_path}")
            label = normalize_label_text(row.get("sample_label_raw", ""))
            if not label:
                label = normalize_label_text(row.get("sample_label_numeric", ""))
            samples.append(
                RequestedSample(
                    group=group_dir.name,
                    sample_id=sample_id,
                    source_split=source_split,
                    assigned_split=assigned_split,
                    sample_index=sample_index,
                    class_name=class_name,
                    label=label,
                    order=len(samples),
                )
            )
    return samples


def compose_indexed_sample_id(dataset: str, group: str, row: dict[str, str]) -> str:
    row_index = str(row.get("index", "")).strip()
    if dataset == "genomic_benchmarks":
        class_name = str(row.get("class_name", "")).strip()
        if class_name:
            return f"{group}:{class_name}:{row_index}"
    return f"{group}:{row_index}"


def build_indexed_lookup(
    *,
    dataset: str,
    indexed_group_dir: Path,
    source_splits: set[str],
) -> dict[tuple[str, str], dict[str, str]]:
    lookup: dict[tuple[str, str], dict[str, str]] = {}
    for split in sorted(source_splits):
        csv_path = indexed_group_dir / f"{split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"坐标索引 CSV 不存在：{csv_path}")
        for row in read_csv_rows(csv_path):
            sample_id = compose_indexed_sample_id(dataset, indexed_group_dir.name, row)
            key = (split, sample_id)
            if key in lookup:
                raise ValueError(f"坐标索引中 sample_id 不唯一：{key}")
            lookup[key] = row
    return lookup


def build_gb_reference_index(
    reference_fasta: Path,
) -> tuple[dict[str, ReferenceContigInfo], dict[str, str]]:
    contigs: dict[str, ReferenceContigInfo] = {}
    alias_to_canonical: dict[str, str] = {}
    current_name: str | None = None
    current_header_text: str | None = None
    current_sequence_offset: int | None = None

    with reference_fasta.open(mode="rb") as handle:
        while True:
            line_offset = handle.tell()
            raw_line = handle.readline()
            if not raw_line:
                if (
                    current_name is not None
                    and current_header_text is not None
                    and current_sequence_offset is not None
                ):
                    aliases = tuple(build_gb_reference_aliases(current_name))
                    contig_info = ReferenceContigInfo(
                        canonical_name=current_name,
                        header_text=current_header_text,
                        sequence_offset=current_sequence_offset,
                        next_header_offset=handle.tell(),
                        aliases=aliases,
                    )
                    contigs[current_name] = contig_info
                    for alias in aliases:
                        alias_to_canonical.setdefault(alias, current_name)
                break

            if not raw_line.startswith(b">"):
                continue

            header_text = raw_line[1:].decode("utf-8").strip()
            canonical_name = header_text.split()[0]
            if (
                current_name is not None
                and current_header_text is not None
                and current_sequence_offset is not None
            ):
                aliases = tuple(build_gb_reference_aliases(current_name))
                contig_info = ReferenceContigInfo(
                    canonical_name=current_name,
                    header_text=current_header_text,
                    sequence_offset=current_sequence_offset,
                    next_header_offset=line_offset,
                    aliases=aliases,
                )
                contigs[current_name] = contig_info
                for alias in aliases:
                    alias_to_canonical.setdefault(alias, current_name)

            current_name = canonical_name
            current_header_text = header_text
            current_sequence_offset = handle.tell()

    return contigs, alias_to_canonical


def get_gb_reference_loader(
    reference_path: Path,
    materialization_dir: Path,
) -> ReferenceContigLoader:
    materialized_reference = materialize_gzip_fasta(reference_path, materialization_dir)
    cached_index = GB_REFERENCE_INDEX_CACHE.get(materialized_reference)
    if cached_index is None:
        cached_index = build_gb_reference_index(materialized_reference)
        GB_REFERENCE_INDEX_CACHE[materialized_reference] = cached_index
    index, alias_to_canonical = cached_index
    return ReferenceContigLoader(
        reference_fasta=materialized_reference,
        index=index,
        alias_to_canonical=alias_to_canonical,
        cache_size=4,
    )


def fetch_nt_sequence(
    pending: PendingRecord,
    materialization_dir: Path,
    loader_cache: dict[Path, ReferenceContigLoader],
) -> str:
    row = pending.indexed_row
    reference_fasta = resolve_repo_path(str(row.get("reference_fasta", "")).strip())
    chromosome = str(row.get("reference_chromosome", "")).strip()
    start = int(str(row.get("expanded_start_0based", "")).strip())
    end = int(str(row.get("expanded_end_0based", "")).strip())
    if not reference_fasta.exists():
        raise FileNotFoundError(f"reference FASTA 不存在：{reference_fasta}")
    loader = loader_cache.get(reference_fasta)
    if loader is None:
        loader = get_nt_reference_loader(reference_fasta, materialization_dir)
        loader_cache[reference_fasta] = loader
    _, chromosome_sequence = loader.load_sequence(chromosome)
    if start < 0 or end > len(chromosome_sequence) or start >= end:
        raise ValueError(f"无效 NT 坐标：{reference_fasta} {chromosome}:{start}-{end}")
    return chromosome_sequence[start:end].upper()


def fetch_gb_sequence(
    pending: PendingRecord,
    materialization_dir: Path,
    loader_cache: dict[Path, ReferenceContigLoader],
) -> str:
    row = pending.indexed_row
    reference_path = resolve_repo_path(str(row.get("reference_path", "")).strip())
    region = str(row.get("region", "")).strip()
    strand = str(row.get("strand", "+") or "+").strip() or "+"
    start = int(str(row.get("expanded_start_0based", "")).strip())
    end = int(str(row.get("expanded_end_0based", "")).strip())
    if not reference_path.exists():
        raise FileNotFoundError(f"reference FASTA 不存在：{reference_path}")
    loader = loader_cache.get(reference_path)
    if loader is None:
        loader = get_gb_reference_loader(reference_path, materialization_dir)
        loader_cache[reference_path] = loader
    _, region_sequence = loader.load_sequence(region)
    if start < 0 or end > len(region_sequence) or start >= end:
        raise ValueError(f"无效 GB 坐标：{reference_path} {region}:{start}-{end}")
    sequence = region_sequence[start:end].upper()
    if strand == "-":
        sequence = reverse_complement(sequence).upper()
    return sequence


def build_raw_record(
    *,
    dataset: str,
    raw_index: int,
    pending: PendingRecord,
    sequence: str,
) -> dict[str, object]:
    request = pending.request
    coord = pending.indexed_row
    label = request.label or normalize_label_text(coord.get("label", ""))
    record: dict[str, object] = {
        "index": raw_index,
        "seq": sequence,
        "label": label,
        "split": request.assigned_split,
        "name": request.sample_id,
        "source_split": request.source_split,
        "source_index": request.sample_index,
        "sample_id": request.sample_id,
        "sequence_length": len(sequence),
    }
    if dataset == "genomic_benchmarks":
        record["label_name"] = str(coord.get("class_name", "")).strip()
        record["reference_path"] = str(coord.get("reference_path", "")).strip()
        record["region"] = str(coord.get("region", "")).strip()
        record["strand"] = str(coord.get("strand", "+") or "+").strip() or "+"
    else:
        record["reference_fasta"] = str(coord.get("reference_fasta", "")).strip()
        record["reference_chromosome"] = str(coord.get("reference_chromosome", "")).strip()
    record["expanded_start_0based"] = str(coord.get("expanded_start_0based", "")).strip()
    record["expanded_end_0based"] = str(coord.get("expanded_end_0based", "")).strip()
    return record


def write_statistic(path: Path, group: str, rows: Sequence[dict[str, object]]) -> None:
    split_counts = Counter(str(row["split"]) for row in rows)
    label_counts = Counter(str(row["label"]) for row in rows)
    split_label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    lengths = [int(row["sequence_length"]) for row in rows]
    for row in rows:
        split_label_counts[str(row["split"])][str(row["label"])] += 1

    classes = sorted(label_counts)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"dataset={group}\n")
        for split in SPLIT_ORDER:
            handle.write(f"n_{split}={split_counts.get(split, 0)}\n")
        handle.write(f"total={len(rows)}\n")
        handle.write(f"n_classes={len(classes)}\n")
        handle.write(f"classes={classes}\n\n")
        if lengths:
            handle.write("sequence_length_statistics:\n")
            handle.write(f"  min={min(lengths)}\n")
            handle.write(f"  max={max(lengths)}\n")
            handle.write(f"  mean={mean(lengths):.2f}\n")
            handle.write(f"  median={median(lengths)}\n\n")
        handle.write("class_distribution:\n")
        header = ["label", *SPLIT_ORDER]
        handle.write("".join(f"{item:>12}" for item in header) + "\n")
        for label in classes:
            values = [label, *[split_label_counts[split].get(label, 0) for split in SPLIT_ORDER]]
            handle.write("".join(f"{str(item):>12}" for item in values) + "\n")


def write_label_mapper(path: Path, rows: Sequence[dict[str, object]]) -> None:
    label_names = {
        str(row.get("label_name", "")).strip(): str(row.get("label", "")).strip()
        for row in rows
        if str(row.get("label_name", "")).strip()
    }
    if not label_names:
        return
    label_mapper = {
        label_name: int(label) if label.isdigit() else label
        for label_name, label in sorted(label_names.items())
    }
    path.write_text(json.dumps(label_mapper, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_int_label(value: str) -> int | None:
    text = str(value).strip()
    if not text:
        return None
    try:
        numeric = float(text)
    except ValueError:
        return None
    if not numeric.is_integer():
        return None
    return int(numeric)


def json_label_value(label: str) -> int | str:
    parsed = parse_int_label(label)
    return parsed if parsed is not None else label


def build_label_mapping(rows: Sequence[dict[str, object]]) -> tuple[dict[str, int | str], dict[str, str]]:
    label_name_pairs = {
        str(row.get("label_name", "")).strip(): str(row.get("label", "")).strip()
        for row in rows
        if str(row.get("label_name", "")).strip()
    }
    if label_name_pairs:
        label_mapping = {
            label_name: json_label_value(label)
            for label_name, label in sorted(label_name_pairs.items())
        }
        inverse_label_mapping = {
            str(json_label_value(label)): label_name
            for label_name, label in sorted(label_name_pairs.items())
        }
        return label_mapping, inverse_label_mapping

    labels = sorted({str(row.get("label", "")).strip() for row in rows})
    label_mapping = {label: json_label_value(label) for label in labels}
    inverse_label_mapping = {str(json_label_value(label)): label for label in labels}
    return label_mapping, inverse_label_mapping


def summarize_lengths(lengths: Sequence[int]) -> dict[str, float | int]:
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


def build_split_summaries(rows: Sequence[dict[str, object]]) -> dict[str, dict[str, float | int]]:
    summaries: dict[str, dict[str, float | int]] = {}
    for split in SPLIT_ORDER:
        split_lengths = [
            int(row["sequence_length"])
            for row in rows
            if str(row.get("split", "")).strip() == split
        ]
        summaries[split] = summarize_lengths(split_lengths)
    return summaries


def write_meta(
    path: Path,
    *,
    dataset: str,
    group: str,
    rows: Sequence[dict[str, object]],
    indexed_group_dir: Path,
    low_group_dir: Path,
    materialization_dir: Path,
    output_format: str,
    parquet_compression: str,
    raw_csv_path: Path | None,
    raw_parquet_path: Path | None,
) -> None:
    label_mapping, inverse_label_mapping = build_label_mapping(rows)
    split_counts = Counter(str(row["split"]) for row in rows)
    labels = sorted({str(row.get("label", "")).strip() for row in rows})
    meta = {
        "task_name": group,
        "dataset": dataset,
        "task_family": "NT" if dataset == "nt" else "genomic_benchmarks",
        "task_type": "classification",
        "input_schema": "single_sequence",
        "num_labels": len(labels),
        "label_mapping": label_mapping,
        "inverse_label_mapping": inverse_label_mapping,
        "storage_format": output_format,
        "parquet_compression": parquet_compression if raw_parquet_path is not None else None,
        "raw_files": {
            "csv": str(raw_csv_path) if raw_csv_path is not None else None,
            "parquet": str(raw_parquet_path) if raw_parquet_path is not None else None,
        },
        "splits": build_split_summaries(rows),
        "split_counts": {split: int(split_counts.get(split, 0)) for split in SPLIT_ORDER},
        "source": {
            "indexed_group_dir": str(indexed_group_dir),
            "low_similarity_group_dir": str(low_group_dir),
            "reference_materialization_dir": str(materialization_dir),
        },
    }
    path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")


def materialize_group(
    *,
    dataset: str,
    low_group_dir: Path,
    indexed_root: Path,
    output_root: Path,
    materialization_dir: Path,
    overwrite: bool,
    progress: bool,
    output_format: str,
    parquet_compression: str,
) -> dict[str, object]:
    indexed_group_dir = indexed_root / low_group_dir.name
    if not indexed_group_dir.exists():
        raise FileNotFoundError(f"坐标索引 group 目录不存在：{indexed_group_dir}")

    requests = load_requested_samples(low_group_dir)
    source_splits = {request.source_split for request in requests}
    indexed_lookup = build_indexed_lookup(
        dataset=dataset,
        indexed_group_dir=indexed_group_dir,
        source_splits=source_splits,
    )

    pending_records: list[PendingRecord] = []
    missing: list[str] = []
    for request in requests:
        indexed_row = indexed_lookup.get((request.source_split, request.sample_id))
        if indexed_row is None:
            missing.append(f"{request.source_split}:{request.sample_id}")
            continue
        pending_records.append(PendingRecord(request=request, indexed_row=indexed_row))
    if missing:
        raise KeyError(f"{low_group_dir.name} 有样本无法回查坐标，示例：{missing[:5]}")

    sort_key = (
        (lambda item: (
            str(resolve_repo_path(str(item.indexed_row.get("reference_fasta", "")).strip())),
            str(item.indexed_row.get("reference_chromosome", "")).strip(),
            int(str(item.indexed_row.get("expanded_start_0based", "0")).strip() or "0"),
            item.request.order,
        ))
        if dataset == "nt"
        else (lambda item: (
            str(resolve_repo_path(str(item.indexed_row.get("reference_path", "")).strip())),
            str(item.indexed_row.get("region", "")).strip(),
            int(str(item.indexed_row.get("expanded_start_0based", "0")).strip() or "0"),
            item.request.order,
        ))
    )
    fetch_sequence = fetch_nt_sequence if dataset == "nt" else fetch_gb_sequence
    sequence_by_order: dict[int, str] = {}
    sorted_pending = sorted(pending_records, key=sort_key)
    loader_cache: dict[Path, ReferenceContigLoader] = {}
    iterator = progress_iter(
        sorted_pending,
        total=len(sorted_pending),
        desc=f"Materialize {dataset} {low_group_dir.name}",
        enabled=progress,
    )
    for pending in iterator:
        sequence_by_order[pending.request.order] = fetch_sequence(
            pending,
            materialization_dir,
            loader_cache,
        )

    raw_rows: list[dict[str, object]] = []
    for raw_index, pending in enumerate(sorted(pending_records, key=lambda item: item.request.order), start=1):
        raw_rows.append(
            build_raw_record(
                dataset=dataset,
                raw_index=raw_index,
                pending=pending,
                sequence=sequence_by_order[pending.request.order],
            )
        )

    output_group_raw_dir = output_root / low_group_dir.name / "raw"
    if output_group_raw_dir.exists():
        if not overwrite:
            raise FileExistsError(f"输出目录已存在：{output_group_raw_dir}。使用 --overwrite 覆盖。")
        shutil.rmtree(output_group_raw_dir)
    output_group_raw_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "index",
        "seq",
        "label",
        "split",
        "name",
        "source_split",
        "source_index",
        "sample_id",
        "sequence_length",
    ]
    if dataset == "genomic_benchmarks":
        fieldnames.extend(["label_name", "reference_path", "region", "strand"])
    else:
        fieldnames.extend(["reference_fasta", "reference_chromosome"])
    fieldnames.extend(["expanded_start_0based", "expanded_end_0based"])

    csv_path = output_group_raw_dir / f"{low_group_dir.name}.csv"
    parquet_path = output_group_raw_dir / f"{low_group_dir.name}.parquet"
    raw_csv_path: Path | None = None
    raw_parquet_path: Path | None = None
    if output_format in {"csv", "both"}:
        write_csv_rows(csv_path, fieldnames, raw_rows)
        raw_csv_path = csv_path
    if output_format in {"parquet", "both"}:
        write_parquet_rows(
            parquet_path,
            fieldnames,
            raw_rows,
            compression=parquet_compression,
            progress=progress,
        )
        raw_parquet_path = parquet_path
    write_statistic(output_group_raw_dir / "statistic.txt", low_group_dir.name, raw_rows)
    write_label_mapper(output_group_raw_dir / "label_mapper.json", raw_rows)
    write_meta(
        output_group_raw_dir / "meta.json",
        dataset=dataset,
        group=low_group_dir.name,
        rows=raw_rows,
        indexed_group_dir=indexed_group_dir,
        low_group_dir=low_group_dir,
        materialization_dir=materialization_dir,
        output_format=output_format,
        parquet_compression=parquet_compression,
        raw_csv_path=raw_csv_path,
        raw_parquet_path=raw_parquet_path,
    )
    split_counts = Counter(str(row["split"]) for row in raw_rows)
    output_paths = [str(path) for path in (raw_csv_path, raw_parquet_path) if path is not None]
    print(
        f"[{dataset}] {low_group_dir.name}: wrote {len(raw_rows)} rows "
        f"(train={split_counts.get('train', 0)}, valid={split_counts.get('valid', 0)}, "
        f"test={split_counts.get('test', 0)}) -> {', '.join(output_paths)}",
        flush=True,
    )
    return {
        "dataset": dataset,
        "group": low_group_dir.name,
        "raw_csv": str(raw_csv_path) if raw_csv_path is not None else "",
        "raw_parquet": str(raw_parquet_path) if raw_parquet_path is not None else "",
        "meta_json": str(output_group_raw_dir / "meta.json"),
        "rows": len(raw_rows),
        "train": split_counts.get("train", 0),
        "valid": split_counts.get("valid", 0),
        "test": split_counts.get("test", 0),
    }


def materialize_dataset(
    *,
    dataset: str,
    indexed_root: Path,
    low_split_root: Path,
    output_root: Path,
    selected_groups: set[str] | None,
    materialization_dir: Path,
    group_workers: int,
    overwrite: bool,
    progress: bool,
    output_format: str,
    parquet_compression: str,
) -> list[dict[str, object]]:
    group_dirs = list_group_dirs(low_split_root, selected_groups, f"{dataset} low_similarity_split")
    manifest_rows: list[dict[str, object]] = []

    if group_workers <= 1 or len(group_dirs) <= 1:
        group_iter = progress_iter(
            group_dirs,
            total=len(group_dirs),
            desc=f"Groups {dataset}",
            enabled=progress,
        )
        for group_dir in group_iter:
            manifest_rows.append(
                materialize_group(
                    dataset=dataset,
                    low_group_dir=group_dir,
                    indexed_root=indexed_root,
                    output_root=output_root,
                    materialization_dir=materialization_dir,
                    overwrite=overwrite,
                    progress=progress,
                    output_format=output_format,
                    parquet_compression=parquet_compression,
                )
            )
    else:
        worker_count = min(group_workers, len(group_dirs))
        print(f"[parallel] {dataset}: group_workers={worker_count}, groups={len(group_dirs)}", flush=True)
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(
                    materialize_group,
                    dataset=dataset,
                    low_group_dir=group_dir,
                    indexed_root=indexed_root,
                    output_root=output_root,
                    materialization_dir=materialization_dir,
                    overwrite=overwrite,
                    progress=False,
                    output_format=output_format,
                    parquet_compression=parquet_compression,
                )
                for group_dir in group_dirs
            ]
            future_iter = progress_iter(
                as_completed(futures),
                total=len(futures),
                desc=f"Groups {dataset}",
                enabled=progress,
            )
            for future in future_iter:
                manifest_rows.append(future.result())

        manifest_rows.sort(key=lambda row: str(row["group"]))

    manifest_path = output_root / "materialized_raw_manifest.csv"
    write_csv_rows(
        manifest_path,
        fieldnames=[
            "dataset",
            "group",
            "raw_csv",
            "raw_parquet",
            "meta_json",
            "rows",
            "train",
            "valid",
            "test",
        ],
        rows=manifest_rows,
    )
    print(f"[done] {dataset} manifest -> {manifest_path}", flush=True)
    return manifest_rows


def main() -> None:
    args = parse_args()
    if args.group_workers < 1:
        raise ValueError(f"--group-workers 必须 >= 1，收到 {args.group_workers}")
    selected_groups = parse_csv_list(args.groups)
    progress = not args.no_progress
    if progress and tqdm is None:
        print("[warning] tqdm 未安装，进度条降级为普通输出。", flush=True)

    tasks: list[tuple[str, Path, Path, Path]] = []
    if args.dataset in {"nt", "all"}:
        nt_output_root = args.output_root / "nt" if args.dataset == "all" else args.output_root
        tasks.append(("nt", args.nt_indexed_root, args.nt_low_split_root, nt_output_root))
    if args.dataset in {"genomic_benchmarks", "all"}:
        gb_output_root = args.output_root / "genomic_benchmarks" if args.dataset == "all" else args.output_root
        tasks.append(("genomic_benchmarks", args.gb_indexed_root, args.gb_low_split_root, gb_output_root))

    for dataset, indexed_root, low_split_root, output_root in tasks:
        materialize_dataset(
            dataset=dataset,
            indexed_root=indexed_root,
            low_split_root=low_split_root,
            output_root=output_root,
            selected_groups=selected_groups,
            materialization_dir=args.reference_materialization_dir,
            group_workers=args.group_workers,
            overwrite=args.overwrite,
            progress=progress,
            output_format=args.output_format,
            parquet_compression=args.parquet_compression,
        )


if __name__ == "__main__":
    main()
