"""Dataset readers shared by Dashing ranking and split materialization.

This module intentionally contains no similarity implementation.  It only turns
the NT/Genomic-Benchmarks source layouts into sequence records and provides the
small FASTA random-access helper needed when materializing indexed coordinates.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import os
import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Iterator, Sequence, TextIO


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GENOMIC_BENCHMARKS_ROOT = (
    REPO_ROOT / "data/processed_download/ood_imageDNA/original_indexed_sequences/genomic_benchmarks"
)
DEFAULT_NT_ROOT = REPO_ROOT / "data/raw_download/nucleotide_transformer_downstream_tasks_revised"
DEFAULT_REFERENCE_MATERIALIZATION_DIR = REPO_ROOT / "data/reference_materialization_cache"
_NT_REFERENCE_INDEX_CACHE: dict[
    Path, tuple[dict[str, "ReferenceContigInfo"], dict[str, str]]
] = {}


@dataclass(frozen=True)
class SequenceRecord:
    dataset: str
    group: str
    sample_id: str
    split: str
    sequence: str
    label_raw: str | None
    label_numeric: float | None


@dataclass(frozen=True)
class ReferenceContigInfo:
    canonical_name: str
    header_text: str
    sequence_offset: int
    next_header_offset: int
    aliases: tuple[str, ...]


def parse_csv_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def group_filter(raw: str) -> set[str] | None:
    values = parse_csv_list(raw)
    return set(values) if values else None


def resolve_num_workers(num_workers: int) -> int:
    return (os.cpu_count() or 1) if num_workers <= 0 else num_workers


def normalize_label(value: object) -> tuple[str | None, float | None]:
    if value is None:
        return None, None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None, None
    try:
        return text, float(text)
    except ValueError:
        return text, None


def maybe_truncate(sequence: str, max_bases: int) -> str:
    return sequence[:max_bases] if max_bases > 0 else sequence


def resolve_repo_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def reverse_complement(sequence: str) -> str:
    return sequence.translate(str.maketrans("ACGTNacgtn", "TGCANtgcan"))[::-1]


def open_text(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, mode="rt", encoding="utf-8")
    return path.open(mode="r", encoding="utf-8")


def iter_fasta_records(path: Path) -> Iterator[tuple[str, str]]:
    with open_text(path) as handle:
        header: str | None = None
        sequence_parts: list[str] = []
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(sequence_parts).upper()
                header = line[1:].strip()
                sequence_parts = []
            else:
                sequence_parts.append(line)
        if header is not None:
            yield header, "".join(sequence_parts).upper()


def iter_csv_rows(path: Path) -> Iterator[dict[str, str]]:
    with path.open(mode="r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            yield {str(key): str(value) for key, value in row.items()}


def read_csv_rows(path: Path, limit: int = 0) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in iter_csv_rows(path):
        rows.append(row)
        if limit > 0 and len(rows) >= limit:
            break
    return rows


def _resolve_group_dirs(
    root: Path,
    selected_groups: set[str] | None,
    *,
    dataset_name: str,
    supported_suffixes: Sequence[str],
) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Missing {dataset_name} root: {root}")

    def has_split(group_dir: Path) -> bool:
        return any(
            (group_dir / f"{split}{suffix}").exists()
            for split in ("train", "valid", "test")
            for suffix in supported_suffixes
        )

    available = [path for path in sorted(root.iterdir()) if path.is_dir() and has_split(path)]
    if selected_groups is None:
        return available
    resolved = [path for path in available if path.name in selected_groups]
    missing = sorted(selected_groups - {path.name for path in resolved})
    if missing:
        raise ValueError(
            f"Unknown groups under {dataset_name}: {missing}. "
            f"Available groups: {[path.name for path in available]}"
        )
    return resolved


def build_gb_reference_aliases(record_id: str) -> tuple[str, ...]:
    aliases = {record_id, record_id.split(".", 1)[0]}
    for alias in tuple(aliases):
        aliases.add(alias[3:] if alias.startswith("chr") else f"chr{alias}")
    if aliases.intersection({"MT", "M", "chrM", "chrMT", "mitochondrion_genome"}):
        aliases.update({"MT", "M", "chrM", "chrMT", "mitochondrion_genome"})
    return tuple(sorted(alias for alias in aliases if alias))


def load_needed_gb_regions(reference_path: Path, regions: Sequence[str]) -> dict[str, str]:
    remaining = set(regions)
    loaded: dict[str, str] = {}
    for header, sequence in iter_fasta_records(reference_path):
        matched = remaining.intersection(build_gb_reference_aliases(header.split()[0]))
        for alias in matched:
            loaded[alias] = sequence
            remaining.remove(alias)
        if not remaining:
            break
    if remaining:
        raise KeyError(f"Reference {reference_path} is missing regions: {sorted(remaining)[:10]}")
    return loaded


def materialize_gzip_fasta(reference_path: Path, materialization_dir: Path) -> Path:
    if reference_path.suffix != ".gz":
        return reference_path
    sibling = reference_path.with_suffix("")
    if sibling.exists():
        return sibling
    materialization_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(str(reference_path.resolve()).encode("utf-8")).hexdigest()[:12]
    destination = materialization_dir / f"{digest}_{reference_path.name[:-3]}"
    if destination.exists():
        return destination
    print(f"[reference] materializing {reference_path} -> {destination}", flush=True)
    with tempfile.NamedTemporaryFile(dir=materialization_dir, suffix=".tmp", delete=False) as handle:
        temporary_path = Path(handle.name)
    try:
        with gzip.open(reference_path, mode="rb") as source, temporary_path.open(mode="wb") as target:
            shutil.copyfileobj(source, target, length=1024 * 1024)
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination


def _build_reference_aliases(canonical_name: str, header_text: str) -> tuple[str, ...]:
    aliases = {canonical_name}
    lowered = header_text.lower()
    if "chromosome" in lowered:
        words = header_text.replace(",", " ").split()
        for index, word in enumerate(words[:-1]):
            if word.lower() == "chromosome":
                token = words[index + 1].strip(".,;:").upper()
                if token == "MT":
                    token = "M"
                aliases.update({token, f"chr{token}"})
                break
    if "mitochondrion" in lowered or "mitochondrial" in lowered:
        aliases.update({"M", "MT", "chrM", "chrMT"})
    return tuple(sorted(alias for alias in aliases if alias))


def _build_reference_index(
    reference_fasta: Path,
) -> tuple[dict[str, ReferenceContigInfo], dict[str, str]]:
    contigs: dict[str, ReferenceContigInfo] = {}
    aliases: dict[str, str] = {}
    current: tuple[str, str, int] | None = None

    def finish(next_header_offset: int) -> None:
        if current is None:
            return
        canonical_name, header_text, sequence_offset = current
        record_aliases = _build_reference_aliases(canonical_name, header_text)
        info = ReferenceContigInfo(
            canonical_name=canonical_name,
            header_text=header_text,
            sequence_offset=sequence_offset,
            next_header_offset=next_header_offset,
            aliases=record_aliases,
        )
        contigs[canonical_name] = info
        for alias in record_aliases:
            aliases.setdefault(alias, canonical_name)

    with reference_fasta.open(mode="rb") as handle:
        while True:
            line_offset = handle.tell()
            raw_line = handle.readline()
            if not raw_line:
                finish(handle.tell())
                break
            if raw_line.startswith(b">"):
                finish(line_offset)
                header_text = raw_line[1:].decode("utf-8").strip()
                current = (header_text.split()[0], header_text, handle.tell())
    return contigs, aliases


class ReferenceContigLoader:
    def __init__(
        self,
        reference_fasta: Path,
        index: dict[str, ReferenceContigInfo],
        alias_to_canonical: dict[str, str],
        cache_size: int = 1,
    ) -> None:
        self.reference_fasta = reference_fasta
        self.index = index
        self.alias_to_canonical = alias_to_canonical
        self.cache_size = max(1, cache_size)
        self._cache: dict[str, str] = {}
        self._cache_order: list[str] = []

    def _resolve(self, chromosome: str) -> ReferenceContigInfo:
        candidates = [chromosome]
        candidates.append(chromosome[3:] if chromosome.startswith("chr") else f"chr{chromosome}")
        canonical = next(
            (
                self.alias_to_canonical[item]
                for item in candidates
                if item in self.alias_to_canonical
            ),
            None,
        )
        if canonical is None:
            raise KeyError(f"Missing chromosome alias {chromosome!r} in {self.reference_fasta}")
        return self.index[canonical]

    def load_sequence(self, chromosome: str) -> tuple[ReferenceContigInfo, str]:
        info = self._resolve(chromosome)
        cached = self._cache.get(info.canonical_name)
        if cached is not None:
            self._cache_order.remove(info.canonical_name)
            self._cache_order.append(info.canonical_name)
            return info, cached
        with self.reference_fasta.open(mode="rb") as handle:
            handle.seek(info.sequence_offset)
            raw = handle.read(info.next_header_offset - info.sequence_offset)
        sequence = b"".join(line.strip() for line in raw.splitlines() if line).decode("utf-8").upper()
        self._cache[info.canonical_name] = sequence
        self._cache_order.append(info.canonical_name)
        if len(self._cache_order) > self.cache_size:
            oldest = self._cache_order.pop(0)
            self._cache.pop(oldest, None)
        return info, sequence


def get_nt_reference_loader(reference_fasta: Path, materialization_dir: Path) -> ReferenceContigLoader:
    materialized = materialize_gzip_fasta(reference_fasta, materialization_dir)
    cached = _NT_REFERENCE_INDEX_CACHE.get(materialized)
    if cached is None:
        cached = _build_reference_index(materialized)
        _NT_REFERENCE_INDEX_CACHE[materialized] = cached
    return ReferenceContigLoader(materialized, cached[0], cached[1])


def _load_gb_records(
    root: Path,
    split: str,
    selected_groups: set[str] | None,
    limit_per_group: int,
    max_bases: int,
) -> list[SequenceRecord]:
    group_dirs = _resolve_group_dirs(
        root,
        selected_groups,
        dataset_name="genomic_benchmarks",
        supported_suffixes=(".csv",),
    )
    needed: DefaultDict[Path, set[str]] = defaultdict(set)
    pending: list[dict[str, object]] = []
    for group_dir in group_dirs:
        csv_path = group_dir / f"{split}.csv"
        if not csv_path.exists():
            continue
        for row in read_csv_rows(csv_path, limit_per_group):
            start_text = row.get("expanded_start_0based", "").strip()
            end_text = row.get("expanded_end_0based", "").strip()
            region = row.get("region", "").strip()
            reference_path = resolve_repo_path(row.get("reference_path", "").strip())
            if not start_text or not end_text or not region:
                continue
            if not reference_path.exists():
                raise FileNotFoundError(f"Missing reference FASTA: {reference_path}")
            class_name = row.get("class_name", "").strip()
            row_index = row.get("index", "").strip()
            pending.append(
                {
                    "group": group_dir.name,
                    "sample_id": (
                        f"{group_dir.name}:{class_name}:{row_index}"
                        if class_name
                        else f"{group_dir.name}:{row_index}"
                    ),
                    "label": row.get("label"),
                    "reference_path": reference_path,
                    "region": region,
                    "strand": row.get("strand", "+").strip() or "+",
                    "start": int(start_text),
                    "end": int(end_text),
                }
            )
            needed[reference_path].add(region)
    loaded = {path: load_needed_gb_regions(path, sorted(regions)) for path, regions in needed.items()}
    records: list[SequenceRecord] = []
    for row in pending:
        sequence = loaded[row["reference_path"]][str(row["region"])]
        start, end = int(row["start"]), int(row["end"])
        if start < 0 or start >= end or end > len(sequence):
            continue
        sliced = sequence[start:end]
        if row["strand"] == "-":
            sliced = reverse_complement(sliced)
        label_raw, label_numeric = normalize_label(row["label"])
        records.append(
            SequenceRecord(
                "genomic_benchmarks",
                str(row["group"]),
                str(row["sample_id"]),
                split,
                maybe_truncate(sliced, max_bases),
                label_raw,
                label_numeric,
            )
        )
    return records


def _raw_sample_id(group: str, split: str, index: int, name: str) -> str:
    return f"{group}:{split}:{index:09d}:{name}"


def _load_nt_fasta(
    group_dir: Path,
    split: str,
    limit: int,
    max_bases: int,
) -> list[SequenceRecord]:
    records: list[SequenceRecord] = []
    for index, (header, sequence) in enumerate(iter_fasta_records(group_dir / f"{split}.fna"), 1):
        label_raw, label_numeric = normalize_label(
            header.rsplit("|", 1)[1] if "|" in header else None
        )
        name = header or f"{split}_{index:09d}"
        records.append(
            SequenceRecord(
                "nt",
                group_dir.name,
                _raw_sample_id(group_dir.name, split, index, name),
                split,
                maybe_truncate(sequence, max_bases),
                label_raw,
                label_numeric,
            )
        )
        if limit > 0 and len(records) >= limit:
            break
    return records


def _load_nt_parquet(
    group_dir: Path,
    split: str,
    limit: int,
    max_bases: int,
) -> list[SequenceRecord]:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - dependency is documented
        raise ImportError("Reading NT parquet requires pandas/pyarrow.") from exc
    frame = pd.read_parquet(group_dir / f"{split}.parquet")
    sequence_column = (
        "sequence"
        if "sequence" in frame.columns
        else "seq" if "seq" in frame.columns else None
    )
    if sequence_column is None:
        raise ValueError(f"NT parquet has no sequence/seq column: {group_dir / f'{split}.parquet'}")
    if limit > 0:
        frame = frame.head(limit)
    records: list[SequenceRecord] = []
    for index, row in enumerate(frame.itertuples(index=False), 1):
        label_raw, label_numeric = normalize_label(getattr(row, "label", None))
        name = str(getattr(row, "name", "")).strip() or f"{split}_{index:09d}"
        sequence = str(getattr(row, sequence_column)).upper()
        records.append(
            SequenceRecord(
                "nt",
                group_dir.name,
                _raw_sample_id(group_dir.name, split, index, name),
                split,
                maybe_truncate(sequence, max_bases),
                label_raw,
                label_numeric,
            )
        )
    return records


def _load_nt_indexed(
    group_dirs: Sequence[Path],
    split: str,
    limit: int,
    max_bases: int,
    materialization_dir: Path,
) -> list[SequenceRecord]:
    loaders: dict[Path, ReferenceContigLoader] = {}
    records: list[SequenceRecord] = []
    for group_dir in group_dirs:
        csv_path = group_dir / f"{split}.csv"
        if not csv_path.exists():
            continue
        for row in read_csv_rows(csv_path, limit):
            reference = resolve_repo_path(row.get("reference_fasta", "").strip())
            chromosome = row.get("reference_chromosome", "").strip()
            start_text = row.get("expanded_start_0based", "").strip()
            end_text = row.get("expanded_end_0based", "").strip()
            if not chromosome or not start_text or not end_text:
                continue
            loader = loaders.get(reference)
            if loader is None:
                loader = get_nt_reference_loader(reference, materialization_dir)
                loaders[reference] = loader
            _, chromosome_sequence = loader.load_sequence(chromosome)
            start, end = int(start_text), int(end_text)
            if start < 0 or start >= end or end > len(chromosome_sequence):
                continue
            label_raw, label_numeric = normalize_label(row.get("label"))
            records.append(
                SequenceRecord(
                    "nt",
                    group_dir.name,
                    f"{group_dir.name}:{row.get('index', '')}",
                    split,
                    maybe_truncate(chromosome_sequence[start:end], max_bases),
                    label_raw,
                    label_numeric,
                )
            )
    return records


def _load_nt_records(
    root: Path,
    split: str,
    selected_groups: set[str] | None,
    limit_per_group: int,
    max_bases: int,
    materialization_dir: Path,
) -> list[SequenceRecord]:
    group_dirs = _resolve_group_dirs(
        root,
        selected_groups,
        dataset_name="nt",
        supported_suffixes=(".csv", ".parquet", ".fna"),
    )
    indexed = [group for group in group_dirs if (group / f"{split}.csv").exists()]
    records = _load_nt_indexed(indexed, split, limit_per_group, max_bases, materialization_dir)
    for group_dir in group_dirs:
        if group_dir in indexed:
            continue
        fasta_path = group_dir / f"{split}.fna"
        parquet_path = group_dir / f"{split}.parquet"
        if fasta_path.exists():
            records.extend(_load_nt_fasta(group_dir, split, limit_per_group, max_bases))
        elif parquet_path.exists():
            records.extend(_load_nt_parquet(group_dir, split, limit_per_group, max_bases))
    return records


def load_merged_dataset_records(
    dataset_name: str,
    nt_root: Path,
    genomic_benchmarks_root: Path,
    selected_groups: set[str] | None,
    limit_per_group: int,
    max_bases: int,
    reference_materialization_dir: Path,
) -> list[SequenceRecord]:
    records: list[SequenceRecord] = []
    for split in ("train", "valid", "test"):
        if dataset_name == "nt":
            records.extend(
                _load_nt_records(
                    nt_root,
                    split,
                    selected_groups,
                    limit_per_group,
                    max_bases,
                    reference_materialization_dir,
                )
            )
        elif dataset_name == "genomic_benchmarks":
            records.extend(
                _load_gb_records(
                    genomic_benchmarks_root,
                    split,
                    selected_groups,
                    limit_per_group,
                    max_bases,
                )
            )
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
    return records


def build_group_map(records: Sequence[SequenceRecord]) -> dict[str, list[SequenceRecord]]:
    grouped: dict[str, list[SequenceRecord]] = defaultdict(list)
    for record in records:
        grouped[record.group].append(record)
    return dict(grouped)
