from __future__ import annotations

import argparse
import csv
import hashlib
import math
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable


DEFAULT_GROUP_COLUMN = "group"
DEFAULT_SCORE_COLUMN = "primary_similarity_score"
DEFAULT_LABEL_COLUMN = "sample_label_raw"
DEFAULT_SAMPLE_ID_COLUMN = "sample_id"
DEFAULT_ASSIGNED_SPLIT_COLUMN = "assigned_split"
DEFAULT_HOLDOUT_FRACTION = 0.2
DEFAULT_SEED = 42
DEFAULT_MANIFEST_NAME = "group_split_manifest.csv"
DEFAULT_SUMMARY_NAME = "group_split_summary.csv"
DEFAULT_PROGRESS_INTERVAL = 10_000
SAFE_FILENAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split similarity-ranking rows within each group. Rows are sorted by "
            "primary similarity score ascending; the least-similar fraction is split "
            "into valid/test stratified by label, and the remaining rows become train."
        )
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input-csv",
        type=Path,
        help="Merged sample_similarity_ranking.csv containing all groups.",
    )
    input_group.add_argument(
        "--input-dir",
        type=Path,
        help="Directory containing per-group CSV files, for example the by_group output.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="Output directory. One folder per group will be created under this root.",
    )
    parser.add_argument(
        "--group-column",
        default=DEFAULT_GROUP_COLUMN,
        help=f"Column used as group name. Default: {DEFAULT_GROUP_COLUMN}.",
    )
    parser.add_argument(
        "--score-column",
        default=DEFAULT_SCORE_COLUMN,
        help=f"Column used for low-to-high similarity sorting. Default: {DEFAULT_SCORE_COLUMN}.",
    )
    parser.add_argument(
        "--label-column",
        default=DEFAULT_LABEL_COLUMN,
        help=f"Column used for valid/test stratification. Default: {DEFAULT_LABEL_COLUMN}.",
    )
    parser.add_argument(
        "--sample-id-column",
        default=DEFAULT_SAMPLE_ID_COLUMN,
        help=f"Column used as deterministic tie-breaker. Default: {DEFAULT_SAMPLE_ID_COLUMN}.",
    )
    parser.add_argument(
        "--assigned-split-column",
        default=DEFAULT_ASSIGNED_SPLIT_COLUMN,
        help=(
            "New column written to each output CSV with values train/valid/test. "
            f"Default: {DEFAULT_ASSIGNED_SPLIT_COLUMN}."
        ),
    )
    parser.add_argument(
        "--holdout-fraction",
        type=float,
        default=DEFAULT_HOLDOUT_FRACTION,
        help=f"Least-similar fraction reserved for valid+test. Default: {DEFAULT_HOLDOUT_FRACTION}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed used when splitting each label into valid/test. Default: {DEFAULT_SEED}.",
    )
    parser.add_argument(
        "--manifest-name",
        default=DEFAULT_MANIFEST_NAME,
        help=f"Manifest filename written inside --output-root. Default: {DEFAULT_MANIFEST_NAME}.",
    )
    parser.add_argument(
        "--summary-name",
        default=DEFAULT_SUMMARY_NAME,
        help=f"Summary filename written inside --output-root. Default: {DEFAULT_SUMMARY_NAME}.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output folders and root manifest/summary.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=DEFAULT_PROGRESS_INTERVAL,
        help=f"Refresh row progress every N rows. Default: {DEFAULT_PROGRESS_INTERVAL}.",
    )
    return parser.parse_args()


def print_progress(prefix: str, current: int, total: int, width: int = 30) -> None:
    if total <= 0:
        sys.stdout.write(f"\r{prefix}: {current}")
        sys.stdout.flush()
        return

    current = min(current, total)
    filled = int(width * current / total)
    bar = "#" * filled + "-" * (width - filled)
    percent = current * 100.0 / total
    sys.stdout.write(f"\r{prefix}: [{bar}] {current}/{total} ({percent:6.2f}%)")
    sys.stdout.flush()


def count_csv_data_rows(path: Path, progress: bool) -> int:
    total_bytes = path.stat().st_size
    if total_bytes == 0:
        return 0

    line_count = 0
    bytes_read = 0
    last_byte = b""
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            bytes_read += len(chunk)
            line_count += chunk.count(b"\n")
            last_byte = chunk[-1:]
            if progress:
                print_progress("[count] input bytes", bytes_read, total_bytes)

    if last_byte and last_byte != b"\n":
        line_count += 1

    if progress:
        print_progress("[count] input bytes", total_bytes, total_bytes)
        sys.stdout.write("\n")
    return max(0, line_count - 1)


def make_safe_name(value: str) -> str:
    normalized = SAFE_FILENAME_PATTERN.sub("_", value.strip())
    normalized = normalized.strip("._")
    return normalized or "missing_group"


def validate_fraction(value: float) -> None:
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"--holdout-fraction must be between 0 and 1, got {value}")


def parse_score(value: str | None) -> float:
    if value is None:
        return math.nan
    stripped = value.strip()
    if not stripped:
        return math.nan
    try:
        return float(stripped)
    except ValueError:
        return math.nan


def similarity_sort_key(
    item: tuple[int, dict[str, str]],
    score_column: str,
    sample_id_column: str,
) -> tuple[bool, float, str, int]:
    original_index, row = item
    score = parse_score(row.get(score_column))
    if math.isnan(score):
        return (True, math.inf, str(row.get(sample_id_column, "")), original_index)
    return (False, score, str(row.get(sample_id_column, "")), original_index)


def deterministic_group_seed(seed: int, group_name: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{group_name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def require_columns(fieldnames: Iterable[str], required_columns: Iterable[str], csv_path: Path) -> None:
    available = list(fieldnames)
    missing = [column for column in required_columns if column not in available]
    if missing:
        raise ValueError(
            f"Missing required columns in {csv_path}: {missing}. "
            f"Available columns: {', '.join(available)}"
        )


def read_rows_from_csv(
    csv_path: Path,
    group_column: str,
    required_columns: list[str],
    progress: bool,
    progress_interval: int,
) -> tuple[list[str], dict[str, list[dict[str, str]]]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {csv_path}")
    if not csv_path.is_file():
        raise ValueError(f"Input path is not a file: {csv_path}")

    total_rows = count_csv_data_rows(csv_path, progress=progress)
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    rows_read = 0
    progress_interval = max(1, progress_interval)

    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Input CSV has no header: {csv_path}")
        require_columns(reader.fieldnames, required_columns, csv_path)
        fieldnames = list(reader.fieldnames)

        for row in reader:
            group_name = str(row.get(group_column, "")).strip() or "missing_group"
            groups[group_name].append(row)
            rows_read += 1

            if progress and rows_read % progress_interval == 0:
                print_progress("[read] rows", rows_read, total_rows)

    if progress:
        print_progress("[read] rows", rows_read, total_rows)
        sys.stdout.write("\n")
    return fieldnames, dict(groups)


def list_input_dir_csvs(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise ValueError(f"Input path is not a directory: {input_dir}")

    csv_paths = [
        path
        for path in sorted(input_dir.glob("*.csv"))
        if path.name not in {DEFAULT_MANIFEST_NAME, DEFAULT_SUMMARY_NAME, "group_split_manifest.csv"}
    ]
    if not csv_paths:
        raise ValueError(f"No CSV files found under {input_dir}")
    return csv_paths


def read_groups_from_input_dir(
    input_dir: Path,
    group_column: str,
    required_columns: list[str],
    progress: bool,
    progress_interval: int,
) -> tuple[list[str], dict[str, list[dict[str, str]]]]:
    all_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    canonical_fieldnames: list[str] | None = None
    csv_paths = list_input_dir_csvs(input_dir)

    for index, csv_path in enumerate(csv_paths, start=1):
        if progress:
            print_progress("[files] group csvs", index - 1, len(csv_paths))
            sys.stdout.write(f"  reading {csv_path.name}\n")

        fieldnames, groups = read_rows_from_csv(
            csv_path=csv_path,
            group_column=group_column,
            required_columns=required_columns,
            progress=progress,
            progress_interval=progress_interval,
        )
        if canonical_fieldnames is None:
            canonical_fieldnames = fieldnames
        elif fieldnames != canonical_fieldnames:
            raise ValueError(
                f"CSV header mismatch in {csv_path}. "
                "All input files must have the same columns."
            )

        for group_name, rows in groups.items():
            all_groups[group_name].extend(rows)

        if progress:
            print_progress("[files] group csvs", index, len(csv_paths))
            sys.stdout.write("\n")

    if canonical_fieldnames is None:
        raise ValueError(f"No readable CSV files found under {input_dir}")
    return canonical_fieldnames, dict(all_groups)


def split_group_rows(
    group_name: str,
    rows: list[dict[str, str]],
    score_column: str,
    label_column: str,
    sample_id_column: str,
    holdout_fraction: float,
    seed: int,
) -> tuple[dict[str, list[dict[str, str]]], dict[str, int]]:
    sorted_rows = sorted(
        enumerate(rows),
        key=lambda item: similarity_sort_key(item, score_column, sample_id_column),
    )
    ordered_rows = [row for _, row in sorted_rows]
    total_count = len(ordered_rows)
    holdout_count = int(math.ceil(total_count * holdout_fraction))
    holdout_rows = ordered_rows[:holdout_count]
    train_rows = ordered_rows[holdout_count:]

    rng = random.Random(deterministic_group_seed(seed, group_name))
    holdout_by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in holdout_rows:
        label = str(row.get(label_column, "")).strip() or "missing_label"
        holdout_by_label[label].append(row)

    valid_rows: list[dict[str, str]] = []
    test_rows: list[dict[str, str]] = []
    for label in sorted(holdout_by_label):
        label_rows = list(holdout_by_label[label])
        rng.shuffle(label_rows)
        valid_count = len(label_rows) // 2
        valid_rows.extend(label_rows[:valid_count])
        test_rows.extend(label_rows[valid_count:])

    outputs = {
        "train": train_rows,
        "valid": valid_rows,
        "test": test_rows,
    }
    stats = {
        "total_rows": total_count,
        "holdout_rows": len(holdout_rows),
        "train_rows": len(train_rows),
        "valid_rows": len(valid_rows),
        "test_rows": len(test_rows),
        "labels_in_holdout": len(holdout_by_label),
    }
    return outputs, stats


def prepare_group_dir(output_root: Path, group_name: str, used_names: set[str], overwrite: bool) -> Path:
    base_name = make_safe_name(group_name)
    safe_name = base_name
    suffix = 2
    while safe_name in used_names:
        safe_name = f"{base_name}_{suffix}"
        suffix += 1
    used_names.add(safe_name)

    group_dir = output_root / safe_name
    if group_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output group directory already exists: {group_dir}. "
                "Use --overwrite to replace it."
            )
        shutil.rmtree(group_dir)
    group_dir.mkdir(parents=True, exist_ok=False)
    return group_dir


def write_split_csv(
    path: Path,
    rows: list[dict[str, str]],
    fieldnames: list[str],
    assigned_split_column: str,
    assigned_split: str,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            output_row = dict(row)
            output_row[assigned_split_column] = assigned_split
            writer.writerow(output_row)


def write_root_csv(
    path: Path,
    rows: list[dict[str, object]],
    fieldnames: list[str],
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output file already exists: {path}. Use --overwrite to replace it.")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def split_groups(
    groups: dict[str, list[dict[str, str]]],
    input_fieldnames: list[str],
    output_root: Path,
    group_column: str,
    score_column: str,
    label_column: str,
    sample_id_column: str,
    assigned_split_column: str,
    holdout_fraction: float,
    seed: int,
    manifest_name: str,
    summary_name: str,
    overwrite: bool,
    progress: bool,
) -> None:
    validate_fraction(holdout_fraction)
    if not groups:
        raise ValueError("No groups were loaded from input.")

    if output_root.exists() and not output_root.is_dir():
        raise ValueError(f"--output-root exists but is not a directory: {output_root}")
    if output_root.exists() and overwrite:
        shutil.rmtree(output_root)
    elif output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"--output-root is not empty: {output_root}. Use --overwrite to replace the whole split output."
        )
    output_root.mkdir(parents=True, exist_ok=True)

    required_columns = [group_column, score_column, label_column]
    require_columns(input_fieldnames, required_columns, Path("<input>"))
    output_fieldnames = list(input_fieldnames)
    if assigned_split_column not in output_fieldnames:
        output_fieldnames.append(assigned_split_column)

    manifest_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    used_group_dir_names: set[str] = set()
    sorted_group_items = sorted(groups.items())

    for index, (group_name, rows) in enumerate(sorted_group_items, start=1):
        if progress:
            print_progress("[split] groups", index - 1, len(sorted_group_items))
            sys.stdout.write(f"  processing {group_name}\n")

        split_rows, stats = split_group_rows(
            group_name=group_name,
            rows=rows,
            score_column=score_column,
            label_column=label_column,
            sample_id_column=sample_id_column,
            holdout_fraction=holdout_fraction,
            seed=seed,
        )
        group_dir = prepare_group_dir(
            output_root=output_root,
            group_name=group_name,
            used_names=used_group_dir_names,
            overwrite=overwrite,
        )

        train_path = group_dir / "train.csv"
        valid_path = group_dir / "valid.csv"
        test_path = group_dir / "test.csv"
        write_split_csv(train_path, split_rows["train"], output_fieldnames, assigned_split_column, "train")
        write_split_csv(valid_path, split_rows["valid"], output_fieldnames, assigned_split_column, "valid")
        write_split_csv(test_path, split_rows["test"], output_fieldnames, assigned_split_column, "test")

        manifest_rows.append(
            {
                "group": group_name,
                "group_dir": str(group_dir),
                "train_csv": str(train_path),
                "valid_csv": str(valid_path),
                "test_csv": str(test_path),
            }
        )
        summary_rows.append(
            {
                "group": group_name,
                **stats,
                "holdout_fraction": holdout_fraction,
                "seed": seed,
            }
        )

        if progress:
            print_progress("[split] groups", index, len(sorted_group_items))
            sys.stdout.write("\n")

    manifest_path = output_root / manifest_name
    summary_path = output_root / summary_name
    write_root_csv(
        manifest_path,
        manifest_rows,
        fieldnames=["group", "group_dir", "train_csv", "valid_csv", "test_csv"],
        overwrite=overwrite,
    )
    write_root_csv(
        summary_path,
        summary_rows,
        fieldnames=[
            "group",
            "total_rows",
            "holdout_rows",
            "train_rows",
            "valid_rows",
            "test_rows",
            "labels_in_holdout",
            "holdout_fraction",
            "seed",
        ],
        overwrite=overwrite,
    )
    print(
        f"[done] wrote {len(groups)} group folders, manifest={manifest_path}, summary={summary_path}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    progress = not bool(args.no_progress)
    required_columns = [args.group_column, args.score_column, args.label_column]
    if args.sample_id_column:
        required_columns.append(args.sample_id_column)

    if args.input_csv:
        fieldnames, groups = read_rows_from_csv(
            csv_path=args.input_csv,
            group_column=args.group_column,
            required_columns=required_columns,
            progress=progress,
            progress_interval=args.progress_interval,
        )
    else:
        fieldnames, groups = read_groups_from_input_dir(
            input_dir=args.input_dir,
            group_column=args.group_column,
            required_columns=required_columns,
            progress=progress,
            progress_interval=args.progress_interval,
        )

    split_groups(
        groups=groups,
        input_fieldnames=fieldnames,
        output_root=args.output_root,
        group_column=args.group_column,
        score_column=args.score_column,
        label_column=args.label_column,
        sample_id_column=args.sample_id_column,
        assigned_split_column=args.assigned_split_column,
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
        manifest_name=args.manifest_name,
        summary_name=args.summary_name,
        overwrite=bool(args.overwrite),
        progress=progress,
    )


if __name__ == "__main__":
    main()
