from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from urllib.request import urlopen

import yaml

try:
    from genomic_benchmarks.utils.datasets import _download_url
    from genomic_benchmarks.utils.datasets import _get_reference_name
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "运行本脚本需要安装 genomic_benchmarks 包。请先在当前环境执行 "
        "`pip install genomic-benchmarks==1.0.0`。"
    ) from exc


@dataclass
class ReferenceJob:
    url: str
    ref_type: str
    output_path: Path
    dataset_classes: List[str] = field(default_factory=list)
    source_urls: List[str] = field(default_factory=list)


OFFICIAL_GB_DATASETS = [
    "demo_coding_vs_intergenomic_seqs",
    "demo_human_or_worm",
    "drosophila_enhancers_stark",
    "dummy_mouse_enhancers_ensembl",
    "human_enhancers_cohn",
    "human_enhancers_ensembl",
    "human_ensembl_regulatory",
    "human_nontata_promoters",
    "human_ocr_ensembl",
]
DEFAULT_METADATA_COMMIT = "605d8539830e16c85abe7826990958303ffc5e1c"
DEFAULT_METADATA_URL_TEMPLATE = (
    "https://raw.githubusercontent.com/ML-Bioinfo-CEITEC/genomic_benchmarks/"
    f"{DEFAULT_METADATA_COMMIT}/datasets/{{dataset}}/metadata.yaml"
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_reference_cache_name(url: str) -> str:
    url_digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return f"{url_digest}_{_get_reference_name(url)}"


def add_bool_flag(
    parser: argparse.ArgumentParser,
    name: str,
    default: bool,
    help_text: str,
) -> None:
    parser.add_argument(
        f"--{name}",
        dest=name.replace("-", "_"),
        action="store_true",
        help=help_text,
    )
    parser.add_argument(
        f"--no-{name}",
        dest=name.replace("-", "_"),
        action="store_false",
        help=f"关闭：{help_text}",
    )
    parser.set_defaults(**{name.replace("-", "_"): default})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "读取 Genomic Benchmarks 各数据集 metadata.yaml 中的 reference url，"
            "自动补齐/刷新 metadata.yaml，并把 reference 提前下载到指定目录，"
            "供 expand_genomic_benchmarks_by_index.py 直接复用。"
        )
    )
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=Path("data/genomic_benchmarks/datasets"),
        help="Genomic Benchmarks 数据集根目录，默认 data/genomic_benchmarks/datasets。",
    )
    parser.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        default=None,
        help="只处理指定数据集。可重复传入多个，例如 --dataset human_ocr_ensembl。",
    )
    parser.add_argument(
        "--metadata-url-template",
        type=str,
        default=DEFAULT_METADATA_URL_TEMPLATE,
        help=(
            "metadata.yaml 下载地址模板，必须包含 {dataset} 占位符。"
            "默认使用 genomic_benchmarks 官方 GitHub raw 地址。"
        ),
    )
    parser.add_argument(
        "--reference-cache-dir",
        type=Path,
        default=Path("data/genomic_benchmarks/reference_cache"),
        help=(
            "reference 下载目录。后续 expand_genomic_benchmarks_by_index.py "
            "只要传入同一个目录，就会直接复用这里的文件。"
        ),
    )
    parser.add_argument(
        "--manifest-json",
        type=Path,
        default=None,
        help="下载清单 JSON 路径。默认写到 reference-cache-dir/download_manifest.json。",
    )
    add_bool_flag(
        parser=parser,
        name="refresh-metadata",
        default=True,
        help_text="是否刷新/重新下载 datasets/<dataset>/metadata.yaml。",
    )
    add_bool_flag(
        parser=parser,
        name="force-download",
        default=False,
        help_text="是否强制重新下载 reference 文件，即使目标文件已经存在。",
    )
    return parser.parse_args()


def resolve_dataset_names(
    datasets_root: Path,
    selected_datasets: Optional[Sequence[str]],
) -> List[str]:
    if datasets_root.exists() and not datasets_root.is_dir():
        raise NotADirectoryError(f"GB 数据集根目录不是文件夹：{datasets_root}")

    if selected_datasets is not None:
        return list(dict.fromkeys(selected_datasets))

    existing_names: List[str] = []
    if datasets_root.exists():
        existing_names = [
            path.name
            for path in sorted(datasets_root.iterdir())
            if path.is_dir()
        ]
    ordered = list(OFFICIAL_GB_DATASETS)
    for name in existing_names:
        if name not in ordered:
            ordered.append(name)
    return ordered


def download_metadata(
    dataset_name: str,
    datasets_root: Path,
    metadata_url_template: str,
    refresh_metadata: bool,
) -> Tuple[Path, str, str, str, Dict[str, str]]:
    metadata_path = datasets_root / dataset_name / "metadata.yaml"
    metadata_url = metadata_url_template.format(dataset=dataset_name)
    if metadata_path.exists() and not refresh_metadata:
        metadata_bytes = metadata_path.read_bytes()
        return metadata_path, "cached", metadata_url, sha256_bytes(metadata_bytes), {}

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(metadata_url) as response:
        metadata_bytes = response.read()
        response_headers = {
            key.lower(): value
            for key, value in response.headers.items()
            if key.lower() in {"etag", "last-modified"}
        }
    metadata_path.write_bytes(metadata_bytes)
    return metadata_path, "downloaded", metadata_url, sha256_bytes(metadata_bytes), response_headers


def load_metadata(metadata_path: Path) -> Dict[str, object]:
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"metadata.yaml 解析结果不是字典：{metadata_path}")
    classes = metadata.get("classes")
    if not isinstance(classes, dict) or not classes:
        raise ValueError(f"metadata.yaml 缺少 classes 定义：{metadata_path}")
    return metadata


def extract_release_number(url: str) -> int:
    match = re.search(r"/release-(\d+)/", url)
    if match is None:
        return -1
    return int(match.group(1))


def build_url_preference_key(url: str) -> Tuple[int, int, str]:
    scheme = url.split("://", 1)[0].lower()
    scheme_rank = {"https": 0, "http": 1, "ftp": 2}.get(scheme, 9)
    return (-extract_release_number(url), scheme_rank, url)


def choose_preferred_url(urls: Sequence[str]) -> str:
    unique_urls = sorted(set(urls), key=build_url_preference_key)
    return unique_urls[0]


def collect_reference_jobs(
    dataset_dirs: Sequence[Path],
    reference_cache_dir: Path,
) -> List[ReferenceJob]:
    jobs_by_output_name: Dict[str, ReferenceJob] = {}

    for dataset_dir in dataset_dirs:
        metadata = load_metadata(dataset_dir / "metadata.yaml")
        classes = metadata["classes"]
        if not isinstance(classes, dict):
            raise ValueError(f"metadata.yaml 中 classes 不是字典：{dataset_dir / 'metadata.yaml'}")

        for class_name, class_config in classes.items():
            if not isinstance(class_config, dict):
                raise ValueError(
                    f"metadata.yaml 中类别 {class_name} 配置不是字典："
                    f"{dataset_dir / 'metadata.yaml'}"
                )
            url = str(class_config["url"])
            ref_type = str(class_config["type"])
            if ref_type != "fa.gz":
                raise ValueError(
                    f"当前仅支持 fa.gz reference，收到 {ref_type}："
                    f"{dataset_dir.name}/{class_name}"
                )

            output_name = build_reference_cache_name(url)
            job = jobs_by_output_name.get(output_name)
            output_path = reference_cache_dir / output_name
            if job is None:
                job = ReferenceJob(
                    url=url,
                    ref_type=ref_type,
                    output_path=output_path,
                    source_urls=[url],
                )
                jobs_by_output_name[output_name] = job
            elif job.ref_type != ref_type:
                raise ValueError(
                    f"同一个输出文件名 {output_name} 对应了不同的 reference 类型："
                    f"{job.ref_type} vs {ref_type}"
                )
            else:
                if url not in job.source_urls:
                    job.source_urls.append(url)
                job.url = choose_preferred_url(job.source_urls)
            job.dataset_classes.append(f"{dataset_dir.name}/{class_name}")

    return [
        jobs_by_output_name[key]
        for key in sorted(jobs_by_output_name)
    ]


def download_job(job: ReferenceJob, force_download: bool) -> str:
    if job.output_path.exists() and not force_download:
        return "cached"

    job.output_path.parent.mkdir(parents=True, exist_ok=True)
    if force_download and job.output_path.exists():
        job.output_path.unlink()
    _download_url(job.url, job.output_path)
    return "downloaded"


def main() -> None:
    args = parse_args()
    dataset_names = resolve_dataset_names(args.datasets_root, args.datasets)
    if "{dataset}" not in args.metadata_url_template:
        raise ValueError("--metadata-url-template 必须包含 {dataset} 占位符。")

    metadata_results = []
    metadata_downloaded = 0
    metadata_cached = 0
    dataset_dirs: List[Path] = []
    for dataset_name in dataset_names:
        print(
            f"[metadata] {dataset_name}",
            flush=True,
        )
        metadata_path, status, metadata_url, metadata_sha256, metadata_headers = download_metadata(
            dataset_name=dataset_name,
            datasets_root=args.datasets_root,
            metadata_url_template=args.metadata_url_template,
            refresh_metadata=args.refresh_metadata,
        )
        if status == "downloaded":
            metadata_downloaded += 1
        else:
            metadata_cached += 1
        metadata_results.append(
            {
                "dataset": dataset_name,
                "status": status,
                "metadata_path": str(metadata_path),
                "metadata_url": metadata_url,
                "metadata_sha256": metadata_sha256,
                "metadata_response_headers": metadata_headers,
            }
        )
        dataset_dirs.append(metadata_path.parent)

    jobs = collect_reference_jobs(
        dataset_dirs=dataset_dirs,
        reference_cache_dir=args.reference_cache_dir,
    )

    results = []
    downloaded = 0
    cached = 0
    for index, job in enumerate(jobs, start=1):
        if len(job.source_urls) > 1:
            print(
                (
                    f"[reference-alias] {job.output_path.name} 命中了多个 metadata url，"
                    f"已自动选择 {job.url} 作为统一下载地址。"
                ),
                flush=True,
            )
        print(
            f"[{index}/{len(jobs)}] reference: {job.output_path.name}",
            flush=True,
        )
        status = download_job(job=job, force_download=args.force_download)
        if status == "downloaded":
            downloaded += 1
        else:
            cached += 1
        reference_sha256 = sha256_file(job.output_path)
        results.append(
            {
                "status": status,
                "url": job.url,
                "output_path": str(job.output_path),
                "output_name": job.output_path.name,
                "reference_sha256": reference_sha256,
                "reference_bytes": job.output_path.stat().st_size,
                "dataset_classes": job.dataset_classes,
                "source_urls": sorted(job.source_urls),
            }
        )

    manifest_json = args.manifest_json or (args.reference_cache_dir / "download_manifest.json")
    manifest_json.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "datasets_root": str(args.datasets_root),
        "reference_cache_dir": str(args.reference_cache_dir),
        "manifest_json": str(manifest_json),
        "datasets": [path.name for path in dataset_dirs],
        "metadata_url_template": args.metadata_url_template,
        "default_metadata_commit": DEFAULT_METADATA_COMMIT,
        "refresh_metadata": args.refresh_metadata,
        "reference_cache_naming": "sha256(url)[:16] + '_' + genomic_benchmarks reference name",
        "metadata_files": len(metadata_results),
        "metadata_downloaded_files": metadata_downloaded,
        "metadata_cached_files": metadata_cached,
        "reference_files": len(jobs),
        "downloaded_files": downloaded,
        "cached_files": cached,
        "metadata_results": metadata_results,
        "results": results,
    }
    manifest_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
