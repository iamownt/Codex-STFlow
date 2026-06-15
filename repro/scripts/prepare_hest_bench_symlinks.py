#!/usr/bin/env python
"""Build a HEST-Benchmark layout from local raw HEST files.

STFlow expects the HEST-Benchmark repository layout:

    <bench>/<task>/{splits,var_50genes.json,adata,patches}

The downloaded raw HEST tree already contains the large files as
`st/<sample>.h5ad` and `patches/<sample>.h5`. This script copies the small
benchmark metadata downloaded from `MahmoodLab/hest-bench` and symlinks the
large files to avoid duplicating about 42 GB of data.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import pandas as pd


def link_file(src: Path, dst: Path, force: bool = False) -> None:
    if not src.is_file():
        raise FileNotFoundError(src)
    if dst.exists() or dst.is_symlink():
        if force:
            dst.unlink()
        else:
            return
    dst.symlink_to(src)


def copy_metadata(src_task: Path, dst_task: Path) -> None:
    for path in src_task.rglob("*"):
        if path.is_dir():
            continue
        rel = path.relative_to(src_task)
        if rel.parts[0] in {"adata", "patches", "patches_vis"}:
            continue
        out = dst_task / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, out)


def collect_sample_ids(split_dir: Path) -> list[str]:
    sample_ids: set[str] = set()
    for split_csv in sorted(split_dir.glob("*.csv")):
        df = pd.read_csv(split_csv)
        sample_ids.update(df["sample_id"].astype(str).tolist())
    return sorted(sample_ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hest-raw-root", default="/home/user/st_data/hest_data")
    parser.add_argument("--bench-meta-root", default="/home/user/st_data/hest_bench_meta")
    parser.add_argument("--output-root", default="/home/user/st_data/hest_bench_linked")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    hest_raw = Path(args.hest_raw_root)
    bench_meta = Path(args.bench_meta_root)
    output = Path(args.output_root)
    output.mkdir(parents=True, exist_ok=True)

    tasks = sorted(
        path for path in bench_meta.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    if not tasks:
        raise RuntimeError(f"No benchmark task folders found under {bench_meta}")

    total_links = 0
    for task in tasks:
        dst_task = output / task.name
        copy_metadata(task, dst_task)
        (dst_task / "adata").mkdir(parents=True, exist_ok=True)
        (dst_task / "patches").mkdir(parents=True, exist_ok=True)

        sample_ids = collect_sample_ids(task / "splits")
        for sample_id in sample_ids:
            link_file(hest_raw / "st" / f"{sample_id}.h5ad", dst_task / "adata" / f"{sample_id}.h5ad", args.force)
            link_file(hest_raw / "patches" / f"{sample_id}.h5", dst_task / "patches" / f"{sample_id}.h5", args.force)
            total_links += 2
        print(f"{task.name}: {len(sample_ids)} samples")

    print(f"Wrote {output} with {total_links} symlinked large files")


if __name__ == "__main__":
    main()
