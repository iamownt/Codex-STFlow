#!/usr/bin/env python
"""Prepare STImage-Bench metadata and 50-gene AnnData files.

The STFlow paper's appendix defines STImage-Bench as human Visium cancer
samples for seven organs, split at slide level into train/validation/test
sets with an 8:1:1 ratio. This script recreates that layout from the
downloaded STimage-1K4M CSV/image tree.

It writes a STFlow-compatible source tree:

    <output>/<Dataset>/splits/{train,val,test}_<seed_index>.csv
    <output>/<Dataset>/var_50genes.json
    <output>/<Dataset>/adata/<slide>.h5ad

Embeddings are produced separately by `extract_stimage_embeddings.py`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


DATASETS = {
    "Breast": "breast",
    "Brain": "brain",
    "Skin": "skin",
    "Mouth": "mouth",
    "Prostate": "prostate",
    "Stomach": "stomach",
    "Colon": "colon",
}


def count_path(root: Path, slide: str) -> Path:
    return root / "Visium" / "gene_exp" / f"{slide}_count.csv"


def coord_path(root: Path, slide: str) -> Path:
    return root / "Visium" / "coord" / f"{slide}_coord.csv"


def image_path(root: Path, slide: str) -> Path:
    return root / "Visium" / "image" / f"{slide}.png"


def get_gene_columns(path: Path) -> set[str]:
    return set(pd.read_csv(path, nrows=0).columns[1:])


def update_welford(count: int, mean: np.ndarray, m2: np.ndarray, values: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    batch_count = values.shape[0]
    if batch_count == 0:
        return count, mean, m2

    batch_mean = values.mean(axis=0, dtype=np.float64)
    centered = values - batch_mean
    batch_m2 = np.square(centered, dtype=np.float64).sum(axis=0)
    if count == 0:
        return batch_count, batch_mean, batch_m2

    total_count = count + batch_count
    delta = batch_mean - mean
    mean = mean + delta * batch_count / total_count
    m2 = m2 + batch_m2 + np.square(delta) * count * batch_count / total_count
    return total_count, mean, m2


def select_hvgs(root: Path, slides: list[str], n_genes: int, chunksize: int) -> list[str]:
    common_genes = None
    for slide in slides:
        genes = get_gene_columns(count_path(root, slide))
        common_genes = genes if common_genes is None else common_genes & genes
    if not common_genes or len(common_genes) < n_genes:
        raise RuntimeError(f"Only {len(common_genes or [])} common genes found for {len(slides)} slides")

    genes = sorted(common_genes)
    usecols = ["Unnamed: 0", *genes]
    count = 0
    mean = np.zeros(len(genes), dtype=np.float64)
    m2 = np.zeros(len(genes), dtype=np.float64)

    for slide in slides:
        for chunk in pd.read_csv(count_path(root, slide), usecols=usecols, chunksize=chunksize):
            values = np.log1p(chunk[genes].to_numpy(dtype=np.float32, copy=False))
            count, mean, m2 = update_welford(count, mean, m2, values)

    variances = m2 / max(count - 1, 1)
    top_idx = np.argsort(variances)[-n_genes:][::-1]
    return [genes[i] for i in top_idx]


def write_h5ad(root: Path, output: Path, slide: str, genes: list[str], chunksize: int) -> None:
    out_path = output / "adata" / f"{slide}.h5ad"
    if out_path.exists():
        return
    usecols = ["Unnamed: 0", *genes]
    rows = []
    barcodes = []
    for chunk in pd.read_csv(count_path(root, slide), usecols=usecols, chunksize=chunksize):
        barcodes.extend(chunk["Unnamed: 0"].astype(str).tolist())
        rows.append(chunk[genes].to_numpy(dtype=np.float32, copy=False))

    x = np.vstack(rows)
    obs = pd.DataFrame(index=pd.Index(barcodes, name="barcode"))
    var = pd.DataFrame(index=pd.Index(genes, name="gene"))

    coords = pd.read_csv(coord_path(root, slide)).rename(columns={"Unnamed: 0": "barcode"})
    coords = coords.set_index("barcode").reindex(barcodes)
    for column in ["xaxis", "yaxis", "r"]:
        if column in coords:
            obs[column] = coords[column].to_numpy()

    adata = ad.AnnData(X=x, obs=obs, var=var)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_path)


def split_slides(slides: list[str], seed: int) -> tuple[list[str], list[str], list[str]]:
    rng = np.random.default_rng(seed)
    slides = np.array(sorted(slides), dtype=object)
    rng.shuffle(slides)
    n = len(slides)
    n_test = max(1, int(round(n * 0.1)))
    n_val = max(1, int(round(n * 0.1)))
    test = sorted(slides[:n_test].tolist())
    val = sorted(slides[n_test:n_test + n_val].tolist())
    train = sorted(slides[n_test + n_val:].tolist())
    return train, val, test


def split_frame(root: Path, dataset: str, slides: list[str]) -> pd.DataFrame:
    return pd.DataFrame({
        "sample_id": slides,
        "expr_path": [f"adata/{slide}.h5ad" for slide in slides],
        "coord_path": [str(coord_path(root, slide)) for slide in slides],
        "image_path": [str(image_path(root, slide)) for slide in slides],
        "dataset": dataset,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stimage-root", default="/home/user/st_data/STimage-1K4M")
    parser.add_argument("--output-root", default="/home/user/st_data/stimage_bench")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--n-genes", type=int, default=50)
    parser.add_argument("--chunksize", type=int, default=512)
    args = parser.parse_args()

    root = Path(args.stimage_root)
    output_root = Path(args.output_root)
    meta = pd.read_csv(root / "meta" / "meta_all_gene02122025.csv")

    for dataset in args.datasets:
        tissue = DATASETS[dataset]
        out = output_root / dataset
        out.mkdir(parents=True, exist_ok=True)
        (out / "splits").mkdir(exist_ok=True)
        (out / "adata").mkdir(exist_ok=True)

        selected = meta[
            (meta["tissue"].str.lower() == tissue)
            & (meta["involve_cancer"] == True)
            & (meta["tech"] == "Visium")
            & (meta["species"] == "human")
        ].copy()
        slides = sorted(selected["slide"].astype(str).tolist())
        if not slides:
            raise RuntimeError(f"No slides selected for {dataset}")

        selected.to_csv(out / "selected_slides.csv", index=False)
        print(f"{dataset}: {len(slides)} human Visium cancer slides")

        genes_path = out / "var_50genes.json"
        if genes_path.exists():
            genes = json.loads(genes_path.read_text())["genes"]
        else:
            genes = select_hvgs(root, slides, args.n_genes, args.chunksize)
            genes_path.write_text(json.dumps({"genes": genes}, indent=2) + "\n")

        for slide in slides:
            write_h5ad(root, out, slide, genes, args.chunksize)

        for split_idx, seed in enumerate(args.seeds):
            train, val, test = split_slides(slides, seed)
            split_frame(root, dataset, train).to_csv(out / "splits" / f"train_{split_idx}.csv", index=False)
            split_frame(root, dataset, val).to_csv(out / "splits" / f"val_{split_idx}.csv", index=False)
            split_frame(root, dataset, test).to_csv(out / "splits" / f"test_{split_idx}.csv", index=False)


if __name__ == "__main__":
    main()
