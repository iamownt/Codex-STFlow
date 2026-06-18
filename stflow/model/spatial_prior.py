from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from stflow.data.normalize_utils import get_normalize_method
from stflow.hest_utils.file_utils import read_assets_from_h5
from stflow.hest_utils.st_dataset import load_adata
from stflow.utils import merge_fold_results


DEFAULT_HEST_TASKS = [
    "IDC",
    "PRAD",
    "PAAD",
    "SKCM",
    "COAD",
    "READ",
    "CCRCC",
    "LUNG",
    "LYMPH_IDC",
]


@dataclass(frozen=True)
class SpatialPriorConfig:
    latent_dim: int = 256
    alpha: float = 30000.0
    smooth_k: int = 8
    smooth_self_weight: float = 0.75
    smooth_radius_percentile: float = 95.0
    fit_intercept: bool = False
    max_iter: int = 1000
    seed: int = 1


@dataclass
class SampleAssets:
    sample_id: str
    embeddings: np.ndarray
    coords: np.ndarray
    expression: np.ndarray
    barcodes: list[str]


def sanitize_json(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {str(key): sanitize_json(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [sanitize_json(value) for value in payload]
    if isinstance(payload, np.ndarray):
        return sanitize_json(payload.tolist())
    if isinstance(payload, np.generic):
        return sanitize_json(payload.item())
    if isinstance(payload, float):
        if not np.isfinite(payload):
            return None
        return payload
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(sanitize_json(payload), f, sort_keys=True, indent=4)


def resolve_embedding_path(
    embed_dataroot: str | Path,
    dataset: str,
    encoder: str,
    sample_id: str,
    precision: str = "fp32",
) -> Path:
    root = Path(embed_dataroot)
    candidates = [
        root / dataset / encoder / precision / f"{sample_id}.h5",
        root / dataset / encoder / f"{sample_id}.h5",
        root / dataset / encoder / "fp32" / f"{sample_id}.h5",
        root / dataset / encoder / "float32" / f"{sample_id}.h5",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    candidate_list = "\n".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Embedding for {dataset}/{sample_id} not found. Tried:\n{candidate_list}")


def load_gene_list(bench_dataset_root: str | Path, gene_list: str) -> list[str]:
    path = Path(bench_dataset_root) / gene_list
    with path.open() as f:
        return json.load(f)["genes"]


def sample_ids_from_split_csv(split_csv: str | Path) -> list[str]:
    return pd.read_csv(split_csv)["sample_id"].astype(str).tolist()


def split_indices(split_dir: str | Path) -> list[int]:
    split_dir = Path(split_dir)
    indices = []
    for path in split_dir.glob("train_*.csv"):
        indices.append(int(path.stem.removeprefix("train_")))
    return sorted(indices)


def _barcode_key(assets: dict[str, np.ndarray]) -> str:
    if "barcodes" in assets:
        return "barcodes"
    if "barcode" in assets:
        return "barcode"
    raise KeyError("Embedding HDF5 must contain either 'barcodes' or 'barcode'")


def _decode_barcodes(raw: np.ndarray) -> list[str]:
    flattened = raw.reshape(-1)
    return [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in flattened]


def resolve_expr_path(bench_dataset_root: str | Path, split_row: pd.Series, sample_id: str) -> Path:
    root = Path(bench_dataset_root)
    if "expr_path" in split_row and isinstance(split_row["expr_path"], str):
        candidate = root / split_row["expr_path"]
        if candidate.is_file():
            return candidate
    candidate = root / "adata" / f"{sample_id}.h5ad"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"Expression file for {sample_id} not found under {root}")


def load_sample_assets(
    sample_id: str,
    bench_dataset_root: str | Path,
    embed_dataroot: str | Path,
    dataset: str,
    encoder: str,
    genes: list[str],
    normalize_method_name: str = "log1p",
    precision: str = "fp32",
    split_row: pd.Series | None = None,
) -> SampleAssets:
    embed_path = resolve_embedding_path(embed_dataroot, dataset, encoder, sample_id, precision=precision)
    assets, _ = read_assets_from_h5(embed_path)
    barcodes = _decode_barcodes(assets[_barcode_key(assets)])
    expr_path = resolve_expr_path(
        bench_dataset_root,
        split_row if split_row is not None else pd.Series(dtype=object),
        sample_id,
    )
    expression = load_adata(
        expr_path,
        genes=genes,
        barcodes=barcodes,
        normalize_method=get_normalize_method(normalize_method_name),
    ).values
    return SampleAssets(
        sample_id=sample_id,
        embeddings=np.asarray(assets["embeddings"], dtype=np.float32),
        coords=np.asarray(assets["coords"], dtype=np.float32),
        expression=np.asarray(expression, dtype=np.float32),
        barcodes=barcodes,
    )


def load_split_assets(
    bench_dataset_root: str | Path,
    embed_dataroot: str | Path,
    dataset: str,
    encoder: str,
    genes: list[str],
    train_csv: str | Path,
    test_csv: str | Path,
    normalize_method_name: str = "log1p",
    precision: str = "fp32",
    cache: dict[str, SampleAssets] | None = None,
) -> tuple[list[str], list[str], dict[str, SampleAssets]]:
    cache = cache if cache is not None else {}
    rows = []
    for csv_path in [train_csv, test_csv]:
        frame = pd.read_csv(csv_path)
        rows.extend(frame.to_dict(orient="records"))

    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id not in cache:
            cache[sample_id] = load_sample_assets(
                sample_id=sample_id,
                bench_dataset_root=bench_dataset_root,
                embed_dataroot=embed_dataroot,
                dataset=dataset,
                encoder=encoder,
                genes=genes,
                normalize_method_name=normalize_method_name,
                precision=precision,
                split_row=pd.Series(row),
            )

    return (
        sample_ids_from_split_csv(train_csv),
        sample_ids_from_split_csv(test_csv),
        cache,
    )


def assemble_arrays(
    sample_ids: list[str],
    assets_by_sample: dict[str, SampleAssets],
) -> tuple[np.ndarray, np.ndarray, dict[str, slice]]:
    x_parts = []
    y_parts = []
    slices = {}
    start = 0
    for sample_id in sample_ids:
        assets = assets_by_sample[sample_id]
        end = start + int(assets.embeddings.shape[0])
        slices[sample_id] = slice(start, end)
        start = end
        x_parts.append(assets.embeddings)
        y_parts.append(assets.expression)
    return (
        np.concatenate(x_parts, axis=0).astype(np.float32, copy=False),
        np.concatenate(y_parts, axis=0).astype(np.float32, copy=False),
        slices,
    )


def weighted_neighbor_context(
    values: np.ndarray,
    coords: np.ndarray,
    k: int,
    radius_percentile: float = 95.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(values, dtype=np.float32)
    coords = np.asarray(coords, dtype=np.float32)
    n_spots = int(values.shape[0])
    if n_spots <= 1 or k <= 0:
        return values.copy(), {
            "n_spots": n_spots,
            "k": int(k),
            "fallback_count": n_spots,
            "radius_cap": None,
            "sigma": None,
            "accepted_count_percentiles": [0, 0, 0, 0, 0],
        }

    n_neighbors = min(int(k) + 1, n_spots)
    knn = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto")
    knn.fit(coords)
    distances, indices = knn.kneighbors(coords, return_distance=True)

    neighbor_indices = np.full((n_spots, int(k)), -1, dtype=np.int64)
    neighbor_distances = np.full((n_spots, int(k)), np.inf, dtype=np.float32)
    for row_idx in range(n_spots):
        keep = indices[row_idx] != row_idx
        kept_indices = indices[row_idx][keep][:k]
        kept_distances = distances[row_idx][keep][:k]
        count = int(kept_indices.size)
        if count > 0:
            neighbor_indices[row_idx, :count] = kept_indices
            neighbor_distances[row_idx, :count] = kept_distances

    kth_column = min(int(k) - 1, neighbor_distances.shape[1] - 1)
    finite_kth = neighbor_distances[:, kth_column][np.isfinite(neighbor_distances[:, kth_column])]
    finite_first = neighbor_distances[:, 0][np.isfinite(neighbor_distances[:, 0])]
    radius_cap = float(np.percentile(finite_kth, radius_percentile)) if finite_kth.size else float("inf")
    positive_first = finite_first[finite_first > 0]
    sigma = float(np.median(positive_first)) if positive_first.size else 1.0
    sigma = max(sigma, 1e-6)

    valid = np.isfinite(neighbor_distances) & (neighbor_indices >= 0) & (neighbor_distances <= radius_cap)
    weights = np.exp(-0.5 * np.square(neighbor_distances / sigma)).astype(np.float32)
    weights[~valid] = 0.0
    weight_sums = weights.sum(axis=1, keepdims=True)

    context = values.copy()
    fallback_count = 0
    for row_idx in range(n_spots):
        if weight_sums[row_idx, 0] <= 0:
            fallback_count += 1
            continue
        row_weights = weights[row_idx] / weight_sums[row_idx, 0]
        context[row_idx] = row_weights @ values[neighbor_indices[row_idx]]

    accepted_counts = valid.sum(axis=1).astype(np.float32)
    diagnostics = {
        "n_spots": n_spots,
        "k": int(k),
        "radius_percentile": float(radius_percentile),
        "radius_cap": radius_cap,
        "sigma": sigma,
        "fallback_count": int(fallback_count),
        "accepted_count_percentiles": [
            float(x) for x in np.percentile(accepted_counts, [0, 25, 50, 75, 100])
        ],
    }
    return context, diagnostics


def smooth_predictions_by_sample(
    preds: np.ndarray,
    sample_ids: list[str],
    sample_slices: dict[str, slice],
    assets_by_sample: dict[str, SampleAssets],
    k: int,
    self_weight: float,
    radius_percentile: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    smoothed = preds.copy()
    diagnostics = {
        "smooth_k": int(k),
        "smooth_self_weight": float(self_weight),
        "smooth_radius_percentile": float(radius_percentile),
        "sample_neighbor_diagnostics": {},
    }
    for sample_id in sample_ids:
        sample_slice = sample_slices[sample_id]
        sample_preds = preds[sample_slice].astype(np.float32, copy=False)
        context, sample_diag = weighted_neighbor_context(
            sample_preds,
            assets_by_sample[sample_id].coords,
            k=k,
            radius_percentile=radius_percentile,
        )
        smoothed[sample_slice] = float(self_weight) * sample_preds + (1.0 - float(self_weight)) * context
        diagnostics["sample_neighbor_diagnostics"][sample_id] = sample_diag
    return smoothed, diagnostics


def metric_func(preds_all: np.ndarray, y_test: np.ndarray, genes: list[str]) -> dict[str, Any]:
    errors = []
    r2_scores = []
    pearson_corrs = []
    pearson_genes = []
    for target, gene in enumerate(genes):
        preds = preds_all[:, target]
        target_vals = y_test[:, target]
        errors.append(float(np.mean(np.square(preds - target_vals))))
        denom = float(np.sum(np.square(target_vals - np.mean(target_vals))))
        if denom <= 0:
            r2_score = float("nan")
        else:
            r2_score = float(1.0 - np.sum(np.square(target_vals - preds)) / denom)
        pearson_corr = float(pearsonr(target_vals, preds).statistic)
        pearson_corrs.append(pearson_corr)
        r2_scores.append(r2_score)
        pearson_genes.append({"name": gene, "pearson_corr": pearson_corr})

    pearson_arr = np.asarray(pearson_corrs, dtype=np.float64)
    r2_arr = np.asarray(r2_scores, dtype=np.float64)
    return {
        "l2_errors": errors,
        "r2_scores": r2_scores,
        "pearson_corrs": pearson_genes,
        "pearson_mean": float(np.nanmean(pearson_arr)),
        "pearson_std": float(np.nanstd(pearson_arr)),
        "l2_error_q1": float(np.percentile(errors, 25)),
        "l2_error_q2": float(np.median(errors)),
        "l2_error_q3": float(np.percentile(errors, 75)),
        "r2_score_q1": float(np.nanpercentile(r2_arr, 25)),
        "r2_score_q2": float(np.nanmedian(r2_arr)),
        "r2_score_q3": float(np.nanpercentile(r2_arr, 75)),
    }


def fit_predict_spatial_prior(
    train_ids: list[str],
    test_ids: list[str],
    assets_by_sample: dict[str, SampleAssets],
    genes: list[str],
    config: SpatialPriorConfig,
    smooth: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    x_train, y_train, _ = assemble_arrays(train_ids, assets_by_sample)
    x_test, y_test, test_slices = assemble_arrays(test_ids, assets_by_sample)
    effective_dim = max(1, min(int(config.latent_dim), int(x_train.shape[0]), int(x_train.shape[1])))

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=effective_dim, random_state=int(config.seed))),
        ]
    )
    x_train_t = pipe.fit_transform(x_train).astype(np.float32, copy=False)
    x_test_t = pipe.transform(x_test).astype(np.float32, copy=False)

    reg = Ridge(
        solver="lsqr",
        alpha=float(config.alpha),
        random_state=int(config.seed),
        fit_intercept=bool(config.fit_intercept),
        max_iter=int(config.max_iter),
    )
    reg.fit(x_train_t, y_train)
    preds = reg.predict(x_test_t).astype(np.float32, copy=False)

    postprocess_diag = {}
    if smooth:
        preds, postprocess_diag = smooth_predictions_by_sample(
            preds,
            test_ids,
            test_slices,
            assets_by_sample,
            k=int(config.smooth_k),
            self_weight=float(config.smooth_self_weight),
            radius_percentile=float(config.smooth_radius_percentile),
        )

    result = metric_func(preds, y_test, genes)
    result.update(
        {
            "n_train": int(y_train.shape[0]),
            "n_test": int(y_test.shape[0]),
            "input_dim_raw": int(x_train.shape[1]),
            "input_dim_model": int(x_train_t.shape[1]),
        }
    )
    diagnostics = {
        "train_samples": train_ids,
        "test_samples": test_ids,
        "preprocess": {
            "scaler": "StandardScaler",
            "pca_requested_dim": int(config.latent_dim),
            "pca_effective_dim": int(effective_dim),
            "pca_explained_variance_ratio_sum": float(
                np.sum(pipe.named_steps["pca"].explained_variance_ratio_)
            ),
        },
        "model": {
            "kind": "ridge",
            "alpha": float(config.alpha),
            "fit_intercept": bool(config.fit_intercept),
        },
        "postprocess": postprocess_diag,
        "prediction_shape": list(preds.shape),
        "target_shape": list(y_test.shape),
    }
    return result, diagnostics, preds


def fit_predict_hest_ridge_baseline(
    train_ids: list[str],
    test_ids: list[str],
    assets_by_sample: dict[str, SampleAssets],
    genes: list[str],
    latent_dim: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    x_train, y_train, _ = assemble_arrays(train_ids, assets_by_sample)
    x_test, y_test, _ = assemble_arrays(test_ids, assets_by_sample)
    effective_dim = max(1, min(int(latent_dim), int(x_train.shape[0]), int(x_train.shape[1])))
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=effective_dim, random_state=int(seed))),
        ]
    )
    x_train_t = pipe.fit_transform(x_train).astype(np.float32, copy=False)
    x_test_t = pipe.transform(x_test).astype(np.float32, copy=False)
    alpha = 100.0 / (float(x_train_t.shape[1]) * float(y_train.shape[1]))
    reg = Ridge(solver="lsqr", alpha=alpha, random_state=int(seed), fit_intercept=False, max_iter=1000)
    reg.fit(x_train_t, y_train)
    preds = reg.predict(x_test_t).astype(np.float32, copy=False)
    result = metric_func(preds, y_test, genes)
    result.update(
        {
            "n_train": int(y_train.shape[0]),
            "n_test": int(y_test.shape[0]),
            "input_dim_raw": int(x_train.shape[1]),
            "input_dim_model": int(x_train_t.shape[1]),
        }
    )
    diagnostics = {
        "train_samples": train_ids,
        "test_samples": test_ids,
        "preprocess": {
            "scaler": "StandardScaler",
            "pca_requested_dim": int(latent_dim),
            "pca_effective_dim": int(effective_dim),
            "pca_explained_variance_ratio_sum": float(
                np.sum(pipe.named_steps["pca"].explained_variance_ratio_)
            ),
        },
        "model": {
            "kind": "hest_ridge",
            "alpha": alpha,
            "fit_intercept": False,
        },
    }
    return result, diagnostics


def merge_split_results(split_results: list[dict[str, Any]]) -> dict[str, Any]:
    return merge_fold_results(split_results)


def ensure_thread_env(default_threads: str = "16") -> None:
    for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ.setdefault(key, default_threads)
