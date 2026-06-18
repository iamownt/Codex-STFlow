from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from stflow.model.spatial_prior import (
    DEFAULT_HEST_TASKS,
    SpatialPriorConfig,
    ensure_thread_env,
    fit_predict_hest_ridge_baseline,
    fit_predict_spatial_prior,
    load_gene_list,
    load_split_assets,
    merge_split_results,
    sanitize_json,
    split_indices,
    weighted_neighbor_context,
    write_json,
)
from stflow.utils import get_current_time, set_random_seed


def load_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    config_path = Path(path)
    with config_path.open() as f:
        if config_path.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(f) or {}
        return json.load(f)


def merge_args_with_config(args: argparse.Namespace, cfg: dict[str, Any]) -> dict[str, Any]:
    merged = dict(cfg)
    for key, value in vars(args).items():
        if key == "config":
            continue
        if value is not None:
            merged[key] = value
    return merged


def method_name(cfg: dict[str, Any]) -> str:
    return (
        f"prior_ridge_pca{int(cfg['latent_dim'])}"
        f"_a{float(cfg['alpha']):g}"
        f"_smooth_k{int(cfg['smooth_k'])}"
        f"_w{float(cfg['smooth_self_weight']):g}"
    )


def _copy_config(config_path: str | None, run_dir: Path) -> None:
    if config_path is None:
        return
    source = Path(config_path)
    if source.is_file():
        shutil.copy2(source, run_dir / source.name)


def _dataset_rows_for_average(dataset_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    methods: dict[str, list[float]] = {}
    for dataset_result in dataset_results:
        for row in dataset_result["results"]:
            methods.setdefault(row["method_name"], []).append(float(row["pearson_mean"]))
    return [
        {
            "method_name": name,
            "mean_pearson": float(np.mean(values)),
            "std_pearson": float(np.std(values)),
            "n_datasets": int(len(values)),
        }
        for name, values in sorted(methods.items(), key=lambda item: float(np.mean(item[1])), reverse=True)
    ]


def _load_reference_baselines(path: str | None) -> dict[str, float]:
    if path is None:
        return {}
    payload = json.loads(Path(path).read_text())
    rows = {}
    for dataset_row in payload.get("results", []):
        dataset = dataset_row.get("dataset")
        if dataset is None:
            dataset = dataset_row.get("dataset_name")
        for result in dataset_row.get("results", []):
            pearson = result.get("pearson_mean")
            if pearson is not None:
                rows[str(dataset)] = float(pearson)
                break
    return rows


def run_dataset(dataset: str, cfg: dict[str, Any], run_dir: Path, reference_baselines: dict[str, float]) -> dict[str, Any]:
    source_root = Path(cfg["source_dataroot"])
    bench_dataset_root = source_root / dataset
    split_dir = bench_dataset_root / "splits"
    encoder = str(cfg["feature_encoder"])
    genes = load_gene_list(bench_dataset_root, str(cfg["gene_list"]))
    indices = split_indices(split_dir)
    dataset_dir = run_dir / dataset / encoder
    sample_cache = {}

    print(f"Running {dataset}: {len(indices)} folds, {len(genes)} genes", flush=True)
    prior_config = SpatialPriorConfig(
        latent_dim=int(cfg["latent_dim"]),
        alpha=float(cfg["alpha"]),
        smooth_k=int(cfg["smooth_k"]),
        smooth_self_weight=float(cfg["smooth_self_weight"]),
        smooth_radius_percentile=float(cfg["smooth_radius_percentile"]),
        fit_intercept=bool(cfg.get("fit_intercept", False)),
        max_iter=int(cfg.get("max_iter", 1000)),
        seed=int(cfg["seed"]),
    )

    baseline_splits = []
    prior_splits = []
    prior_diagnostics = []
    baseline_diagnostics = []

    for split_idx in indices:
        train_csv = split_dir / f"train_{split_idx}.csv"
        test_csv = split_dir / f"test_{split_idx}.csv"
        train_ids, test_ids, sample_cache = load_split_assets(
            bench_dataset_root=bench_dataset_root,
            embed_dataroot=cfg["embed_dataroot"],
            dataset=dataset,
            encoder=encoder,
            genes=genes,
            train_csv=train_csv,
            test_csv=test_csv,
            normalize_method_name=str(cfg["normalize_method"]),
            precision=str(cfg["precision"]),
            cache=sample_cache,
        )

        split_dir_out = dataset_dir / f"split{split_idx}"
        if bool(cfg.get("run_hest_baseline", True)):
            baseline_result, baseline_diag = fit_predict_hest_ridge_baseline(
                train_ids=train_ids,
                test_ids=test_ids,
                assets_by_sample=sample_cache,
                genes=genes,
                latent_dim=int(cfg["latent_dim"]),
                seed=int(cfg["seed"]),
            )
            baseline_splits.append(baseline_result)
            baseline_diagnostics.append(baseline_diag)
            write_json(split_dir_out / "hest_ridge_pca256_summary.json", baseline_result)
            write_json(split_dir_out / "hest_ridge_pca256_diagnostics.json", baseline_diag)

        prior_result, prior_diag, _ = fit_predict_spatial_prior(
            train_ids=train_ids,
            test_ids=test_ids,
            assets_by_sample=sample_cache,
            genes=genes,
            config=prior_config,
            smooth=True,
        )
        prior_splits.append(prior_result)
        prior_diagnostics.append(prior_diag)
        write_json(split_dir_out / f"{method_name(cfg)}_summary.json", prior_result)
        write_json(split_dir_out / f"{method_name(cfg)}_diagnostics.json", prior_diag)

    rows = []
    reference = reference_baselines.get(dataset)
    if baseline_splits:
        baseline_kfold = merge_split_results(baseline_splits)
        write_json(dataset_dir / "hest_ridge_pca256_results_kfold.json", baseline_kfold)
        write_json(dataset_dir / "hest_ridge_pca256_diagnostics_kfold.json", baseline_diagnostics)
        rows.append(
            {
                "method_name": "hest_ridge_pca256",
                "encoder_name": encoder,
                "pearson_mean": baseline_kfold["pearson_mean"],
                "pearson_std": baseline_kfold["pearson_std"],
                "mean_per_split": baseline_kfold["mean_per_split"],
                "delta_vs_reference_uni2": (
                    float(baseline_kfold["pearson_mean"]) - reference if reference is not None else None
                ),
            }
        )

    prior_kfold = merge_split_results(prior_splits)
    write_json(dataset_dir / f"{method_name(cfg)}_results_kfold.json", prior_kfold)
    write_json(dataset_dir / f"{method_name(cfg)}_diagnostics_kfold.json", prior_diagnostics)
    prior_row = {
        "method_name": method_name(cfg),
        "encoder_name": encoder,
        "pearson_mean": prior_kfold["pearson_mean"],
        "pearson_std": prior_kfold["pearson_std"],
        "mean_per_split": prior_kfold["mean_per_split"],
        "delta_vs_hest_ridge_pca256": (
            float(prior_kfold["pearson_mean"]) - float(rows[0]["pearson_mean"]) if rows else None
        ),
        "delta_vs_reference_uni2": float(prior_kfold["pearson_mean"]) - reference if reference is not None else None,
    }
    rows.append(prior_row)
    rows = sorted(rows, key=lambda item: item["pearson_mean"], reverse=True)
    write_json(dataset_dir / "comparison.json", rows)
    write_json(
        dataset_dir / "dataset_metadata.json",
        {
            "dataset": dataset,
            "encoder": encoder,
            "n_splits": len(indices),
            "genes": genes,
            "sample_shapes": {
                sample_id: {
                    "embeddings": list(assets.embeddings.shape),
                    "coords": list(assets.coords.shape),
                    "expression": list(assets.expression.shape),
                }
                for sample_id, assets in sample_cache.items()
            },
        },
    )
    return {
        "dataset": dataset,
        "encoder": encoder,
        "reference_uni2_pearson": reference,
        "results": rows,
    }


def run(cfg: dict[str, Any], config_path: str | None = None) -> Path:
    ensure_thread_env(str(cfg.get("threads", 16)))
    set_random_seed(int(cfg["seed"]))
    datasets = cfg.get("datasets") or DEFAULT_HEST_TASKS
    if isinstance(datasets, str):
        datasets = DEFAULT_HEST_TASKS if datasets == "all" else [datasets]

    exp_code = str(cfg.get("exp_code", "prior_guided_uni2_h_all_tasks"))
    run_dir = Path(cfg["save_dir"]) / f"{exp_code}::{get_current_time()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    _copy_config(config_path, run_dir)
    write_json(
        run_dir / "run_metadata.json",
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "python": sys.version,
            "platform": platform.platform(),
            "config": cfg,
        },
    )

    reference_baselines = _load_reference_baselines(cfg.get("baseline_results_path"))
    dataset_results = []
    for dataset in datasets:
        dataset_results.append(run_dataset(str(dataset), cfg, run_dir, reference_baselines))

    average_rows = _dataset_rows_for_average(dataset_results)
    write_json(
        run_dir / "dataset_results.json",
        {
            "results": dataset_results,
            "average": average_rows,
            "reference": {
                "baseline_results_path": cfg.get("baseline_results_path"),
                "reference_baselines_loaded": reference_baselines,
            },
        },
    )
    pd.DataFrame(
        [
            {
                "dataset": row["dataset"],
                "method_name": result["method_name"],
                "encoder_name": result["encoder_name"],
                "pearson_mean": result["pearson_mean"],
                "pearson_std": result["pearson_std"],
                "delta_vs_reference_uni2": result.get("delta_vs_reference_uni2"),
                "delta_vs_hest_ridge_pca256": result.get("delta_vs_hest_ridge_pca256"),
            }
            for row in dataset_results
            for result in row["results"]
        ]
    ).to_csv(run_dir / "dataset_results.csv", index=False)
    pd.DataFrame(average_rows).to_csv(run_dir / "average_results.csv", index=False)
    print(f"Saved results to {run_dir}", flush=True)
    return run_dir


def self_test() -> None:
    values = np.arange(20, dtype=np.float32).reshape(10, 2)
    coords = np.stack([np.arange(10, dtype=np.float32), np.zeros(10, dtype=np.float32)], axis=1)
    context, diagnostics = weighted_neighbor_context(values, coords, k=2, radius_percentile=100)
    assert context.shape == values.shape
    assert diagnostics["fallback_count"] == 0
    assert np.all(np.isfinite(context))
    assert diagnostics["accepted_count_percentiles"][0] >= 1
    print("prior_guided self-test passed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run STFlow prior-guided UNI2-h spatial Ridge benchmark.")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--source_dataroot", type=str, default=None)
    parser.add_argument("--embed_dataroot", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--baseline_results_path", type=str, default=None)
    parser.add_argument("--feature_encoder", type=str, default=None)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--gene_list", type=str, default=None)
    parser.add_argument("--normalize_method", type=str, default=None)
    parser.add_argument("--precision", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--latent_dim", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--smooth_k", type=int, default=None)
    parser.add_argument("--smooth_self_weight", type=float, default=None)
    parser.add_argument("--smooth_radius_percentile", type=float, default=None)
    parser.add_argument("--exp_code", type=str, default=None)
    parser.add_argument("--run_hest_baseline", action="store_true", default=None)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    cfg = merge_args_with_config(args, load_config(args.config))
    defaults = {
        "source_dataroot": "/home/user/st_data/hest_uni2_repro/official_bench_all",
        "embed_dataroot": "/home/user/st_data/hest_uni2_repro/official_embeddings_all",
        "save_dir": "/home/user/st_data/Codex-STFlow/repro/results/prior_guided_hest",
        "baseline_results_path": "/home/user/st_data/hest_uni2_repro/results/uni2_h_all_tasks_official::26-06-16-20-57-29/dataset_results.json",
        "feature_encoder": "uni_v2",
        "datasets": DEFAULT_HEST_TASKS,
        "gene_list": "var_50genes.json",
        "normalize_method": "log1p",
        "precision": "fp32",
        "seed": 1,
        "latent_dim": 256,
        "alpha": 30000.0,
        "smooth_k": 8,
        "smooth_self_weight": 0.75,
        "smooth_radius_percentile": 95.0,
        "fit_intercept": False,
        "max_iter": 1000,
        "run_hest_baseline": True,
        "threads": 16,
        "exp_code": "prior_guided_uni2_h_all_tasks",
    }
    for key, value in defaults.items():
        cfg.setdefault(key, value)
    run(cfg, config_path=args.config)


if __name__ == "__main__":
    main()
