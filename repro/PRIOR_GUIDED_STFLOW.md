# Prior-Guided STFlow Recipe

This experiment ports the strongest UNI2-h HEST recipe into the STFlow
workspace as a deterministic prior-guided flow baseline.

## Method

The method is:

- UNI2-h spot embeddings from the verified HEST benchmark cache.
- StandardScaler and PCA(256), fit only on each official train split.
- Multi-output Ridge with fixed `alpha = 30000`.
- Spatial smoothing of predictions inside each held-out test slide:
  `final = 0.75 * own_prediction + 0.25 * weighted_neighbor_prediction`.
- K = 8 non-self spatial neighbors, with the same 95th percentile distance cap
  used by the HEST UNI2 reproduction workspace.

No test expression is used by the model or by the spatial smoother.

The intended research framing is a foundation-prior flow recipe: establish a
strong, train-only morphology prior, then use spatial flow modeling for residual
structure instead of asking the flow model to learn the full expression map from
limited HEST labels. This first implementation verifies the prior component and
provides the reusable loader/smoother for residual-flow follow-up work.

## Command

Run from `/home/user/st_data/Codex-STFlow`:

```bash
OPENBLAS_NUM_THREADS=16 OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16 \
.venv/bin/python -m stflow.app.flow.prior_guided \
  --config repro/configs/prior_guided_uni2_h_all_tasks.yaml
```

## Result

Result directory:

```text
repro/results/prior_guided_hest/prior_guided_uni2_h_all_tasks::26-06-18-10-25-00
```

| Task | HEST UNI2 Ridge | Prior-Guided STFlow | Delta |
|---|---:|---:|---:|
| IDC | 0.5898 | 0.6018 | +0.0120 |
| PRAD | 0.3566 | 0.3949 | +0.0383 |
| PAAD | 0.5002 | 0.5036 | +0.0034 |
| SKCM | 0.6609 | 0.6811 | +0.0202 |
| COAD | 0.3018 | 0.3579 | +0.0562 |
| READ | 0.2227 | 0.2512 | +0.0285 |
| CCRCC | 0.2640 | 0.2815 | +0.0175 |
| LUNG | 0.5589 | 0.6046 | +0.0458 |
| LYMPH_IDC | 0.2727 | 0.3053 | +0.0326 |
| Average | 0.4142 | 0.4424 | +0.0283 |

The prior-guided recipe improves all 9/9 HEST tasks over the reproduced UNI2-h
Ridge baseline.

## Artifacts

- Runner: `stflow/app/flow/prior_guided.py`
- Prior module: `stflow/model/spatial_prior.py`
- Config: `repro/configs/prior_guided_uni2_h_all_tasks.yaml`
- Summary: `repro/results/prior_guided_hest/prior_guided_uni2_h_all_tasks::26-06-18-10-25-00/dataset_results.json`
- Split-level k-fold outputs: each dataset has one `hest_ridge_pca256` result
  and one `prior_ridge_pca256_a30000_smooth_k8_w0.75` result.

## Verification

Checks run:

```bash
.venv/bin/python -m py_compile stflow/model/spatial_prior.py stflow/app/flow/prior_guided.py
.venv/bin/python -m stflow.app.flow.prior_guided --self-test
```

Result coverage:

- `18` k-fold result files.
- `18` k-fold diagnostic files.
- Coverage is `9` tasks x `2` methods.
