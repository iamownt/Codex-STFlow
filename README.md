# Codex STFlow Reproduction

This repository is a reproducibility workspace for the STFlow paper:

> Tinglin Huang, Tianyu Liu, Mehrtash Babadi, Wengong Jin, and Rex Ying (2025). Scalable Generation of Spatial Transcriptomics from Histology Images via Whole-Slide Flow Matching.

Upstream code: https://github.com/Graph-and-Geometric-Learning/STFlow  
Paper: https://arxiv.org/abs/2506.05361

## Current Local Progress

The machine has the downloaded datasets under `/home/user/st_data`:

- Raw HEST-1k: `/home/user/st_data/hest_data`
- STimage-1K4M: `/home/user/st_data/STimage-1K4M`
- HEST-Benchmark metadata plus symlinked raw files: `/home/user/st_data/hest_bench_linked`
- UNI weights: `/home/user/st_data/weights_root/uni/pytorch_model.bin`

Completed STFlow runs:

| Benchmark | Dataset | Encoder | Local result | Paper Table 1 |
| --- | --- | --- | --- | --- |
| HEST | HCC | UNI | `0.1172 +/- 0.0019` over 3 seeds | `0.124 +/- 0.004` |
| STImage | Colon | UNI | `0.5506 +/- 0.0903` over 3 constructed splits | `0.323 +/- 0.015` |

The HEST HCC run is close to the paper value. The STImage Colon run uses a locally reconstructed split/HVG protocol because upstream STFlow does not ship official STImage splits or gene lists. It matches the appendix selection rule for Colon: 4 human Visium cancer slides, average 4024 spots.

Detailed metrics are tracked in `repro/results_summary.json`. Full JSON outputs are local under ignored `repro/results/`.

## Environment

The working virtualenv is `.venv`. PyTorch is pinned to CUDA 12.8 because the default PyPI wheel installed CUDA 13.0 and failed with the installed NVIDIA driver.

```bash
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r requirements.txt -e .
```

Verified environment:

```text
Python 3.12.7
torch 2.7.1+cu128
CUDA available: true
GPU used for runs: CUDA_VISIBLE_DEVICES=1, logical cuda:0
```

## HEST Setup

Download only the small HEST-Benchmark metadata:

```bash
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="MahmoodLab/hest-bench",
    repo_type="dataset",
    local_dir="/home/user/st_data/hest_bench_meta",
    allow_patterns=["*/splits/*.csv", "*/var_50genes.json", "*/mean_50genes.json", ".gitattributes"],
)
PY
```

Create a benchmark layout that symlinks large files from raw HEST:

```bash
.venv/bin/python repro/scripts/prepare_hest_bench_symlinks.py \
  --hest-raw-root /home/user/st_data/hest_data \
  --bench-meta-root /home/user/st_data/hest_bench_meta \
  --output-root /home/user/st_data/hest_bench_linked
```

Download UNI:

```bash
.venv/bin/python - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin", local_dir="/home/user/st_data/weights_root/uni")
PY
```

Extract HEST HCC UNI embeddings:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m stflow.app.hest.benchmark \
  --datasets HCC \
  --encoders uni_v1_official \
  --weights_root /home/user/st_data/weights_root \
  --source_dataroot /home/user/st_data/hest_bench_linked \
  --embed_dataroot /home/user/st_data/stflow_embeddings \
  --results_dir /home/user/st_data/Codex-STFlow/repro/results/hest_linprobe \
  --exp_code hcc_uni_smoke \
  --batch_size 128 \
  --num_workers 2 \
  --method ridge \
  --skip_download
```

Run HEST HCC STFlow seeds:

```bash
for seed in 1 2 3; do
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m stflow.app.flow.train \
    --datasets HCC \
    --feature_encoder uni_v1_official \
    --source_dataroot /home/user/st_data/hest_bench_linked \
    --embed_dataroot /home/user/st_data/stflow_embeddings \
    --save_dir /home/user/st_data/Codex-STFlow/repro/results/stflow_hest \
    --exp_code hcc_seed${seed} \
    --device cuda:0 \
    --seed ${seed} \
    --batch_size 2 \
    --sample_times 10 \
    --epochs 100 \
    --eval_step 1 \
    --n_layers 4 \
    --n_sample_steps 5
done
```

## STImage Setup

Prepare STImage Colon from raw STimage-1K4M:

```bash
.venv/bin/python repro/scripts/prepare_stimage_bench.py \
  --stimage-root /home/user/st_data/STimage-1K4M \
  --output-root /home/user/st_data/stimage_bench \
  --datasets Colon \
  --seeds 1 2 3 \
  --chunksize 512
```

Extract UNI embeddings directly from the raw STImage PNGs:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python repro/scripts/extract_stimage_embeddings.py \
  --source-dataroot /home/user/st_data/stimage_bench \
  --embed-dataroot /home/user/st_data/stimage_embeddings \
  --weights-root /home/user/st_data/weights_root \
  --datasets Colon \
  --feature-encoder uni_v1_official \
  --precision fp32 \
  --batch-size 128 \
  --patch-size 224 \
  --device cuda:0
```

Run STImage Colon STFlow:

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m stflow.app.flow.train \
  --datasets Colon \
  --feature_encoder uni_v1_official \
  --source_dataroot /home/user/st_data/stimage_bench \
  --embed_dataroot /home/user/st_data/stimage_embeddings \
  --save_dir /home/user/st_data/Codex-STFlow/repro/results/stflow_stimage \
  --exp_code colon_splits123 \
  --device cuda:0 \
  --seed 1 \
  --batch_size 2 \
  --sample_times 10 \
  --epochs 100 \
  --eval_step 1 \
  --n_layers 4 \
  --n_sample_steps 5
```

## Code Changes From Upstream

- Fixed package discovery and included model config JSON files in editable installs.
- Added GPU device selection for feature extraction/training.
- Replaced the `scvi-tools` ZINB dependency with a small Torch implementation.
- Fixed denoiser initialization, JSON serialization of metric scalars, and optional STImage validation/test evaluation.
- Added scripts for HEST symlink layout, STImage-Bench construction, and direct STImage embedding extraction.

## Next Runs

HEST can now be extended by extracting embeddings and running STFlow for the remaining Table 1 tasks. The highest-priority next tasks are `LUNG`, `IDC`, and `SKCM` because they are prominent in the paper and exercise larger slide sizes.

For STImage, repeat the same prep/extract/train flow for `Breast`, `Brain`, `Skin`, `Mouth`, `Prostate`, and `Stomach`. Exact reproduction of the STImage row still depends on matching the authors' unpublished split seeds and HVG lists.
