#!/usr/bin/env python
"""Extract STImage spot embeddings without materializing patch H5 files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from stflow.hest_utils.encoder import load_encoder
from stflow.hest_utils.file_utils import save_hdf5


Image.MAX_IMAGE_PIXELS = None


def resolve_device(device_arg: str) -> torch.device:
    if str(device_arg).isdigit():
        device_arg = f"cuda:{device_arg}" if torch.cuda.is_available() else "cpu"
    if str(device_arg).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    device = torch.device(device_arg)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device


def crop_patch(image: Image.Image, x: float, y: float, patch_size: int) -> Image.Image:
    half = patch_size // 2
    left = int(round(x)) - half
    top = int(round(y)) - half
    right = left + patch_size
    bottom = top + patch_size

    crop_left = max(left, 0)
    crop_top = max(top, 0)
    crop_right = min(right, image.width)
    crop_bottom = min(bottom, image.height)

    patch = Image.new("RGB", (patch_size, patch_size), (255, 255, 255))
    crop = image.crop((crop_left, crop_top, crop_right, crop_bottom)).convert("RGB")
    patch.paste(crop, (crop_left - left, crop_top - top))
    return patch


def embed_slide(
    sample_id: str,
    image_path: Path,
    coord_path: Path,
    output_path: Path,
    model: torch.nn.Module,
    transform,
    device: torch.device,
    batch_size: int,
    patch_size: int,
    precision: torch.dtype,
) -> None:
    coord_df = pd.read_csv(coord_path).rename(columns={"Unnamed: 0": "barcode"})
    barcodes = coord_df["barcode"].astype(str).to_numpy()
    coords = coord_df[["xaxis", "yaxis"]].to_numpy(dtype=np.float32)

    image = Image.open(image_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    model.eval()
    for start in tqdm(range(0, len(coord_df), batch_size), desc=sample_id, ncols=100):
        chunk = coord_df.iloc[start:start + batch_size]
        patches = [
            transform(crop_patch(image, row.xaxis, row.yaxis, patch_size))
            for row in chunk.itertuples(index=False)
        ]
        imgs = torch.stack(patches).to(device)
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=precision, enabled=device.type == "cuda" and precision != torch.float32):
            embeddings = model(imgs)

        mode = "w" if start == 0 else "a"
        save_hdf5(
            output_path,
            {
                "embeddings": embeddings.cpu().numpy(),
                "barcodes": barcodes[start:start + len(chunk)],
                "coords": coords[start:start + len(chunk)],
            },
            mode=mode,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dataroot", default="/home/user/st_data/stimage_bench")
    parser.add_argument("--embed-dataroot", default="/home/user/st_data/stimage_embeddings")
    parser.add_argument("--weights-root", default="/home/user/st_data/weights_root")
    parser.add_argument("--datasets", nargs="+", default=["Breast", "Brain", "Skin", "Mouth", "Prostate", "Stomach", "Colon"])
    parser.add_argument("--feature-encoder", default="uni_v1_official")
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patch-size", type=int, default=224)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    device = resolve_device(args.device)
    precision = {"fp32": torch.float32, "fp16": torch.float16}[args.precision]
    model, transform, _ = load_encoder(args.feature_encoder, device, args.weights_root, private_weights_root=None)

    source_root = Path(args.source_dataroot)
    embed_root = Path(args.embed_dataroot)
    for dataset in args.datasets:
        split_dir = source_root / dataset / "splits"
        sample_rows = []
        for split_csv in sorted(split_dir.glob("*.csv")):
            sample_rows.append(pd.read_csv(split_csv))
        samples = pd.concat(sample_rows, ignore_index=True).drop_duplicates("sample_id")

        for row in samples.itertuples(index=False):
            output_path = embed_root / dataset / args.feature_encoder / args.precision / f"{row.sample_id}.h5"
            if output_path.exists() and not args.overwrite:
                print(f"Skipping {row.sample_id}: {output_path} exists")
                continue
            embed_slide(
                sample_id=row.sample_id,
                image_path=Path(row.image_path),
                coord_path=Path(row.coord_path),
                output_path=output_path,
                model=model,
                transform=transform,
                device=device,
                batch_size=args.batch_size,
                patch_size=args.patch_size,
                precision=precision,
            )


if __name__ == "__main__":
    main()
