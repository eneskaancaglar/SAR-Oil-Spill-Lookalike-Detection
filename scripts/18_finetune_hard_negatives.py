from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from losses import BCEDiceLoss
from oil_spill_dataset import OilSpillDataset
from unet import UNet

SOS_MANIFEST = PROJECT_ROOT / "data" / "metadata" / "dataset_manifest.csv"
DARTIS_MANIFEST = PROJECT_ROOT / "data" / "metadata" / "dartis_no_oil_scene_split.csv"
DARTIS_RAW_DIR = PROJECT_ROOT / "data" / "external" / "dartis" / "raw"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "full_baseline" / "best.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SOS + DARTIS hard-negative verisiyle U-Net fine-tuning."
    )
    parser.add_argument("--run-name", default="hard_negative_finetune")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--dartis-repeat", type=int, default=2)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--max-dartis-val-images", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ImageMaskOnlyDataset(Dataset):
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.dataset[index]
        return {"image": sample["image"], "mask": sample["mask"]}


class DartisHardNegativeDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        raw_dir: Path,
        split: str,
        crop_size: int,
        repeat: int,
    ) -> None:
        frame = pd.read_csv(manifest_path, encoding="utf-8-sig")
        self.frame = frame[frame["split"] == split].reset_index(drop=True)
        if self.frame.empty:
            raise RuntimeError(f"DARTIS split bos: {split}")
        if crop_size <= 0 or repeat <= 0:
            raise ValueError("crop-size ve dartis-repeat sifirdan buyuk olmali.")
        self.raw_dir = raw_dir
        self.crop_size = crop_size
        self.repeat = repeat

    def __len__(self) -> int:
        return len(self.frame) * self.repeat

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.frame.iloc[index % len(self.frame)]
        path = self.raw_dir / str(row["image_set"]) / str(row["image_name"])
        if not path.exists():
            raise FileNotFoundError(path)

        with Image.open(path) as image:
            array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0

        height, width = array.shape
        if height < self.crop_size or width < self.crop_size:
            resized = Image.fromarray((array * 255).astype(np.uint8)).resize(
                (self.crop_size, self.crop_size), Image.Resampling.BILINEAR
            )
            crop = np.asarray(resized, dtype=np.float32) / 255.0
        else:
            top = random.randint(0, height - self.crop_size)
            left = random.randint(0, width - self.crop_size)
            crop = array[top : top + self.crop_size, left : left + self.crop_size]

        image_tensor = torch.from_numpy(crop.copy()).unsqueeze(0)
        return {"image": image_tensor, "mask": torch.zeros_like(image_tensor)}


def segmentation_metrics(tp: int, fp: int, fn: int) -> dict[str, float]:
    eps = 1e-7
    tp_f, fp_f, fn_f = float(tp), float(fp), float(fn)
    return {
        "dice": (2 * tp_f + eps) / (2 * tp_f + fp_f + fn_f + eps),
        "iou": (tp_f + eps) / (tp_f + fp_f + fn_f + eps),
        "precision": (tp_f + eps) / (tp_f + fp_f + eps),
        "recall": (tp_f + eps) / (tp_f + fn_f + eps),
    }


def evaluate_sos(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> dict[str, float]:
    model.eval()
    tp = fp = fn = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["mask"].to(device, non_blocking=True) >= 0.5
            predictions = torch.sigmoid(model(images)) >= threshold
            tp += int((predictions & targets).sum().item())
            fp += int((predictions & ~targets).sum().item())
            fn += int((~predictions & targets).sum().item())
    return segmentation_metrics(tp, fp, fn)


def new_alarm_state() -> dict[str, int]:
    return {
        "images": 0,
        "pixels": 0,
        "positive": 0,
        "any": 0,
        "ge_0_1": 0,
        "ge_1": 0,
    }


def update_alarm(state: dict[str, int], positive: int, total: int) -> None:
    ratio = positive / total
    state["images"] += 1
    state["pixels"] += total
    state["positive"] += positive
    state["any"] += int(positive > 0)
    state["ge_0_1"] += int(ratio >= 0.001)
    state["ge_1"] += int(ratio >= 0.01)


def finalize_alarm(state: dict[str, int]) -> dict[str, float | int]:
    images = state["images"]
    pixels = state["pixels"]
    pct = lambda count: count / images * 100.0 if images else 0.0
    return {
        "image_count": images,
        "any_alarm_percentage": pct(state["any"]),
        "alarm_ge_0_1_percentage": pct(state["ge_0_1"]),
        "alarm_ge_1_percentage": pct(state["ge_1"]),
        "false_positive_pixel_percentage": (
            state["positive"] / pixels * 100.0 if pixels else 0.0
        ),
    }


def evaluate_dartis(
    model: torch.nn.Module,
    frame: pd.DataFrame,
    device: torch.device,
    threshold: float,
    max_images: int,
) -> dict[str, dict[str, float | int]]:
    model.eval()
    if max_images > 0:
        frame = frame.head(max_images)
    states: dict[str, dict[str, int]] = defaultdict(new_alarm_state)

    with torch.no_grad():
        for row in frame.itertuples(index=False):
            image_set = str(row.image_set)
            path = DARTIS_RAW_DIR / image_set / str(row.image_name)
            with Image.open(path) as image:
                array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
            tensor = torch.from_numpy(array.copy()).unsqueeze(0).unsqueeze(0).to(device)
            prediction = torch.sigmoid(model(tensor))[0, 0] >= threshold
            positive = int(prediction.sum().item())
            total = int(prediction.numel())
            for group in ("overall", image_set):
                update_alarm(states[group], positive, total)

    return {group: finalize_alarm(state) for group, state in states.items()}


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    metrics: dict[str, Any],
    base_channels: int,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": {**vars(args), "base_channels": base_channels},
            "metrics": metrics,
            "source_checkpoint": str(args.checkpoint),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    for required in (SOS_MANIFEST, DARTIS_MANIFEST, args.checkpoint):
        if not required.exists():
            raise FileNotFoundError(required)

    device = torch.device(
        args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )
    source = torch.load(args.checkpoint, map_location=device, weights_only=False)
    base_channels = int(source.get("args", {}).get("base_channels", 16))

    model = UNet(in_channels=1, out_channels=1, base_channels=base_channels).to(device)
    model.load_state_dict(source["model_state_dict"])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_function = BCEDiceLoss()

    sos_train = ImageMaskOnlyDataset(
        OilSpillDataset(manifest_path=SOS_MANIFEST, split="train")
    )
    sos_val = OilSpillDataset(manifest_path=SOS_MANIFEST, split="val")
    dartis_train = DartisHardNegativeDataset(
        DARTIS_MANIFEST,
        DARTIS_RAW_DIR,
        "hard_negative_train",
        args.crop_size,
        args.dartis_repeat,
    )
    mixed_train = ConcatDataset([sos_train, dartis_train])

    train_loader = DataLoader(
        mixed_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    sos_val_loader = DataLoader(
        sos_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    split_frame = pd.read_csv(DARTIS_MANIFEST, encoding="utf-8-sig")
    dartis_val = split_frame[split_frame["split"] == "hard_negative_val"].reset_index(drop=True)
    if dartis_val.empty:
        raise RuntimeError("hard_negative_val bos.")

    checkpoint_dir = PROJECT_ROOT / "checkpoints" / args.run_name
    output_dir = PROJECT_ROOT / "outputs" / args.run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("HARD-NEGATIVE FINE-TUNING")
    print("=" * 75)
    print("Run:", args.run_name)
    print("Cihaz:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(device))
    print("Baslangic checkpoint:", args.checkpoint)
    print("SOS train:", len(sos_train))
    print("DARTIS train (repeat dahil):", len(dartis_train))
    print("Karisik train:", len(mixed_train))
    print("SOS val:", len(sos_val))
    print("DARTIS val:", len(dartis_val))
    print("External test kullanilmiyor.")

    history: list[dict[str, Any]] = []
    best_score = float("-inf")
    total_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        loss_sum = 0.0
        steps = 0

        for step, batch in enumerate(train_loader, start=1):
            if args.max_train_steps > 0 and step > args.max_train_steps:
                break
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(images), masks)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.item())
            steps += 1

        if steps == 0:
            raise RuntimeError("Egitim adimi tamamlanmadi.")

        sos_metrics = evaluate_sos(model, sos_val_loader, device, args.threshold)
        dartis_metrics = evaluate_dartis(
            model,
            dartis_val,
            device,
            args.threshold,
            args.max_dartis_val_images,
        )
        overall = dartis_metrics["overall"]
        severe_alarm = float(overall["alarm_ge_1_percentage"]) / 100.0
        balanced_score = sos_metrics["dice"] * (1.0 - severe_alarm)

        metrics: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": loss_sum / steps,
            "sos_validation": sos_metrics,
            "dartis_validation": dartis_metrics,
            "balanced_score": balanced_score,
        }
        history.append(metrics)

        save_checkpoint(
            checkpoint_dir / "last.pth",
            model,
            optimizer,
            epoch,
            args,
            metrics,
            base_channels,
        )
        is_best = balanced_score > best_score
        if is_best:
            best_score = balanced_score
            save_checkpoint(
                checkpoint_dir / "best_balanced.pth",
                model,
                optimizer,
                epoch,
                args,
                metrics,
                base_channels,
            )

        print()
        print(f"Epoch {epoch:02d}/{args.epochs} | {time.time() - epoch_start:.1f} saniye")
        print(f"Train loss: {metrics['train_loss']:.4f}")
        print(
            "SOS val | "
            f"Dice {sos_metrics['dice']:.4f} | IoU {sos_metrics['iou']:.4f} | "
            f"P {sos_metrics['precision']:.4f} | R {sos_metrics['recall']:.4f}"
        )
        print(
            "DARTIS val | "
            f"Any alarm %{overall['any_alarm_percentage']:.2f} | "
            f">=0.1% %{overall['alarm_ge_0_1_percentage']:.2f} | "
            f">=1% %{overall['alarm_ge_1_percentage']:.2f} | "
            f"FP pixel %{overall['false_positive_pixel_percentage']:.4f}"
        )
        print(f"Balanced score: {balanced_score:.4f}")
        if is_best:
            print("Yeni en iyi dengeli checkpoint kaydedildi.")

    history_path = output_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print("=" * 75)
    print("FINE-TUNING TAMAMLANDI")
    print("=" * 75)
    print(f"Toplam sure: {time.time() - total_start:.1f} saniye")
    print(f"En iyi balanced score: {best_score:.4f}")
    print("Checkpoint:", checkpoint_dir.resolve())
    print("History:", history_path.resolve())


if __name__ == "__main__":
    main()
