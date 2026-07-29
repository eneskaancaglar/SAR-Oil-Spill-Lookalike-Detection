from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = ROOT / "scripts" / "68_evaluate_dartis_water_transfer_gate.py"
DEFAULT_MANIFEST = ROOT / "data" / "metadata" / "v06_dartis_water_active_training_manifest.csv"
DEFAULT_SPLIT = ROOT / "data" / "metadata" / "v06_dartis_water_locked_split.csv"
DEFAULT_SOURCE = ROOT / "checkpoints" / "water_unet_flood_v06_robust" / "best.pth"
CHECKPOINT_DIR = ROOT / "checkpoints" / "water_unet_dartis_active_v06"
OUTPUT_DIR = ROOT / "outputs" / "water_unet_dartis_active_v06"


def load_base():
    if not BASE_SCRIPT.exists():
        raise FileNotFoundError(f"Gerekli script bulunamadı: {BASE_SCRIPT}")
    spec = importlib.util.spec_from_file_location("dartis_base_v76", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("68 scripti modül olarak yüklenemedi.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_base()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT)
    p.add_argument("--source-checkpoint", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--repeat-factor", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--train-size", type=int, default=384)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--maximum-pos-weight", type=float, default=3.0)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260729)
    p.add_argument("--minimum-validation-precision", type=float, default=0.95)
    p.add_argument("--maximum-validation-land-leakage", type=float, default=0.05)
    return p.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize(image: np.ndarray) -> np.ndarray:
    x = image.astype(np.float32)
    lo, hi = np.percentile(x, [2, 98])
    if hi <= lo:
        return (x / 255.0).astype(np.float32)
    return np.clip((x - lo) / (hi - lo), 0, 1).astype(np.float32)


def load_manifest(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Manifest bulunamadı: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    required = {"sample_id", "coastal_group", "annotation_image_path", "manual_mask_path", "hard_pass"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Manifestte eksik sütunlar: {sorted(missing)}")
    df["hard_pass"] = df["hard_pass"].astype(str).str.lower().isin(["true", "1"])
    df = df[df["hard_pass"]].copy()
    if len(df) != 32:
        raise RuntimeError(f"32 doğrulanmış maske bekleniyor, bulunan: {len(df)}")
    counts = df.groupby("coastal_group").size().to_dict()
    if counts != {"nc": 16, "oc": 16}:
        raise RuntimeError(f"Beklenen dağılım nc=16, oc=16; bulunan: {counts}")
    if df["sample_id"].duplicated().any():
        raise RuntimeError("Tekrarlı sample_id bulundu.")
    for col in ("annotation_image_path", "manual_mask_path"):
        df[f"resolved_{col}"] = df[col].map(lambda x: str(resolve_path(x)))
        bad = df[~df[f"resolved_{col}"].map(lambda x: Path(x).exists())]
        if not bad.empty:
            raise FileNotFoundError(f"Eksik {col}: {', '.join(bad['sample_id'].astype(str).head(5))}")
    return df.reset_index(drop=True)


def create_or_load_split(df: pd.DataFrame, path: Path, seed: int) -> pd.DataFrame:
    expected = {
        ("nc", "train"): 10,
        ("nc", "validation"): 2,
        ("nc", "locked_test"): 4,
        ("oc", "train"): 10,
        ("oc", "validation"): 2,
        ("oc", "locked_test"): 4,
    }
    if path.exists():
        saved = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        if not {"sample_id", "split"}.issubset(saved.columns):
            raise RuntimeError("Mevcut split manifesti geçersiz.")
        merged = df.merge(saved[["sample_id", "split"]], on="sample_id", how="left", validate="one_to_one")
        if merged["split"].isna().any():
            raise RuntimeError("Split manifesti 32 örneğin tamamını kapsamıyor.")
        counts = merged.groupby(["coastal_group", "split"]).size().to_dict()
        if counts != expected:
            raise RuntimeError(f"Mevcut split dağılımı geçersiz: {counts}")
        return merged

    rng = np.random.default_rng(seed)
    parts = []
    for group in ("nc", "oc"):
        part = df[df["coastal_group"].eq(group)].copy().reset_index(drop=True)
        indices = np.arange(len(part))
        rng.shuffle(indices)
        test_ids = set(indices[:4].tolist())
        val_ids = set(indices[4:6].tolist())
        part["split"] = [
            "locked_test" if i in test_ids else "validation" if i in val_ids else "train"
            for i in range(len(part))
        ]
        parts.append(part)

    out = pd.concat(parts, ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "sample_id", "coastal_group", "split", "annotation_image_path", "manual_mask_path",
        "land_fraction", "water_fraction", "ignore_fraction"
    ]
    out[[c for c in cols if c in out.columns]].to_csv(path, index=False, encoding="utf-8-sig")
    return out


class WaterDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, train: bool, crop_size: int, repeat_factor: int, seed: int):
        self.records = frame.to_dict("records")
        self.train = train
        self.crop_size = crop_size
        self.repeat_factor = repeat_factor if train else 1
        self.seed = seed

    def __len__(self):
        return len(self.records) * self.repeat_factor

    def __getitem__(self, index):
        rec_index = index % len(self.records)
        rec = self.records[rec_index]
        image = np.array(Image.open(rec["resolved_annotation_image_path"]).convert("L"))
        mask = np.array(Image.open(rec["resolved_manual_mask_path"]).convert("L"))
        if image.shape != mask.shape:
            raise RuntimeError(f"Boyut uyuşmazlığı: {rec['sample_id']} {image.shape} != {mask.shape}")

        if self.train:
            rng = np.random.default_rng(self.seed + index * 1009 + rec_index * 7919)
            h, w = image.shape
            if h < self.crop_size or w < self.crop_size:
                nh, nw = max(h, self.crop_size), max(w, self.crop_size)
                image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_CUBIC)
                mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
                h, w = image.shape
            y = int(rng.integers(0, h - self.crop_size + 1))
            x = int(rng.integers(0, w - self.crop_size + 1))
            image = image[y:y+self.crop_size, x:x+self.crop_size]
            mask = mask[y:y+self.crop_size, x:x+self.crop_size]
            if rng.random() < 0.5:
                image, mask = np.fliplr(image), np.fliplr(mask)
            if rng.random() < 0.5:
                image, mask = np.flipud(image), np.flipud(mask)
            k = int(rng.integers(0, 4))
            if k:
                image, mask = np.rot90(image, k), np.rot90(mask, k)
            image = normalize(image)
            if rng.random() < 0.8:
                image = np.power(image, float(rng.uniform(0.85, 1.15)))
            if rng.random() < 0.5:
                image = np.clip(image * float(rng.uniform(0.90, 1.10)) + float(rng.uniform(-0.04, 0.04)), 0, 1)
            if rng.random() < 0.35:
                image = np.clip(image + rng.normal(0, float(rng.uniform(0.005, 0.02)), image.shape), 0, 1)
        else:
            image = normalize(image)

        image = np.ascontiguousarray(image.astype(np.float32))
        mask = np.ascontiguousarray(mask.astype(np.uint8))
        two_channel = np.stack([image, image], axis=0).astype(np.float32)
        return torch.from_numpy(two_channel), torch.from_numpy(mask), str(rec["sample_id"])


def masked_loss(logits, raw_mask, pos_weight):
    valid = (raw_mask != 128).float()
    target = (raw_mask == 255).float()
    valid_count = valid.sum().clamp_min(1.0)
    bce = (F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight, reduction="none") * valid).sum() / valid_count
    prob = torch.sigmoid(logits)
    intersection = (prob * target * valid).sum()
    denominator = (prob * valid).sum() + (target * valid).sum()
    dice_loss = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
    return 0.55 * bce + 0.45 * dice_loss


def calculate_pos_weight(frame: pd.DataFrame, maximum: float) -> float:
    water = 0
    land = 0
    for row in frame.itertuples():
        mask = np.array(Image.open(row.resolved_manual_mask_path).convert("L"))
        water += int((mask == 255).sum())
        land += int((mask == 0).sum())
    if water == 0:
        return 1.0
    return float(min(land / water, maximum))


def collect_predictions(model, loader, device):
    model.eval()
    outputs = []
    with torch.no_grad():
        for images, masks, sample_ids in loader:
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
            probs = torch.sigmoid(logits[:, 0].float()).cpu().numpy()
            masks_np = masks.numpy()
            for i, sample_id in enumerate(sample_ids):
                outputs.append((str(sample_id), probs[i], masks_np[i]))
    return outputs


def threshold_metrics(predictions, threshold: float):
    tp = fp = fn = tn = 0
    per_iou, per_recall, per_leak = [], [], []
    for _, prob, mask in predictions:
        pred = prob >= threshold
        water = mask == 255
        land = mask == 0
        a = int((pred & water).sum())
        b = int((pred & land).sum())
        c = int(((~pred) & water).sum())
        d = int(((~pred) & land).sum())
        tp += a; fp += b; fn += c; tn += d
        per_iou.append(a / (a + b + c) if a + b + c else 1.0)
        per_recall.append(a / (a + c) if a + c else 1.0)
        per_leak.append(b / (b + d) if b + d else 0.0)
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    dice = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    leakage = fp / (fp + tn) if fp + tn else 0.0
    return {
        "threshold": float(threshold), "dice": float(dice), "iou": float(iou),
        "precision": float(precision), "recall": float(recall), "land_leakage": float(leakage),
        "minimum_per_image_iou": float(min(per_iou)),
        "minimum_per_image_recall": float(min(per_recall)),
        "maximum_per_image_land_leakage": float(max(per_leak)),
    }


def choose_operating_point(predictions, minimum_precision: float, maximum_leakage: float):
    candidates = [threshold_metrics(predictions, float(t)) for t in np.arange(0.30, 0.751, 0.05)]
    safe = [x for x in candidates if x["precision"] >= minimum_precision and x["land_leakage"] <= maximum_leakage]
    if safe:
        return max(safe, key=lambda x: (x["iou"], x["recall"], -x["threshold"])), True
    return max(candidates, key=lambda x: (x["iou"] + 0.30*x["precision"] - 0.75*x["land_leakage"], x["precision"])), False


def sample_hash(frame: pd.DataFrame, split: str) -> str:
    payload = "\n".join(sorted(frame.loc[frame["split"].eq(split), "sample_id"].astype(str))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main():
    args = parse_args()
    set_seed(args.seed)
    manifest_path = resolve_path(args.manifest)
    split_path = resolve_path(args.split_manifest)
    source_path = resolve_path(args.source_checkpoint)

    frame = load_manifest(manifest_path)
    split = create_or_load_split(frame, split_path, args.seed)
    train_df = split[split["split"].eq("train")].copy()
    val_df = split[split["split"].eq("validation")].copy()
    test_df = split[split["split"].eq("locked_test")].copy()
    if (len(train_df), len(val_df), len(test_df)) != (20, 4, 8):
        raise RuntimeError(f"Beklenen split 20/4/8, bulunan: {len(train_df)}/{len(val_df)}/{len(test_df)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    base_channels = int(checkpoint.get("config", {}).get("base_channels", 16))
    model = BASE.SmallUNet(in_channels=2, base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    train_ds = WaterDataset(train_df, True, args.train_size, args.repeat_factor, args.seed)
    val_ds = WaterDataset(val_df, False, args.train_size, 1, args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=device.type == "cuda", generator=generator)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.num_workers,
                            pin_memory=device.type == "cuda")

    pos_weight_value = calculate_pos_weight(train_df, args.maximum_pos_weight)
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3, min_lr=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / "best.pth"
    history = []
    best_score = -1e9
    best_epoch = 0
    best_metrics = None
    best_safe = False
    no_improvement = 0
    start = time.time()

    print("=" * 78)
    print("v0.6 DARTIS ACTIVE WATER ADAPTATION")
    print("=" * 78)
    print(f"Train: {len(train_df)} | Validation: {len(val_df)} | Locked test: {len(test_df)}")
    print("Locked test bu aşamada açılmayacak.")
    print(f"Cihaz: {device}")
    print(f"Train crop: {args.train_size}x{args.train_size}")
    print(f"Train örnek/epoch: {len(train_ds)}")
    print(f"Pos weight: {pos_weight_value:.4f}\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for images, masks, _ in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)[:, 0]
                loss = masked_loss(logits, masks, pos_weight)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().item())
            batches += 1

        predictions = collect_predictions(model, val_loader, device)
        metrics, safe = choose_operating_point(
            predictions,
            args.minimum_validation_precision,
            args.maximum_validation_land_leakage,
        )
        score = (1.0 if safe else 0.0) + metrics["iou"] + 0.20*metrics["recall"] - 0.50*metrics["land_leakage"]
        scheduler.step(score)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(batches, 1),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "validation_safe_point": safe,
            "selection_score": score,
            **{f"validation_{k}": v for k, v in metrics.items()},
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | Loss {row['train_loss']:.4f} | "
            f"Val IoU {metrics['iou']:.4f} P {metrics['precision']:.4f} R {metrics['recall']:.4f} "
            f"Leak {metrics['land_leakage']:.4f} | thr {metrics['threshold']:.2f} | safe {safe}"
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_metrics = dict(metrics)
            best_safe = safe
            no_improvement = 0
            torch.save({
                "stage": "v06_dartis_active_water_adaptation",
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "selected_threshold": metrics["threshold"],
                "validation_metrics": metrics,
                "validation_safe_point": safe,
                "source_checkpoint": relative(source_path),
                "split_manifest": relative(split_path),
                "config": {
                    "base_channels": base_channels,
                    "in_channels": 2,
                    "train_size": args.train_size,
                    "learning_rate": args.learning_rate,
                    "repeat_factor": args.repeat_factor,
                    "batch_size": args.batch_size,
                    "seed": args.seed,
                },
                "locked_test_used": False,
            }, checkpoint_path)
        else:
            no_improvement += 1

        if no_improvement >= args.patience:
            print(f"Early stopping: {args.patience} epoch iyileşme olmadı.")
            break

    history_path = OUTPUT_DIR / "training_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False, encoding="utf-8-sig")
    if best_metrics is None:
        raise RuntimeError("Checkpoint kaydedilemedi.")

    summary = {
        "stage": "v06_dartis_active_water_adaptation",
        "train_count": int(len(train_df)),
        "validation_count": int(len(val_df)),
        "locked_test_count": int(len(test_df)),
        "group_split_counts": {
            f"{group}_{part}": int(count)
            for (group, part), count in split.groupby(["coastal_group", "split"]).size().items()
        },
        "split_manifest": relative(split_path),
        "locked_test_sample_hash": sample_hash(split, "locked_test"),
        "source_checkpoint": relative(source_path),
        "best_checkpoint": relative(checkpoint_path),
        "best_epoch": int(best_epoch),
        "best_validation_safe_point": bool(best_safe),
        "best_validation_metrics": best_metrics,
        "pos_weight": float(pos_weight_value),
        "training_config": {
            "requested_epochs": args.epochs,
            "completed_epochs": len(history),
            "repeat_factor": args.repeat_factor,
            "batch_size": args.batch_size,
            "train_size": args.train_size,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
        },
        "training_history": relative(history_path),
        "training_seconds": float(time.time() - start),
        "locked_test_used": False,
        "official_sen1floods11_test_used": False,
        "scientific_note": (
            "Sekiz yeni DARTIS örneği split oluşturulduktan sonra kilitli test olarak ayrılmış, "
            "eğitim ve validation datasetleri tarafından açılmamıştır. Test sonraki ayrı aşamada bir kez kullanılacaktır."
        ),
    }
    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 78)
    print("DARTIS ACTIVE ADAPTATION TAMAMLANDI")
    print("=" * 78)
    print(f"Best epoch: {best_epoch}")
    print(f"Validation safe point: {best_safe}")
    print(f"Best validation: {best_metrics}")
    print(f"Checkpoint: {checkpoint_path.resolve()}")
    print(f"Özet: {summary_path.resolve()}")
    print("Locked test kullanıldı mı: False")


if __name__ == "__main__":
    main()
