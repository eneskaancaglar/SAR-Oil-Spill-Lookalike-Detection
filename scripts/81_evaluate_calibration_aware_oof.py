from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = ROOT / "scripts" / "68_evaluate_dartis_water_transfer_gate.py"
MANIFEST = ROOT / "data" / "metadata" / "v06_dartis_water_combined_development_manifest.csv"
CV_SUMMARY = ROOT / "outputs" / "water_unet_dartis_combined_landaware_cv_v06" / "summary.json"
CHECKPOINT_DIR = ROOT / "checkpoints" / "water_unet_dartis_combined_landaware_cv_v06"
OUTPUT_DIR = ROOT / "outputs" / "v06_dartis_calibration_aware_oof"


def load_base_module():
    spec = importlib.util.spec_from_file_location("dartis_base_v81", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("68 scripti yüklenemedi.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_base_module()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


def normalize_percentile(image: np.ndarray) -> np.ndarray:
    array = image.astype(np.float32)
    low = float(np.percentile(array, 2))
    high = float(np.percentile(array, 98))
    if high <= low:
        return (array / 255.0).astype(np.float32)
    return np.clip((array - low) / (high - low), 0.0, 1.0).astype(np.float32)


def load_model(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    base_channels = int(checkpoint.get("config", {}).get("base_channels", 16))
    model = BASE.SmallUNet(in_channels=2, base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    threshold = float(
        checkpoint.get("validation_metrics", {}).get("threshold", 0.5)
    )
    return model, threshold, int(checkpoint.get("best_epoch", 0))


def align_probability(probability: np.ndarray, threshold: float) -> np.ndarray:
    p = np.clip(probability, 1e-6, 1.0 - 1e-6)
    t = float(np.clip(threshold, 1e-4, 1.0 - 1e-4))
    shifted = np.log(p / (1.0 - p)) - math.log(t / (1.0 - t))
    return (1.0 / (1.0 + np.exp(-np.clip(shifted, -30.0, 30.0)))).astype(
        np.float32
    )


def evaluate(rows: list[dict], key: str, threshold: float):
    total_tp = total_fp = total_fn = total_tn = 0
    per_image = []

    for row in rows:
        probability = row[key]
        mask = row["mask"]
        valid = mask != 128
        water = mask == 255
        land = mask == 0
        predicted = probability >= threshold

        tp = int((predicted & water & valid).sum())
        fp = int((predicted & land & valid).sum())
        fn = int(((~predicted) & water & valid).sum())
        tn = int(((~predicted) & land & valid).sum())

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

        precision = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 1.0
        iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
        leakage = fp / (fp + tn) if fp + tn else 0.0

        per_image.append(
            {
                "sample_id": row["sample_id"],
                "fold": row["fold"],
                "fold_threshold": row["fold_threshold"],
                "precision": precision,
                "recall": recall,
                "iou": iou,
                "land_leakage": leakage,
            }
        )

    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 1.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 1.0
    iou = (
        total_tp / (total_tp + total_fp + total_fn)
        if total_tp + total_fp + total_fn
        else 1.0
    )
    leakage = (
        total_fp / (total_fp + total_tn)
        if total_fp + total_tn
        else 0.0
    )

    metrics = {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "land_leakage": leakage,
        "minimum_per_image_iou": min(item["iou"] for item in per_image),
        "minimum_per_image_recall": min(item["recall"] for item in per_image),
        "maximum_per_image_land_leakage": max(
            item["land_leakage"] for item in per_image
        ),
    }
    return metrics, per_image


def main():
    frame = pd.read_csv(MANIFEST, encoding="utf-8-sig", low_memory=False)
    if len(frame) != 40:
        raise RuntimeError(f"40 development örneği bekleniyordu, bulunan: {len(frame)}")

    frame["resolved_image"] = frame["annotation_image_path"].map(resolve_path)
    frame["resolved_mask"] = frame["manual_mask_path"].map(resolve_path)

    cv_summary = json.loads(CV_SUMMARY.read_text(encoding="utf-8-sig"))
    raw_global_threshold = float(cv_summary["oof_metrics"]["threshold"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    fold_report = []

    print("=" * 78)
    print("v0.6 CALIBRATION-AWARE OOF")
    print("=" * 78)
    print("Development: 40")
    print("Kilitli test kullanılmayacak.")
    print("Cihaz:", device)
    print()

    for fold_index in range(5):
        fold_number = fold_index + 1
        checkpoint_path = CHECKPOINT_DIR / f"fold_{fold_number:02d}_best.pth"
        model, fold_threshold, best_epoch = load_model(checkpoint_path, device)
        validation = frame[frame["cv_fold"].eq(fold_index)].copy()

        if len(validation) != 8:
            raise RuntimeError(f"Fold {fold_number} için 8 OOF örneği bulunamadı.")

        fold_rows = []

        with torch.no_grad():
            for item in validation.itertuples():
                image = np.array(Image.open(item.resolved_image).convert("L"))
                mask = np.array(Image.open(item.resolved_mask).convert("L"))
                normalized = normalize_percentile(image)
                tensor = torch.from_numpy(
                    np.stack([normalized, normalized], axis=0).astype(np.float32)
                ).unsqueeze(0).to(device)

                with torch.amp.autocast(
                    device_type=device.type,
                    enabled=device.type == "cuda",
                ):
                    logits = model(tensor)[0, 0]

                raw_probability = torch.sigmoid(logits.float()).cpu().numpy()
                aligned_probability = align_probability(
                    raw_probability,
                    fold_threshold,
                )

                record = {
                    "sample_id": str(item.sample_id),
                    "fold": fold_number,
                    "fold_threshold": fold_threshold,
                    "raw_probability": raw_probability,
                    "aligned_probability": aligned_probability,
                    "mask": mask,
                }
                rows.append(record)
                fold_rows.append(record)

        fold_metrics, _ = evaluate(
            fold_rows,
            "aligned_probability",
            0.5,
        )
        fold_report.append(
            {
                "fold": fold_number,
                "best_epoch": best_epoch,
                "original_threshold": fold_threshold,
                **fold_metrics,
            }
        )

        print(
            f"Fold {fold_number}: original thr {fold_threshold:.2f} | "
            f"P {fold_metrics['precision']:.4f} "
            f"R {fold_metrics['recall']:.4f} "
            f"IoU {fold_metrics['iou']:.4f} "
            f"Leak {fold_metrics['land_leakage']:.4f}"
        )

    raw_metrics, _ = evaluate(rows, "raw_probability", raw_global_threshold)
    aligned_metrics, per_image = evaluate(rows, "aligned_probability", 0.5)

    criteria = {
        "precision": aligned_metrics["precision"] >= 0.95,
        "iou": aligned_metrics["iou"] >= 0.60,
        "land_leakage": aligned_metrics["land_leakage"] <= 0.05,
        "per_image_land_leakage": (
            aligned_metrics["maximum_per_image_land_leakage"] <= 0.15
        ),
        "per_image_iou": aligned_metrics["minimum_per_image_iou"] >= 0.35,
        "per_image_recall": aligned_metrics["minimum_per_image_recall"] >= 0.50,
    }
    failed = [name for name, passed in criteria.items() if not passed]
    gate = not failed

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fold_report).to_csv(
        OUTPUT_DIR / "fold_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(per_image).sort_values(
        ["iou", "land_leakage"],
        ascending=[True, False],
    ).to_csv(
        OUTPUT_DIR / "per_image_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "stage": "v06_dartis_calibration_aware_oof",
        "development_count": 40,
        "locked_test_used": False,
        "training_performed": False,
        "raw_global_metrics": raw_metrics,
        "calibration_aware_metrics": aligned_metrics,
        "gate_passed": gate,
        "failed_reasons": failed,
        "calibration_method": (
            "Her foldun seçilmiş olasılık eşiği logit uzayında hizalanıp "
            "0.5'e taşındı."
        ),
        "scientific_note": (
            "Bu yalnız development tanısıdır. Fold eşiği aynı foldun validation "
            "etiketlerinden seçildiği için bağımsız final test kanıtı değildir."
        ),
    }
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("CALIBRATION-AWARE OOF SONUCU")
    print("=" * 78)
    print(
        "Raw global:",
        f"P {raw_metrics['precision']:.4f}",
        f"R {raw_metrics['recall']:.4f}",
        f"IoU {raw_metrics['iou']:.4f}",
        f"Leak {raw_metrics['land_leakage']:.4f}",
        f"Min IoU {raw_metrics['minimum_per_image_iou']:.4f}",
    )
    print(
        "Aligned:",
        f"P {aligned_metrics['precision']:.4f}",
        f"R {aligned_metrics['recall']:.4f}",
        f"IoU {aligned_metrics['iou']:.4f}",
        f"Leak {aligned_metrics['land_leakage']:.4f}",
        f"Worst leak {aligned_metrics['maximum_per_image_land_leakage']:.4f}",
        f"Min IoU {aligned_metrics['minimum_per_image_iou']:.4f}",
        f"Min recall {aligned_metrics['minimum_per_image_recall']:.4f}",
    )
    print("Gate:", "PASS" if gate else "FAIL")
    print("Başarısız koşullar:", failed)
    print("Kilitli test kullanıldı mı: False")
    print("Özet:", (OUTPUT_DIR / "summary.json").resolve())


if __name__ == "__main__":
    main()
