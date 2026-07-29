from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]

BASE_SCRIPT = (
    ROOT
    / "scripts"
    / "68_evaluate_dartis_water_transfer_gate.py"
)

DEFAULT_OOF = (
    ROOT
    / "outputs"
    / "water_unet_dartis_landaware_cv_v06"
    / "oof_per_image_metrics.csv"
)

DEFAULT_CV = (
    ROOT
    / "outputs"
    / "water_unet_dartis_landaware_cv_v06"
    / "development_cv_folds.csv"
)

DEFAULT_MANIFEST = (
    ROOT
    / "data"
    / "metadata"
    / "v06_dartis_water_active_training_manifest.csv"
)

DEFAULT_SUMMARY = (
    ROOT
    / "outputs"
    / "water_unet_dartis_landaware_cv_v06"
    / "summary.json"
)

DEFAULT_CHECKPOINT_DIR = (
    ROOT
    / "checkpoints"
    / "water_unet_dartis_landaware_cv_v06"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "v06_dartis_oof_failure_diagnosis"
)


def load_base_module():
    if not BASE_SCRIPT.exists():
        raise FileNotFoundError(
            f"Gerekli 68 scripti bulunamadı: {BASE_SCRIPT}"
        )

    spec = importlib.util.spec_from_file_location(
        "dartis_water_base_v78",
        BASE_SCRIPT,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "68 scripti modül olarak yüklenemedi."
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_base_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "DARTIS land-aware CV sonucundaki en kritik OOF "
            "başarısızlıklarını görselleştirir. Maske değiştirmez, "
            "eğitim yapmaz ve kilitli testi açmaz."
        )
    )

    parser.add_argument(
        "--oof",
        type=Path,
        default=DEFAULT_OOF,
    )
    parser.add_argument(
        "--cv",
        type=Path,
        default=DEFAULT_CV,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
    )
    parser.add_argument(
        "--top-leakage",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--top-low-iou",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--contact-width",
        type=int,
        default=1800,
    )

    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)

    if not path.is_absolute():
        path = ROOT / path

    return path.resolve()


def relative(path: Path) -> str:
    try:
        return str(
            path.resolve().relative_to(ROOT.resolve())
        ).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def normalize_percentile(
    image: np.ndarray,
) -> np.ndarray:
    array = image.astype(np.float32)

    low = float(np.percentile(array, 2))
    high = float(np.percentile(array, 98))

    if high <= low:
        return (
            array / 255.0
        ).astype(np.float32)

    return np.clip(
        (array - low)
        / (high - low),
        0.0,
        1.0,
    ).astype(np.float32)


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    base_channels = int(
        checkpoint.get(
            "config",
            {},
        ).get(
            "base_channels",
            16,
        )
    )

    model = BASE.SmallUNet(
        in_channels=2,
        base_channels=base_channels,
    ).to(device)

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()
    return model


def colorize_ground_truth(
    mask: np.ndarray,
) -> np.ndarray:
    rgb = np.zeros(
        (
            mask.shape[0],
            mask.shape[1],
            3,
        ),
        dtype=np.uint8,
    )

    rgb[mask == 255] = (
        0,
        255,
        255,
    )

    rgb[mask == 128] = (
        255,
        0,
        255,
    )

    return rgb


def colorize_prediction(
    predicted_water: np.ndarray,
) -> np.ndarray:
    rgb = np.zeros(
        (
            predicted_water.shape[0],
            predicted_water.shape[1],
            3,
        ),
        dtype=np.uint8,
    )

    rgb[predicted_water] = (
        255,
        255,
        255,
    )

    return rgb


def create_overlay(
    image: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    rgb = np.stack(
        [image, image, image],
        axis=2,
    ).astype(np.float32)

    painted = rgb.copy()
    painted[mask == 255] = (
        0,
        255,
        255,
    )
    painted[mask == 128] = (
        255,
        0,
        255,
    )

    return np.clip(
        0.60 * rgb + 0.40 * painted,
        0,
        255,
    ).astype(np.uint8)


def create_error_map(
    raw_mask: np.ndarray,
    predicted_water: np.ndarray,
) -> np.ndarray:
    rgb = np.zeros(
        (
            raw_mask.shape[0],
            raw_mask.shape[1],
            3,
        ),
        dtype=np.uint8,
    )

    valid = raw_mask != 128
    water = raw_mask == 255
    land = raw_mask == 0

    true_positive = (
        predicted_water
        & water
        & valid
    )

    false_positive = (
        predicted_water
        & land
        & valid
    )

    false_negative = (
        (~predicted_water)
        & water
        & valid
    )

    true_negative = (
        (~predicted_water)
        & land
        & valid
    )

    rgb[true_positive] = (
        0,
        200,
        0,
    )

    rgb[false_positive] = (
        255,
        0,
        0,
    )

    rgb[false_negative] = (
        0,
        80,
        255,
    )

    rgb[true_negative] = (
        30,
        30,
        30,
    )

    rgb[raw_mask == 128] = (
        255,
        0,
        255,
    )

    return rgb


def add_title(
    image: Image.Image,
    title: str,
    width: int,
) -> Image.Image:
    ratio = width / image.width

    resized = image.resize(
        (
            width,
            max(
                1,
                int(
                    image.height * ratio
                ),
            ),
        ),
        Image.Resampling.LANCZOS,
    )

    header_height = 44

    canvas = Image.new(
        "RGB",
        (
            width,
            resized.height + header_height,
        ),
        "black",
    )

    canvas.paste(
        resized,
        (0, header_height),
    )

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    draw.text(
        (8, 8),
        title,
        fill="white",
        font=font,
    )

    return canvas


def main() -> None:
    args = parse_args()

    oof_path = resolve_path(args.oof)
    cv_path = resolve_path(args.cv)
    manifest_path = resolve_path(
        args.manifest
    )
    summary_path = resolve_path(
        args.summary
    )
    checkpoint_dir = resolve_path(
        args.checkpoint_dir
    )

    for path in (
        oof_path,
        cv_path,
        manifest_path,
        summary_path,
        checkpoint_dir,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Gerekli kaynak bulunamadı: {path}"
            )

    oof = pd.read_csv(
        oof_path,
        encoding="utf-8-sig",
        low_memory=False,
    )

    cv = pd.read_csv(
        cv_path,
        encoding="utf-8-sig",
        low_memory=False,
    )

    manifest = pd.read_csv(
        manifest_path,
        encoding="utf-8-sig",
        low_memory=False,
    )

    summary = json.loads(
        summary_path.read_text(
            encoding="utf-8-sig"
        )
    )

    threshold = float(
        summary[
            "oof_metrics"
        ][
            "threshold"
        ]
    )

    combined = (
        oof.merge(
            cv,
            on="sample_id",
            how="left",
            validate="one_to_one",
        )
        .merge(
            manifest[
                [
                    "sample_id",
                    "annotation_image_path",
                    "manual_mask_path",
                ]
            ],
            on="sample_id",
            how="left",
            validate="one_to_one",
        )
    )

    leakage_ids = (
        combined.sort_values(
            "land_leakage",
            ascending=False,
        )
        .head(
            args.top_leakage
        )[
            "sample_id"
        ]
        .astype(str)
        .tolist()
    )

    low_iou_ids = (
        combined.sort_values(
            "iou",
            ascending=True,
        )
        .head(
            args.top_low_iou
        )[
            "sample_id"
        ]
        .astype(str)
        .tolist()
    )

    selected_ids = []

    for sample_id in (
        leakage_ids
        + low_iou_ids
    ):
        if sample_id not in selected_ids:
            selected_ids.append(
                sample_id
            )

    selected = combined[
        combined[
            "sample_id"
        ].astype(str).isin(
            selected_ids
        )
    ].copy()

    selected[
        "selection_reason"
    ] = selected[
        "sample_id"
    ].map(
        lambda sample_id: "|".join(
            reason
            for reason, group
            in (
                (
                    "high_land_leakage",
                    leakage_ids,
                ),
                (
                    "low_iou",
                    low_iou_ids,
                ),
            )
            if str(sample_id) in group
        )
    )

    selected = selected.sort_values(
        [
            "land_leakage",
            "iou",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(drop=True)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []
    report_rows: list[
        dict[str, Any]
    ] = []

    model_cache: dict[
        int,
        torch.nn.Module
    ] = {}

    for item in selected.itertuples():
        fold_number = int(
            item.cv_fold
        ) + 1

        checkpoint_path = (
            checkpoint_dir
            / f"fold_{fold_number:02d}_best.pth"
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Fold checkpoint bulunamadı: {checkpoint_path}"
            )

        if fold_number not in model_cache:
            model_cache[
                fold_number
            ] = load_model(
                checkpoint_path,
                device,
            )

        model = model_cache[
            fold_number
        ]

        image_path = resolve_path(
            item.annotation_image_path
        )
        mask_path = resolve_path(
            item.manual_mask_path
        )

        image = np.array(
            Image.open(
                image_path
            ).convert("L")
        )

        raw_mask = np.array(
            Image.open(
                mask_path
            ).convert("L")
        )

        if image.shape != raw_mask.shape:
            raise RuntimeError(
                f"Boyut uyuşmazlığı: {item.sample_id}"
            )

        normalized = normalize_percentile(
            image
        )

        tensor = torch.from_numpy(
            np.stack(
                [
                    normalized,
                    normalized,
                ],
                axis=0,
            ).astype(np.float32)
        ).unsqueeze(0).to(device)

        with torch.no_grad():
            with torch.amp.autocast(
                device_type=device.type,
                enabled=(
                    device.type
                    == "cuda"
                ),
            ):
                logits = model(
                    tensor
                )[0, 0]

            probability = torch.sigmoid(
                logits.float()
            ).cpu().numpy()

        predicted_water = (
            probability >= threshold
        )

        image_rgb = np.stack(
            [image, image, image],
            axis=2,
        )

        probability_uint8 = (
            np.clip(
                probability,
                0.0,
                1.0,
            )
            * 255.0
        ).astype(np.uint8)

        heatmap = cv2.applyColorMap(
            probability_uint8,
            cv2.COLORMAP_TURBO,
        )

        heatmap = cv2.cvtColor(
            heatmap,
            cv2.COLOR_BGR2RGB,
        )

        panels = [
            image_rgb,
            colorize_ground_truth(
                raw_mask
            ),
            heatmap,
            colorize_prediction(
                predicted_water
            ),
            create_overlay(
                image,
                raw_mask,
            ),
            create_error_map(
                raw_mask,
                predicted_water,
            ),
        ]

        panel_height = min(
            panel.shape[0]
            for panel in panels
        )

        normalized_panels = []

        for panel in panels:
            if panel.shape[0] != panel_height:
                scale = (
                    panel_height
                    / panel.shape[0]
                )

                panel = cv2.resize(
                    panel,
                    (
                        max(
                            1,
                            int(
                                panel.shape[1]
                                * scale
                            ),
                        ),
                        panel_height,
                    ),
                    interpolation=cv2.INTER_AREA,
                )

            normalized_panels.append(
                panel
            )

        row_image = np.concatenate(
            normalized_panels,
            axis=1,
        )

        title = (
            f"{item.sample_id} | fold {fold_number} | "
            f"P={float(item.precision):.4f} "
            f"R={float(item.recall):.4f} "
            f"IoU={float(item.iou):.4f} "
            f"Leak={float(item.land_leakage):.4f} "
            f"| thr={threshold:.2f} "
            f"| {item.selection_reason}"
        )

        rows.append(
            add_title(
                Image.fromarray(
                    row_image
                ),
                title,
                args.contact_width,
            )
        )

        report_rows.append(
            {
                "sample_id": str(
                    item.sample_id
                ),
                "fold": fold_number,
                "coastal_group": str(
                    item.coastal_group
                ),
                "selection_reason": str(
                    item.selection_reason
                ),
                "threshold": threshold,
                "land_fraction": float(
                    item.land_fraction
                ),
                "water_fraction": float(
                    item.water_fraction
                ),
                "precision": float(
                    item.precision
                ),
                "recall": float(
                    item.recall
                ),
                "iou": float(
                    item.iou
                ),
                "land_leakage": float(
                    item.land_leakage
                ),
                "checkpoint": relative(
                    checkpoint_path
                ),
                "image_path": relative(
                    image_path
                ),
                "mask_path": relative(
                    mask_path
                ),
            }
        )

    sheet = Image.new(
        "RGB",
        (
            args.contact_width,
            sum(
                row.height
                for row in rows
            ),
        ),
        "black",
    )

    y = 0

    for row in rows:
        sheet.paste(
            row,
            (0, y),
        )
        y += row.height

    contact_sheet_path = (
        OUTPUT_DIR
        / "critical_oof_failures.jpg"
    )

    sheet.save(
        contact_sheet_path,
        quality=92,
    )

    report_path = (
        OUTPUT_DIR
        / "critical_oof_failures.csv"
    )

    pd.DataFrame(
        report_rows
    ).to_csv(
        report_path,
        index=False,
        encoding="utf-8-sig",
    )

    diagnosis = {
        "stage": (
            "v06_dartis_oof_failure_diagnosis"
        ),
        "selected_count": len(
            report_rows
        ),
        "selected_samples": [
            row["sample_id"]
            for row in report_rows
        ],
        "global_oof_threshold": (
            threshold
        ),
        "contact_sheet": relative(
            contact_sheet_path
        ),
        "report_csv": relative(
            report_path
        ),
        "panel_order": [
            "original_sar",
            "manual_ground_truth",
            "probability_heatmap",
            "prediction_at_global_threshold",
            "ground_truth_overlay",
            (
                "error_map: green=TP, red=FP, "
                "blue=FN, dark=TN, magenta=ignore"
            ),
        ],
        "training_performed": False,
        "masks_modified": False,
        "locked_test_used": False,
    }

    diagnosis_path = (
        OUTPUT_DIR
        / "summary.json"
    )

    diagnosis_path.write_text(
        json.dumps(
            diagnosis,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("=" * 78)
    print(
        "DARTIS OOF KRİTİK HATA GÖRSELLEŞTİRMESİ"
    )
    print("=" * 78)
    print(
        "Seçilen görüntü:",
        len(report_rows),
    )
    print(
        "Örnekler:",
        [
            row["sample_id"]
            for row in report_rows
        ],
    )
    print(
        "Global threshold:",
        threshold,
    )
    print(
        "Görsel:",
        contact_sheet_path.resolve(),
    )
    print(
        "Rapor:",
        report_path.resolve(),
    )
    print(
        "Locked test kullanıldı mı: False"
    )
    print(
        "Maske değiştirildi mi: False"
    )


if __name__ == "__main__":
    main()
