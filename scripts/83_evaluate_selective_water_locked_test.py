from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from datetime import datetime, timezone
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

ACTIVE_MANIFEST = (
    ROOT
    / "data"
    / "metadata"
    / "v06_dartis_water_active_training_manifest.csv"
)

LOCKED_SPLIT = (
    ROOT
    / "data"
    / "metadata"
    / "v06_dartis_water_locked_split.csv"
)

POLICY_SUMMARY = (
    ROOT
    / "outputs"
    / "v06_dartis_selective_water_gate"
    / "summary.json"
)

CHECKPOINT_DIR = (
    ROOT
    / "checkpoints"
    / "water_unet_dartis_combined_landaware_cv_v06"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "v06_dartis_selective_water_locked_test"
)

CONSUMED_MARKER = (
    ROOT
    / "data"
    / "metadata"
    / "v06_dartis_locked_test_consumed.json"
)


def load_base_module():
    if not BASE_SCRIPT.exists():
        raise FileNotFoundError(
            f"Gerekli 68 scripti bulunamadı: {BASE_SCRIPT}"
        )

    spec = importlib.util.spec_from_file_location(
        "dartis_water_base_v83",
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
            "Geliştirme aşamasında kilitlenen sekiz DARTIS test "
            "örneğini, beş fold modelinden oluşan kalibrasyonlu "
            "ensemble ve önceden seçilmiş güvenlik politikasıyla "
            "yalnız bir kez değerlendirir."
        )
    )

    parser.add_argument(
        "--consume-locked-test",
        action="store_true",
        help=(
            "Bu bayrak testin tek seferlik nihai kullanımını "
            "bilinçli olarak onaylar."
        ),
    )
    parser.add_argument(
        "--minimum-accepted-scenes",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--minimum-water-precision",
        type=float,
        default=0.995,
    )
    parser.add_argument(
        "--maximum-land-leakage",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--maximum-per-image-land-leakage",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--minimum-mean-accepted-recall",
        type=float,
        default=0.60,
    )
    parser.add_argument(
        "--minimum-per-image-water-precision",
        type=float,
        default=0.95,
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
        (array - low) / (high - low),
        0.0, 1.0,
    ).astype(np.float32)


def align_probability(
    probability: np.ndarray,
    threshold: float,
) -> np.ndarray:
    probability = np.clip(
        probability,
        1e-6,
        1.0 - 1e-6,
    )

    threshold = float(
        np.clip(
            threshold,
            1e-4,
            1.0 - 1e-4,
        )
    )

    probability_logit = np.log(
        probability / (1.0 - probability)
    )

    threshold_logit = math.log(
        threshold / (1.0 - threshold)
    )

    shifted = probability_logit - threshold_logit

    return (
        1.0
        / (
            1.0
            + np.exp(
                -np.clip(
                    shifted,
                    -30.0,
                    30.0,
                )
            )
        )
    ).astype(np.float32)


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, float]:
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
        checkpoint["model_state_dict"]
    )

    model.eval()

    threshold = float(
        checkpoint.get(
            "validation_metrics",
            {},
        ).get(
            "threshold",
            0.5,
        )
    )

    return model, threshold


def load_locked_test() -> pd.DataFrame:
    active = pd.read_csv(
        ACTIVE_MANIFEST,
        encoding="utf-8-sig",
        low_memory=False,
    )

    split = pd.read_csv(
        LOCKED_SPLIT,
        encoding="utf-8-sig",
        low_memory=False,
    )

    frame = active.merge(
        split[
            [
                "sample_id",
                "split",
            ]
        ],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )

    locked = frame[
        frame["split"].eq(
            "locked_test"
        )
    ].copy()

    if len(locked) != 8:
        raise RuntimeError(
            "Kilitli test tam sekiz örnek içermiyor. "
            f"Bulunan: {len(locked)}"
        )

    counts = (
        locked.groupby(
            "coastal_group"
        )
        .size()
        .to_dict()
    )

    if counts != {
        "nc": 4,
        "oc": 4,
    }:
        raise RuntimeError(
            "Kilitli test dağılımı nc=4, oc=4 değil: "
            f"{counts}"
        )

    locked["resolved_image_path"] = (
        locked[
            "annotation_image_path"
        ].map(resolve_path)
    )

    locked["resolved_mask_path"] = (
        locked[
            "manual_mask_path"
        ].map(resolve_path)
    )

    for column in (
        "resolved_image_path",
        "resolved_mask_path",
    ):
        missing = locked[
            ~locked[column].map(
                lambda value: Path(value).exists()
            )
        ]

        if not missing.empty:
            raise FileNotFoundError(
                f"{column} için eksik dosya bulundu."
            )

    return locked.reset_index(drop=True)


def compute_scene_metrics(
    mask: np.ndarray,
    safe_water: np.ndarray,
) -> dict[str, float]:
    valid = mask != 128
    water = mask == 255
    land = mask == 0

    tp = int(
        (
            safe_water
            & water
            & valid
        ).sum()
    )

    fp = int(
        (
            safe_water
            & land
            & valid
        ).sum()
    )

    fn = int(
        (
            (~safe_water)
            & water
            & valid
        ).sum()
    )

    tn = int(
        (
            (~safe_water)
            & land
            & valid
        ).sum()
    )

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 1.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 1.0
    )

    iou = (
        tp / (tp + fp + fn)
        if tp + fp + fn > 0
        else 1.0
    )

    leakage = (
        fp / (fp + tn)
        if fp + tn > 0
        else 0.0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "water_precision": float(
            precision
        ),
        "water_recall": float(
            recall
        ),
        "water_iou": float(iou),
        "land_leakage": float(
            leakage
        ),
    }


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


def colorize_error(
    mask: np.ndarray,
    safe_water: np.ndarray,
) -> np.ndarray:
    rgb = np.zeros(
        (
            mask.shape[0],
            mask.shape[1],
            3,
        ),
        dtype=np.uint8,
    )

    valid = mask != 128
    water = mask == 255
    land = mask == 0

    rgb[
        safe_water
        & water
        & valid
    ] = (
        0,
        200,
        0,
    )

    rgb[
        safe_water
        & land
        & valid
    ] = (
        255,
        0,
        0,
    )

    rgb[
        (~safe_water)
        & water
        & valid
    ] = (
        0,
        80,
        255,
    )

    rgb[
        (~safe_water)
        & land
        & valid
    ] = (
        25,
        25,
        25,
    )

    rgb[mask == 128] = (
        255,
        0,
        255,
    )

    return rgb


def add_header(
    image: Image.Image,
    text: str,
    width: int,
) -> Image.Image:
    ratio = width / image.width

    resized = image.resize(
        (
            width,
            max(
                1,
                int(
                    image.height
                    * ratio
                ),
            ),
        ),
        Image.Resampling.LANCZOS,
    )

    header_height = 42

    canvas = Image.new(
        "RGB",
        (
            width,
            resized.height
            + header_height,
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
        (8, 10),
        text,
        fill="white",
        font=font,
    )

    return canvas


def main() -> None:
    args = parse_args()

    if not args.consume_locked_test:
        raise SystemExit(
            "Bu nihai testtir. Çalıştırmak için "
            "--consume-locked-test bayrağını ekleyin."
        )

    if CONSUMED_MARKER.exists():
        previous = CONSUMED_MARKER.read_text(
            encoding="utf-8-sig"
        )

        raise SystemExit(
            "Kilitli test daha önce tüketilmiş. "
            "Tekrar çalıştırılmadı.\n"
            + previous
        )

    for path in (
        ACTIVE_MANIFEST,
        LOCKED_SPLIT,
        POLICY_SUMMARY,
        CHECKPOINT_DIR,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Gerekli kaynak bulunamadı: {path}"
            )

    policy_summary = json.loads(
        POLICY_SUMMARY.read_text(
            encoding="utf-8-sig"
        )
    )

    if not policy_summary.get(
        "policy_passed",
        False,
    ):
        raise RuntimeError(
            "Development seçici politikası PASS değil."
        )

    policy = policy_summary[
        "selected_policy"
    ]

    water_threshold = float(
        policy[
            "water_threshold"
        ]
    )

    land_threshold = float(
        policy[
            "land_threshold"
        ]
    )

    minimum_water_fraction = float(
        policy[
            "minimum_predicted_water_fraction"
        ]
    )

    maximum_uncertain_fraction = float(
        policy[
            "maximum_uncertain_fraction"
        ]
    )

    locked = load_locked_test()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    models = []
    thresholds = []

    for fold_number in range(1, 6):
        checkpoint_path = (
            CHECKPOINT_DIR
            / f"fold_{fold_number:02d}_best.pth"
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint bulunamadı: {checkpoint_path}"
            )

        model, threshold = load_model(
            checkpoint_path,
            device,
        )

        models.append(model)
        thresholds.append(threshold)

    print("=" * 78)
    print(
        "v0.6 SELECTIVE WATER LOCKED TEST"
    )
    print("=" * 78)
    print(
        "Locked test:",
        len(locked),
    )
    print(
        "Ensemble model:",
        len(models),
    )
    print(
        "Water threshold:",
        water_threshold,
    )
    print(
        "Land threshold:",
        land_threshold,
    )
    print(
        "Bu test yalnız bir kez tüketilecek."
    )
    print(
        "Cihaz:",
        device,
    )
    print()

    scene_rows: list[
        dict[str, Any]
    ] = []

    contact_rows = []

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    accepted_metrics = []

    for item in locked.itertuples():
        image = np.array(
            Image.open(
                item.resolved_image_path
            ).convert("L")
        )

        mask = np.array(
            Image.open(
                item.resolved_mask_path
            ).convert("L")
        )

        if image.shape != mask.shape:
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

        aligned_probabilities = []

        with torch.no_grad():
            for model, threshold in zip(
                models,
                thresholds,
            ):
                with torch.amp.autocast(
                    device_type=device.type,
                    enabled=(
                        device.type == "cuda"
                    ),
                ):
                    logits = model(
                        tensor
                    )[0, 0]

                probability = torch.sigmoid(
                    logits.float()
                ).cpu().numpy()

                aligned_probabilities.append(
                    align_probability(
                        probability,
                        threshold,
                    )
                )

        stack = np.stack(
            aligned_probabilities,
            axis=0,
        )

        ensemble_probability = np.median(
            stack,
            axis=0,
        ).astype(np.float32)

        ensemble_std = np.std(
            stack,
            axis=0,
        ).astype(np.float32)

        valid = mask != 128

        safe_water = (
            ensemble_probability
            >= water_threshold
        )

        safe_land = (
            ensemble_probability
            <= land_threshold
        )

        uncertain = (
            valid
            & (~safe_water)
            & (~safe_land)
        )

        valid_count = max(
            int(valid.sum()),
            1,
        )

        predicted_water_fraction = float(
            (
                safe_water
                & valid
            ).sum()
            / valid_count
        )

        uncertain_fraction = float(
            uncertain.sum()
            / valid_count
        )

        accepted = (
            predicted_water_fraction
            >= minimum_water_fraction
            and uncertain_fraction
            <= maximum_uncertain_fraction
        )

        metrics = compute_scene_metrics(
            mask,
            safe_water,
        )

        if accepted:
            accepted_metrics.append(
                metrics
            )

            total_tp += int(
                metrics["tp"]
            )
            total_fp += int(
                metrics["fp"]
            )
            total_fn += int(
                metrics["fn"]
            )
            total_tn += int(
                metrics["tn"]
            )

        scene_row = {
            "sample_id": str(
                item.sample_id
            ),
            "coastal_group": str(
                item.coastal_group
            ),
            "accepted": bool(
                accepted
            ),
            "decision": (
                "ACCEPT_WATER"
                if accepted
                else "UNCERTAIN_BLOCK"
            ),
            "predicted_water_fraction": (
                predicted_water_fraction
            ),
            "uncertain_fraction": (
                uncertain_fraction
            ),
            "mean_ensemble_std": float(
                ensemble_std.mean()
            ),
            **{
                key: value
                for key, value
                in metrics.items()
                if key not in {
                    "tp",
                    "fp",
                    "fn",
                    "tn",
                }
            },
        }

        scene_rows.append(
            scene_row
        )

        image_rgb = np.stack(
            [image, image, image],
            axis=2,
        )

        probability_uint8 = (
            np.clip(
                ensemble_probability,
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

        internal_mask = np.zeros_like(
            image_rgb
        )

        internal_mask[
            safe_water
        ] = (
            255,
            255,
            255,
        )

        internal_mask[
            uncertain
        ] = (
            255,
            0,
            255,
        )

        operational_mask = np.zeros_like(
            image_rgb
        )

        if accepted:
            operational_mask[
                safe_water
            ] = (
                255,
                255,
                255,
            )

        error_map = colorize_error(
            mask,
            safe_water,
        )

        panel = np.concatenate(
            [
                image_rgb,
                colorize_ground_truth(
                    mask
                ),
                heatmap,
                internal_mask,
                operational_mask,
                error_map,
            ],
            axis=1,
        )

        title = (
            f"{item.sample_id} | "
            f"{scene_row['decision']} | "
            f"water={predicted_water_fraction:.3f} "
            f"uncertain={uncertain_fraction:.3f} "
            f"P={metrics['water_precision']:.4f} "
            f"R={metrics['water_recall']:.4f} "
            f"IoU={metrics['water_iou']:.4f} "
            f"Leak={metrics['land_leakage']:.4f}"
        )

        contact_rows.append(
            add_header(
                Image.fromarray(panel),
                title,
                1800,
            )
        )

        print(
            f"{item.sample_id}: "
            f"{scene_row['decision']} "
            f"| P {metrics['water_precision']:.4f} "
            f"R {metrics['water_recall']:.4f} "
            f"IoU {metrics['water_iou']:.4f} "
            f"Leak {metrics['land_leakage']:.4f}"
        )

    accepted_count = len(
        accepted_metrics
    )

    if accepted_count > 0:
        precision = (
            total_tp
            / (total_tp + total_fp)
            if total_tp + total_fp > 0
            else 1.0
        )

        pixel_recall = (
            total_tp
            / (total_tp + total_fn)
            if total_tp + total_fn > 0
            else 1.0
        )

        iou = (
            total_tp
            / (
                total_tp
                + total_fp
                + total_fn
            )
            if (
                total_tp
                + total_fp
                + total_fn
            ) > 0
            else 1.0
        )

        leakage = (
            total_fp
            / (total_fp + total_tn)
            if total_fp + total_tn > 0
            else 0.0
        )

        mean_recall = float(
            np.mean(
                [
                    item[
                        "water_recall"
                    ]
                    for item in accepted_metrics
                ]
            )
        )

        minimum_precision = float(
            min(
                item[
                    "water_precision"
                ]
                for item in accepted_metrics
            )
        )

        maximum_leakage = float(
            max(
                item[
                    "land_leakage"
                ]
                for item in accepted_metrics
            )
        )
    else:
        precision = 0.0
        pixel_recall = 0.0
        iou = 0.0
        leakage = 1.0
        mean_recall = 0.0
        minimum_precision = 0.0
        maximum_leakage = 1.0

    checks = {
        "minimum_accepted_scenes": (
            accepted_count
            >= args.minimum_accepted_scenes
        ),
        "minimum_water_precision": (
            precision
            >= args.minimum_water_precision
        ),
        "maximum_land_leakage": (
            leakage
            <= args.maximum_land_leakage
        ),
        "maximum_per_image_land_leakage": (
            maximum_leakage
            <= args.maximum_per_image_land_leakage
        ),
        "minimum_mean_accepted_recall": (
            mean_recall
            >= args.minimum_mean_accepted_recall
        ),
        "minimum_per_image_water_precision": (
            minimum_precision
            >= args.minimum_per_image_water_precision
        ),
    }

    failed = [
        name
        for name, passed in checks.items()
        if not passed
    ]

    final_gate = len(failed) == 0

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    per_scene_path = (
        OUTPUT_DIR
        / "locked_test_per_scene.csv"
    )

    pd.DataFrame(
        scene_rows
    ).to_csv(
        per_scene_path,
        index=False,
        encoding="utf-8-sig",
    )

    contact_sheet = Image.new(
        "RGB",
        (
            1800,
            sum(
                row.height
                for row in contact_rows
            ),
        ),
        "black",
    )

    y = 0

    for row in contact_rows:
        contact_sheet.paste(
            row,
            (0, y),
        )
        y += row.height

    contact_sheet_path = (
        OUTPUT_DIR
        / "locked_test_contact_sheet.jpg"
    )

    contact_sheet.save(
        contact_sheet_path,
        quality=92,
    )

    summary = {
        "stage": (
            "v06_dartis_selective_water_locked_test"
        ),
        "locked_test_count": 8,
        "locked_test_consumed": True,
        "ensemble_model_count": 5,
        "ensemble_aggregation": (
            "median of per-model logit-aligned probabilities"
        ),
        "policy": {
            "water_threshold": (
                water_threshold
            ),
            "land_threshold": (
                land_threshold
            ),
            "minimum_predicted_water_fraction": (
                minimum_water_fraction
            ),
            "maximum_uncertain_fraction": (
                maximum_uncertain_fraction
            ),
        },
        "accepted_count": int(
            accepted_count
        ),
        "rejected_count": int(
            8 - accepted_count
        ),
        "accepted_rate": float(
            accepted_count / 8
        ),
        "accepted_scene_metrics": {
            "water_precision": float(
                precision
            ),
            "mean_water_recall": float(
                mean_recall
            ),
            "pixel_water_recall": float(
                pixel_recall
            ),
            "water_iou": float(iou),
            "land_leakage": float(
                leakage
            ),
            "minimum_per_image_water_precision": float(
                minimum_precision
            ),
            "maximum_per_image_land_leakage": float(
                maximum_leakage
            ),
        },
        "final_gate_passed": bool(
            final_gate
        ),
        "failed_reasons": failed,
        "criteria": {
            "minimum_accepted_scenes": (
                args.minimum_accepted_scenes
            ),
            "minimum_water_precision": (
                args.minimum_water_precision
            ),
            "maximum_land_leakage": (
                args.maximum_land_leakage
            ),
            "maximum_per_image_land_leakage": (
                args.maximum_per_image_land_leakage
            ),
            "minimum_mean_accepted_recall": (
                args.minimum_mean_accepted_recall
            ),
            "minimum_per_image_water_precision": (
                args.minimum_per_image_water_precision
            ),
        },
        "per_scene_results": relative(
            per_scene_path
        ),
        "contact_sheet": relative(
            contact_sheet_path
        ),
        "operational_rule": (
            "ACCEPT_WATER sahnelerinde yalnız yüksek güvenli "
            "su pikselleri petrol analizine aktarılır. "
            "UNCERTAIN_BLOCK sahnelerinde petrol analizi durdurulur "
            "ve petrol maskesi üretilmez."
        ),
        "scientific_limitation": (
            "Kilitli test yalnız sekiz aktif-öğrenme zorluk "
            "örneğinden oluşur; sonuçlar prototip doğrulamasıdır "
            "ve geniş ölçekli operasyonel genelleme kanıtı değildir."
        ),
    }

    summary_path = (
        OUTPUT_DIR
        / "summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    marker = {
        "consumed_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "script": (
            "83_evaluate_selective_water_locked_test.py"
        ),
        "summary": relative(
            summary_path
        ),
        "final_gate_passed": bool(
            final_gate
        ),
    }

    CONSUMED_MARKER.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    CONSUMED_MARKER.write_text(
        json.dumps(
            marker,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print(
        "LOCKED TEST NİHAİ SONUCU"
    )
    print("=" * 78)
    print(
        "Final gate:",
        "PASS"
        if final_gate
        else "FAIL",
    )
    print(
        "Accepted scenes:",
        f"{accepted_count}/8",
    )
    print(
        "Water precision:",
        f"{precision:.4f}",
    )
    print(
        "Mean accepted recall:",
        f"{mean_recall:.4f}",
    )
    print(
        "Pixel recall:",
        f"{pixel_recall:.4f}",
    )
    print(
        "Water IoU:",
        f"{iou:.4f}",
    )
    print(
        "Land leakage:",
        f"{leakage:.4f}",
    )
    print(
        "Worst accepted leakage:",
        f"{maximum_leakage:.4f}",
    )
    print(
        "Min accepted precision:",
        f"{minimum_precision:.4f}",
    )
    print(
        "Failed:",
        failed,
    )
    print(
        "Kilitli test tüketildi: True"
    )
    print(
        "Özet:",
        summary_path.resolve(),
    )
    print(
        "Görsel:",
        contact_sheet_path.resolve(),
    )


if __name__ == "__main__":
    main()
