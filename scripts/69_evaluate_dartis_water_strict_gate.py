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


ROOT = Path(__file__).resolve().parents[1]

BASE_SCRIPT = (
    ROOT
    / "scripts"
    / "68_evaluate_dartis_water_transfer_gate.py"
)

DEFAULT_CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "water_unet_flood_v06_robust"
    / "best.pth"
)

DEFAULT_DARTIS_MANIFEST = (
    ROOT
    / "data"
    / "metadata"
    / "v06_dartis_water_masks_final.csv"
)

DEFAULT_MANUAL_MANIFEST = (
    ROOT
    / "data"
    / "metadata"
    / "v06_manual_land_corrections.csv"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "v06_dartis_water_strict_gate"
)


def load_base_module():
    if not BASE_SCRIPT.exists():
        raise FileNotFoundError(
            f"Gerekli 68 scripti bulunamadı: {BASE_SCRIPT}"
        )

    spec = importlib.util.spec_from_file_location(
        "dartis_transfer_base",
        BASE_SCRIPT,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "68 scripti modül olarak yüklenemedi."
        )

    module = importlib.util.module_from_spec(
        spec
    )

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    return module


BASE = load_base_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "DARTIS kara-su aktarımını sıkı güvenlik koşullarıyla "
            "yeniden değerlendirir. Yalnız kıyı tamponu ve bileşen "
            "filtresi etkin olan conservative postprocessing "
            "kombinasyonlarını kabul eder."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )

    parser.add_argument(
        "--dartis-manifest",
        type=Path,
        default=DEFAULT_DARTIS_MANIFEST,
    )

    parser.add_argument(
        "--manual-manifest",
        type=Path,
        default=DEFAULT_MANUAL_MANIFEST,
    )

    parser.add_argument(
        "--minimum-water-precision",
        type=float,
        default=0.95,
    )

    parser.add_argument(
        "--minimum-water-iou",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--maximum-land-leakage",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--maximum-per-image-land-leakage",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--coast-buffer-pixels",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--minimum-component-pixels",
        type=int,
        default=256,
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
            path.resolve().relative_to(
                ROOT.resolve()
            )
        ).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def load_records(
    dartis_manifest_path: Path,
    manual_manifest_path: Path,
) -> pd.DataFrame:
    return BASE.load_calibration_records(
        dartis_manifest_path,
        manual_manifest_path,
    )


def resize_image_to_mask(
    image: np.ndarray,
    mask_shape: tuple[int, int],
) -> np.ndarray:
    if image.shape == mask_shape:
        return image

    target_height, target_width = (
        mask_shape
    )

    interpolation = (
        cv2.INTER_AREA
        if (
            image.shape[0] > target_height
            or image.shape[1] > target_width
        )
        else cv2.INTER_CUBIC
    )

    return cv2.resize(
        image,
        (
            target_width,
            target_height,
        ),
        interpolation=interpolation,
    )


def evaluate_configuration(
    records: pd.DataFrame,
    cached: dict[
        tuple[str, str],
        dict[str, np.ndarray],
    ],
    preprocessing: str,
    threshold: float,
    coast_buffer_pixels: int,
    minimum_component_pixels: int,
) -> tuple[
    dict[str, float],
    list[dict[str, Any]],
]:
    aggregate_counts = (
        BASE.empty_counts()
    )

    sample_rows: list[
        dict[str, Any]
    ] = []

    for row in records.to_dict(
        orient="records"
    ):
        sample_id = str(
            row["sample_id"]
        )

        base = cached[
            (
                sample_id,
                "base",
            )
        ]

        probability = cached[
            (
                sample_id,
                preprocessing,
            )
        ]["probability"]

        prediction = (
            probability >= threshold
        )

        prediction = (
            BASE.apply_postprocessing(
                prediction=prediction,
                mode="conservative",
                coast_buffer_pixels=(
                    coast_buffer_pixels
                ),
                minimum_component_pixels=(
                    minimum_component_pixels
                ),
            )
        )

        sample_counts = (
            BASE.empty_counts()
        )

        BASE.update_counts(
            counts=sample_counts,
            prediction=prediction,
            truth_water=base[
                "safe_water"
            ],
            valid=base["valid"],
        )

        BASE.update_counts(
            counts=aggregate_counts,
            prediction=prediction,
            truth_water=base[
                "safe_water"
            ],
            valid=base["valid"],
        )

        sample_metrics = (
            BASE.finalize_counts(
                sample_counts
            )
        )

        sample_rows.append(
            {
                "sample_id": sample_id,
                "preprocessing": (
                    preprocessing
                ),
                "postprocessing": (
                    "conservative"
                ),
                "threshold": threshold,
                **sample_metrics,
            }
        )

    aggregate_metrics = (
        BASE.finalize_counts(
            aggregate_counts
        )
    )

    return (
        aggregate_metrics,
        sample_rows,
    )


def violation_score(
    row: pd.Series,
    minimum_water_precision: float,
    minimum_water_iou: float,
    maximum_land_leakage: float,
    maximum_per_image_land_leakage: float,
) -> float:
    precision_deficit = max(
        0.0,
        minimum_water_precision
        - float(
            row["water_precision"]
        ),
    )

    iou_deficit = max(
        0.0,
        minimum_water_iou
        - float(row["iou"]),
    )

    leakage_excess = max(
        0.0,
        float(
            row["land_leakage_rate"]
        )
        - maximum_land_leakage,
    )

    worst_excess = max(
        0.0,
        float(
            row[
                "worst_per_image_land_leakage"
            ]
        )
        - maximum_per_image_land_leakage,
    )

    violating_fraction = float(
        row[
            "per_image_violation_count"
        ]
    ) / max(
        float(
            row["sample_count"]
        ),
        1.0,
    )

    return float(
        precision_deficit
        / max(
            minimum_water_precision,
            1e-8,
        )
        + iou_deficit
        / max(
            minimum_water_iou,
            1e-8,
        )
        + leakage_excess
        / max(
            maximum_land_leakage,
            1e-8,
        )
        + worst_excess
        / max(
            maximum_per_image_land_leakage,
            1e-8,
        )
        + violating_fraction
    )


def save_selected_previews(
    records: pd.DataFrame,
    cached: dict[
        tuple[str, str],
        dict[str, np.ndarray],
    ],
    preprocessing: str,
    threshold: float,
    coast_buffer_pixels: int,
    minimum_component_pixels: int,
) -> Path:
    preview_directory = (
        OUTPUT_DIR
        / "selected_previews"
    )

    preview_paths: list[Path] = []

    for row in records.to_dict(
        orient="records"
    ):
        sample_id = str(
            row["sample_id"]
        )

        base = cached[
            (
                sample_id,
                "base",
            )
        ]

        probability = cached[
            (
                sample_id,
                preprocessing,
            )
        ]["probability"]

        prediction = (
            probability >= threshold
        )

        prediction = (
            BASE.apply_postprocessing(
                prediction=prediction,
                mode="conservative",
                coast_buffer_pixels=(
                    coast_buffer_pixels
                ),
                minimum_component_pixels=(
                    minimum_component_pixels
                ),
            )
        )

        preview_path = (
            preview_directory
            / (
                sample_id.replace(
                    ":",
                    "__",
                )
                + ".png"
            )
        )

        BASE.save_preview(
            image=base["image"],
            truth_water=base[
                "safe_water"
            ],
            valid=base["valid"],
            probability=probability,
            prediction=prediction,
            output_path=preview_path,
        )

        preview_paths.append(
            preview_path
        )

    contact_sheet_path = (
        OUTPUT_DIR
        / "selected_contact_sheet.jpg"
    )

    BASE.create_contact_sheet(
        preview_paths,
        contact_sheet_path,
    )

    return contact_sheet_path


def main() -> None:
    args = parse_args()

    checkpoint_path = resolve_path(
        args.checkpoint
    )

    dartis_manifest_path = (
        resolve_path(
            args.dartis_manifest
        )
    )

    manual_manifest_path = (
        resolve_path(
            args.manual_manifest
        )
    )

    for path in (
        checkpoint_path,
        dartis_manifest_path,
        manual_manifest_path,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Gerekli dosya bulunamadı: {path}"
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    config = checkpoint.get(
        "config",
        {},
    )

    base_channels = int(
        config.get(
            "base_channels",
            16,
        )
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
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

    records = load_records(
        dartis_manifest_path,
        manual_manifest_path,
    )

    cached: dict[
        tuple[str, str],
        dict[str, np.ndarray],
    ] = {}

    print("=" * 78)
    print(
        "v0.6 DARTIS STRICT WATER GATE"
    )
    print("=" * 78)
    print(
        "Manuel geliştirme örneği:",
        len(records),
    )
    print(
        "Cihaz:",
        device,
    )
    print(
        "Koşullar:"
    )
    print(
        "  Aggregate precision >=",
        args.minimum_water_precision,
    )
    print(
        "  Aggregate IoU >=",
        args.minimum_water_iou,
    )
    print(
        "  Aggregate land leakage <=",
        args.maximum_land_leakage,
    )
    print(
        "  Her görüntü land leakage <=",
        args.maximum_per_image_land_leakage,
    )
    print(
        "  Postprocessing: conservative"
    )
    print()

    for row in records.to_dict(
        orient="records"
    ):
        sample_id = str(
            row["sample_id"]
        )

        image_path = resolve_path(
            row["image_path"]
        )

        land_path = resolve_path(
            row["land_mask_path"]
        )

        water_path = resolve_path(
            row[
                "safe_water_mask_path"
            ]
        )

        image = BASE.read_grayscale(
            image_path
        )

        land = (
            BASE.read_grayscale(
                land_path
            )
            > 127
        )

        safe_water = (
            BASE.read_grayscale(
                water_path
            )
            > 127
        )

        if land.shape != safe_water.shape:
            raise RuntimeError(
                "Manuel land ve water maskeleri farklı boyutta: "
                f"{sample_id}"
            )

        original_shape = image.shape

        image = resize_image_to_mask(
            image,
            land.shape,
        )

        if original_shape != image.shape:
            print(
                "Yeniden boyutlandırıldı:",
                sample_id,
                f"{original_shape} -> {image.shape}",
            )

        overlap = (
            land & safe_water
        )

        if overlap.any():
            raise RuntimeError(
                "Manuel land ve water maskeleri çakışıyor: "
                f"{sample_id}"
            )

        valid = land | safe_water

        if not valid.any():
            raise RuntimeError(
                "Geçerli değerlendirme pikseli yok: "
                f"{sample_id}"
            )

        cached[
            (
                sample_id,
                "base",
            )
        ] = {
            "image": image,
            "land": land,
            "safe_water": safe_water,
            "valid": valid,
        }

        for preprocessing in (
            BASE.PREPROCESSING_MODES
        ):
            probability = (
                BASE.infer_probability(
                    model=model,
                    device=device,
                    image=image,
                    preprocessing_mode=(
                        preprocessing
                    ),
                )
            )

            cached[
                (
                    sample_id,
                    preprocessing,
                )
            ] = {
                "probability": probability,
            }

    candidate_rows: list[
        dict[str, Any]
    ] = []

    all_sample_rows: list[
        dict[str, Any]
    ] = []

    for preprocessing in (
        BASE.PREPROCESSING_MODES
    ):
        for threshold in (
            BASE.THRESHOLDS
        ):
            (
                aggregate_metrics,
                sample_rows,
            ) = evaluate_configuration(
                records=records,
                cached=cached,
                preprocessing=(
                    preprocessing
                ),
                threshold=threshold,
                coast_buffer_pixels=(
                    args.coast_buffer_pixels
                ),
                minimum_component_pixels=(
                    args.minimum_component_pixels
                ),
            )

            sample_frame = (
                pd.DataFrame(
                    sample_rows
                )
            )

            worst_leakage = float(
                sample_frame[
                    "land_leakage_rate"
                ].max()
            )

            violation_count = int(
                (
                    sample_frame[
                        "land_leakage_rate"
                    ]
                    > args.maximum_per_image_land_leakage
                ).sum()
            )

            passes = bool(
                aggregate_metrics[
                    "water_precision"
                ]
                >= args.minimum_water_precision
                and aggregate_metrics[
                    "iou"
                ]
                >= args.minimum_water_iou
                and aggregate_metrics[
                    "land_leakage_rate"
                ]
                <= args.maximum_land_leakage
                and violation_count == 0
            )

            candidate_rows.append(
                {
                    "preprocessing": (
                        preprocessing
                    ),
                    "postprocessing": (
                        "conservative"
                    ),
                    "threshold": threshold,
                    **aggregate_metrics,
                    "worst_per_image_land_leakage": (
                        worst_leakage
                    ),
                    "per_image_violation_count": (
                        violation_count
                    ),
                    "sample_count": int(
                        len(sample_frame)
                    ),
                    "strict_pass": passes,
                }
            )

            all_sample_rows.extend(
                sample_rows
            )

    candidates = pd.DataFrame(
        candidate_rows
    )

    candidates[
        "violation_score"
    ] = candidates.apply(
        lambda row: violation_score(
            row=row,
            minimum_water_precision=(
                args.minimum_water_precision
            ),
            minimum_water_iou=(
                args.minimum_water_iou
            ),
            maximum_land_leakage=(
                args.maximum_land_leakage
            ),
            maximum_per_image_land_leakage=(
                args.maximum_per_image_land_leakage
            ),
        ),
        axis=1,
    )

    passing = candidates[
        candidates[
            "strict_pass"
        ].astype(bool)
    ].copy()

    if passing.empty:
        selected = (
            candidates.sort_values(
                [
                    "violation_score",
                    "land_leakage_rate",
                    "worst_per_image_land_leakage",
                    "iou",
                ],
                ascending=[
                    True,
                    True,
                    True,
                    False,
                ],
            )
            .iloc[0]
        )

        gate_passed = False
    else:
        selected = (
            passing.sort_values(
                [
                    "iou",
                    "water_precision",
                    "land_leakage_rate",
                    "water_recall",
                ],
                ascending=[
                    False,
                    False,
                    True,
                    False,
                ],
            )
            .iloc[0]
        )

        gate_passed = True

    selected_preprocessing = str(
        selected["preprocessing"]
    )

    selected_threshold = float(
        selected["threshold"]
    )

    all_samples = pd.DataFrame(
        all_sample_rows
    )

    selected_samples = all_samples[
        (
            all_samples[
                "preprocessing"
            ].eq(
                selected_preprocessing
            )
        )
        & (
            np.isclose(
                all_samples[
                    "threshold"
                ].astype(float),
                selected_threshold,
            )
        )
    ].copy()

    candidate_path = (
        OUTPUT_DIR
        / "strict_candidate_ranking.csv"
    )

    all_samples_path = (
        OUTPUT_DIR
        / "strict_all_per_sample.csv"
    )

    selected_samples_path = (
        OUTPUT_DIR
        / "strict_selected_per_sample.csv"
    )

    candidates.sort_values(
        [
            "strict_pass",
            "violation_score",
            "iou",
        ],
        ascending=[
            False,
            True,
            False,
        ],
    ).to_csv(
        candidate_path,
        index=False,
        encoding="utf-8-sig",
    )

    all_samples.to_csv(
        all_samples_path,
        index=False,
        encoding="utf-8-sig",
    )

    selected_samples.sort_values(
        "land_leakage_rate",
        ascending=False,
    ).to_csv(
        selected_samples_path,
        index=False,
        encoding="utf-8-sig",
    )

    contact_sheet_path = (
        save_selected_previews(
            records=records,
            cached=cached,
            preprocessing=(
                selected_preprocessing
            ),
            threshold=(
                selected_threshold
            ),
            coast_buffer_pixels=(
                args.coast_buffer_pixels
            ),
            minimum_component_pixels=(
                args.minimum_component_pixels
            ),
        )
    )

    failed_reasons: list[str] = []

    if (
        float(
            selected[
                "water_precision"
            ]
        )
        < args.minimum_water_precision
    ):
        failed_reasons.append(
            "aggregate_precision"
        )

    if (
        float(selected["iou"])
        < args.minimum_water_iou
    ):
        failed_reasons.append(
            "aggregate_iou"
        )

    if (
        float(
            selected[
                "land_leakage_rate"
            ]
        )
        > args.maximum_land_leakage
    ):
        failed_reasons.append(
            "aggregate_land_leakage"
        )

    if (
        int(
            selected[
                "per_image_violation_count"
            ]
        )
        > 0
    ):
        failed_reasons.append(
            "per_image_land_leakage"
        )

    summary = {
        "stage": (
            "v06_dartis_water_strict_gate"
        ),
        "gate_passed": bool(
            gate_passed
        ),
        "failed_reasons": (
            failed_reasons
        ),
        "manual_development_count": int(
            len(records)
        ),
        "manual_development_note": (
            "Bu 7 örnek hedef-domain geliştirme kümesidir; "
            "bağımsız final test değildir."
        ),
        "criteria": {
            "minimum_water_precision": float(
                args.minimum_water_precision
            ),
            "minimum_water_iou": float(
                args.minimum_water_iou
            ),
            "maximum_land_leakage": float(
                args.maximum_land_leakage
            ),
            "maximum_per_image_land_leakage": float(
                args.maximum_per_image_land_leakage
            ),
            "required_postprocessing": (
                "conservative"
            ),
            "coast_buffer_pixels": int(
                args.coast_buffer_pixels
            ),
            "minimum_component_pixels": int(
                args.minimum_component_pixels
            ),
        },
        "selected_configuration": {
            "preprocessing": (
                selected_preprocessing
            ),
            "postprocessing": (
                "conservative"
            ),
            "threshold": (
                selected_threshold
            ),
        },
        "selected_metrics": {
            key: float(
                selected[key]
            )
            for key in (
                "dice",
                "iou",
                "water_precision",
                "water_recall",
                "accuracy",
                "land_leakage_rate",
                "worst_per_image_land_leakage",
                "violation_score",
            )
        },
        "per_image_violation_count": int(
            selected[
                "per_image_violation_count"
            ]
        ),
        "candidate_ranking": relative(
            candidate_path
        ),
        "selected_per_sample": relative(
            selected_samples_path
        ),
        "selected_contact_sheet": relative(
            contact_sheet_path
        ),
        "all_dartis_masks_generated": False,
        "training_performed": False,
        "official_sen1floods11_test_used": False,
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

    print()
    print("=" * 78)
    print(
        "STRICT DARTIS WATER GATE SONUCU"
    )
    print("=" * 78)
    print(
        "Gate:",
        (
            "PASS"
            if gate_passed
            else "FAIL"
        ),
    )
    print(
        "Preprocessing:",
        selected_preprocessing,
    )
    print(
        "Postprocessing: conservative"
    )
    print(
        "Threshold:",
        f"{selected_threshold:.2f}",
    )
    print(
        "Water precision:",
        f"{float(selected['water_precision']):.4f}",
    )
    print(
        "Water recall:",
        f"{float(selected['water_recall']):.4f}",
    )
    print(
        "Water IoU:",
        f"{float(selected['iou']):.4f}",
    )
    print(
        "Aggregate land leakage:",
        f"{float(selected['land_leakage_rate']):.4f}",
    )
    print(
        "Worst per-image land leakage:",
        f"{float(selected['worst_per_image_land_leakage']):.4f}",
    )
    print(
        "Per-image violation count:",
        int(
            selected[
                "per_image_violation_count"
            ]
        ),
    )

    if failed_reasons:
        print(
            "Başarısız koşullar:",
            ", ".join(
                failed_reasons
            ),
        )

    print(
        "Özet:",
        summary_path.resolve(),
    )
    print(
        "Görsel:",
        contact_sheet_path.resolve(),
    )
    print()
    print(
        "Bu aşamada tüm DARTIS görüntülerine maske uygulanmadı."
    )


if __name__ == "__main__":
    main()
