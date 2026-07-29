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
    / "v06_dartis_water_balanced_gate"
)

COAST_BUFFERS = (2, 4, 6, 8)
MINIMUM_COMPONENTS = (64, 128, 256)


def load_base_module():
    if not BASE_SCRIPT.exists():
        raise FileNotFoundError(
            f"Gerekli 68 scripti bulunamadı: {BASE_SCRIPT}"
        )

    spec = importlib.util.spec_from_file_location(
        "dartis_transfer_base_v70",
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
            "DARTIS su aktarımında hem kara sızıntısını hem de her "
            "görüntüdeki su kapsamasını zorunlu tutar. Kıyı tamponu "
            "ve bileşen filtresi güçlerini birlikte tarar."
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
        "--minimum-per-image-water-recall",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--minimum-per-image-water-iou",
        type=float,
        default=0.35,
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


def resize_image_to_mask(
    image: np.ndarray,
    mask_shape: tuple[int, int],
) -> np.ndarray:
    if image.shape == mask_shape:
        return image

    target_height, target_width = mask_shape

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
        (target_width, target_height),
        interpolation=interpolation,
    )


def evaluate_candidate(
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
    aggregate = BASE.empty_counts()
    sample_rows: list[dict[str, Any]] = []

    for row in records.to_dict(
        orient="records"
    ):
        sample_id = str(row["sample_id"])
        base = cached[(sample_id, "base")]

        probability = cached[
            (sample_id, preprocessing)
        ]["probability"]

        prediction = probability >= threshold

        prediction = BASE.apply_postprocessing(
            prediction=prediction,
            mode="conservative",
            coast_buffer_pixels=(
                coast_buffer_pixels
            ),
            minimum_component_pixels=(
                minimum_component_pixels
            ),
        )

        sample_counts = BASE.empty_counts()

        BASE.update_counts(
            counts=sample_counts,
            prediction=prediction,
            truth_water=base["safe_water"],
            valid=base["valid"],
        )

        BASE.update_counts(
            counts=aggregate,
            prediction=prediction,
            truth_water=base["safe_water"],
            valid=base["valid"],
        )

        sample_rows.append(
            {
                "sample_id": sample_id,
                "preprocessing": preprocessing,
                "postprocessing": "conservative",
                "threshold": threshold,
                "coast_buffer_pixels": (
                    coast_buffer_pixels
                ),
                "minimum_component_pixels": (
                    minimum_component_pixels
                ),
                **BASE.finalize_counts(
                    sample_counts
                ),
            }
        )

    return (
        BASE.finalize_counts(aggregate),
        sample_rows,
    )


def normalized_excess(
    value: float,
    limit: float,
) -> float:
    return max(
        0.0,
        value - limit,
    ) / max(limit, 1e-8)


def normalized_deficit(
    value: float,
    limit: float,
) -> float:
    return max(
        0.0,
        limit - value,
    ) / max(limit, 1e-8)


def violation_score(
    candidate: pd.Series,
    args: argparse.Namespace,
) -> float:
    sample_count = max(
        int(candidate["sample_count"]),
        1,
    )

    return float(
        normalized_deficit(
            float(
                candidate[
                    "water_precision"
                ]
            ),
            args.minimum_water_precision,
        )
        + normalized_deficit(
            float(candidate["iou"]),
            args.minimum_water_iou,
        )
        + normalized_excess(
            float(
                candidate[
                    "land_leakage_rate"
                ]
            ),
            args.maximum_land_leakage,
        )
        + normalized_excess(
            float(
                candidate[
                    "worst_per_image_land_leakage"
                ]
            ),
            args.maximum_per_image_land_leakage,
        )
        + normalized_deficit(
            float(
                candidate[
                    "minimum_per_image_water_recall"
                ]
            ),
            args.minimum_per_image_water_recall,
        )
        + normalized_deficit(
            float(
                candidate[
                    "minimum_per_image_water_iou"
                ]
            ),
            args.minimum_per_image_water_iou,
        )
        + float(
            candidate[
                "land_leakage_violation_count"
            ]
        )
        / sample_count
        + float(
            candidate[
                "water_recall_violation_count"
            ]
        )
        / sample_count
        + float(
            candidate[
                "water_iou_violation_count"
            ]
        )
        / sample_count
    )


def save_selected_previews(
    records: pd.DataFrame,
    cached: dict[
        tuple[str, str],
        dict[str, np.ndarray],
    ],
    selected: pd.Series,
) -> Path:
    preview_directory = (
        OUTPUT_DIR
        / "selected_previews"
    )

    preview_paths: list[Path] = []

    preprocessing = str(
        selected["preprocessing"]
    )
    threshold = float(
        selected["threshold"]
    )
    coast_buffer_pixels = int(
        selected["coast_buffer_pixels"]
    )
    minimum_component_pixels = int(
        selected[
            "minimum_component_pixels"
        ]
    )

    for row in records.to_dict(
        orient="records"
    ):
        sample_id = str(row["sample_id"])
        base = cached[(sample_id, "base")]

        probability = cached[
            (sample_id, preprocessing)
        ]["probability"]

        prediction = probability >= threshold

        prediction = BASE.apply_postprocessing(
            prediction=prediction,
            mode="conservative",
            coast_buffer_pixels=(
                coast_buffer_pixels
            ),
            minimum_component_pixels=(
                minimum_component_pixels
            ),
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
            truth_water=base["safe_water"],
            valid=base["valid"],
            probability=probability,
            prediction=prediction,
            output_path=preview_path,
        )

        preview_paths.append(preview_path)

    contact_sheet = (
        OUTPUT_DIR
        / "selected_contact_sheet.jpg"
    )

    BASE.create_contact_sheet(
        preview_paths,
        contact_sheet,
    )

    return contact_sheet


def main() -> None:
    args = parse_args()

    checkpoint_path = resolve_path(
        args.checkpoint
    )
    dartis_manifest_path = resolve_path(
        args.dartis_manifest
    )
    manual_manifest_path = resolve_path(
        args.manual_manifest
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

    base_channels = int(
        checkpoint.get(
            "config",
            {},
        ).get(
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

    records = (
        BASE.load_calibration_records(
            dartis_manifest_path,
            manual_manifest_path,
        )
    )

    cached: dict[
        tuple[str, str],
        dict[str, np.ndarray],
    ] = {}

    print("=" * 78)
    print(
        "v0.6 DARTIS BALANCED WATER GATE"
    )
    print("=" * 78)
    print(
        "Manuel geliştirme örneği:",
        len(records),
    )
    print("Cihaz:", device)
    print(
        "Kıyı tamponları:",
        COAST_BUFFERS,
    )
    print(
        "Bileşen filtreleri:",
        MINIMUM_COMPONENTS,
    )
    print()

    for row in records.to_dict(
        orient="records"
    ):
        sample_id = str(row["sample_id"])

        image = BASE.read_grayscale(
            resolve_path(
                row["image_path"]
            )
        )

        land = (
            BASE.read_grayscale(
                resolve_path(
                    row["land_mask_path"]
                )
            )
            > 127
        )

        safe_water = (
            BASE.read_grayscale(
                resolve_path(
                    row[
                        "safe_water_mask_path"
                    ]
                )
            )
            > 127
        )

        if land.shape != safe_water.shape:
            raise RuntimeError(
                "Manuel land ve water maskeleri "
                f"farklı boyutta: {sample_id}"
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
                f"{original_shape} -> "
                f"{image.shape}",
            )

        if (land & safe_water).any():
            raise RuntimeError(
                "Manuel land ve water maskeleri "
                f"çakışıyor: {sample_id}"
            )

        valid = land | safe_water

        if not valid.any():
            raise RuntimeError(
                "Geçerli değerlendirme pikseli "
                f"yok: {sample_id}"
            )

        cached[(sample_id, "base")] = {
            "image": image,
            "land": land,
            "safe_water": safe_water,
            "valid": valid,
        }

        for preprocessing in (
            BASE.PREPROCESSING_MODES
        ):
            cached[
                (sample_id, preprocessing)
            ] = {
                "probability": (
                    BASE.infer_probability(
                        model=model,
                        device=device,
                        image=image,
                        preprocessing_mode=(
                            preprocessing
                        ),
                    )
                )
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
        for threshold in BASE.THRESHOLDS:
            for coast_buffer in (
                COAST_BUFFERS
            ):
                for minimum_component in (
                    MINIMUM_COMPONENTS
                ):
                    (
                        aggregate_metrics,
                        sample_rows,
                    ) = evaluate_candidate(
                        records=records,
                        cached=cached,
                        preprocessing=(
                            preprocessing
                        ),
                        threshold=threshold,
                        coast_buffer_pixels=(
                            coast_buffer
                        ),
                        minimum_component_pixels=(
                            minimum_component
                        ),
                    )

                    sample_frame = (
                        pd.DataFrame(
                            sample_rows
                        )
                    )

                    worst_land_leakage = float(
                        sample_frame[
                            "land_leakage_rate"
                        ].max()
                    )

                    minimum_recall = float(
                        sample_frame[
                            "water_recall"
                        ].min()
                    )

                    minimum_iou = float(
                        sample_frame[
                            "iou"
                        ].min()
                    )

                    land_violation_count = int(
                        (
                            sample_frame[
                                "land_leakage_rate"
                            ]
                            > args.maximum_per_image_land_leakage
                        ).sum()
                    )

                    recall_violation_count = int(
                        (
                            sample_frame[
                                "water_recall"
                            ]
                            < args.minimum_per_image_water_recall
                        ).sum()
                    )

                    iou_violation_count = int(
                        (
                            sample_frame["iou"]
                            < args.minimum_per_image_water_iou
                        ).sum()
                    )

                    strict_pass = bool(
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
                        and land_violation_count == 0
                        and recall_violation_count == 0
                        and iou_violation_count == 0
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
                            "coast_buffer_pixels": (
                                coast_buffer
                            ),
                            "minimum_component_pixels": (
                                minimum_component
                            ),
                            **aggregate_metrics,
                            "worst_per_image_land_leakage": (
                                worst_land_leakage
                            ),
                            "minimum_per_image_water_recall": (
                                minimum_recall
                            ),
                            "minimum_per_image_water_iou": (
                                minimum_iou
                            ),
                            "land_leakage_violation_count": (
                                land_violation_count
                            ),
                            "water_recall_violation_count": (
                                recall_violation_count
                            ),
                            "water_iou_violation_count": (
                                iou_violation_count
                            ),
                            "sample_count": int(
                                len(sample_frame)
                            ),
                            "balanced_pass": (
                                strict_pass
                            ),
                        }
                    )

                    all_sample_rows.extend(
                        sample_rows
                    )

    candidates = pd.DataFrame(
        candidate_rows
    )

    candidates["violation_score"] = (
        candidates.apply(
            lambda row: violation_score(
                row,
                args,
            ),
            axis=1,
        )
    )

    passing = candidates[
        candidates[
            "balanced_pass"
        ].astype(bool)
    ].copy()

    if passing.empty:
        selected = (
            candidates.sort_values(
                [
                    "violation_score",
                    "water_recall_violation_count",
                    "water_iou_violation_count",
                    "land_leakage_rate",
                    "iou",
                ],
                ascending=[
                    True,
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
                    "minimum_per_image_water_recall",
                    "water_precision",
                    "land_leakage_rate",
                ],
                ascending=[
                    False,
                    False,
                    False,
                    True,
                ],
            )
            .iloc[0]
        )
        gate_passed = True

    all_samples = pd.DataFrame(
        all_sample_rows
    )

    selected_samples = all_samples[
        (
            all_samples[
                "preprocessing"
            ].eq(
                str(
                    selected[
                        "preprocessing"
                    ]
                )
            )
        )
        & (
            np.isclose(
                all_samples[
                    "threshold"
                ].astype(float),
                float(
                    selected[
                        "threshold"
                    ]
                ),
            )
        )
        & (
            all_samples[
                "coast_buffer_pixels"
            ].astype(int).eq(
                int(
                    selected[
                        "coast_buffer_pixels"
                    ]
                )
            )
        )
        & (
            all_samples[
                "minimum_component_pixels"
            ].astype(int).eq(
                int(
                    selected[
                        "minimum_component_pixels"
                    ]
                )
            )
        )
    ].copy()

    ranking_path = (
        OUTPUT_DIR
        / "balanced_candidate_ranking.csv"
    )

    selected_samples_path = (
        OUTPUT_DIR
        / "balanced_selected_per_sample.csv"
    )

    candidates.sort_values(
        [
            "balanced_pass",
            "violation_score",
            "iou",
        ],
        ascending=[
            False,
            True,
            False,
        ],
    ).to_csv(
        ranking_path,
        index=False,
        encoding="utf-8-sig",
    )

    selected_samples.sort_values(
        [
            "water_recall",
            "iou",
        ],
        ascending=[
            True,
            True,
        ],
    ).to_csv(
        selected_samples_path,
        index=False,
        encoding="utf-8-sig",
    )

    contact_sheet = save_selected_previews(
        records=records,
        cached=cached,
        selected=selected,
    )

    failed_reasons: list[str] = []

    checks = (
        (
            float(
                selected[
                    "water_precision"
                ]
            )
            >= args.minimum_water_precision,
            "aggregate_precision",
        ),
        (
            float(selected["iou"])
            >= args.minimum_water_iou,
            "aggregate_iou",
        ),
        (
            float(
                selected[
                    "land_leakage_rate"
                ]
            )
            <= args.maximum_land_leakage,
            "aggregate_land_leakage",
        ),
        (
            int(
                selected[
                    "land_leakage_violation_count"
                ]
            )
            == 0,
            "per_image_land_leakage",
        ),
        (
            int(
                selected[
                    "water_recall_violation_count"
                ]
            )
            == 0,
            "per_image_water_recall",
        ),
        (
            int(
                selected[
                    "water_iou_violation_count"
                ]
            )
            == 0,
            "per_image_water_iou",
        ),
    )

    for passed, name in checks:
        if not passed:
            failed_reasons.append(name)

    summary = {
        "stage": (
            "v06_dartis_water_balanced_gate"
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
            "minimum_per_image_water_recall": float(
                args.minimum_per_image_water_recall
            ),
            "minimum_per_image_water_iou": float(
                args.minimum_per_image_water_iou
            ),
            "coast_buffers_tested": list(
                COAST_BUFFERS
            ),
            "minimum_components_tested": list(
                MINIMUM_COMPONENTS
            ),
        },
        "selected_configuration": {
            "preprocessing": str(
                selected["preprocessing"]
            ),
            "postprocessing": (
                "conservative"
            ),
            "threshold": float(
                selected["threshold"]
            ),
            "coast_buffer_pixels": int(
                selected[
                    "coast_buffer_pixels"
                ]
            ),
            "minimum_component_pixels": int(
                selected[
                    "minimum_component_pixels"
                ]
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
                "minimum_per_image_water_recall",
                "minimum_per_image_water_iou",
                "violation_score",
            )
        },
        "violation_counts": {
            "land_leakage": int(
                selected[
                    "land_leakage_violation_count"
                ]
            ),
            "water_recall": int(
                selected[
                    "water_recall_violation_count"
                ]
            ),
            "water_iou": int(
                selected[
                    "water_iou_violation_count"
                ]
            ),
        },
        "candidate_ranking": relative(
            ranking_path
        ),
        "selected_per_sample": relative(
            selected_samples_path
        ),
        "selected_contact_sheet": relative(
            contact_sheet
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
        "BALANCED DARTIS WATER GATE SONUCU"
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
        selected[
            "preprocessing"
        ],
    )
    print(
        "Threshold:",
        f"{float(selected['threshold']):.2f}",
    )
    print(
        "Coast buffer:",
        int(
            selected[
                "coast_buffer_pixels"
            ]
        ),
    )
    print(
        "Minimum component:",
        int(
            selected[
                "minimum_component_pixels"
            ]
        ),
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
        "Worst land leakage:",
        f"{float(selected['worst_per_image_land_leakage']):.4f}",
    )
    print(
        "Minimum per-image recall:",
        f"{float(selected['minimum_per_image_water_recall']):.4f}",
    )
    print(
        "Minimum per-image IoU:",
        f"{float(selected['minimum_per_image_water_iou']):.4f}",
    )
    print(
        "Recall violation count:",
        int(
            selected[
                "water_recall_violation_count"
            ]
        ),
    )
    print(
        "IoU violation count:",
        int(
            selected[
                "water_iou_violation_count"
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

    print("Özet:", summary_path.resolve())
    print("Görsel:", contact_sheet.resolve())
    print()
    print(
        "Bu aşamada tüm DARTIS görüntülerine "
        "maske uygulanmadı."
    )


if __name__ == "__main__":
    main()
