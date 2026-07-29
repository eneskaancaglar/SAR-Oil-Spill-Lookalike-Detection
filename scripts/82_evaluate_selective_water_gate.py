from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

BASE_SCRIPT = (
    ROOT
    / "scripts"
    / "68_evaluate_dartis_water_transfer_gate.py"
)

DEFAULT_MANIFEST = (
    ROOT
    / "data"
    / "metadata"
    / "v06_dartis_water_combined_development_manifest.csv"
)

DEFAULT_CHECKPOINT_DIR = (
    ROOT
    / "checkpoints"
    / "water_unet_dartis_combined_landaware_cv_v06"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "v06_dartis_selective_water_gate"
)


def load_base_module():
    if not BASE_SCRIPT.exists():
        raise FileNotFoundError(
            f"Gerekli 68 scripti bulunamadı: {BASE_SCRIPT}"
        )

    spec = importlib.util.spec_from_file_location(
        "dartis_water_base_v82",
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
            "Mevcut beş OOF modelini yeniden eğitmeden, yüksek güvenli "
            "su piksellerini kabul edip kalan sahneleri UNCERTAIN olarak "
            "reddeden seçici su kapısını kalibre eder. Kilitli testi açmaz."
        )
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT_DIR,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--minimum-accepted-scenes",
        type=int,
        default=20,
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
        0.0,
        1.0,
    ).astype(np.float32)


def align_probability(
    probability: np.ndarray,
    fold_threshold: float,
) -> np.ndarray:
    probability = np.clip(
        probability,
        1e-6,
        1.0 - 1e-6,
    )

    threshold = float(
        np.clip(
            fold_threshold,
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
) -> tuple[
    torch.nn.Module,
    float,
]:
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


def load_manifest(
    manifest_path: Path,
) -> pd.DataFrame:
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Birleşik manifest bulunamadı: {manifest_path}"
        )

    frame = pd.read_csv(
        manifest_path,
        encoding="utf-8-sig",
        low_memory=False,
    )

    required = {
        "sample_id",
        "cv_fold",
        "coastal_group",
        "source_batch",
        "annotation_image_path",
        "manual_mask_path",
    }

    missing = required - set(frame.columns)

    if missing:
        raise RuntimeError(
            "Manifestte eksik sütunlar: "
            f"{sorted(missing)}"
        )

    if len(frame) != 40:
        raise RuntimeError(
            "Bu aşama 40 development örneği bekliyor. "
            f"Bulunan: {len(frame)}"
        )

    frame = frame.copy()

    frame["resolved_image_path"] = (
        frame["annotation_image_path"].map(
            resolve_path
        )
    )

    frame["resolved_mask_path"] = (
        frame["manual_mask_path"].map(
            resolve_path
        )
    )

    for column in (
        "resolved_image_path",
        "resolved_mask_path",
    ):
        missing_rows = frame[
            ~frame[column].map(
                lambda value: Path(value).exists()
            )
        ]

        if not missing_rows.empty:
            raise FileNotFoundError(
                f"{column} için eksik dosya: "
                + ", ".join(
                    missing_rows[
                        "sample_id"
                    ].astype(str).head(5)
                )
            )

    return frame.reset_index(drop=True)


def collect_oof_probabilities(
    frame: pd.DataFrame,
    checkpoint_dir: Path,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for fold_index in range(5):
        fold_number = fold_index + 1

        checkpoint_path = (
            checkpoint_dir
            / f"fold_{fold_number:02d}_best.pth"
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Fold checkpoint bulunamadı: {checkpoint_path}"
            )

        model, fold_threshold = load_model(
            checkpoint_path,
            device,
        )

        validation = frame[
            frame["cv_fold"].eq(
                fold_index
            )
        ].copy()

        if len(validation) != 8:
            raise RuntimeError(
                f"Fold {fold_number} için 8 OOF örnek bekleniyor."
            )

        with torch.no_grad():
            for item in validation.itertuples():
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

                with torch.amp.autocast(
                    device_type=device.type,
                    enabled=(
                        device.type == "cuda"
                    ),
                ):
                    logits = model(
                        tensor
                    )[0, 0]

                raw_probability = torch.sigmoid(
                    logits.float()
                ).cpu().numpy()

                aligned_probability = (
                    align_probability(
                        raw_probability,
                        fold_threshold,
                    )
                )

                rows.append(
                    {
                        "sample_id": str(
                            item.sample_id
                        ),
                        "fold": fold_number,
                        "coastal_group": str(
                            item.coastal_group
                        ),
                        "source_batch": str(
                            item.source_batch
                        ),
                        "fold_threshold": float(
                            fold_threshold
                        ),
                        "probability": (
                            aligned_probability
                        ),
                        "mask": mask,
                    }
                )

    if len(rows) != 40:
        raise RuntimeError(
            f"OOF tahmin sayısı 40 değil: {len(rows)}"
        )

    return rows


def evaluate_policy(
    rows: list[dict[str, Any]],
    water_threshold: float,
    land_threshold: float,
    minimum_predicted_water_fraction: float,
    maximum_uncertain_fraction: float,
) -> dict[str, Any]:
    accepted_rows = []
    all_scene_rows = []

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    for row in rows:
        probability = row["probability"]
        mask = row["mask"]

        valid = mask != 128
        water = mask == 255
        land = mask == 0

        safe_water = (
            probability >= water_threshold
        )

        safe_land = (
            probability <= land_threshold
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
            >= minimum_predicted_water_fraction
            and uncertain_fraction
            <= maximum_uncertain_fraction
        )

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

        scene_result = {
            "sample_id": row["sample_id"],
            "fold": row["fold"],
            "coastal_group": (
                row["coastal_group"]
            ),
            "source_batch": (
                row["source_batch"]
            ),
            "accepted": bool(accepted),
            "predicted_water_fraction": (
                predicted_water_fraction
            ),
            "uncertain_fraction": (
                uncertain_fraction
            ),
            "water_precision": float(
                precision
            ),
            "water_recall": float(recall),
            "water_iou": float(iou),
            "land_leakage": float(
                leakage
            ),
        }

        all_scene_rows.append(
            scene_result
        )

        if accepted:
            accepted_rows.append(
                scene_result
            )

            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

    accepted_count = len(
        accepted_rows
    )

    if accepted_count == 0:
        return {
            "accepted_count": 0,
            "accepted_rate": 0.0,
            "water_precision": 0.0,
            "mean_water_recall": 0.0,
            "pixel_water_recall": 0.0,
            "water_iou": 0.0,
            "land_leakage": 1.0,
            "minimum_per_image_water_precision": 0.0,
            "maximum_per_image_land_leakage": 1.0,
            "mean_confident_coverage": 0.0,
            "scene_rows": all_scene_rows,
        }

    precision = (
        total_tp / (total_tp + total_fp)
        if total_tp + total_fp > 0
        else 1.0
    )

    pixel_recall = (
        total_tp / (total_tp + total_fn)
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
        total_fp / (total_fp + total_tn)
        if total_fp + total_tn > 0
        else 0.0
    )

    mean_recall = float(
        np.mean(
            [
                item[
                    "water_recall"
                ]
                for item in accepted_rows
            ]
        )
    )

    mean_confident_coverage = float(
        np.mean(
            [
                1.0
                - item[
                    "uncertain_fraction"
                ]
                for item in accepted_rows
            ]
        )
    )

    return {
        "accepted_count": int(
            accepted_count
        ),
        "accepted_rate": float(
            accepted_count / len(rows)
        ),
        "water_precision": float(
            precision
        ),
        "mean_water_recall": (
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
            min(
                item[
                    "water_precision"
                ]
                for item in accepted_rows
            )
        ),
        "maximum_per_image_land_leakage": float(
            max(
                item[
                    "land_leakage"
                ]
                for item in accepted_rows
            )
        ),
        "mean_confident_coverage": (
            mean_confident_coverage
        ),
        "scene_rows": all_scene_rows,
    }


def policy_passes(
    result: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[
    bool,
    list[str],
]:
    checks = {
        "accepted_scenes": (
            result["accepted_count"]
            >= args.minimum_accepted_scenes
        ),
        "water_precision": (
            result["water_precision"]
            >= args.minimum_water_precision
        ),
        "land_leakage": (
            result["land_leakage"]
            <= args.maximum_land_leakage
        ),
        "per_image_land_leakage": (
            result[
                "maximum_per_image_land_leakage"
            ]
            <= args.maximum_per_image_land_leakage
        ),
        "mean_accepted_recall": (
            result["mean_water_recall"]
            >= args.minimum_mean_accepted_recall
        ),
        "per_image_water_precision": (
            result[
                "minimum_per_image_water_precision"
            ]
            >= args.minimum_per_image_water_precision
        ),
    }

    failed = [
        name
        for name, passed in checks.items()
        if not passed
    ]

    return (
        len(failed) == 0,
        failed,
    )


def main() -> None:
    args = parse_args()

    manifest_path = resolve_path(
        args.manifest
    )

    checkpoint_dir = resolve_path(
        args.checkpoint_dir
    )

    frame = load_manifest(
        manifest_path
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 78)
    print(
        "v0.6 SELECTIVE HIGH-CONFIDENCE WATER GATE"
    )
    print("=" * 78)
    print(
        "Development:",
        len(frame),
    )
    print(
        "Kilitli test kullanılmayacak."
    )
    print(
        "Eğitim yapılmayacak."
    )
    print(
        "Cihaz:",
        device,
    )
    print()

    rows = collect_oof_probabilities(
        frame,
        checkpoint_dir,
        device,
    )

    candidates = []

    for water_threshold in np.arange(
        0.55,
        0.951,
        0.05,
    ):
        for land_threshold in (
            0.05,
            0.10,
            0.15,
            0.20,
        ):
            if land_threshold >= water_threshold:
                continue

            for minimum_water_fraction in (
                0.005,
                0.01,
                0.02,
                0.05,
            ):
                for maximum_uncertain_fraction in (
                    0.25,
                    0.40,
                    0.55,
                    0.70,
                ):
                    result = evaluate_policy(
                        rows=rows,
                        water_threshold=float(
                            water_threshold
                        ),
                        land_threshold=float(
                            land_threshold
                        ),
                        minimum_predicted_water_fraction=float(
                            minimum_water_fraction
                        ),
                        maximum_uncertain_fraction=float(
                            maximum_uncertain_fraction
                        ),
                    )

                    passed, failed = (
                        policy_passes(
                            result,
                            args,
                        )
                    )

                    score = (
                        (10.0 if passed else 0.0)
                        + result[
                            "accepted_rate"
                        ]
                        + 0.80
                        * result[
                            "water_precision"
                        ]
                        + 0.40
                        * result[
                            "mean_water_recall"
                        ]
                        + 0.20
                        * result[
                            "mean_confident_coverage"
                        ]
                        - 2.0
                        * result[
                            "land_leakage"
                        ]
                        - 0.50
                        * result[
                            "maximum_per_image_land_leakage"
                        ]
                    )

                    candidates.append(
                        {
                            "water_threshold": float(
                                water_threshold
                            ),
                            "land_threshold": float(
                                land_threshold
                            ),
                            "minimum_predicted_water_fraction": float(
                                minimum_water_fraction
                            ),
                            "maximum_uncertain_fraction": float(
                                maximum_uncertain_fraction
                            ),
                            "passed": bool(
                                passed
                            ),
                            "failed_reasons": (
                                failed
                            ),
                            "score": float(
                                score
                            ),
                            **{
                                key: value
                                for key, value
                                in result.items()
                                if key != "scene_rows"
                            },
                            "scene_rows": (
                                result[
                                    "scene_rows"
                                ]
                            ),
                        }
                    )

    selected = max(
        candidates,
        key=lambda item: item["score"],
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    scene_rows = selected.pop(
        "scene_rows"
    )

    scene_path = (
        OUTPUT_DIR
        / "selected_policy_per_scene.csv"
    )

    pd.DataFrame(
        scene_rows
    ).sort_values(
        [
            "accepted",
            "land_leakage",
            "water_recall",
        ],
        ascending=[
            False,
            False,
            True,
        ],
    ).to_csv(
        scene_path,
        index=False,
        encoding="utf-8-sig",
    )

    candidate_path = (
        OUTPUT_DIR
        / "policy_sweep.csv"
    )

    candidate_rows = []

    for candidate in candidates:
        row = {
            key: value
            for key, value
            in candidate.items()
            if key not in {
                "scene_rows",
                "failed_reasons",
            }
        }

        row["failed_reasons"] = "|".join(
            candidate[
                "failed_reasons"
            ]
        )

        candidate_rows.append(row)

    pd.DataFrame(
        candidate_rows
    ).sort_values(
        "score",
        ascending=False,
    ).to_csv(
        candidate_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "stage": (
            "v06_dartis_selective_water_gate"
        ),
        "development_count": 40,
        "locked_test_used": False,
        "training_performed": False,
        "policy_passed": bool(
            selected["passed"]
        ),
        "selected_policy": {
            key: value
            for key, value
            in selected.items()
            if key not in {
                "score",
                "failed_reasons",
                "passed",
            }
        },
        "failed_reasons": (
            selected[
                "failed_reasons"
            ]
        ),
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
            scene_path
        ),
        "policy_sweep": relative(
            candidate_path
        ),
        "operational_rule": (
            "Yalnız yüksek güvenli su pikselleri petrol analizine "
            "aktarılır. Politika koşullarını karşılamayan sahne "
            "UNCERTAIN olarak reddedilir ve petrol maskesi üretilmez."
        ),
        "scientific_note": (
            "Bu politika yalnız development OOF tahminleriyle "
            "kalibre edilmiştir; bağımsız final kanıt değildir. "
            "Kilitli sekiz DARTIS test örneği kullanılmamıştır."
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

    print("=" * 78)
    print(
        "SELECTIVE WATER GATE SONUCU"
    )
    print("=" * 78)
    print(
        "Policy:",
        "PASS"
        if selected["passed"]
        else "FAIL",
    )
    print(
        "Water threshold:",
        f"{selected['water_threshold']:.2f}",
    )
    print(
        "Land threshold:",
        f"{selected['land_threshold']:.2f}",
    )
    print(
        "Accepted scenes:",
        f"{selected['accepted_count']}/40",
    )
    print(
        "Water precision:",
        f"{selected['water_precision']:.4f}",
    )
    print(
        "Mean accepted recall:",
        f"{selected['mean_water_recall']:.4f}",
    )
    print(
        "Pixel recall:",
        f"{selected['pixel_water_recall']:.4f}",
    )
    print(
        "Land leakage:",
        f"{selected['land_leakage']:.4f}",
    )
    print(
        "Worst accepted leakage:",
        f"{selected['maximum_per_image_land_leakage']:.4f}",
    )
    print(
        "Min accepted precision:",
        f"{selected['minimum_per_image_water_precision']:.4f}",
    )
    print(
        "Failed:",
        selected["failed_reasons"],
    )
    print(
        "Kilitli test kullanıldı mı: False"
    )
    print(
        "Özet:",
        summary_path.resolve(),
    )


if __name__ == "__main__":
    main()
