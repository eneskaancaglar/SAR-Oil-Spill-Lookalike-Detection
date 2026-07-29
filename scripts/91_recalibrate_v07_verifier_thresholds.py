from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_DIR = (
    ROOT
    / "checkpoints"
    / "verifier_v07"
)

CHECKPOINT_PATH = (
    CHECKPOINT_DIR
    / "best.pth"
)

OLD_CONFIG_PATH = (
    CHECKPOINT_DIR
    / "calibration_config.json"
)

NEW_CONFIG_PATH = (
    CHECKPOINT_DIR
    / "calibration_config_v2.json"
)

PREDICTIONS_PATH = (
    ROOT
    / "outputs"
    / "verifier_v07"
    / "calibration_predictions.csv"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "verifier_v07_recalibration"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "summary.json"
)

RECALIBRATED_PREDICTIONS_PATH = (
    OUTPUT_DIR
    / "recalibrated_predictions.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "v0.7 verifier modelini yeniden eğitmeden, calibration "
            "olasılıklarında güvenli LOOK_ALIKE / UNCERTAIN / "
            "CONFIRMED_OIL eşiklerini yeniden seçer."
        )
    )

    parser.add_argument(
        "--minimum-confirmed-oil-precision",
        type=float,
        default=0.99,
    )

    parser.add_argument(
        "--minimum-confirmed-oil-recall",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--minimum-lookalike-precision",
        type=float,
        default=0.95,
    )

    parser.add_argument(
        "--minimum-lookalike-recall",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--max-negative-confirm-rate",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--max-positive-lookalike-rate",
        type=float,
        default=0.01,
    )

    return parser.parse_args()


def load_inputs() -> tuple[
    pd.DataFrame,
    dict[str, Any],
    dict[str, Any],
]:
    for path in (
        CHECKPOINT_PATH,
        OLD_CONFIG_PATH,
        PREDICTIONS_PATH,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Gerekli dosya bulunamadı: {path}"
            )

    predictions = pd.read_csv(
        PREDICTIONS_PATH,
        encoding="utf-8-sig",
        low_memory=False,
    )

    required = {
        "candidate_id",
        "scene_id",
        "label",
        "logit",
    }

    missing = required - set(
        predictions.columns
    )

    if missing:
        raise RuntimeError(
            "Calibration tahminlerinde eksik sütunlar: "
            f"{sorted(missing)}"
        )

    old_config = json.loads(
        OLD_CONFIG_PATH.read_text(
            encoding="utf-8-sig"
        )
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    return (
        predictions,
        old_config,
        checkpoint,
    )


def calibrated_probabilities(
    logits: np.ndarray,
    temperature: float,
) -> np.ndarray:
    scaled = np.clip(
        logits.astype(np.float64)
        / float(temperature),
        -30.0,
        30.0,
    )

    return (
        1.0
        / (
            1.0
            + np.exp(
                -scaled
            )
        )
    )


def threshold_candidates(
    probabilities: np.ndarray,
) -> np.ndarray:
    unique = np.unique(
        probabilities.astype(
            np.float64
        )
    )

    candidates = [
        -1e-9,
        0.0,
    ]

    if len(unique) > 1:
        midpoints = (
            unique[:-1]
            + unique[1:]
        ) / 2.0

        candidates.extend(
            midpoints.tolist()
        )

    candidates.extend(
        unique.tolist()
    )

    candidates.extend(
        [
            1.0,
            1.0 + 1e-9,
        ]
    )

    return np.unique(
        np.asarray(
            candidates,
            dtype=np.float64,
        )
    )


def safe_divide(
    numerator: int,
    denominator: int,
    empty_value: float = 1.0,
) -> float:
    if denominator == 0:
        return float(
            empty_value
        )

    return float(
        numerator
        / denominator
    )


def evaluate_pair(
    labels: np.ndarray,
    probabilities: np.ndarray,
    low_threshold: float,
    high_threshold: float,
) -> dict[str, Any]:
    decisions = np.full(
        len(probabilities),
        "UNCERTAIN",
        dtype=object,
    )

    decisions[
        probabilities
        <= low_threshold
    ] = "LOOK_ALIKE"

    decisions[
        probabilities
        >= high_threshold
    ] = "CONFIRMED_OIL"

    positive = (
        labels == 1
    )

    negative = (
        labels == 0
    )

    confirmed = (
        decisions
        == "CONFIRMED_OIL"
    )

    lookalike = (
        decisions
        == "LOOK_ALIKE"
    )

    uncertain = (
        decisions
        == "UNCERTAIN"
    )

    confirmed_tp = int(
        (
            confirmed
            & positive
        ).sum()
    )

    confirmed_fp = int(
        (
            confirmed
            & negative
        ).sum()
    )

    lookalike_tn = int(
        (
            lookalike
            & negative
        ).sum()
    )

    lookalike_fn = int(
        (
            lookalike
            & positive
        ).sum()
    )

    positive_count = int(
        positive.sum()
    )

    negative_count = int(
        negative.sum()
    )

    confirmed_precision = safe_divide(
        confirmed_tp,
        confirmed_tp
        + confirmed_fp,
    )

    confirmed_recall = safe_divide(
        confirmed_tp,
        positive_count,
        empty_value=0.0,
    )

    lookalike_precision = safe_divide(
        lookalike_tn,
        lookalike_tn
        + lookalike_fn,
    )

    lookalike_recall = safe_divide(
        lookalike_tn,
        negative_count,
        empty_value=0.0,
    )

    negative_confirm_rate = safe_divide(
        confirmed_fp,
        negative_count,
        empty_value=0.0,
    )

    positive_lookalike_rate = safe_divide(
        lookalike_fn,
        positive_count,
        empty_value=0.0,
    )

    return {
        "lookalike_threshold": float(
            low_threshold
        ),
        "confirmed_oil_threshold": float(
            high_threshold
        ),
        "confirmed_oil_precision": (
            confirmed_precision
        ),
        "confirmed_oil_recall": (
            confirmed_recall
        ),
        "lookalike_precision": (
            lookalike_precision
        ),
        "lookalike_recall": (
            lookalike_recall
        ),
        "negative_confirm_rate": (
            negative_confirm_rate
        ),
        "positive_lookalike_rate": (
            positive_lookalike_rate
        ),
        "confirmed_true_positive_count": (
            confirmed_tp
        ),
        "confirmed_false_positive_count": (
            confirmed_fp
        ),
        "lookalike_true_negative_count": (
            lookalike_tn
        ),
        "lookalike_false_negative_count": (
            lookalike_fn
        ),
        "confirmed_oil_count": int(
            confirmed.sum()
        ),
        "lookalike_count": int(
            lookalike.sum()
        ),
        "uncertain_count": int(
            uncertain.sum()
        ),
        "overall_uncertain_rate": float(
            uncertain.mean()
        ),
        "definite_decision_rate": float(
            1.0
            - uncertain.mean()
        ),
        "positive_uncertain_rate": float(
            uncertain[
                positive
            ].mean()
        ),
        "negative_uncertain_rate": float(
            uncertain[
                negative
            ].mean()
        ),
        "decisions": decisions,
    }


def passes(
    metrics: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[
    bool,
    list[str],
]:
    checks = {
        "confirmed_oil_precision": (
            metrics[
                "confirmed_oil_precision"
            ]
            >= args.minimum_confirmed_oil_precision
        ),
        "confirmed_oil_recall": (
            metrics[
                "confirmed_oil_recall"
            ]
            >= args.minimum_confirmed_oil_recall
        ),
        "lookalike_precision": (
            metrics[
                "lookalike_precision"
            ]
            >= args.minimum_lookalike_precision
        ),
        "lookalike_recall": (
            metrics[
                "lookalike_recall"
            ]
            >= args.minimum_lookalike_recall
        ),
        "negative_confirm_rate": (
            metrics[
                "negative_confirm_rate"
            ]
            <= args.max_negative_confirm_rate
            + 1e-12
        ),
        "positive_lookalike_rate": (
            metrics[
                "positive_lookalike_rate"
            ]
            <= args.max_positive_lookalike_rate
            + 1e-12
        ),
    }

    failed = [
        name
        for name, passed
        in checks.items()
        if not passed
    ]

    return (
        len(failed) == 0,
        failed,
    )


def choose_thresholds(
    labels: np.ndarray,
    probabilities: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    candidates = threshold_candidates(
        probabilities
    )

    valid_pairs = []

    for low_threshold in candidates:
        for high_threshold in candidates:
            if low_threshold >= high_threshold:
                continue

            metrics = evaluate_pair(
                labels=labels,
                probabilities=probabilities,
                low_threshold=float(
                    low_threshold
                ),
                high_threshold=float(
                    high_threshold
                ),
            )

            passed, failed = passes(
                metrics,
                args,
            )

            if not passed:
                continue

            utility = (
                4.0
                * metrics[
                    "confirmed_oil_precision"
                ]
                + 3.0
                * metrics[
                    "lookalike_precision"
                ]
                + 2.0
                * metrics[
                    "confirmed_oil_recall"
                ]
                + 2.0
                * metrics[
                    "lookalike_recall"
                ]
                + 0.5
                * metrics[
                    "definite_decision_rate"
                ]
                - 5.0
                * metrics[
                    "negative_confirm_rate"
                ]
                - 3.0
                * metrics[
                    "positive_lookalike_rate"
                ]
            )

            valid_pairs.append(
                {
                    **{
                        key: value
                        for key, value
                        in metrics.items()
                        if key != "decisions"
                    },
                    "utility": float(
                        utility
                    ),
                    "failed_checks": (
                        failed
                    ),
                    "decisions": (
                        metrics[
                            "decisions"
                        ]
                    ),
                }
            )

    if not valid_pairs:
        raise RuntimeError(
            "Belirlenen güvenlik ölçütlerinin tamamını sağlayan "
            "eşik çifti bulunamadı. Model yeniden eğitilmeden "
            "güvenli kalibrasyon yapılamıyor."
        )

    return max(
        valid_pairs,
        key=lambda item: (
            item["utility"],
            item[
                "definite_decision_rate"
            ],
            item[
                "confirmed_oil_recall"
            ],
            item[
                "lookalike_recall"
            ],
            -item[
                "overall_uncertain_rate"
            ],
        ),
    )


def main() -> None:
    args = parse_args()

    (
        predictions,
        old_config,
        checkpoint,
    ) = load_inputs()

    temperature = float(
        old_config.get(
            "temperature",
            checkpoint.get(
                "temperature",
                1.0,
            ),
        )
    )

    labels = predictions[
        "label"
    ].to_numpy(
        dtype=np.int64
    )

    logits = predictions[
        "logit"
    ].to_numpy(
        dtype=np.float64
    )

    probabilities = calibrated_probabilities(
        logits=logits,
        temperature=temperature,
    )

    selected = choose_thresholds(
        labels=labels,
        probabilities=probabilities,
        args=args,
    )

    decisions = selected.pop(
        "decisions"
    )

    output_predictions = (
        predictions.copy()
    )

    output_predictions[
        "calibrated_probability"
    ] = probabilities

    output_predictions[
        "recalibrated_decision"
    ] = decisions

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_predictions.to_csv(
        RECALIBRATED_PREDICTIONS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    gate_passed, failed = passes(
        selected,
        args,
    )

    new_config = {
        "stage": (
            "v07_lookalike_aware_verifier_recalibration_v2"
        ),
        "model_checkpoint": str(
            CHECKPOINT_PATH.relative_to(
                ROOT
            )
        ).replace(
            "\\",
            "/",
        ),
        "temperature": float(
            temperature
        ),
        "lookalike_threshold": float(
            selected[
                "lookalike_threshold"
            ]
        ),
        "confirmed_oil_threshold": float(
            selected[
                "confirmed_oil_threshold"
            ]
        ),
        "decision_rule": {
            "LOOK_ALIKE": (
                "probability <= lookalike_threshold"
            ),
            "UNCERTAIN": (
                "lookalike_threshold < probability "
                "< confirmed_oil_threshold"
            ),
            "CONFIRMED_OIL": (
                "probability >= confirmed_oil_threshold"
            ),
        },
        "calibration_metrics": {
            key: value
            for key, value
            in selected.items()
            if key not in {
                "utility",
                "failed_checks",
            }
        },
        "gate_passed": bool(
            gate_passed
        ),
        "failed_checks": failed,
        "criteria": {
            "minimum_confirmed_oil_precision": (
                args.minimum_confirmed_oil_precision
            ),
            "minimum_confirmed_oil_recall": (
                args.minimum_confirmed_oil_recall
            ),
            "minimum_lookalike_precision": (
                args.minimum_lookalike_precision
            ),
            "minimum_lookalike_recall": (
                args.minimum_lookalike_recall
            ),
            "max_negative_confirm_rate": (
                args.max_negative_confirm_rate
            ),
            "max_positive_lookalike_rate": (
                args.max_positive_lookalike_rate
            ),
        },
        "previous_config": str(
            OLD_CONFIG_PATH.relative_to(
                ROOT
            )
        ).replace(
            "\\",
            "/",
        ),
        "fresh_end_to_end_holdout_used": False,
        "scientific_note": (
            "Model yeniden eğitilmemiştir. Aynı calibration split "
            "üzerinde güvenlik kriterlerini doğrudan içeren eşik "
            "optimizasyonu yapılmıştır. Bu sonuç bağımsız test kanıtı "
            "değildir; taze holdout hâlâ kullanılmamıştır."
        ),
    }

    NEW_CONFIG_PATH.write_text(
        json.dumps(
            new_config,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    summary = {
        "stage": (
            "v07_verifier_safe_recalibration"
        ),
        "training_performed": False,
        "checkpoint_changed": False,
        "temperature": float(
            temperature
        ),
        "old_thresholds": {
            "lookalike_threshold": float(
                old_config[
                    "lookalike_threshold"
                ]
            ),
            "confirmed_oil_threshold": float(
                old_config[
                    "confirmed_oil_threshold"
                ]
            ),
        },
        "new_thresholds": {
            "lookalike_threshold": float(
                selected[
                    "lookalike_threshold"
                ]
            ),
            "confirmed_oil_threshold": float(
                selected[
                    "confirmed_oil_threshold"
                ]
            ),
        },
        "metrics": {
            key: value
            for key, value
            in selected.items()
            if key not in {
                "utility",
                "failed_checks",
            }
        },
        "gate_passed": bool(
            gate_passed
        ),
        "failed_checks": failed,
        "new_config": str(
            NEW_CONFIG_PATH.relative_to(
                ROOT
            )
        ).replace(
            "\\",
            "/",
        ),
        "recalibrated_predictions": str(
            RECALIBRATED_PREDICTIONS_PATH.relative_to(
                ROOT
            )
        ).replace(
            "\\",
            "/",
        ),
        "fresh_end_to_end_holdout_used": False,
    }

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("=" * 78)
    print(
        "v0.7 GÜVENLİ EŞİK YENİDEN KALİBRASYONU"
    )
    print("=" * 78)

    print(
        "Model yeniden eğitildi mi: False"
    )

    print(
        "Gate:",
        (
            "PASS"
            if gate_passed
            else "FAIL"
        ),
    )

    print(
        "Eski LOOK_ALIKE threshold:",
        f"{float(old_config['lookalike_threshold']):.6f}",
    )

    print(
        "Yeni LOOK_ALIKE threshold:",
        f"{selected['lookalike_threshold']:.6f}",
    )

    print(
        "Eski CONFIRMED_OIL threshold:",
        f"{float(old_config['confirmed_oil_threshold']):.6f}",
    )

    print(
        "Yeni CONFIRMED_OIL threshold:",
        f"{selected['confirmed_oil_threshold']:.6f}",
    )

    print(
        "Confirmed precision:",
        f"{selected['confirmed_oil_precision']:.4f}",
    )

    print(
        "Confirmed recall:",
        f"{selected['confirmed_oil_recall']:.4f}",
    )

    print(
        "Look-alike precision:",
        f"{selected['lookalike_precision']:.4f}",
    )

    print(
        "Look-alike recall:",
        f"{selected['lookalike_recall']:.4f}",
    )

    print(
        "Uncertain rate:",
        f"{selected['overall_uncertain_rate']:.4f}",
    )

    print(
        "Confirmed false positive:",
        selected[
            "confirmed_false_positive_count"
        ],
    )

    print(
        "Look-alike false negative:",
        selected[
            "lookalike_false_negative_count"
        ],
    )

    print(
        "Yeni config:",
        NEW_CONFIG_PATH.resolve(),
    )

    print(
        "Özet:",
        SUMMARY_PATH.resolve(),
    )


if __name__ == "__main__":
    main()
