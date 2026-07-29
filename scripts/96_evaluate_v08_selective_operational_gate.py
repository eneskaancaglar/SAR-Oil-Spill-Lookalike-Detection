from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

TRAINING_SUMMARY = (
    ROOT
    / "outputs"
    / "verifier_v08"
    / "summary.json"
)

CALIBRATION_CONFIG = (
    ROOT
    / "checkpoints"
    / "verifier_v08"
    / "calibration_config.json"
)

CALIBRATION_PREDICTIONS = (
    ROOT
    / "outputs"
    / "verifier_v08"
    / "calibration_predictions.csv"
)

CHECKPOINT_PATH = (
    ROOT
    / "checkpoints"
    / "verifier_v08"
    / "best.pth"
)

FRESH_HOLDOUT_MANIFEST = (
    ROOT
    / "data"
    / "metadata"
    / "v08_fresh_negative_holdout_scenes.csv"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "verifier_v08_operational_gate"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "summary.json"
)

OPERATIONAL_CONFIG = (
    ROOT
    / "checkpoints"
    / "verifier_v08"
    / "operational_gate_config.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "v0.8 verifier'ı üç sınıflı genel sınıflandırıcı olarak değil, "
            "yalnız yüksek güvenli CONFIRMED_OIL adaylarının nihai maskeye "
            "girmesine izin veren seçici güvenlik kapısı olarak değerlendirir. "
            "Modeli ve eşikleri değiştirmez."
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
        "--maximum-confirmed-negative-count",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--maximum-positive-lookalike-rate",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--minimum-negative-safe-block-rate",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--minimum-definite-decision-rate",
        type=float,
        default=0.75,
    )

    parser.add_argument(
        "--maximum-failure-calibration-confirmed",
        type=int,
        default=0,
    )

    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Gerekli dosya bulunamadı: {path}"
        )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(
            encoding="utf-8-sig"
        )
    )


def relative(path: Path) -> str:
    return str(
        path.resolve().relative_to(
            ROOT.resolve()
        )
    ).replace("\\", "/")


def main() -> None:
    args = parse_args()

    for path in (
        TRAINING_SUMMARY,
        CALIBRATION_CONFIG,
        CALIBRATION_PREDICTIONS,
        CHECKPOINT_PATH,
        FRESH_HOLDOUT_MANIFEST,
    ):
        require_file(path)

    training_summary = load_json(
        TRAINING_SUMMARY
    )

    calibration_config = load_json(
        CALIBRATION_CONFIG
    )

    predictions = pd.read_csv(
        CALIBRATION_PREDICTIONS,
        encoding="utf-8-sig",
        low_memory=False,
    )

    required = {
        "label",
        "decision",
    }

    missing = required - set(
        predictions.columns
    )

    if missing:
        raise RuntimeError(
            "Calibration tahminlerinde eksik sütunlar: "
            f"{sorted(missing)}"
        )

    labels = predictions[
        "label"
    ].astype(int)

    decisions = predictions[
        "decision"
    ].astype(str)

    positive = labels.eq(1)
    negative = labels.eq(0)

    confirmed = decisions.eq(
        "CONFIRMED_OIL"
    )

    lookalike = decisions.eq(
        "LOOK_ALIKE"
    )

    uncertain = decisions.eq(
        "UNCERTAIN"
    )

    positive_count = int(
        positive.sum()
    )

    negative_count = int(
        negative.sum()
    )

    confirmed_true_positive = int(
        (
            confirmed
            & positive
        ).sum()
    )

    confirmed_false_positive = int(
        (
            confirmed
            & negative
        ).sum()
    )

    lookalike_false_negative = int(
        (
            lookalike
            & positive
        ).sum()
    )

    negative_safe_block_count = int(
        (
            (
                lookalike
                | uncertain
            )
            & negative
        ).sum()
    )

    definite_count = int(
        (
            confirmed
            | lookalike
        ).sum()
    )

    confirmed_precision = (
        confirmed_true_positive
        / (
            confirmed_true_positive
            + confirmed_false_positive
        )
        if (
            confirmed_true_positive
            + confirmed_false_positive
        ) > 0
        else 1.0
    )

    confirmed_recall = (
        confirmed_true_positive
        / positive_count
        if positive_count > 0
        else 0.0
    )

    positive_lookalike_rate = (
        lookalike_false_negative
        / positive_count
        if positive_count > 0
        else 0.0
    )

    negative_safe_block_rate = (
        negative_safe_block_count
        / negative_count
        if negative_count > 0
        else 0.0
    )

    definite_decision_rate = (
        definite_count
        / len(
            predictions
        )
        if len(
            predictions
        ) > 0
        else 0.0
    )

    failure_metrics = training_summary.get(
        "failure_calibration_metrics",
        {}
    )

    failure_confirmed_count = int(
        failure_metrics.get(
            "confirmed_oil_count",
            0,
        )
    )

    fresh_holdout_used = bool(
        training_summary.get(
            "fresh_v08_negative_holdout_used",
            training_summary.get(
                "fresh_end_to_end_holdout_used",
                False,
            ),
        )
    )

    checks = {
        "fresh_holdout_not_used": (
            not fresh_holdout_used
        ),
        "confirmed_oil_precision": (
            confirmed_precision
            >= args.minimum_confirmed_oil_precision
        ),
        "confirmed_oil_recall": (
            confirmed_recall
            >= args.minimum_confirmed_oil_recall
        ),
        "confirmed_negative_count": (
            confirmed_false_positive
            <= args.maximum_confirmed_negative_count
        ),
        "positive_lookalike_rate": (
            positive_lookalike_rate
            <= args.maximum_positive_lookalike_rate
        ),
        "negative_safe_block_rate": (
            negative_safe_block_rate
            >= args.minimum_negative_safe_block_rate
        ),
        "definite_decision_rate": (
            definite_decision_rate
            >= args.minimum_definite_decision_rate
        ),
        "failure_calibration_confirmed": (
            failure_confirmed_count
            <= args.maximum_failure_calibration_confirmed
        ),
    }

    failed_checks = [
        name
        for name, passed
        in checks.items()
        if not passed
    ]

    gate_passed = (
        len(
            failed_checks
        )
        == 0
    )

    metrics = {
        "calibration_count": int(
            len(
                predictions
            )
        ),
        "positive_count": (
            positive_count
        ),
        "negative_count": (
            negative_count
        ),
        "confirmed_oil_precision": float(
            confirmed_precision
        ),
        "confirmed_oil_recall": float(
            confirmed_recall
        ),
        "confirmed_negative_count": (
            confirmed_false_positive
        ),
        "positive_lookalike_count": (
            lookalike_false_negative
        ),
        "positive_lookalike_rate": float(
            positive_lookalike_rate
        ),
        "negative_safe_block_count": (
            negative_safe_block_count
        ),
        "negative_safe_block_rate": float(
            negative_safe_block_rate
        ),
        "definite_decision_rate": float(
            definite_decision_rate
        ),
        "overall_uncertain_rate": float(
            uncertain.mean()
        ),
        "failure_calibration_confirmed_count": (
            failure_confirmed_count
        ),
    }

    criteria = {
        "minimum_confirmed_oil_precision": float(
            args.minimum_confirmed_oil_precision
        ),
        "minimum_confirmed_oil_recall": float(
            args.minimum_confirmed_oil_recall
        ),
        "maximum_confirmed_negative_count": int(
            args.maximum_confirmed_negative_count
        ),
        "maximum_positive_lookalike_rate": float(
            args.maximum_positive_lookalike_rate
        ),
        "minimum_negative_safe_block_rate": float(
            args.minimum_negative_safe_block_rate
        ),
        "minimum_definite_decision_rate": float(
            args.minimum_definite_decision_rate
        ),
        "maximum_failure_calibration_confirmed": int(
            args.maximum_failure_calibration_confirmed
        ),
    }

    operational_config = {
        "stage": (
            "v08_selective_confirmed_oil_operational_gate"
        ),
        "model_checkpoint": relative(
            CHECKPOINT_PATH
        ),
        "temperature": float(
            calibration_config[
                "temperature"
            ]
        ),
        "lookalike_threshold": float(
            calibration_config[
                "lookalike_threshold"
            ]
        ),
        "confirmed_oil_threshold": float(
            calibration_config[
                "confirmed_oil_threshold"
            ]
        ),
        "decision_rule": {
            "LOOK_ALIKE": (
                "probability <= lookalike_threshold; "
                "nihai petrol maskesine girmez"
            ),
            "UNCERTAIN": (
                "lookalike_threshold < probability "
                "< confirmed_oil_threshold; "
                "nihai petrol maskesine girmez"
            ),
            "CONFIRMED_OIL": (
                "probability >= confirmed_oil_threshold; "
                "yalnız bu aday nihai petrol maskesine girebilir"
            ),
        },
        "operational_gate_passed": bool(
            gate_passed
        ),
        "failed_checks": (
            failed_checks
        ),
        "metrics": metrics,
        "criteria": criteria,
        "fresh_v08_negative_holdout_used": (
            False
        ),
        "scientific_note": (
            "Bu değerlendirme LOOK_ALIKE sınıfının genel sınıflandırma "
            "başarısını nihai kapı ölçütü yapmaz. Operasyonel amaç, negatif "
            "adayların CONFIRMED_OIL olarak nihai maskeye girmesini "
            "engellerken yeterli sayıda gerçek petrol adayını yalnız yüksek "
            "güvenle onaylamaktır. LOOK_ALIKE ve UNCERTAIN kararlarının ikisi "
            "de nihai petrol maskesini güvenli biçimde engeller. Bu yalnız "
            "calibration geliştirme sonucudur; bağımsız v0.8 holdout henüz "
            "kullanılmamıştır."
        ),
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OPERATIONAL_CONFIG.write_text(
        json.dumps(
            operational_config,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    summary = {
        "stage": (
            "v08_selective_operational_gate_evaluation"
        ),
        "training_performed": False,
        "thresholds_changed": False,
        "operational_gate_passed": bool(
            gate_passed
        ),
        "failed_checks": (
            failed_checks
        ),
        "metrics": metrics,
        "criteria": criteria,
        "source_training_gate_passed": bool(
            training_summary.get(
                "gate_passed",
                False,
            )
        ),
        "source_training_failed_checks": (
            training_summary.get(
                "failed_checks",
                [],
            )
        ),
        "operational_config": relative(
            OPERATIONAL_CONFIG
        ),
        "fresh_v08_negative_holdout_used": (
            False
        ),
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
        "v0.8 SEÇİCİ OPERASYONEL PETROL KAPISI"
    )
    print("=" * 78)

    print(
        "Model yeniden eğitildi mi: False"
    )

    print(
        "Eşikler değiştirildi mi: False"
    )

    print(
        "Operational gate:",
        (
            "PASS"
            if gate_passed
            else "FAIL"
        ),
    )

    print(
        "Confirmed-oil precision:",
        f"{confirmed_precision:.4f}",
    )

    print(
        "Confirmed-oil recall:",
        f"{confirmed_recall:.4f}",
    )

    print(
        "Confirmed negative:",
        confirmed_false_positive,
        "/",
        negative_count,
    )

    print(
        "Negative safe block:",
        f"{negative_safe_block_rate:.4f}",
    )

    print(
        "Positive look-alike rate:",
        f"{positive_lookalike_rate:.4f}",
    )

    print(
        "Definite decision rate:",
        f"{definite_decision_rate:.4f}",
    )

    print(
        "Failure calibration confirmed:",
        failure_confirmed_count,
    )

    print(
        "Failed:",
        failed_checks,
    )

    print(
        "Fresh v0.8 holdout kullanıldı mı: False"
    )

    print(
        "Config:",
        OPERATIONAL_CONFIG.resolve(),
    )

    print(
        "Özet:",
        SUMMARY_PATH.resolve(),
    )


if __name__ == "__main__":
    main()
