from __future__ import annotations

import argparse
import hashlib
import json
import math
import runpy
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]

BASE_TRAINING_SCRIPT = (
    ROOT
    / "scripts"
    / "90_train_lookalike_aware_verifier.py"
)

SOURCE_DATASET = (
    ROOT
    / "data"
    / "metadata"
    / "v08_verifier_binary_dataset.csv"
)

TRAINING_SPLIT_MANIFEST = (
    ROOT
    / "data"
    / "metadata"
    / "v08_verifier_training_split.csv"
)

SPLIT_AUDIT_PATH = (
    ROOT
    / "outputs"
    / "verifier_v08"
    / "split_audit.json"
)

V07_CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "verifier_v07"
    / "best.pth"
)

CHECKPOINT_DIR = (
    ROOT
    / "checkpoints"
    / "verifier_v08"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "verifier_v08"
)

BEST_CHECKPOINT = (
    CHECKPOINT_DIR
    / "best.pth"
)

CALIBRATION_CONFIG = (
    CHECKPOINT_DIR
    / "calibration_config.json"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "summary.json"
)

HISTORY_PATH = (
    OUTPUT_DIR
    / "training_history.csv"
)

INTERNAL_PREDICTIONS = (
    OUTPUT_DIR
    / "internal_validation_predictions.csv"
)

CALIBRATION_PREDICTIONS = (
    OUTPUT_DIR
    / "calibration_predictions.csv"
)

CONTACT_SHEET = (
    OUTPUT_DIR
    / "calibration_decision_contact_sheet.jpg"
)

FAILURE_SOURCE_LABEL = (
    "V07_FRESH_NEGATIVE_FALSE_ALARM"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "v0.7 bağımsız negatif holdout'ta sistemi yanıltan kıyısal "
            "hard-negative'leri yüksek ağırlıkla kullanarak v0.8 verifier "
            "modelini eğitir. Eski calibration bölümü yeniden kullanılmaz; "
            "sahne bazında yeni bir v0.8 split oluşturulur."
        )
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.00005,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0001,
    )

    parser.add_argument(
        "--internal-validation-scene-ratio",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--calibration-scene-ratio",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--hard-failure-calibration-scenes",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--samples-per-epoch",
        type=int,
        default=3200,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
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

    parser.add_argument(
        "--minimum-confirmed-oil-recall",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--minimum-lookalike-recall",
        type=float,
        default=0.75,
    )

    parser.add_argument(
        "--minimum-confirmed-oil-precision",
        type=float,
        default=0.99,
    )

    parser.add_argument(
        "--minimum-lookalike-precision",
        type=float,
        default=0.95,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260729,
    )

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cuda",
            "cpu",
        ],
        default="auto",
    )

    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Gerekli dosya bulunamadı: {path}"
        )


def deterministic_rank(value: str) -> int:
    digest = hashlib.sha256(
        value.encode("utf-8")
    ).digest()

    return int.from_bytes(
        digest[:8],
        "big",
    )


def build_v08_scene_split(
    args: argparse.Namespace,
) -> tuple[
    pd.DataFrame,
    set[str],
    set[str],
]:
    require_file(
        SOURCE_DATASET
    )

    frame = pd.read_csv(
        SOURCE_DATASET,
        encoding="utf-8-sig",
        low_memory=False,
    )

    required = {
        "candidate_id",
        "scene_id",
        "label",
        "class_name",
        "split_role",
        "crop_path",
    }

    missing = required - set(
        frame.columns
    )

    if missing:
        raise RuntimeError(
            "v0.8 veri setinde eksik sütunlar: "
            f"{sorted(missing)}"
        )

    frame = frame.copy()

    frame[
        "scene_id"
    ] = frame[
        "scene_id"
    ].astype(str)

    frame[
        "label"
    ] = frame[
        "label"
    ].astype(int)

    if (
        "source_row_label"
        not in frame.columns
    ):
        frame[
            "source_row_label"
        ] = ""

    failure_rows = frame[
        frame[
            "source_row_label"
        ].astype(str).eq(
            FAILURE_SOURCE_LABEL
        )
    ]

    failure_scenes = sorted(
        failure_rows[
            "scene_id"
        ].unique()
    )

    if len(
        failure_scenes
    ) < 3:
        raise RuntimeError(
            "v0.8 veri setinde yeterli v0.7 failure sahnesi yok. "
            f"Bulunan: {len(failure_scenes)}"
        )

    calibration_failure_count = min(
        int(
            args.hard_failure_calibration_scenes
        ),
        len(
            failure_scenes
        ) - 1,
    )

    ranked_failure_scenes = sorted(
        failure_scenes,
        key=lambda scene_id: (
            deterministic_rank(
                f"v08-failure-calibration::{scene_id}"
            )
        ),
    )

    failure_calibration_scenes = set(
        ranked_failure_scenes[
            :calibration_failure_count
        ]
    )

    failure_train_scenes = set(
        failure_scenes
    ) - failure_calibration_scenes

    excluded_mask = frame[
        "split_role"
    ].astype(str).str.contains(
        "excluded",
        case=False,
        regex=False,
    )

    frame.loc[
        excluded_mask,
        "split_role",
    ] = "excluded_seen_audit"

    eligible = frame[
        ~excluded_mask
    ].copy()

    split_by_scene: dict[str, str] = {}

    for scene_id in failure_train_scenes:
        split_by_scene[
            scene_id
        ] = "train"

    for scene_id in failure_calibration_scenes:
        split_by_scene[
            scene_id
        ] = "calibration"

    for label in (
        0,
        1,
    ):
        class_scenes = sorted(
            set(
                eligible[
                    eligible[
                        "label"
                    ].eq(label)
                ][
                    "scene_id"
                ].unique()
            )
            - set(
                failure_scenes
            )
        )

        if len(
            class_scenes
        ) < 2:
            raise RuntimeError(
                f"Etiket {label} için yeni train/calibration split "
                "oluşturacak yeterli sahne yok."
            )

        ranked = sorted(
            class_scenes,
            key=lambda scene_id: (
                deterministic_rank(
                    f"v08-main-calibration::{label}::{scene_id}"
                )
            ),
        )

        calibration_count = max(
            1,
            int(
                round(
                    len(
                        ranked
                    )
                    * float(
                        args.calibration_scene_ratio
                    )
                )
            ),
        )

        calibration_count = min(
            calibration_count,
            len(
                ranked
            ) - 1,
        )

        calibration_scenes = set(
            ranked[
                :calibration_count
            ]
        )

        for scene_id in ranked:
            split_by_scene[
                scene_id
            ] = (
                "calibration"
                if scene_id
                in calibration_scenes
                else "train"
            )

    eligible[
        "split_role"
    ] = eligible[
        "scene_id"
    ].map(
        split_by_scene
    )

    if eligible[
        "split_role"
    ].isna().any():
        missing_scenes = sorted(
            eligible[
                eligible[
                    "split_role"
                ].isna()
            ][
                "scene_id"
            ].unique()
        )

        raise RuntimeError(
            "Split atanamayan sahneler: "
            + ", ".join(
                missing_scenes[
                    :10
                ]
            )
        )

    combined = pd.concat(
        [
            eligible,
            frame[
                excluded_mask
            ],
        ],
        ignore_index=True,
    )

    leakage = (
        combined[
            combined[
                "split_role"
            ].isin(
                [
                    "train",
                    "calibration",
                ]
            )
        ]
        .groupby(
            "scene_id"
        )[
            "split_role"
        ]
        .nunique()
    )

    if int(
        (
            leakage
            > 1
        ).sum()
    ) != 0:
        raise RuntimeError(
            "v0.8 sahne bazlı train/calibration leakage oluştu."
        )

    for split_name in (
        "train",
        "calibration",
    ):
        labels = set(
            combined[
                combined[
                    "split_role"
                ].eq(
                    split_name
                )
            ][
                "label"
            ].unique()
        )

        if labels != {
            0,
            1,
        }:
            raise RuntimeError(
                f"{split_name} iki sınıfı da içermiyor: {labels}"
            )

    TRAINING_SPLIT_MANIFEST.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined.to_csv(
        TRAINING_SPLIT_MANIFEST,
        index=False,
        encoding="utf-8-sig",
    )

    audit = {
        "stage": (
            "v08_scene_isolated_split"
        ),
        "source_dataset": str(
            SOURCE_DATASET.relative_to(
                ROOT
            )
        ).replace(
            "\\",
            "/",
        ),
        "output_manifest": str(
            TRAINING_SPLIT_MANIFEST.relative_to(
                ROOT
            )
        ).replace(
            "\\",
            "/",
        ),
        "failure_scene_count": int(
            len(
                failure_scenes
            )
        ),
        "failure_train_scenes": sorted(
            failure_train_scenes
        ),
        "failure_calibration_scenes": sorted(
            failure_calibration_scenes
        ),
        "split_counts": {
            str(key): int(value)
            for key, value in combined.groupby(
                "split_role"
            ).size().to_dict().items()
        },
        "split_class_counts": {
            split_name: {
                str(key): int(value)
                for key, value in combined[
                    combined[
                        "split_role"
                    ].eq(
                        split_name
                    )
                ].groupby(
                    "class_name"
                ).size().to_dict().items()
            }
            for split_name in (
                "train",
                "calibration",
            )
        },
        "scene_isolation_verified": True,
    }

    SPLIT_AUDIT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SPLIT_AUDIT_PATH.write_text(
        json.dumps(
            audit,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return (
        combined,
        failure_train_scenes,
        failure_calibration_scenes,
    )


def main() -> None:
    args = parse_args()

    for path in (
        BASE_TRAINING_SCRIPT,
        SOURCE_DATASET,
        V07_CHECKPOINT,
    ):
        require_file(path)

    (
        split_frame,
        protected_failure_train_scenes,
        failure_calibration_scenes,
    ) = build_v08_scene_split(
        args
    )

    module = runpy.run_path(
        str(
            BASE_TRAINING_SCRIPT
        ),
        run_name=(
            "v08_training_base_module"
        ),
    )

    # runpy.run_path() dönen sözlüğün bir kopyasını verebilir.
    # Fonksiyonların gerçek global sözlüğünü doğrudan güncellemeliyiz.
    base_globals = module[
        "main"
    ].__globals__

    base_globals[
        "DATASET_MANIFEST"
    ] = TRAINING_SPLIT_MANIFEST

    base_globals[
        "CHECKPOINT_DIR"
    ] = CHECKPOINT_DIR

    base_globals[
        "OUTPUT_DIR"
    ] = OUTPUT_DIR

    base_globals[
        "BEST_CHECKPOINT"
    ] = BEST_CHECKPOINT

    base_globals[
        "CALIBRATION_CONFIG"
    ] = CALIBRATION_CONFIG

    base_globals[
        "SUMMARY_PATH"
    ] = SUMMARY_PATH

    base_globals[
        "HISTORY_PATH"
    ] = HISTORY_PATH

    base_globals[
        "INTERNAL_PREDICTIONS"
    ] = INTERNAL_PREDICTIONS

    base_globals[
        "CALIBRATION_PREDICTIONS"
    ] = CALIBRATION_PREDICTIONS

    base_globals[
        "CONTACT_SHEET"
    ] = CONTACT_SHEET

    base_load_old_verifier = module[
        "load_old_verifier"
    ]

    def load_v07_initialized_verifier(
        device: torch.device,
    ) -> tuple[
        dict[str, Any],
        torch.nn.Module,
        dict[str, Any],
        Any,
    ]:
        (
            detector,
            model,
            verifier_config,
            transform,
        ) = base_load_old_verifier(
            device
        )

        checkpoint = torch.load(
            V07_CHECKPOINT,
            map_location="cpu",
            weights_only=False,
        )

        state_dict = {
            key: value.detach()
            .clone()
            for key, value
            in checkpoint[
                "model_state_dict"
            ].items()
        }

        model.load_state_dict(
            state_dict,
            strict=True,
        )

        model.to(
            device
        )

        model.train()

        return (
            detector,
            model,
            verifier_config,
            transform,
        )

    base_globals[
        "load_old_verifier"
    ] = load_v07_initialized_verifier

    def weighted_scene_class_weights(
        frame: pd.DataFrame,
    ) -> torch.Tensor:
        class_scene_counts = (
            frame[
                [
                    "label",
                    "scene_id",
                ]
            ]
            .drop_duplicates()
            .groupby(
                "label"
            )
            .size()
            .to_dict()
        )

        crop_counts = (
            frame.groupby(
                [
                    "label",
                    "scene_id",
                ]
            )
            .size()
            .to_dict()
        )

        weights = []

        for row in frame.itertuples():
            label = int(
                row.label
            )

            scene_id = str(
                row.scene_id
            )

            base_weight = (
                1.0
                / max(
                    int(
                        class_scene_counts[
                            label
                        ]
                    ),
                    1,
                )
                / max(
                    int(
                        crop_counts[
                            (
                                label,
                                scene_id,
                            )
                        ]
                    ),
                    1,
                )
            )

            hard_weight = getattr(
                row,
                "hard_negative_weight",
                1.0,
            )

            try:
                hard_weight = float(
                    hard_weight
                )
            except (
                TypeError,
                ValueError,
            ):
                hard_weight = 1.0

            if not math.isfinite(
                hard_weight
            ):
                hard_weight = 1.0

            hard_weight = max(
                hard_weight,
                1.0,
            )

            weights.append(
                base_weight
                * hard_weight
            )

        return torch.tensor(
            weights,
            dtype=torch.double,
        )

    base_globals[
        "build_scene_class_balanced_weights"
    ] = weighted_scene_class_weights

    def split_internal_validation_without_failure_leakage(
        train_frame: pd.DataFrame,
        scene_ratio: float,
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
    ]:
        validation_scenes: set[str] = set()

        for label in (
            0,
            1,
        ):
            class_scenes = sorted(
                set(
                    train_frame[
                        train_frame[
                            "label"
                        ].eq(
                            label
                        )
                    ][
                        "scene_id"
                    ].unique()
                )
                - (
                    protected_failure_train_scenes
                    if label == 0
                    else set()
                )
            )

            if len(
                class_scenes
            ) < 2:
                raise RuntimeError(
                    f"Etiket {label} için korumalı internal validation "
                    "oluşturacak yeterli sahne yok."
                )

            ranked = sorted(
                class_scenes,
                key=lambda scene_id: (
                    deterministic_rank(
                        f"v08-internal-validation::{label}::{scene_id}"
                    )
                ),
            )

            validation_count = max(
                1,
                int(
                    round(
                        len(
                            ranked
                        )
                        * scene_ratio
                    )
                ),
            )

            validation_count = min(
                validation_count,
                len(
                    ranked
                ) - 1,
            )

            validation_scenes.update(
                ranked[
                    :validation_count
                ]
            )

        internal_validation = train_frame[
            train_frame[
                "scene_id"
            ].isin(
                validation_scenes
            )
        ].copy()

        actual_train = train_frame[
            ~train_frame[
                "scene_id"
            ].isin(
                validation_scenes
            )
        ].copy()

        if (
            set(
                actual_train[
                    "scene_id"
                ]
            )
            & set(
                internal_validation[
                    "scene_id"
                ]
            )
        ):
            raise RuntimeError(
                "v0.8 internal train/validation sahne sızıntısı oluştu."
            )

        if not protected_failure_train_scenes.issubset(
            set(
                actual_train[
                    "scene_id"
                ]
            )
        ):
            raise RuntimeError(
                "Korumalı failure train sahnelerinden biri internal "
                "validation'a sızdı."
            )

        for name, subset in (
            (
                "actual_train",
                actual_train,
            ),
            (
                "internal_validation",
                internal_validation,
            ),
        ):
            labels = set(
                subset[
                    "label"
                ].unique()
            )

            if labels != {
                0,
                1,
            }:
                raise RuntimeError(
                    f"{name} iki sınıfı da içermiyor: {labels}"
                )

        return (
            actual_train.reset_index(
                drop=True
            ),
            internal_validation.reset_index(
                drop=True
            ),
        )

    base_globals[
        "split_internal_validation"
    ] = split_internal_validation_without_failure_leakage

    base_args = argparse.Namespace(
        epochs=int(
            args.epochs
        ),
        batch_size=int(
            args.batch_size
        ),
        learning_rate=float(
            args.learning_rate
        ),
        weight_decay=float(
            args.weight_decay
        ),
        internal_validation_scene_ratio=float(
            args.internal_validation_scene_ratio
        ),
        samples_per_epoch=int(
            args.samples_per_epoch
        ),
        patience=int(
            args.patience
        ),
        num_workers=int(
            args.num_workers
        ),
        max_negative_confirm_rate=float(
            args.max_negative_confirm_rate
        ),
        max_positive_lookalike_rate=float(
            args.max_positive_lookalike_rate
        ),
        minimum_confirmed_oil_recall=float(
            args.minimum_confirmed_oil_recall
        ),
        minimum_lookalike_recall=float(
            args.minimum_lookalike_recall
        ),
        minimum_confirmed_oil_precision=float(
            args.minimum_confirmed_oil_precision
        ),
        minimum_lookalike_precision=float(
            args.minimum_lookalike_precision
        ),
        seed=int(
            args.seed
        ),
        device=str(
            args.device
        ),
    )

    base_globals[
        "parse_args"
    ] = lambda: base_args

    print("=" * 78)
    print(
        "v0.8 FAILURE-AWARE VERIFIER TRAINING"
    )
    print("=" * 78)

    print(
        "v0.7 checkpoint başlangıç ağırlığı olarak kullanılacak."
    )

    print(
        "Failure train scenes:",
        sorted(
            protected_failure_train_scenes
        ),
    )

    print(
        "Failure calibration scenes:",
        sorted(
            failure_calibration_scenes
        ),
    )

    print(
        "Fresh v0.8 negative holdout kullanılmayacak."
    )

    print()

    base_globals[
        "main"
    ]()

    for path in (
        SUMMARY_PATH,
        CALIBRATION_CONFIG,
        BEST_CHECKPOINT,
        CALIBRATION_PREDICTIONS,
    ):
        require_file(path)

    summary = json.loads(
        SUMMARY_PATH.read_text(
            encoding="utf-8-sig"
        )
    )

    config = json.loads(
        CALIBRATION_CONFIG.read_text(
            encoding="utf-8-sig"
        )
    )

    predictions = pd.read_csv(
        CALIBRATION_PREDICTIONS,
        encoding="utf-8-sig",
        low_memory=False,
    )

    hard_failure_predictions = predictions[
        predictions[
            "scene_id"
        ].astype(str).isin(
            failure_calibration_scenes
        )
    ].copy()

    if hard_failure_predictions.empty:
        raise RuntimeError(
            "Failure calibration sahneleri calibration tahminlerinde "
            "bulunamadı."
        )

    hard_failure_confirmed_count = int(
        hard_failure_predictions[
            "decision"
        ].astype(str).eq(
            "CONFIRMED_OIL"
        ).sum()
    )

    hard_failure_lookalike_count = int(
        hard_failure_predictions[
            "decision"
        ].astype(str).eq(
            "LOOK_ALIKE"
        ).sum()
    )

    hard_failure_uncertain_count = int(
        hard_failure_predictions[
            "decision"
        ].astype(str).eq(
            "UNCERTAIN"
        ).sum()
    )

    hard_failure_gate = (
        hard_failure_confirmed_count
        == 0
    )

    failed_checks = list(
        summary.get(
            "failed_checks",
            [],
        )
    )

    if not hard_failure_gate:
        failed_checks.append(
            "failure_calibration_confirmed_oil"
        )

    failed_checks = list(
        dict.fromkeys(
            failed_checks
        )
    )

    final_gate_passed = (
        bool(
            summary.get(
                "gate_passed",
                False,
            )
        )
        and hard_failure_gate
    )

    failure_metrics = {
        "scene_count": int(
            len(
                failure_calibration_scenes
            )
        ),
        "candidate_count": int(
            len(
                hard_failure_predictions
            )
        ),
        "confirmed_oil_count": (
            hard_failure_confirmed_count
        ),
        "uncertain_count": (
            hard_failure_uncertain_count
        ),
        "lookalike_count": (
            hard_failure_lookalike_count
        ),
        "zero_confirmed_oil_gate": bool(
            hard_failure_gate
        ),
    }

    summary[
        "stage"
    ] = (
        "v08_failure_aware_verifier_training"
    )

    summary[
        "source_dataset"
    ] = str(
        SOURCE_DATASET.relative_to(
            ROOT
        )
    ).replace(
        "\\",
        "/",
    )

    summary[
        "training_split_manifest"
    ] = str(
        TRAINING_SPLIT_MANIFEST.relative_to(
            ROOT
        )
    ).replace(
        "\\",
        "/",
    )

    summary[
        "initialized_from_checkpoint"
    ] = str(
        V07_CHECKPOINT.relative_to(
            ROOT
        )
    ).replace(
        "\\",
        "/",
    )

    summary[
        "failure_train_scenes"
    ] = sorted(
        protected_failure_train_scenes
    )

    summary[
        "failure_calibration_scenes"
    ] = sorted(
        failure_calibration_scenes
    )

    summary[
        "failure_calibration_metrics"
    ] = failure_metrics

    summary[
        "hard_negative_weighting_enabled"
    ] = True

    summary[
        "gate_passed"
    ] = bool(
        final_gate_passed
    )

    summary[
        "failed_checks"
    ] = failed_checks

    summary[
        "fresh_v08_negative_holdout_used"
    ] = False

    summary[
        "scientific_note"
    ] = (
        "v0.8 verifier v0.7 checkpoint'inden başlatılmıştır. v0.7 bağımsız "
        "negatif holdout'taki altı hata sahnesinin dördü yüksek ağırlıklı "
        "training hard-negative, ikisi sahne izolasyonlu calibration "
        "hard-negative olarak kullanılmıştır. v0.8 için ayrılan yeni "
        "20 nc + 20 nw holdout bu aşamada açılmamış ve inference "
        "çalıştırılmamıştır."
    )

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    config[
        "stage"
    ] = (
        "v08_failure_aware_verifier"
    )

    config[
        "gate_passed"
    ] = bool(
        final_gate_passed
    )

    config[
        "failed_checks"
    ] = failed_checks

    config[
        "failure_calibration_metrics"
    ] = failure_metrics

    config[
        "initialized_from_checkpoint"
    ] = str(
        V07_CHECKPOINT.relative_to(
            ROOT
        )
    ).replace(
        "\\",
        "/",
    )

    CALIBRATION_CONFIG.write_text(
        json.dumps(
            config,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    checkpoint = torch.load(
        BEST_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    checkpoint[
        "stage"
    ] = (
        "v08_failure_aware_verifier"
    )

    checkpoint[
        "gate_passed"
    ] = bool(
        final_gate_passed
    )

    checkpoint[
        "failed_checks"
    ] = failed_checks

    checkpoint[
        "failure_calibration_metrics"
    ] = failure_metrics

    checkpoint[
        "initialized_from_checkpoint"
    ] = str(
        V07_CHECKPOINT.relative_to(
            ROOT
        )
    ).replace(
        "\\",
        "/",
    )

    torch.save(
        checkpoint,
        BEST_CHECKPOINT,
    )

    print()
    print("=" * 78)
    print(
        "v0.8 NİHAİ EĞİTİM KONTROLÜ"
    )
    print("=" * 78)

    print(
        "Final gate:",
        (
            "PASS"
            if final_gate_passed
            else "FAIL"
        ),
    )

    print(
        "Failure calibration confirmed oil:",
        hard_failure_confirmed_count,
    )

    print(
        "Failure calibration uncertain:",
        hard_failure_uncertain_count,
    )

    print(
        "Failure calibration look-alike:",
        hard_failure_lookalike_count,
    )

    print(
        "Fresh v0.8 holdout kullanıldı mı: False"
    )

    print(
        "Özet:",
        SUMMARY_PATH.resolve(),
    )

    print(
        "Checkpoint:",
        BEST_CHECKPOINT.resolve(),
    )


if __name__ == "__main__":
    main()
