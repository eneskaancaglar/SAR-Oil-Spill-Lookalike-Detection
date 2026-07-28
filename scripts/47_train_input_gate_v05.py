from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler


ROOT = Path(__file__).resolve().parents[1]

BASE_SCRIPT = (
    ROOT
    / "scripts"
    / "40_train_input_gate.py"
)

MANIFEST_PATH = (
    ROOT
    / "data"
    / "metadata"
    / "v05_input_gate_dataset.csv"
)

INITIAL_CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "input_gate_v04"
    / "best_input_gate.pth"
)

CHECKPOINT_DIR = (
    ROOT
    / "checkpoints"
    / "input_gate_v05"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "v05_input_gate_training"
)

REPORT_PATH = (
    ROOT
    / "reports"
    / "v05_input_gate_training.md"
)

BEST_CHECKPOINT = (
    CHECKPOINT_DIR
    / "best_input_gate_v05.pth"
)

LAST_CHECKPOINT = (
    CHECKPOINT_DIR
    / "last_input_gate_v05.pth"
)

HISTORY_PATH = (
    OUTPUT_DIR
    / "training_history.csv"
)

VALIDATION_PREDICTIONS_PATH = (
    OUTPUT_DIR
    / "validation_predictions.csv"
)

GROUP_METRICS_PATH = (
    OUTPUT_DIR
    / "validation_group_metrics.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "training_summary.json"
)

LOSS_PLOT_PATH = (
    OUTPUT_DIR
    / "loss_curve.png"
)

METRICS_PLOT_PATH = (
    OUTPUT_DIR
    / "validation_metrics.png"
)


def load_base_module():
    if not BASE_SCRIPT.exists():
        raise FileNotFoundError(
            f"Temel eğitim scripti bulunamadı: {BASE_SCRIPT}"
        )

    spec = importlib.util.spec_from_file_location(
        "input_gate_v04_training",
        BASE_SCRIPT,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "40_train_input_gate.py yüklenemedi."
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    module.MANIFEST_PATH = MANIFEST_PATH
    module.CHECKPOINT_DIR = CHECKPOINT_DIR
    module.OUTPUT_DIR = OUTPUT_DIR
    module.REPORT_PATH = REPORT_PATH

    module.BEST_CHECKPOINT = BEST_CHECKPOINT
    module.LAST_CHECKPOINT = LAST_CHECKPOINT

    module.HISTORY_PATH = HISTORY_PATH
    module.VALIDATION_PREDICTIONS_PATH = (
        VALIDATION_PREDICTIONS_PATH
    )
    module.GROUP_METRICS_PATH = (
        GROUP_METRICS_PATH
    )
    module.SUMMARY_PATH = SUMMARY_PATH

    module.LOSS_PLOT_PATH = LOSS_PLOT_PATH
    module.METRICS_PLOT_PATH = (
        METRICS_PLOT_PATH
    )

    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "v0.4 input-gate ağırlıklarını yeni optik ve "
            "deniz dışı SAR hard-negative verileriyle "
            "v0.5 olarak fine-tune eder."
        )
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=[
            "auto",
            "cuda",
            "cpu",
        ],
    )

    return parser.parse_args()


def load_checkpoint_payload(
    path: Path,
    device: torch.device,
):
    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=False,
        )

    except TypeError:
        return torch.load(
            path,
            map_location=device,
        )


def create_sampler_weights(
    train_dataframe: pd.DataFrame,
) -> torch.Tensor:
    dataframe = train_dataframe.copy()

    if "sample_weight" not in dataframe.columns:
        dataframe["sample_weight"] = 1.0

    dataframe["sample_weight"] = (
        pd.to_numeric(
            dataframe["sample_weight"],
            errors="coerce",
        )
        .fillna(1.0)
        .clip(lower=0.1, upper=20.0)
    )

    labels = dataframe[
        "binary_label"
    ].astype(int)

    class_weight_totals = (
        dataframe
        .groupby(labels)[
            "sample_weight"
        ]
        .transform("sum")
    )

    normalized_weights = (
        dataframe["sample_weight"]
        / class_weight_totals
    )

    weights_array = normalized_weights.to_numpy(
        dtype=np.float64,
        copy=True,
    )

    if not np.isfinite(
        weights_array
    ).all():
        raise RuntimeError(
            "Sampler ağırlıklarında geçersiz değer bulundu."
        )

    if (
        weights_array <= 0
    ).any():
        raise RuntimeError(
            "Sampler ağırlıkları pozitif olmalıdır."
        )

    return torch.tensor(
        weights_array,
        dtype=torch.double,
    )


def write_report(
    summary: dict,
) -> None:
    metrics = summary[
        "best_validation_metrics"
    ]

    lines = [
        "# v0.5 Input-Gate Fine-Tuning",
        "",
        "## Amaç",
        "",
        "Yalnız DARTIS benzeri deniz ve kıyı SAR "
        "görüntülerinin petrol pipeline'ına gönderilmesini "
        "sağlayan muhafazakâr giriş modeli geliştirmek.",
        "",
        "## Başlangıç modeli",
        "",
        "- v0.4 input-gate ResNet18",
        "- Bütün katmanlar düşük öğrenme oranıyla fine-tune edildi",
        "",
        "## Yeni negatif veri",
        "",
        "- Tüketilmiş UC Merced hard-negative görüntüleri",
        "- EuroSAT optik kara, River ve diğer optik sınıflar",
        "- OpenSARUrban gerçek fakat deniz olmayan VV SAR",
        "",
        "## Kullanılan split'ler",
        "",
        f"- Train: {summary['train_samples']}",
        f"- Validation: {summary['validation_samples']}",
        "- Calibration: kullanılmadı",
        "- test_negative_locked: kullanılmadı",
        "",
        "## Geçici validation sonucu — threshold 0.50",
        "",
        f"- Accuracy: {metrics['accuracy']:.4f}",
        f"- Balanced accuracy: "
        f"{metrics['balanced_accuracy']:.4f}",
        f"- Desteklenen SAR recall: "
        f"{metrics['supported_recall']:.4f}",
        f"- Desteklenmeyen reddetme: "
        f"{metrics['unsupported_rejection_rate']:.4f}",
        f"- False acceptance rate: "
        f"{metrics['false_acceptance_rate']:.4f}",
        f"- False rejection rate: "
        f"{metrics['false_rejection_rate']:.4f}",
        f"- F1: {metrics['f1']:.4f}",
        "",
        "## Bilimsel durum",
        "",
        "Bu aşamadaki 0.50 eşiği nihai eşik değildir.",
        "",
        "Calibration ve test_negative_locked bölümleri "
        "eğitim sırasında kullanılmamıştır.",
        "",
        "Nihai KABUL, BELİRSİZ ve RED eşikleri sonraki "
        "kalibrasyon aşamasında dondurulacaktır.",
        "",
    ]

    REPORT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    base = load_base_module()

    base.set_seed(
        args.seed
    )

    for required_path in (
        MANIFEST_PATH,
        INITIAL_CHECKPOINT,
    ):
        if not required_path.exists():
            raise FileNotFoundError(
                f"Gerekli dosya bulunamadı: {required_path}"
            )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = pd.read_csv(
        MANIFEST_PATH,
        encoding="utf-8-sig",
        low_memory=False,
    )

    required_columns = {
        "sample_id",
        "binary_label",
        "split",
        "source_dataset",
        "source_group",
        "image_path",
        "sample_weight",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise RuntimeError(
            "Manifestte eksik sütunlar: "
            f"{sorted(missing_columns)}"
        )

    train_dataframe = dataframe[
        dataframe[
            "split"
        ].eq(
            "train"
        )
    ].copy()

    validation_dataframe = dataframe[
        dataframe[
            "split"
        ].eq(
            "validation"
        )
    ].copy()

    if train_dataframe.empty:
        raise RuntimeError(
            "Train split boş."
        )

    if validation_dataframe.empty:
        raise RuntimeError(
            "Validation split boş."
        )

    for split_name, subset in (
        (
            "train",
            train_dataframe,
        ),
        (
            "validation",
            validation_dataframe,
        ),
    ):
        labels = set(
            subset[
                "binary_label"
            ].astype(int).unique()
        )

        if labels != {0, 1}:
            raise RuntimeError(
                f"{split_name} iki sınıfı da içermiyor: "
                f"{sorted(labels)}"
            )

    (
        training_transform,
        evaluation_transform,
    ) = base.build_transforms()

    train_dataset = base.InputGateDataset(
        train_dataframe,
        training_transform,
    )

    validation_dataset = (
        base.InputGateDataset(
            validation_dataframe,
            evaluation_transform,
        )
    )

    sampler_weights = create_sampler_weights(
        train_dataframe
    )

    sampler = WeightedRandomSampler(
        weights=sampler_weights,
        num_samples=len(
            train_dataset
        ),
        replacement=True,
        generator=(
            torch.Generator()
            .manual_seed(
                args.seed
            )
        ),
    )

    device = base.select_device(
        args.device
    )

    pin_memory = (
        device.type == "cuda"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=pin_memory,
        persistent_workers=(
            args.workers > 0
        ),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
        persistent_workers=(
            args.workers > 0
        ),
    )

    model, _ = base.build_model(
        use_pretrained=False
    )

    initial_payload = load_checkpoint_payload(
        INITIAL_CHECKPOINT,
        device,
    )

    model.load_state_dict(
        initial_payload[
            "model_state_dict"
        ],
        strict=True,
    )

    initialization = (
        "v04_input_gate_finetune"
    )

    model.to(
        device
    )

    criterion = (
        torch.nn.BCEWithLogitsLoss()
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = (
        torch.optim.lr_scheduler
        .ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=1,
            min_lr=1e-6,
        )
    )

    amp_enabled = (
        device.type == "cuda"
    )

    try:
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=amp_enabled,
        )

    except (AttributeError, TypeError):
        scaler = (
            torch.cuda.amp.GradScaler(
                enabled=amp_enabled
            )
        )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("=" * 78)
    print(
        "ROBUST BINARY OIL DETECTOR v0.5"
    )
    print(
        "INPUT-GATE HARD-NEGATIVE FINE-TUNING"
    )
    print("=" * 78)

    print(
        "Cihaz:",
        device,
    )

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print(
        "Başlangıç:",
        initialization,
    )

    print(
        "Parametre:",
        f"{parameter_count:,}",
    )

    print(
        "Train:",
        len(train_dataset),
    )

    print(
        "Validation:",
        len(validation_dataset),
    )

    print(
        "Train supported:",
        int(
            train_dataframe[
                "binary_label"
            ].eq(1).sum()
        ),
    )

    print(
        "Train unsupported:",
        int(
            train_dataframe[
                "binary_label"
            ].eq(0).sum()
        ),
    )

    print(
        "Calibration kullanımı: YOK"
    )

    print(
        "test_negative_locked kullanımı: YOK"
    )

    print()

    history_rows = []

    best_score = -1.0
    best_balanced_accuracy = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        train_loss = base.train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            scaler,
            device,
        )

        (
            validation_loss,
            validation_metrics,
            _,
        ) = base.evaluate(
            model,
            validation_loader,
            criterion,
            device,
            threshold=0.50,
        )

        supported_recall = float(
            validation_metrics[
                "supported_recall"
            ]
        )

        rejection_rate = float(
            validation_metrics[
                "unsupported_rejection_rate"
            ]
        )

        balanced_accuracy = float(
            validation_metrics[
                "balanced_accuracy"
            ]
        )

        safety_score = min(
            supported_recall,
            rejection_rate,
        )

        scheduler.step(
            safety_score
        )

        learning_rate = float(
            optimizer.param_groups[
                0
            ]["lr"]
        )

        row = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": float(
                train_loss
            ),
            "validation_loss": float(
                validation_loss
            ),
            "validation_accuracy": (
                validation_metrics[
                    "accuracy"
                ]
            ),
            "validation_balanced_accuracy": (
                balanced_accuracy
            ),
            "validation_supported_recall": (
                supported_recall
            ),
            "validation_unsupported_rejection_rate": (
                rejection_rate
            ),
            "validation_false_acceptance_rate": (
                validation_metrics[
                    "false_acceptance_rate"
                ]
            ),
            "validation_false_rejection_rate": (
                validation_metrics[
                    "false_rejection_rate"
                ]
            ),
            "validation_f1": (
                validation_metrics[
                    "f1"
                ]
            ),
            "validation_safety_score": (
                safety_score
            ),
        }

        history_rows.append(
            row
        )

        base.save_checkpoint(
            LAST_CHECKPOINT,
            epoch,
            model,
            optimizer,
            scheduler,
            validation_metrics,
            args,
            initialization,
        )

        improved = (
            safety_score
            > best_score + 1e-6
            or (
                abs(
                    safety_score
                    - best_score
                ) <= 1e-6
                and balanced_accuracy
                > best_balanced_accuracy
                + 1e-6
            )
        )

        if improved:
            best_score = safety_score
            best_balanced_accuracy = (
                balanced_accuracy
            )
            best_epoch = epoch
            epochs_without_improvement = 0

            base.save_checkpoint(
                BEST_CHECKPOINT,
                epoch,
                model,
                optimizer,
                scheduler,
                validation_metrics,
                args,
                initialization,
            )

        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} "
            f"| train_loss={train_loss:.4f} "
            f"| val_loss={validation_loss:.4f} "
            f"| bal_acc={balanced_accuracy:.4f} "
            f"| SAR_recall={supported_recall:.4f} "
            f"| reject={rejection_rate:.4f} "
            f"| FAR="
            f"{validation_metrics['false_acceptance_rate']:.4f} "
            f"| safety={safety_score:.4f} "
            f"| lr={learning_rate:.2e}"
            f"{' BEST' if improved else ''}"
        )

        pd.DataFrame(
            history_rows
        ).to_csv(
            HISTORY_PATH,
            index=False,
            encoding="utf-8-sig",
        )

        if (
            epochs_without_improvement
            >= args.patience
        ):
            print()
            print(
                "Early stopping:",
                f"{args.patience} epoch boyunca "
                "iyileşme olmadı.",
            )

            break

    checkpoint_payload = base.load_checkpoint(
        BEST_CHECKPOINT,
        model,
        device,
    )

    (
        best_validation_loss,
        best_validation_metrics,
        validation_predictions,
    ) = base.evaluate(
        model,
        validation_loader,
        criterion,
        device,
        threshold=0.50,
    )

    validation_metadata = (
        validation_dataframe[
            [
                "sample_id",
                "source_dataset",
                "source_group",
                "scene_id",
                "image_path",
                "difficulty_type",
            ]
        ]
        .drop_duplicates(
            subset=[
                "sample_id"
            ]
        )
    )

    validation_predictions = (
        validation_predictions.merge(
            validation_metadata,
            on="sample_id",
            how="left",
            validate="one_to_one",
        )
    )

    validation_predictions.to_csv(
        VALIDATION_PREDICTIONS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    group_metrics = (
        base.calculate_group_metrics(
            validation_predictions
        )
    )

    group_metrics.to_csv(
        GROUP_METRICS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    history_dataframe = pd.DataFrame(
        history_rows
    )

    base.save_plots(
        history_dataframe
    )

    summary = {
        "model_version": (
            "input_gate_v05"
        ),
        "architecture": (
            "resnet18_grayscale_input_gate"
        ),
        "initialization": initialization,
        "parameter_count": int(
            parameter_count
        ),
        "device": str(
            device
        ),
        "train_samples": int(
            len(train_dataset)
        ),
        "validation_samples": int(
            len(validation_dataset)
        ),
        "best_epoch": int(
            best_epoch
        ),
        "checkpoint_epoch": int(
            checkpoint_payload[
                "epoch"
            ]
        ),
        "best_validation_loss": float(
            best_validation_loss
        ),
        "best_validation_metrics": (
            best_validation_metrics
        ),
        "best_safety_score": float(
            best_score
        ),
        "temporary_threshold": 0.50,
        "calibration_used": False,
        "test_negative_locked_used": False,
        "initial_checkpoint": str(
            INITIAL_CHECKPOINT
        ),
        "best_checkpoint": str(
            BEST_CHECKPOINT
        ),
        "validation_predictions": str(
            VALIDATION_PREDICTIONS_PATH
        ),
        "validation_group_metrics": str(
            GROUP_METRICS_PATH
        ),
    }

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    write_report(
        summary
    )

    print()
    print("=" * 78)
    print(
        "v0.5 INPUT-GATE EĞİTİM SONUCU"
    )
    print("=" * 78)

    print(
        "En iyi epoch:",
        best_epoch,
    )

    print(
        "Validation loss:",
        f"{best_validation_loss:.6f}",
    )

    print(
        "Balanced accuracy:",
        f"{best_validation_metrics['balanced_accuracy']:.4f}",
    )

    print(
        "Desteklenen SAR recall:",
        f"{best_validation_metrics['supported_recall']:.4f}",
    )

    print(
        "Desteklenmeyen reddetme:",
        f"{best_validation_metrics['unsupported_rejection_rate']:.4f}",
    )

    print(
        "False acceptance rate:",
        f"{best_validation_metrics['false_acceptance_rate']:.4f}",
    )

    print(
        "False rejection rate:",
        f"{best_validation_metrics['false_rejection_rate']:.4f}",
    )

    print(
        "Safety score:",
        f"{best_score:.4f}",
    )

    print()
    print(
        "Best checkpoint:",
        BEST_CHECKPOINT.resolve(),
    )

    print(
        "Validation tahminleri:",
        VALIDATION_PREDICTIONS_PATH.resolve(),
    )

    print(
        "Grup sonuçları:",
        GROUP_METRICS_PATH.resolve(),
    )

    print(
        "Özet:",
        SUMMARY_PATH.resolve(),
    )

    print(
        "Rapor:",
        REPORT_PATH.resolve(),
    )


if __name__ == "__main__":
    main()
