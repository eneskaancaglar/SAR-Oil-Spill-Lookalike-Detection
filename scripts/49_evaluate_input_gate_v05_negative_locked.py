from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode


ROOT = Path(__file__).resolve().parents[1]

CALIBRATION_SCRIPT = (
    ROOT
    / "scripts"
    / "48_calibrate_input_gate_v05.py"
)

MANIFEST_PATH = (
    ROOT
    / "data"
    / "metadata"
    / "v05_input_gate_dataset.csv"
)

CHECKPOINT_PATH = (
    ROOT
    / "checkpoints"
    / "input_gate_v05"
    / "best_input_gate_v05.pth"
)

CONFIG_PATH = (
    ROOT
    / "checkpoints"
    / "input_gate_v05"
    / "input_gate_config_v05.json"
)

PROTOTYPES_PATH = (
    ROOT
    / "checkpoints"
    / "input_gate_v05"
    / "supported_prototypes_v05.npz"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "v05_input_gate_negative_locked_test"
)

PREDICTIONS_PATH = (
    OUTPUT_DIR
    / "locked_negative_predictions.csv"
)

GROUP_METRICS_PATH = (
    OUTPUT_DIR
    / "locked_negative_group_metrics.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "locked_negative_summary.json"
)

REVIEW_DIR = (
    OUTPUT_DIR
    / "review_images"
)

REPORT_PATH = (
    ROOT
    / "reports"
    / "v05_input_gate_negative_locked_test.md"
)


def load_calibration_module():
    spec = importlib.util.spec_from_file_location(
        "input_gate_v05_calibration",
        CALIBRATION_SCRIPT,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Kalibrasyon modülü yüklenemedi."
        )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


def resolve_path(value: Any) -> Path:
    path = Path(
        str(value).strip()
    )

    if not path.is_absolute():
        path = ROOT / path

    return path.resolve()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def copy_review_images(
    dataframe: pd.DataFrame,
    maximum_per_category: int = 50,
) -> None:
    if REVIEW_DIR.exists():
        shutil.rmtree(
            REVIEW_DIR
        )

    REVIEW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    categories = {
        "false_accept": "ACCEPT",
        "uncertain": "UNCERTAIN",
    }

    for folder_name, decision in categories.items():
        subset = dataframe[
            dataframe[
                "decision"
            ].eq(
                decision
            )
        ].copy()

        subset = subset.sort_values(
            [
                "supported_probability",
                "prototype_similarity",
            ],
            ascending=False,
        )

        destination_directory = (
            REVIEW_DIR
            / folder_name
        )

        destination_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        for index, row in enumerate(
            subset.head(
                maximum_per_category
            ).to_dict(
                orient="records"
            ),
            start=1,
        ):
            source = resolve_path(
                row["image_path"]
            )

            if not source.exists():
                continue

            destination_name = (
                f"{index:03d}"
                f"__{row['source_dataset']}"
                f"__{row['source_group']}"
                f"__p_{row['supported_probability']:.6f}"
                f"__s_{row['prototype_similarity']:.6f}"
                f"__{source.name}"
            )

            shutil.copy2(
                source,
                destination_directory
                / destination_name,
            )


def create_group_metrics(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for (
        source_dataset,
        source_group,
    ), subset in dataframe.groupby(
        [
            "source_dataset",
            "source_group",
        ],
        dropna=False,
    ):
        accept_count = int(
            subset[
                "decision"
            ].eq(
                "ACCEPT"
            ).sum()
        )

        uncertain_count = int(
            subset[
                "decision"
            ].eq(
                "UNCERTAIN"
            ).sum()
        )

        reject_count = int(
            subset[
                "decision"
            ].eq(
                "REJECT"
            ).sum()
        )

        total = int(
            len(subset)
        )

        rows.append(
            {
                "source_dataset": str(
                    source_dataset
                ),
                "source_group": str(
                    source_group
                ),
                "sample_count": total,
                "accept_count": accept_count,
                "uncertain_count": uncertain_count,
                "reject_count": reject_count,
                "false_accept_rate": float(
                    accept_count / total
                ),
                "safe_block_rate": float(
                    (
                        uncertain_count
                        + reject_count
                    )
                    / total
                ),
                "mean_supported_probability": float(
                    subset[
                        "supported_probability"
                    ].mean()
                ),
                "maximum_supported_probability": float(
                    subset[
                        "supported_probability"
                    ].max()
                ),
                "mean_prototype_similarity": float(
                    subset[
                        "prototype_similarity"
                    ].mean()
                ),
                "maximum_prototype_similarity": float(
                    subset[
                        "prototype_similarity"
                    ].max()
                ),
            }
        )

    return pd.DataFrame(
        rows
    ).sort_values(
        [
            "false_accept_rate",
            "accept_count",
            "maximum_supported_probability",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    ).reset_index(
        drop=True
    )


def write_report(
    summary: dict[str, Any],
    group_metrics: pd.DataFrame,
) -> None:
    lines = [
        "# v0.5 Input-Gate Yeni Negatif Kilitli Test",
        "",
        "## Protokol",
        "",
        "- Model ağırlıkları donduruldu.",
        "- Temperature değeri donduruldu.",
        "- KABUL, BELİRSİZ ve RED eşikleri donduruldu.",
        "- Kilitli test sonuçlarına göre eşik değiştirilmedi.",
        "- Kilitli görüntüler eğitim ve calibration sırasında kullanılmadı.",
        "",
        "## Genel sonuç",
        "",
        f"- Toplam negatif görüntü: "
        f"{summary['total_negative_samples']}",
        f"- Yanlış kabul: "
        f"{summary['false_accept_count']}",
        f"- Belirsiz: "
        f"{summary['uncertain_count']}",
        f"- Red: "
        f"{summary['reject_count']}",
        f"- Yanlış kabul oranı: "
        f"{summary['false_accept_rate']:.6f}",
        f"- Güvenli engelleme oranı: "
        f"{summary['safe_block_rate']:.6f}",
        f"- Güvenlik testi: "
        f"{summary['safety_test_result']}",
        "",
        "## Kaynak ve grup sonuçları",
        "",
        "| Veri | Grup | Örnek | Kabul | Belirsiz | Red | "
        "Yanlış kabul oranı |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]

    for row in group_metrics.to_dict(
        orient="records"
    ):
        lines.append(
            f"| {row['source_dataset']} "
            f"| {row['source_group']} "
            f"| {row['sample_count']} "
            f"| {row['accept_count']} "
            f"| {row['uncertain_count']} "
            f"| {row['reject_count']} "
            f"| {row['false_accept_rate']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Sınırlama",
            "",
            "Bu sürüm için yeni ve bağımsız pozitif deniz-SAR "
            "test kaynağı bulunmamaktadır. Bu nedenle bu test "
            "yalnız desteklenmeyen girişlerin güvenli biçimde "
            "engellenmesini ölçmektedir.",
            "",
            "Calibration skorlarının 0 veya 1'e çok yakın olması "
            "skorların gerçek dünya olasılığı olduğu anlamına gelmez.",
            "",
        ]
    )

    REPORT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    for required_path in (
        CALIBRATION_SCRIPT,
        MANIFEST_PATH,
        CHECKPOINT_PATH,
        CONFIG_PATH,
        PROTOTYPES_PATH,
    ):
        if not required_path.exists():
            raise FileNotFoundError(
                f"Gerekli dosya bulunamadı: "
                f"{required_path}"
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    base = load_calibration_module()

    dataframe = pd.read_csv(
        MANIFEST_PATH,
        encoding="utf-8-sig",
        low_memory=False,
    )

    locked = dataframe[
        dataframe[
            "split"
        ].eq(
            "test_negative_locked"
        )
    ].copy()

    if locked.empty:
        raise RuntimeError(
            "test_negative_locked split boş."
        )

    labels = set(
        locked[
            "binary_label"
        ].astype(int).unique()
    )

    if labels != {0}:
        raise RuntimeError(
            "Yeni negatif kilitli test yalnız "
            "binary_label=0 içermelidir. "
            f"Bulunan etiketler: {sorted(labels)}"
        )

    missing_files = []

    for value in locked[
        "image_path"
    ]:
        path = resolve_path(
            value
        )

        if not path.exists():
            missing_files.append(
                str(path)
            )

    if missing_files:
        raise FileNotFoundError(
            f"{len(missing_files)} görüntü bulunamadı.\n"
            + "\n".join(
                missing_files[:20]
            )
        )

    config = read_json(
        CONFIG_PATH
    )

    device = base.select_device(
        "auto"
    )

    model, checkpoint_payload = (
        base.build_model(
            device
        )
    )

    prototype_data = np.load(
        PROTOTYPES_PATH,
        allow_pickle=False,
    )

    prototype_names = prototype_data[
        "group_names"
    ].astype(str)

    prototypes = prototype_data[
        "centroids"
    ].astype(
        np.float32
    )

    mean = tuple(
        checkpoint_payload.get(
            "normalization_mean",
            [0.5],
        )
    )

    std = tuple(
        checkpoint_payload.get(
            "normalization_std",
            [0.25],
        )
    )

    input_size = int(
        checkpoint_payload.get(
            "input_size",
            224,
        )
    )

    transform = transforms.Compose(
        [
            transforms.Resize(
                (
                    input_size,
                    input_size,
                ),
                interpolation=(
                    InterpolationMode.BILINEAR
                ),
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=mean,
                std=std,
            ),
        ]
    )

    dataset = base.GateDataset(
        locked,
        transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    print("=" * 78)
    print(
        "ROBUST BINARY OIL DETECTOR v0.5"
    )
    print(
        "YENİ NEGATİF KİLİTLİ INPUT-GATE TESTİ"
    )
    print("=" * 78)

    print(
        "Cihaz:",
        device,
    )

    print(
        "Toplam negatif:",
        len(dataset),
    )

    print(
        "Eşikler: DONDURULDU"
    )

    print(
        "Model yeniden eğitilmiyor."
    )

    print()

    (
        logits,
        features,
        labels_array,
        sample_ids,
        source_groups,
    ) = base.extract_outputs(
        model,
        loader,
        device,
    )

    temperature = float(
        config["temperature"]
    )

    probabilities = base.sigmoid(
        logits
        / temperature
    )

    (
        similarities,
        nearest_indices,
    ) = base.maximum_similarity(
        features,
        prototypes,
    )

    decisions = base.assign_decisions(
        probabilities,
        similarities,
        float(
            config[
                "accept_probability_threshold"
            ]
        ),
        float(
            config[
                "accept_similarity_threshold"
            ]
        ),
        float(
            config[
                "reject_probability_threshold"
            ]
        ),
        float(
            config[
                "reject_similarity_threshold"
            ]
        ),
    )

    predictions = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "binary_label": labels_array,
            "raw_logit": logits,
            "supported_probability": (
                probabilities
            ),
            "prototype_similarity": (
                similarities
            ),
            "nearest_prototype": [
                prototype_names[index]
                for index in nearest_indices
            ],
            "decision": decisions,
            "pipeline_allowed": (
                decisions == "ACCEPT"
            ),
        }
    )

    metadata = (
        locked[
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
            subset=["sample_id"]
        )
    )

    predictions = predictions.merge(
        metadata,
        on="sample_id",
        how="left",
        validate="one_to_one",
    )

    predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    group_metrics = create_group_metrics(
        predictions
    )

    group_metrics.to_csv(
        GROUP_METRICS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    copy_review_images(
        predictions
    )

    false_accept_count = int(
        predictions[
            "decision"
        ].eq(
            "ACCEPT"
        ).sum()
    )

    uncertain_count = int(
        predictions[
            "decision"
        ].eq(
            "UNCERTAIN"
        ).sum()
    )

    reject_count = int(
        predictions[
            "decision"
        ].eq(
            "REJECT"
        ).sum()
    )

    total = int(
        len(predictions)
    )

    summary = {
        "model_version": (
            "input_gate_v05"
        ),
        "test_type": (
            "new_negative_locked_test"
        ),
        "total_negative_samples": total,
        "false_accept_count": (
            false_accept_count
        ),
        "uncertain_count": (
            uncertain_count
        ),
        "reject_count": (
            reject_count
        ),
        "false_accept_rate": float(
            false_accept_count / total
        ),
        "safe_block_rate": float(
            (
                uncertain_count
                + reject_count
            )
            / total
        ),
        "safety_test_passed": bool(
            false_accept_count == 0
        ),
        "safety_test_result": (
            "BAŞARILI"
            if false_accept_count == 0
            else "BAŞARISIZ"
        ),
        "configuration_frozen": True,
        "locked_test_consumed": True,
        "positive_locked_test_available": False,
        "temperature": temperature,
        "accept_probability_threshold": float(
            config[
                "accept_probability_threshold"
            ]
        ),
        "accept_similarity_threshold": float(
            config[
                "accept_similarity_threshold"
            ]
        ),
        "reject_probability_threshold": float(
            config[
                "reject_probability_threshold"
            ]
        ),
        "reject_similarity_threshold": float(
            config[
                "reject_similarity_threshold"
            ]
        ),
        "predictions_path": str(
            PREDICTIONS_PATH.resolve()
        ),
        "group_metrics_path": str(
            GROUP_METRICS_PATH.resolve()
        ),
        "review_images_path": str(
            REVIEW_DIR.resolve()
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
        summary,
        group_metrics,
    )

    print("=" * 78)
    print(
        "v0.5 YENİ NEGATİF KİLİTLİ TEST SONUCU"
    )
    print("=" * 78)

    print(
        "Toplam negatif:",
        total,
    )

    print(
        "Yanlış ACCEPT:",
        false_accept_count,
    )

    print(
        "UNCERTAIN:",
        uncertain_count,
    )

    print(
        "REJECT:",
        reject_count,
    )

    print(
        "Yanlış kabul oranı:",
        f"{false_accept_count / total:.6f}",
    )

    print(
        "Güvenli engelleme:",
        f"{(uncertain_count + reject_count) / total:.6f}",
    )

    print(
        "Güvenlik testi:",
        (
            "BAŞARILI"
            if false_accept_count == 0
            else "BAŞARISIZ"
        ),
    )

    print()
    print(
        "Grup sonuçları:"
    )

    print(
        group_metrics[
            [
                "source_dataset",
                "source_group",
                "sample_count",
                "accept_count",
                "uncertain_count",
                "reject_count",
                "false_accept_rate",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Tahminler:",
        PREDICTIONS_PATH.resolve(),
    )

    print(
        "Grup metrikleri:",
        GROUP_METRICS_PATH.resolve(),
    )

    print(
        "İnceleme görüntüleri:",
        REVIEW_DIR.resolve(),
    )

    print(
        "Özet:",
        SUMMARY_PATH.resolve(),
    )

    print(
        "Rapor:",
        REPORT_PATH.resolve(),
    )

    print()
    print(
        "ÖNEMLİ: Bu yeni negatif kilitli test "
        "artık tüketilmiştir."
    )


if __name__ == "__main__":
    main()
