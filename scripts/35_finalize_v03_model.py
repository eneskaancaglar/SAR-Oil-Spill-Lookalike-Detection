from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

EVALUATION_SUMMARY = (
    ROOT
    / "outputs"
    / "v03_end_to_end_locked_evaluation"
    / "evaluation_summary.json"
)

POSITIVE_RESULTS = (
    ROOT
    / "outputs"
    / "v03_end_to_end_locked_evaluation"
    / "positive_locked_results.csv"
)

NEGATIVE_RESULTS = (
    ROOT
    / "outputs"
    / "v03_end_to_end_locked_evaluation"
    / "negative_external_results.csv"
)

CALIBRATION_SUMMARY = (
    ROOT
    / "outputs"
    / "verifier_v03"
    / "calibration"
    / "calibration_summary.json"
)

THRESHOLD_SUMMARY = (
    ROOT
    / "outputs"
    / "verifier_v03"
    / "threshold_sweep"
    / "threshold_summary.json"
)

SOURCE_SEGMENTATION_MODEL = (
    ROOT
    / "checkpoints"
    / "final_model"
    / "oil_spill_unet_t060.pth"
)

SOURCE_VERIFIER_MODEL = (
    ROOT
    / "checkpoints"
    / "verifier_v03"
    / "best.pth"
)

SOURCE_CALIBRATION_CONFIG = (
    ROOT
    / "checkpoints"
    / "verifier_v03"
    / "calibration_config.json"
)

FINAL_MODEL_DIR = (
    ROOT
    / "checkpoints"
    / "final_model_v03"
)

FINAL_SEGMENTATION_MODEL = (
    FINAL_MODEL_DIR
    / "oil_segmentation_unet.pth"
)

FINAL_VERIFIER_MODEL = (
    FINAL_MODEL_DIR
    / "oil_candidate_verifier_resnet18.pth"
)

FINAL_CALIBRATION_CONFIG = (
    FINAL_MODEL_DIR
    / "calibration_config.json"
)

FINAL_DETECTOR_CONFIG = (
    FINAL_MODEL_DIR
    / "detector_config.json"
)

FINAL_MODEL_CARD = (
    FINAL_MODEL_DIR
    / "model_card.json"
)

FINAL_MANIFEST = (
    FINAL_MODEL_DIR
    / "package_manifest.json"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "v03_finalization"
)

FINAL_SUMMARY = (
    OUTPUT_DIR
    / "final_summary.json"
)

FINAL_REPORT = (
    ROOT
    / "reports"
    / "v03_final_model_report.md"
)


def read_json(
    path: Path,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Gerekli JSON bulunamadı: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def require_file(
    path: Path,
) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Gerekli dosya bulunamadı: {path}"
        )

    if not path.is_file():
        raise RuntimeError(
            f"Beklenen yol dosya değil: {path}"
        )


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def copy_file(
    source: Path,
    destination: Path,
) -> None:
    require_file(source)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = destination.with_suffix(
        destination.suffix + ".part"
    )

    shutil.copy2(
        source,
        temporary,
    )

    temporary.replace(
        destination
    )


def relative_path(
    path: Path,
) -> str:
    try:
        return str(
            path.resolve().relative_to(
                ROOT.resolve()
            )
        )

    except ValueError:
        return str(path.resolve())


def percent(
    value: float,
    digits: int = 2,
) -> str:
    return (
        f"%{float(value):.{digits}f}"
    )


def read_optional_csv(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(
        path,
        encoding="utf-8-sig",
        low_memory=False,
    )


def calculate_additional_statistics(
    positive_results: pd.DataFrame,
    negative_results: pd.DataFrame,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "positive": {},
        "negative": {},
    }

    if not positive_results.empty:
        detected = (
            positive_results[
                "oil_detected"
            ]
            .astype(bool)
        )

        result["positive"] = {
            "images": int(
                len(positive_results)
            ),
            "mean_final_coverage_percent": float(
                positive_results[
                    "final_coverage_ratio"
                ].mean()
                * 100.0
            ),
            "median_final_coverage_percent": float(
                positive_results[
                    "final_coverage_ratio"
                ].median()
                * 100.0
            ),
            "mean_maximum_candidate_probability": float(
                positive_results[
                    "maximum_candidate_probability"
                ].mean()
            ),
            "missed_image_names": (
                positive_results.loc[
                    ~detected,
                    "image_name",
                ]
                .astype(str)
                .head(30)
                .tolist()
            ),
        }

    if not negative_results.empty:
        false_alarm = (
            negative_results[
                "oil_detected"
            ]
            .astype(bool)
        )

        result["negative"] = {
            "images": int(
                len(negative_results)
            ),
            "false_alarm_images": int(
                false_alarm.sum()
            ),
            "clean_images": int(
                (
                    ~false_alarm
                ).sum()
            ),
            "mean_final_coverage_percent": float(
                negative_results[
                    "final_coverage_ratio"
                ].mean()
                * 100.0
            ),
            "median_final_coverage_percent": float(
                negative_results[
                    "final_coverage_ratio"
                ].median()
                * 100.0
            ),
            "worst_false_alarm_images": (
                negative_results
                .sort_values(
                    "final_coverage_ratio",
                    ascending=False,
                )
                [
                    [
                        "image_set",
                        "image_name",
                        "final_coverage_ratio",
                    ]
                ]
                .head(20)
                .assign(
                    final_coverage_percent=(
                        lambda frame:
                        frame[
                            "final_coverage_ratio"
                        ]
                        * 100.0
                    )
                )
                [
                    [
                        "image_set",
                        "image_name",
                        "final_coverage_percent",
                    ]
                ]
                .to_dict(
                    orient="records"
                )
            ),
        }

    return result


def create_detector_config(
    evaluation: dict[str, Any],
    calibration: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_version": (
            "robust_binary_oil_detector_v03"
        ),
        "task": (
            "binary_oil_spill_segmentation"
        ),
        "positive_class": "oil",
        "negative_class": "non_oil",
        "segmenter": {
            "architecture": "unet",
            "checkpoint": relative_path(
                FINAL_SEGMENTATION_MODEL
            ),
            "input_channels": 1,
            "segmentation_threshold": float(
                evaluation[
                    "segmentation_threshold"
                ]
            ),
        },
        "candidate_extraction": {
            "connectivity": 8,
            "minimum_component_ratio": (
                0.0005
            ),
            "minimum_component_pixels": 16,
            "maximum_components": 50,
            "context_ratio": 0.25,
            "minimum_crop_size": 64,
        },
        "verifier": {
            "architecture": (
                "resnet18_grayscale_binary"
            ),
            "checkpoint": relative_path(
                FINAL_VERIFIER_MODEL
            ),
            "input_channels": 1,
            "input_size": 224,
            "normalization_mean": [
                0.449
            ],
            "normalization_std": [
                0.226
            ],
            "probability_mode": (
                evaluation[
                    "verifier_probability_mode"
                ]
            ),
            "temperature": float(
                evaluation[
                    "verifier_temperature"
                ]
            ),
            "raw_threshold": float(
                evaluation[
                    "verifier_raw_threshold"
                ]
            ),
            "calibrated_threshold": float(
                evaluation[
                    "verifier_calibrated_threshold"
                ]
            ),
        },
        "output": {
            "oil_detected": (
                "Final mask contains at least "
                "one accepted component."
            ),
            "coverage_percent": (
                "Final oil pixels divided by "
                "all image pixels."
            ),
            "confidence": (
                "Maximum accepted candidate "
                "verifier probability."
            ),
        },
        "calibration": {
            "method": (
                "temperature_scaling"
            ),
            "calibration_samples": int(
                calibration[
                    "calibration_samples"
                ]
            ),
            "test_data_used": False,
        },
        "configuration_frozen": True,
        "locked_test_consumed": True,
    }


def create_model_card(
    evaluation: dict[str, Any],
    calibration: dict[str, Any],
    additional: dict[str, Any],
) -> dict[str, Any]:
    positive = evaluation[
        "positive_locked"
    ]

    negative = evaluation[
        "negative_external"
    ]

    return {
        "model_name": (
            "Robust Binary Oil Detector v0.3"
        ),
        "version": "v0.3",
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "task": (
            "SAR görüntülerinde petrol / "
            "petrol değil ayrımı ve petrol maskesi"
        ),
        "pipeline": [
            (
                "U-Net ile petrol aday "
                "segmentasyonu"
            ),
            (
                "Bağlantılı bileşen analizi"
            ),
            (
                "ResNet18 ile aday petrol "
                "doğrulaması"
            ),
            (
                "Verifier tarafından reddedilen "
                "bölgelerin maskeden silinmesi"
            ),
        ],
        "locked_test_metrics": {
            "positive_image_count": int(
                positive["images"]
            ),
            "positive_image_recall": float(
                positive["image_recall"]
            ),
            "positive_box_recall": float(
                positive["box_recall"]
            ),
            "negative_image_count": int(
                negative["overall"]["images"]
            ),
            "negative_any_alarm_percent": float(
                negative[
                    "overall"
                ][
                    "any_alarm_percent"
                ]
            ),
            "negative_severe_alarm_percent": float(
                negative[
                    "overall"
                ][
                    "alarm_ge_1_percent"
                ]
            ),
            "negative_fp_pixel_percent": float(
                negative[
                    "overall"
                ][
                    "aggregate_fp_pixel_percent"
                ]
            ),
            "nc_severe_alarm_percent": float(
                negative[
                    "nc"
                ][
                    "alarm_ge_1_percent"
                ]
            ),
            "nw_severe_alarm_percent": float(
                negative[
                    "nw"
                ][
                    "alarm_ge_1_percent"
                ]
            ),
        },
        "calibration_metrics": {
            "temperature": float(
                calibration["temperature"]
            ),
            "raw_threshold": float(
                calibration["raw_threshold"]
            ),
            "calibrated_threshold": float(
                calibration[
                    "calibrated_threshold"
                ]
            ),
            "nll_before": float(
                calibration[
                    "before_calibration"
                ]["nll"]
            ),
            "nll_after": float(
                calibration[
                    "after_calibration"
                ]["nll"]
            ),
            "brier_before": float(
                calibration[
                    "before_calibration"
                ]["brier"]
            ),
            "brier_after": float(
                calibration[
                    "after_calibration"
                ]["brier"]
            ),
            "ece_before": float(
                calibration[
                    "before_calibration"
                ]["ece"]
            ),
            "ece_after": float(
                calibration[
                    "after_calibration"
                ]["ece"]
            ),
        },
        "additional_statistics": (
            additional
        ),
        "intended_use": [
            (
                "SAR görüntülerindeki olası "
                "petrol tabakalarını göstermek"
            ),
            (
                "Araştırma ve staj sunumu "
                "amaçlı karar destek prototipi"
            ),
            (
                "Petrol maskesi ve görüntü "
                "kaplama oranı üretmek"
            ),
        ],
        "not_intended_for": [
            (
                "Tek başına kesin çevresel "
                "olay doğrulaması"
            ),
            (
                "Uzman incelemesi olmadan "
                "operasyonel alarm kararı"
            ),
            (
                "Petrol miktarını ton veya "
                "litre olarak hesaplama"
            ),
        ],
        "limitations": [
            (
                "Bağımsız kara-deniz maskesi "
                "bulunmamaktadır."
            ),
            (
                "Kaplama yüzdesi bütün görüntü "
                "alanına göre hesaplanmaktadır."
            ),
            (
                "DARTIS pozitif anotasyonları "
                "piksel maskesi değil bounding box'tır."
            ),
            (
                "Verifier güven skoru aday bölge "
                "seviyesindedir; bütün görüntünün "
                "sertifikalı petrol olasılığı değildir."
            ),
            (
                "Kilitli test artık kullanılmıştır; "
                "sonuçlara bakarak mevcut model "
                "yeniden ayarlanmamalıdır."
            ),
        ],
    }


def write_report(
    model_card: dict[str, Any],
    package_entries: list[
        dict[str, Any]
    ],
) -> None:
    metrics = model_card[
        "locked_test_metrics"
    ]

    calibration = model_card[
        "calibration_metrics"
    ]

    lines = [
        "# Robust Binary Oil Detector v0.3",
        "# Final Model Raporu",
        "",
        "## Sistem ne yapıyor?",
        "",
        "Sistem bir gri seviye SAR görüntüsünü alır, "
        "petrol olabilecek bölgeleri U-Net ile maskeler "
        "ve bu bölgeleri ResNet18 verifier ile "
        "`petrol / petrol değil` şeklinde doğrular.",
        "",
        "Nihai çıktı:",
        "",
        "- Petrol var / yok kararı",
        "- Nihai petrol maskesi",
        "- Görüntü üzerindeki petrol kaplama yüzdesi",
        "- Aday bölge güven skoru",
        "- Görsel overlay",
        "",
        "## Kilitli test sonuçları",
        "",
        f"- Kilitli petrollü görüntü: "
        f"{metrics['positive_image_count']}",
        f"- Pozitif görüntü recall: "
        f"{metrics['positive_image_recall']:.4f}",
        f"- XML kutu recall: "
        f"{metrics['positive_box_recall']:.4f}",
        f"- Petrolsüz external-test görüntüsü: "
        f"{metrics['negative_image_count']}",
        f"- Herhangi yanlış alarm: "
        f"{percent(metrics['negative_any_alarm_percent'])}",
        f"- Ciddi yanlış alarm (≥%1): "
        f"{percent(metrics['negative_severe_alarm_percent'])}",
        f"- Yanlış pozitif piksel: "
        f"{percent(metrics['negative_fp_pixel_percent'], 4)}",
        f"- Kıyı `nc` ciddi alarm: "
        f"{percent(metrics['nc_severe_alarm_percent'])}",
        f"- Açık deniz `nw` ciddi alarm: "
        f"{percent(metrics['nw_severe_alarm_percent'])}",
        "",
        "## Verifier kalibrasyonu",
        "",
        f"- Temperature: "
        f"{calibration['temperature']:.6f}",
        f"- Ham threshold: "
        f"{calibration['raw_threshold']:.6f}",
        f"- Kalibre threshold: "
        f"{calibration['calibrated_threshold']:.6f}",
        f"- NLL: "
        f"{calibration['nll_before']:.6f} → "
        f"{calibration['nll_after']:.6f}",
        f"- Brier: "
        f"{calibration['brier_before']:.6f} → "
        f"{calibration['brier_after']:.6f}",
        f"- ECE: "
        f"{calibration['ece_before']:.6f} → "
        f"{calibration['ece_after']:.6f}",
        "",
        "ECE küçük ölçüde yükseldiği için verifier "
        "çıktısı kesin başarı olasılığı değil, "
        "kalibre edilmiş aday-petrol güven skoru "
        "olarak sunulmalıdır.",
        "",
        "## Final paket dosyaları",
        "",
        "| Dosya | Boyut | SHA-256 |",
        "|---|---:|---|",
    ]

    for entry in package_entries:
        lines.append(
            f"| `{entry['relative_path']}` "
            f"| {entry['size_bytes']} bayt "
            f"| `{entry['sha256']}` |"
        )

    lines.extend(
        [
            "",
            "## Kullanım sınırları",
            "",
            "- Model araştırma prototipidir.",
            "- Kara-deniz maskesi bulunmamaktadır.",
            "- Kaplama yüzdesi bütün görüntüye göredir.",
            "- Güven skoru aday bölge seviyesindedir.",
            "- Kıyı görüntüleri açık denizden daha zordur.",
            "- Kilitli test sonuçlarına bakılarak mevcut "
            "model yeniden ayarlanmamalıdır.",
            "",
            "## Bilimsel durum",
            "",
            "Bu model ve eşikler dondurulmuştur. Yeni bir "
            "iyileştirme yapılacaksa yeni geliştirme verisi, "
            "yeni validation/calibration bölümü ve yeni bir "
            "kilitli test seti oluşturulmalıdır.",
            "",
        ]
    )

    FINAL_REPORT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    required_files = [
        EVALUATION_SUMMARY,
        POSITIVE_RESULTS,
        NEGATIVE_RESULTS,
        CALIBRATION_SUMMARY,
        THRESHOLD_SUMMARY,
        SOURCE_SEGMENTATION_MODEL,
        SOURCE_VERIFIER_MODEL,
        SOURCE_CALIBRATION_CONFIG,
    ]

    for path in required_files:
        require_file(path)

    FINAL_MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FINAL_REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    evaluation = read_json(
        EVALUATION_SUMMARY
    )

    calibration = read_json(
        CALIBRATION_SUMMARY
    )

    threshold = read_json(
        THRESHOLD_SUMMARY
    )

    if not evaluation.get(
        "model_configuration_frozen",
        False,
    ):
        raise RuntimeError(
            "Evaluation summary modelin dondurulduğunu "
            "belirtmiyor."
        )

    if not evaluation.get(
        "test_locked_used",
        False,
    ):
        raise RuntimeError(
            "Kilitli pozitif test tamamlanmamış görünüyor."
        )

    if not evaluation.get(
        "negative_external_test_used",
        False,
    ):
        raise RuntimeError(
            "Negatif external test tamamlanmamış görünüyor."
        )

    positive_results = read_optional_csv(
        POSITIVE_RESULTS
    )

    negative_results = read_optional_csv(
        NEGATIVE_RESULTS
    )

    if positive_results.empty:
        raise RuntimeError(
            "Pozitif kilitli sonuç CSV'si boş."
        )

    if negative_results.empty:
        raise RuntimeError(
            "Negatif external sonuç CSV'si boş."
        )

    print("=" * 78)
    print(
        "ROBUST BINARY OIL DETECTOR v0.3"
    )
    print("FINAL MODEL PAKETLEME")
    print("=" * 78)

    print(
        "Pozitif kilitli sonuç:",
        len(positive_results),
    )

    print(
        "Negatif external sonuç:",
        len(negative_results),
    )

    copy_file(
        SOURCE_SEGMENTATION_MODEL,
        FINAL_SEGMENTATION_MODEL,
    )

    copy_file(
        SOURCE_VERIFIER_MODEL,
        FINAL_VERIFIER_MODEL,
    )

    copy_file(
        SOURCE_CALIBRATION_CONFIG,
        FINAL_CALIBRATION_CONFIG,
    )

    additional_statistics = (
        calculate_additional_statistics(
            positive_results,
            negative_results,
        )
    )

    detector_config = (
        create_detector_config(
            evaluation,
            calibration,
        )
    )

    with FINAL_DETECTOR_CONFIG.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            detector_config,
            file,
            indent=2,
            ensure_ascii=False,
        )

    model_card = create_model_card(
        evaluation,
        calibration,
        additional_statistics,
    )

    with FINAL_MODEL_CARD.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            model_card,
            file,
            indent=2,
            ensure_ascii=False,
        )

    package_files = [
        FINAL_SEGMENTATION_MODEL,
        FINAL_VERIFIER_MODEL,
        FINAL_CALIBRATION_CONFIG,
        FINAL_DETECTOR_CONFIG,
        FINAL_MODEL_CARD,
    ]

    package_entries = []

    for path in package_files:
        package_entries.append(
            {
                "filename": path.name,
                "relative_path": (
                    relative_path(path)
                ),
                "size_bytes": int(
                    path.stat().st_size
                ),
                "sha256": sha256_file(
                    path
                ),
            }
        )

    package_manifest = {
        "package_name": (
            "robust_binary_oil_detector_v03"
        ),
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "configuration_frozen": True,
        "locked_test_consumed": True,
        "files": package_entries,
    }

    with FINAL_MANIFEST.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            package_manifest,
            file,
            indent=2,
            ensure_ascii=False,
        )

    final_summary = {
        "model_version": "v0.3",
        "status": "finalized",
        "configuration_frozen": True,
        "locked_test_consumed": True,
        "evaluation": evaluation,
        "calibration": calibration,
        "threshold_selection": threshold,
        "additional_statistics": (
            additional_statistics
        ),
        "final_model_directory": (
            relative_path(
                FINAL_MODEL_DIR
            )
        ),
        "final_report": (
            relative_path(
                FINAL_REPORT
            )
        ),
        "package_manifest": (
            relative_path(
                FINAL_MANIFEST
            )
        ),
    }

    with FINAL_SUMMARY.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            final_summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    write_report(
        model_card,
        package_entries,
    )

    print()
    print("=" * 78)
    print("FINAL PAKET OLUŞTURULDU")
    print("=" * 78)

    metrics = model_card[
        "locked_test_metrics"
    ]

    print(
        "Pozitif görüntü recall:",
        f"{metrics['positive_image_recall']:.4f}",
    )

    print(
        "Kutu recall:",
        f"{metrics['positive_box_recall']:.4f}",
    )

    print(
        "Ciddi yanlış alarm >=%1:",
        percent(
            metrics[
                "negative_severe_alarm_percent"
            ]
        ),
    )

    print(
        "FP piksel:",
        percent(
            metrics[
                "negative_fp_pixel_percent"
            ],
            4,
        ),
    )

    print()
    print(
        "Final model klasörü:",
        FINAL_MODEL_DIR.resolve(),
    )

    print(
        "Model kartı:",
        FINAL_MODEL_CARD.resolve(),
    )

    print(
        "Detector config:",
        FINAL_DETECTOR_CONFIG.resolve(),
    )

    print(
        "Paket manifesti:",
        FINAL_MANIFEST.resolve(),
    )

    print(
        "Final rapor:",
        FINAL_REPORT.resolve(),
    )

    print()
    print(
        "Model ve threshold donduruldu."
    )

    print(
        "Bir sonraki aşama: sunum için "
        "dosya yüklemeli demo arayüzü."
    )


if __name__ == "__main__":
    main()
