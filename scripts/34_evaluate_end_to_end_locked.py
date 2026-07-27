from __future__ import annotations

import argparse
import json
import math
import runpy
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

DETECTOR_SCRIPT = (
    ROOT
    / "scripts"
    / "33_run_end_to_end_oil_detector.py"
)

VERIFIER_MANIFEST = (
    ROOT
    / "data"
    / "metadata"
    / "v03_verifier_dataset.csv"
)

POSITIVE_CROP_MANIFEST = (
    ROOT
    / "data"
    / "metadata"
    / "dartis_positive_verifier_crops.csv"
)

NEGATIVE_SCENE_MANIFEST = (
    ROOT
    / "data"
    / "metadata"
    / "dartis_no_oil_scene_split.csv"
)

DARTIS_RAW_ROOT = (
    ROOT
    / "data"
    / "external"
    / "dartis"
    / "raw"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "v03_end_to_end_locked_evaluation"
)

POSITIVE_RESULTS_PATH = (
    OUTPUT_DIR
    / "positive_locked_results.csv"
)

NEGATIVE_RESULTS_PATH = (
    OUTPUT_DIR
    / "negative_external_results.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "evaluation_summary.json"
)

ERROR_EXAMPLES_DIR = (
    OUTPUT_DIR
    / "error_examples"
)

REPORT_PATH = (
    ROOT
    / "reports"
    / "v03_end_to_end_locked_evaluation.md"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Kilitli pozitif DARTIS sahneleri ve dokunulmamış "
            "negatif external-test sahneleri üzerinde uçtan uca "
            "petrol tespit sistemini değerlendirir."
        )
    )

    parser.add_argument(
        "--positive-limit",
        type=int,
        default=0,
        help="0 bütün kilitli pozitif görüntüleri işler.",
    )

    parser.add_argument(
        "--negative-limit",
        type=int,
        default=0,
        help="0 bütün external-test negatiflerini işler.",
    )

    parser.add_argument(
        "--segmentation-threshold",
        type=float,
        default=0.60,
    )

    parser.add_argument(
        "--minimum-component-ratio",
        type=float,
        default=0.0005,
    )

    parser.add_argument(
        "--minimum-component-pixels",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--context",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--minimum-crop-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--maximum-components",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--error-examples",
        type=int,
        default=12,
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


def clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if text.lower() in {
        "",
        "nan",
        "none",
        "<na>",
    }:
        return ""

    return text


def normalize_identifier(value: Any) -> str:
    text = clean_text(value)

    if not text:
        return ""

    try:
        number = float(text)

        if number.is_integer():
            return str(
                int(number)
            )

    except ValueError:
        pass

    return text


def resolve_path(value: Any) -> Path:
    text = clean_text(value)

    if not text:
        return Path()

    path = Path(text)

    if not path.is_absolute():
        path = ROOT / path

    return path


def find_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
    required: bool = True,
) -> str | None:
    column_map = {
        str(column).strip().lower(): str(column)
        for column in dataframe.columns
    }

    for candidate in candidates:
        normalized = candidate.strip().lower()

        if normalized in column_map:
            return column_map[
                normalized
            ]

    if required:
        raise RuntimeError(
            f"Sütun bulunamadı: {candidates}\n"
            f"Mevcut sütunlar: {list(dataframe.columns)}"
        )

    return None


def load_detector_api() -> dict[str, Any]:
    if not DETECTOR_SCRIPT.exists():
        raise FileNotFoundError(
            f"Script 33 bulunamadı: {DETECTOR_SCRIPT}"
        )

    return runpy.run_path(
        str(DETECTOR_SCRIPT),
        run_name="end_to_end_detector_module",
    )


def find_dartis_image(
    group_name: str,
    image_name: str,
    explicit_path: Any = "",
) -> Path | None:
    explicit = resolve_path(
        explicit_path
    )

    if (
        str(explicit)
        and explicit.exists()
        and explicit.is_file()
    ):
        return explicit

    direct = (
        DARTIS_RAW_ROOT
        / group_name
        / image_name
    )

    if direct.exists():
        return direct

    directory = (
        DARTIS_RAW_ROOT
        / group_name
    )

    if not directory.exists():
        return None

    stem = Path(
        image_name
    ).stem

    for suffix in (
        ".jpg",
        ".jpeg",
        ".png",
        ".tif",
        ".tiff",
    ):
        candidate = (
            directory
            / f"{stem}{suffix}"
        )

        if candidate.exists():
            return candidate

    return None


def read_boxes(
    xml_path: Path,
    image_width: int,
    image_height: int,
) -> list[
    tuple[int, int, int, int]
]:
    if not xml_path.exists():
        raise FileNotFoundError(
            f"XML bulunamadı: {xml_path}"
        )

    root = ET.parse(
        xml_path
    ).getroot()

    boxes = []

    for bbox in root.findall(
        ".//bndbox"
    ):
        values = {}

        for name in (
            "xmin",
            "ymin",
            "xmax",
            "ymax",
        ):
            element = bbox.find(name)

            if (
                element is None
                or element.text is None
            ):
                values = {}
                break

            values[name] = float(
                element.text
                .strip()
                .replace(",", ".")
            )

        if not values:
            continue

        left = int(
            math.floor(
                values["xmin"]
            )
        )

        top = int(
            math.floor(
                values["ymin"]
            )
        )

        if left >= 1:
            left -= 1

        if top >= 1:
            top -= 1

        right = int(
            math.ceil(
                values["xmax"]
            )
        )

        bottom = int(
            math.ceil(
                values["ymax"]
            )
        )

        left = max(
            0,
            min(
                left,
                image_width - 1,
            ),
        )

        top = max(
            0,
            min(
                top,
                image_height - 1,
            ),
        )

        right = max(
            left + 1,
            min(
                right,
                image_width,
            ),
        )

        bottom = max(
            top + 1,
            min(
                bottom,
                image_height,
            ),
        )

        boxes.append(
            (
                left,
                top,
                right,
                bottom,
            )
        )

    return boxes


def run_single_image(
    detector: dict[str, Any],
    segmentation_model,
    verifier_model,
    verifier_transform,
    verifier_config: dict[str, Any],
    image_path: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> dict[str, Any]:
    image, image_array = detector[
        "load_grayscale_image"
    ](
        image_path
    )

    image_width, image_height = (
        image.size
    )

    total_pixels = (
        image_width
        * image_height
    )

    probability = detector[
        "run_segmentation"
    ](
        segmentation_model,
        image_array,
        device,
    )

    raw_mask = (
        probability
        >= args.segmentation_threshold
    ).astype(
        np.uint8
    )

    components = detector[
        "connected_components"
    ](
        raw_mask
    )

    minimum_area = max(
        args.minimum_component_pixels,
        int(
            math.ceil(
                total_pixels
                * args.minimum_component_ratio
            )
        ),
    )

    eligible = [
        component
        for component in components
        if component[
            "area_pixels"
        ]
        >= minimum_area
    ][
        :args.maximum_components
    ]

    final_mask = np.zeros(
        raw_mask.shape,
        dtype=np.uint8,
    )

    candidate_results = []

    for component_index, component in enumerate(
        eligible,
        start=1,
    ):
        crop_box = detector[
            "expand_box"
        ](
            component["bbox"],
            image_width,
            image_height,
            args.context,
            args.minimum_crop_size,
        )

        crop = image.crop(
            crop_box
        )

        verifier_result = detector[
            "run_verifier"
        ](
            verifier_model,
            crop,
            verifier_transform,
            verifier_config,
            device,
        )

        accepted = bool(
            verifier_result[
                "accepted"
            ]
        )

        if accepted:
            for y, x in component[
                "pixels"
            ]:
                final_mask[
                    y,
                    x,
                ] = 1

        candidate_results.append(
            {
                "component_index": (
                    component_index
                ),
                "area_pixels": int(
                    component[
                        "area_pixels"
                    ]
                ),
                "bbox": tuple(
                    int(value)
                    for value
                    in component[
                        "bbox"
                    ]
                ),
                "crop_box": tuple(
                    int(value)
                    for value
                    in crop_box
                ),
                "accepted": accepted,
                "selected_probability": float(
                    verifier_result[
                        "selected_probability"
                    ]
                ),
                "selected_threshold": float(
                    verifier_result[
                        "selected_threshold"
                    ]
                ),
            }
        )

    raw_positive_pixels = int(
        raw_mask.sum()
    )

    final_positive_pixels = int(
        final_mask.sum()
    )

    maximum_probability = max(
        (
            candidate[
                "selected_probability"
            ]
            for candidate
            in candidate_results
        ),
        default=0.0,
    )

    return {
        "image": image,
        "probability": probability,
        "raw_mask": raw_mask,
        "final_mask": final_mask,
        "candidate_results": (
            candidate_results
        ),
        "image_width": image_width,
        "image_height": image_height,
        "total_pixels": total_pixels,
        "raw_positive_pixels": (
            raw_positive_pixels
        ),
        "final_positive_pixels": (
            final_positive_pixels
        ),
        "raw_coverage_ratio": float(
            raw_positive_pixels
            / total_pixels
        ),
        "final_coverage_ratio": float(
            final_positive_pixels
            / total_pixels
        ),
        "candidate_count": int(
            len(candidate_results)
        ),
        "accepted_candidate_count": int(
            sum(
                candidate[
                    "accepted"
                ]
                for candidate
                in candidate_results
            )
        ),
        "maximum_candidate_probability": float(
            maximum_probability
        ),
        "oil_detected": bool(
            final_positive_pixels > 0
        ),
    }


def save_error_example(
    detector: dict[str, Any],
    evaluation: dict[str, Any],
    destination_directory: Path,
    filename_stem: str,
) -> None:
    destination_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    detector[
        "create_overlay"
    ](
        evaluation["image"],
        evaluation["final_mask"],
    ).save(
        destination_directory
        / f"{filename_stem}__final_overlay.png"
    )

    detector[
        "create_candidate_overlay"
    ](
        evaluation["image"],
        evaluation[
            "candidate_results"
        ],
    ).save(
        destination_directory
        / f"{filename_stem}__candidates.png"
    )

    detector[
        "save_summary_figure"
    ](
        evaluation["image"],
        evaluation["probability"],
        evaluation["raw_mask"],
        evaluation["final_mask"],
        destination_directory
        / f"{filename_stem}__summary.png",
    )


def build_positive_test_table() -> pd.DataFrame:
    verifier = pd.read_csv(
        VERIFIER_MANIFEST,
        encoding="utf-8-sig",
        low_memory=False,
    )

    locked = verifier[
        verifier["split"].eq(
            "test_locked"
        )
        & verifier[
            "binary_label"
        ].eq(1)
    ].copy()

    if locked.empty:
        raise RuntimeError(
            "test_locked pozitif örnek bulunamadı."
        )

    locked[
        "normalized_scene_id"
    ] = locked[
        "scene_id"
    ].map(
        normalize_identifier
    )

    locked_scene_ids = set(
        locked[
            "normalized_scene_id"
        ]
    )

    locked_image_names = set(
        locked[
            "source_image_name"
        ].map(
            clean_text
        )
    )

    positive = pd.read_csv(
        POSITIVE_CROP_MANIFEST,
        encoding="utf-8-sig",
        low_memory=False,
    )

    positive[
        "normalized_scene_id"
    ] = positive[
        "scene_id"
    ].map(
        normalize_identifier
    )

    selected = positive[
        positive[
            "normalized_scene_id"
        ].isin(
            locked_scene_ids
        )
        | positive[
            "source_image_name"
        ].map(
            clean_text
        ).isin(
            locked_image_names
        )
    ].copy()

    selected = (
        selected
        .drop_duplicates(
            subset=[
                "image_set",
                "source_image_name",
            ]
        )
        .reset_index(drop=True)
    )

    if selected.empty:
        raise RuntimeError(
            "Kilitli pozitif görüntüler pozitif manifestle "
            "eşleştirilemedi."
        )

    return selected


def build_negative_test_table() -> pd.DataFrame:
    dataframe = pd.read_csv(
        NEGATIVE_SCENE_MANIFEST,
        encoding="utf-8-sig",
        low_memory=False,
    )

    split_column = find_column(
        dataframe,
        [
            "split",
            "scene_split",
        ],
    )

    group_column = find_column(
        dataframe,
        [
            "image_set",
            "group",
            "source_group",
        ],
    )

    image_column = find_column(
        dataframe,
        [
            "image_name",
            "sample_name",
            "filename",
        ],
    )

    scene_column = find_column(
        dataframe,
        [
            "scene_id",
            "ID_3",
            "id_3",
        ],
        required=False,
    )

    path_column = find_column(
        dataframe,
        [
            "local_image_path",
            "image_path",
        ],
        required=False,
    )

    selected = dataframe[
        dataframe[
            split_column
        ].astype(str).str.strip().eq(
            "external_test"
        )
    ].copy()

    selected[
        "evaluation_group"
    ] = (
        selected[group_column]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    selected = selected[
        selected[
            "evaluation_group"
        ].isin(
            [
                "nc",
                "nw",
            ]
        )
    ].copy()

    selected[
        "evaluation_image_name"
    ] = selected[
        image_column
    ].map(
        clean_text
    )

    selected[
        "evaluation_scene_id"
    ] = (
        selected[
            scene_column
        ].map(
            normalize_identifier
        )
        if scene_column
        else ""
    )

    selected[
        "evaluation_image_path"
    ] = (
        selected[path_column]
        if path_column
        else ""
    )

    selected = (
        selected
        .drop_duplicates(
            subset=[
                "evaluation_group",
                "evaluation_image_name",
            ]
        )
        .reset_index(drop=True)
    )

    if selected.empty:
        raise RuntimeError(
            "External-test negatif görüntü bulunamadı."
        )

    return selected


def calculate_negative_summary(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    if dataframe.empty:
        return {
            "images": 0,
            "any_alarm_percent": 0.0,
            "alarm_ge_001_percent": 0.0,
            "alarm_ge_01_percent": 0.0,
            "alarm_ge_1_percent": 0.0,
            "aggregate_fp_pixel_percent": 0.0,
        }

    coverage = dataframe[
        "final_coverage_ratio"
    ].to_numpy(
        dtype=float
    )

    total_positive_pixels = int(
        dataframe[
            "final_positive_pixels"
        ].sum()
    )

    total_pixels = int(
        dataframe[
            "total_pixels"
        ].sum()
    )

    return {
        "images": int(
            len(dataframe)
        ),
        "any_alarm_percent": float(
            (
                coverage > 0.0
            ).mean()
            * 100.0
        ),
        "alarm_ge_001_percent": float(
            (
                coverage >= 0.0001
            ).mean()
            * 100.0
        ),
        "alarm_ge_01_percent": float(
            (
                coverage >= 0.001
            ).mean()
            * 100.0
        ),
        "alarm_ge_1_percent": float(
            (
                coverage >= 0.01
            ).mean()
            * 100.0
        ),
        "aggregate_fp_pixel_percent": float(
            (
                total_positive_pixels
                / total_pixels
                * 100.0
            )
            if total_pixels > 0
            else 0.0
        ),
        "mean_fp_pixel_percent": float(
            coverage.mean()
            * 100.0
        ),
        "maximum_fp_pixel_percent": float(
            coverage.max()
            * 100.0
        ),
    }


def write_report(
    summary: dict[str, Any],
) -> None:
    positive = summary[
        "positive_locked"
    ]

    negative = summary[
        "negative_external"
    ]

    lines = [
        "# Robust Binary Oil Detector v0.3",
        "# Kilitli Uçtan Uca Değerlendirme",
        "",
        "## Bilimsel protokol",
        "",
        "- Verifier threshold validation verisiyle seçildi.",
        "- Temperature calibration yalnız calibration split ile yapıldı.",
        "- Bu aşamada test_locked pozitif sahneler ilk kez kullanıldı.",
        "- DARTIS external_test negatif sahneleri eğitimde kullanılmadı.",
        "- Bu testten sonra model veya eşikler bu sonuçlara göre "
        "değiştirilmemelidir.",
        "",
        "## Kilitli pozitif sonuçları",
        "",
        f"- Pozitif görüntü: {positive['images']}",
        f"- Petrol tespit edilen görüntü: "
        f"{positive['detected_images']}",
        f"- Kaçırılan görüntü: "
        f"{positive['missed_images']}",
        f"- Görüntü seviyesi recall: "
        f"{positive['image_recall']:.4f}",
        f"- XML petrol kutusu: "
        f"{positive['ground_truth_boxes']}",
        f"- Maskeyle kesişen kutu: "
        f"{positive['hit_boxes']}",
        f"- Kutu seviyesi recall: "
        f"{positive['box_recall']:.4f}",
        "",
        "XML anotasyonları bounding box olduğundan bu bölümde "
        "Dice veya IoU hesaplanmamıştır.",
        "",
        "## Petrolsüz external-test sonuçları",
        "",
        "| Grup | Görüntü | Herhangi alarm | ≥%0.01 | "
        "≥%0.1 | ≥%1 | FP piksel |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for group_name in (
        "overall",
        "nc",
        "nw",
    ):
        result = negative[
            group_name
        ]

        lines.append(
            f"| {group_name} "
            f"| {result['images']} "
            f"| %{result['any_alarm_percent']:.2f} "
            f"| %{result['alarm_ge_001_percent']:.2f} "
            f"| %{result['alarm_ge_01_percent']:.2f} "
            f"| %{result['alarm_ge_1_percent']:.2f} "
            f"| %{result['aggregate_fp_pixel_percent']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Yorumlama",
            "",
            "Pozitif recall, sistemin gerçek petrol içeren kilitli "
            "görüntüleri yakalama oranını gösterir.",
            "",
            "Negatif sonuçlardaki `≥%1`, petrol bulunmayan bir görüntünün "
            "en az yüzde birinin yanlışlıkla petrol olarak maskelendiği "
            "ciddi yanlış alarm oranıdır.",
            "",
            "Bu sürümde bağımsız kara-deniz maskesi bulunmadığından "
            "kıyı grubu `nc` hâlâ en zor gruptur.",
            "",
        ]
    )

    REPORT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    required_paths = [
        DETECTOR_SCRIPT,
        VERIFIER_MANIFEST,
        POSITIVE_CROP_MANIFEST,
        NEGATIVE_SCENE_MANIFEST,
    ]

    for required_path in required_paths:
        if not required_path.exists():
            raise FileNotFoundError(
                f"Gerekli dosya bulunamadı: "
                f"{required_path}"
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ERROR_EXAMPLES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    detector = load_detector_api()

    device = detector[
        "select_device"
    ](
        args.device
    )

    segmentation_model = detector[
        "build_segmentation_model"
    ](
        device
    )

    (
        verifier_model,
        verifier_config,
    ) = detector[
        "build_verifier_model"
    ](
        device
    )

    verifier_transform = detector[
        "build_verifier_transform"
    ](
        verifier_config
    )

    positive_table = (
        build_positive_test_table()
    )

    negative_table = (
        build_negative_test_table()
    )

    if args.positive_limit > 0:
        positive_table = (
            positive_table
            .head(
                args.positive_limit
            )
            .copy()
        )

    if args.negative_limit > 0:
        negative_table = (
            negative_table
            .head(
                args.negative_limit
            )
            .copy()
        )

    print("=" * 78)
    print(
        "ROBUST BINARY OIL DETECTOR v0.3"
    )
    print(
        "KİLİTLİ END-TO-END DEĞERLENDİRME"
    )
    print("=" * 78)

    print("Cihaz:", device)
    print(
        "Kilitli pozitif görüntü:",
        len(positive_table),
    )
    print(
        "External-test negatif görüntü:",
        len(negative_table),
    )
    print(
        "Model ve eşikler: DONDURULDU"
    )
    print()

    start_time = time.perf_counter()

    positive_rows = []
    positive_error_count = 0

    total_boxes = 0
    total_hit_boxes = 0

    print("POZİTİF KİLİTLİ TEST")

    for position, row in enumerate(
        positive_table.to_dict(
            orient="records"
        ),
        start=1,
    ):
        group_name = clean_text(
            row.get(
                "image_set",
                "",
            )
        ).lower()

        image_name = clean_text(
            row.get(
                "source_image_name",
                "",
            )
        )

        scene_id = normalize_identifier(
            row.get(
                "scene_id",
                "",
            )
        )

        image_path = find_dartis_image(
            group_name,
            image_name,
            row.get(
                "source_image_path",
                "",
            ),
        )

        xml_path = resolve_path(
            row.get(
                "annotation_path",
                "",
            )
        )

        if image_path is None:
            raise FileNotFoundError(
                f"Pozitif görüntü bulunamadı: "
                f"{group_name}/{image_name}"
            )

        evaluation = run_single_image(
            detector,
            segmentation_model,
            verifier_model,
            verifier_transform,
            verifier_config,
            image_path,
            device,
            args,
        )

        boxes = read_boxes(
            xml_path,
            evaluation[
                "image_width"
            ],
            evaluation[
                "image_height"
            ],
        )

        box_hits = []

        for (
            left,
            top,
            right,
            bottom,
        ) in boxes:
            hit = bool(
                evaluation[
                    "final_mask"
                ][
                    top:bottom,
                    left:right,
                ].any()
            )

            box_hits.append(
                hit
            )

        hit_box_count = int(
            sum(box_hits)
        )

        total_boxes += len(boxes)
        total_hit_boxes += (
            hit_box_count
        )

        positive_rows.append(
            {
                "image_set": group_name,
                "scene_id": scene_id,
                "image_name": image_name,
                "image_path": str(
                    image_path
                ),
                "annotation_path": str(
                    xml_path
                ),
                "ground_truth_box_count": int(
                    len(boxes)
                ),
                "hit_box_count": (
                    hit_box_count
                ),
                "box_recall": float(
                    hit_box_count
                    / len(boxes)
                )
                if boxes
                else 0.0,
                "oil_detected": bool(
                    evaluation[
                        "oil_detected"
                    ]
                ),
                "raw_positive_pixels": int(
                    evaluation[
                        "raw_positive_pixels"
                    ]
                ),
                "final_positive_pixels": int(
                    evaluation[
                        "final_positive_pixels"
                    ]
                ),
                "final_coverage_ratio": float(
                    evaluation[
                        "final_coverage_ratio"
                    ]
                ),
                "candidate_count": int(
                    evaluation[
                        "candidate_count"
                    ]
                ),
                "accepted_candidate_count": int(
                    evaluation[
                        "accepted_candidate_count"
                    ]
                ),
                "maximum_candidate_probability": float(
                    evaluation[
                        "maximum_candidate_probability"
                    ]
                ),
            }
        )

        detected_text = (
            "DETECTED"
            if evaluation[
                "oil_detected"
            ]
            else "MISSED"
        )

        print(
            f"[P {position:03d}/"
            f"{len(positive_table):03d}] "
            f"{detected_text} "
            f"{group_name}/{image_name} "
            f"| boxes={hit_box_count}/{len(boxes)} "
            f"| coverage="
            f"{evaluation['final_coverage_ratio'] * 100.0:.4f}%"
        )

        if (
            not evaluation[
                "oil_detected"
            ]
            and positive_error_count
            < args.error_examples
        ):
            save_error_example(
                detector,
                evaluation,
                ERROR_EXAMPLES_DIR
                / "positive_missed",
                Path(
                    image_name
                ).stem,
            )

            positive_error_count += 1

    positive_results = pd.DataFrame(
        positive_rows
    )

    positive_results.to_csv(
        POSITIVE_RESULTS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("NEGATİF EXTERNAL TEST")

    negative_rows = []
    negative_error_count = 0

    for position, row in enumerate(
        negative_table.to_dict(
            orient="records"
        ),
        start=1,
    ):
        group_name = clean_text(
            row[
                "evaluation_group"
            ]
        ).lower()

        image_name = clean_text(
            row[
                "evaluation_image_name"
            ]
        )

        scene_id = normalize_identifier(
            row[
                "evaluation_scene_id"
            ]
        )

        image_path = find_dartis_image(
            group_name,
            image_name,
            row.get(
                "evaluation_image_path",
                "",
            ),
        )

        if image_path is None:
            raise FileNotFoundError(
                f"Negatif görüntü bulunamadı: "
                f"{group_name}/{image_name}"
            )

        evaluation = run_single_image(
            detector,
            segmentation_model,
            verifier_model,
            verifier_transform,
            verifier_config,
            image_path,
            device,
            args,
        )

        negative_rows.append(
            {
                "image_set": group_name,
                "scene_id": scene_id,
                "image_name": image_name,
                "image_path": str(
                    image_path
                ),
                "oil_detected": bool(
                    evaluation[
                        "oil_detected"
                    ]
                ),
                "total_pixels": int(
                    evaluation[
                        "total_pixels"
                    ]
                ),
                "raw_positive_pixels": int(
                    evaluation[
                        "raw_positive_pixels"
                    ]
                ),
                "final_positive_pixels": int(
                    evaluation[
                        "final_positive_pixels"
                    ]
                ),
                "raw_coverage_ratio": float(
                    evaluation[
                        "raw_coverage_ratio"
                    ]
                ),
                "final_coverage_ratio": float(
                    evaluation[
                        "final_coverage_ratio"
                    ]
                ),
                "candidate_count": int(
                    evaluation[
                        "candidate_count"
                    ]
                ),
                "accepted_candidate_count": int(
                    evaluation[
                        "accepted_candidate_count"
                    ]
                ),
                "maximum_candidate_probability": float(
                    evaluation[
                        "maximum_candidate_probability"
                    ]
                ),
            }
        )

        alarm_text = (
            "FALSE-ALARM"
            if evaluation[
                "oil_detected"
            ]
            else "CLEAN"
        )

        print(
            f"[N {position:03d}/"
            f"{len(negative_table):03d}] "
            f"{alarm_text} "
            f"{group_name}/{image_name} "
            f"| coverage="
            f"{evaluation['final_coverage_ratio'] * 100.0:.4f}%"
        )

        if (
            evaluation[
                "oil_detected"
            ]
            and negative_error_count
            < args.error_examples
        ):
            save_error_example(
                detector,
                evaluation,
                ERROR_EXAMPLES_DIR
                / "negative_false_alarm"
                / group_name,
                Path(
                    image_name
                ).stem,
            )

            negative_error_count += 1

    negative_results = pd.DataFrame(
        negative_rows
    )

    negative_results.to_csv(
        NEGATIVE_RESULTS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    detected_positive_images = int(
        positive_results[
            "oil_detected"
        ].sum()
    )

    positive_image_count = int(
        len(
            positive_results
        )
    )

    positive_summary = {
        "images": (
            positive_image_count
        ),
        "detected_images": (
            detected_positive_images
        ),
        "missed_images": int(
            positive_image_count
            - detected_positive_images
        ),
        "image_recall": float(
            detected_positive_images
            / positive_image_count
        )
        if positive_image_count > 0
        else 0.0,
        "ground_truth_boxes": int(
            total_boxes
        ),
        "hit_boxes": int(
            total_hit_boxes
        ),
        "missed_boxes": int(
            total_boxes
            - total_hit_boxes
        ),
        "box_recall": float(
            total_hit_boxes
            / total_boxes
        )
        if total_boxes > 0
        else 0.0,
    }

    overall_negative = (
        calculate_negative_summary(
            negative_results
        )
    )

    nc_negative = (
        calculate_negative_summary(
            negative_results[
                negative_results[
                    "image_set"
                ].eq("nc")
            ]
        )
    )

    nw_negative = (
        calculate_negative_summary(
            negative_results[
                negative_results[
                    "image_set"
                ].eq("nw")
            ]
        )
    )

    elapsed_seconds = (
        time.perf_counter()
        - start_time
    )

    summary = {
        "model_version": (
            "robust_binary_oil_detector_v03"
        ),
        "segmentation_threshold": float(
            args.segmentation_threshold
        ),
        "verifier_probability_mode": (
            verifier_config.get(
                "probability_mode"
            )
        ),
        "verifier_temperature": float(
            verifier_config.get(
                "temperature",
                1.0,
            )
        ),
        "verifier_raw_threshold": float(
            verifier_config.get(
                "raw_threshold",
                0.15,
            )
        ),
        "verifier_calibrated_threshold": float(
            verifier_config.get(
                "calibrated_threshold",
                0.173478,
            )
        ),
        "positive_locked": (
            positive_summary
        ),
        "negative_external": {
            "overall": (
                overall_negative
            ),
            "nc": nc_negative,
            "nw": nw_negative,
        },
        "elapsed_seconds": float(
            elapsed_seconds
        ),
        "model_configuration_frozen": True,
        "test_locked_used": True,
        "negative_external_test_used": True,
        "important_note": (
            "The locked test results must not be used to tune "
            "the current model or thresholds."
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
        "KİLİTLİ DEĞERLENDİRME SONUCU"
    )
    print("=" * 78)

    print(
        "Pozitif görüntü:",
        positive_summary[
            "images"
        ],
    )

    print(
        "Pozitif tespit:",
        positive_summary[
            "detected_images"
        ],
    )

    print(
        "Görüntü recall:",
        f"{positive_summary['image_recall']:.4f}",
    )

    print(
        "Kutu recall:",
        f"{positive_summary['box_recall']:.4f}",
    )

    print()
    print(
        "Negatif external görüntü:",
        overall_negative[
            "images"
        ],
    )

    print(
        "Herhangi yanlış alarm:",
        f"%{overall_negative['any_alarm_percent']:.2f}",
    )

    print(
        "Ciddi yanlış alarm >=%1:",
        f"%{overall_negative['alarm_ge_1_percent']:.2f}",
    )

    print(
        "FP piksel:",
        f"%{overall_negative['aggregate_fp_pixel_percent']:.4f}",
    )

    print()
    print(
        "NC ciddi alarm >=%1:",
        f"%{nc_negative['alarm_ge_1_percent']:.2f}",
    )

    print(
        "NW ciddi alarm >=%1:",
        f"%{nw_negative['alarm_ge_1_percent']:.2f}",
    )

    print()
    print(
        "Süre:",
        f"{elapsed_seconds:.1f} saniye",
    )

    print()
    print(
        "ÖNEMLİ: Model ve threshold artık bu test "
        "sonuçlarına göre değiştirilmemelidir."
    )

    print()
    print(
        "Pozitif sonuçlar:",
        POSITIVE_RESULTS_PATH.resolve(),
    )

    print(
        "Negatif sonuçlar:",
        NEGATIVE_RESULTS_PATH.resolve(),
    )

    print(
        "Hata örnekleri:",
        ERROR_EXAMPLES_DIR.resolve(),
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
