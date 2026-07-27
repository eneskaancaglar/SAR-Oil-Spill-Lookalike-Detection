from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]

POSITIVE_MANIFEST = (
    ROOT
    / "data"
    / "metadata"
    / "dartis_positive_oil_manifest.csv"
)

RAW_ROOT = (
    ROOT
    / "data"
    / "external"
    / "dartis"
    / "raw"
)

ANNOTATION_ROOT = (
    ROOT
    / "data"
    / "external"
    / "dartis"
    / "annotations"
)

CROP_ROOT = (
    ROOT
    / "data"
    / "external"
    / "dartis"
    / "processed"
    / "verifier_positive"
)

OUTPUT_MANIFEST = (
    ROOT
    / "data"
    / "metadata"
    / "dartis_positive_verifier_crops.csv"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "v03_dartis_positive_crops"
)

QC_DIR = (
    OUTPUT_DIR
    / "qc_overlays"
)

SUMMARY_JSON = (
    OUTPUT_DIR
    / "positive_crop_summary.json"
)

REPORT_PATH = (
    ROOT
    / "reports"
    / "v03_dartis_positive_crops.md"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "DARTIS pozitif XML bounding box anotasyonlarını "
            "verifier modeli için pozitif crop görüntülerine dönüştürür."
        )
    )

    parser.add_argument(
        "--context",
        type=float,
        default=0.25,
        help=(
            "Bounding box çevresine eklenecek bağlam oranı. "
            "Varsayılan: 0.25"
        ),
    )

    parser.add_argument(
        "--size",
        type=int,
        default=224,
        help="Kaydedilecek crop boyutu. Varsayılan: 224",
    )

    parser.add_argument(
        "--minimum-box-size",
        type=int,
        default=8,
        help="Kabul edilen minimum kutu genişliği/yüksekliği.",
    )

    parser.add_argument(
        "--minimum-crop-size",
        type=int,
        default=64,
        help="Bağlam eklendikten sonraki minimum crop boyutu.",
    )

    parser.add_argument(
        "--qc-images",
        type=int,
        default=24,
        help="Kutuları çizilerek kaydedilecek örnek görüntü sayısı.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="İşlenecek görüntü sayısı. 0 bütün görüntüleri işler.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Mevcut crop dosyalarını yeniden oluşturur.",
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


def find_existing_file(
    root: Path,
    group_name: str,
    expected_name: str,
    extensions: tuple[str, ...],
) -> Path | None:
    direct_path = (
        root
        / group_name
        / expected_name
    )

    if direct_path.exists():
        return direct_path

    directory = (
        root
        / group_name
    )

    if not directory.exists():
        return None

    stem = Path(expected_name).stem

    for extension in extensions:
        candidate = (
            directory
            / f"{stem}{extension}"
        )

        if candidate.exists():
            return candidate

    return None


def find_child_text(
    element: ET.Element,
    names: tuple[str, ...],
) -> str:
    normalized_names = {
        name.lower()
        for name in names
    }

    for child in list(element):
        tag_name = (
            str(child.tag)
            .split("}")[-1]
            .strip()
            .lower()
        )

        if tag_name in normalized_names:
            return (
                child.text.strip()
                if child.text
                else ""
            )

    return ""


def parse_number(value: str) -> float:
    cleaned = (
        str(value)
        .strip()
        .replace(",", ".")
    )

    return float(cleaned)


def parse_bbox(
    bbox_element: ET.Element,
) -> tuple[float, float, float, float]:
    xmin_text = find_child_text(
        bbox_element,
        (
            "xmin",
            "x_min",
            "left",
            "x1",
        ),
    )

    ymin_text = find_child_text(
        bbox_element,
        (
            "ymin",
            "y_min",
            "top",
            "y1",
        ),
    )

    xmax_text = find_child_text(
        bbox_element,
        (
            "xmax",
            "x_max",
            "right",
            "x2",
        ),
    )

    ymax_text = find_child_text(
        bbox_element,
        (
            "ymax",
            "y_max",
            "bottom",
            "y2",
        ),
    )

    if not all(
        [
            xmin_text,
            ymin_text,
            xmax_text,
            ymax_text,
        ]
    ):
        raise ValueError(
            "Bounding box koordinatlarından biri eksik."
        )

    return (
        parse_number(xmin_text),
        parse_number(ymin_text),
        parse_number(xmax_text),
        parse_number(ymax_text),
    )


def parse_xml_annotation(
    xml_path: Path,
) -> dict[str, Any]:
    root = ET.parse(
        xml_path
    ).getroot()

    xml_width = None
    xml_height = None

    size_element = root.find(".//size")

    if size_element is not None:
        width_text = find_child_text(
            size_element,
            ("width",),
        )

        height_text = find_child_text(
            size_element,
            ("height",),
        )

        if width_text:
            xml_width = int(
                round(
                    parse_number(width_text)
                )
            )

        if height_text:
            xml_height = int(
                round(
                    parse_number(height_text)
                )
            )

    objects: list[dict[str, Any]] = []

    object_elements = root.findall(
        ".//object"
    )

    for object_index, object_element in enumerate(
        object_elements,
        start=1,
    ):
        label = find_child_text(
            object_element,
            (
                "name",
                "label",
                "class",
            ),
        )

        if not label:
            label = "oil"

        bbox_element = object_element.find(
            ".//bndbox"
        )

        if bbox_element is None:
            continue

        bbox = parse_bbox(
            bbox_element
        )

        objects.append(
            {
                "object_index": object_index,
                "object_label": label,
                "bbox": bbox,
            }
        )

    if not objects:
        bbox_elements = root.findall(
            ".//bndbox"
        )

        for object_index, bbox_element in enumerate(
            bbox_elements,
            start=1,
        ):
            bbox = parse_bbox(
                bbox_element
            )

            objects.append(
                {
                    "object_index": object_index,
                    "object_label": "oil",
                    "bbox": bbox,
                }
            )

    return {
        "xml_width": xml_width,
        "xml_height": xml_height,
        "objects": objects,
    }


def convert_voc_box(
    bbox: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    xmin, ymin, xmax, ymax = bbox

    # PASCAL VOC koordinatları çoğunlukla 1 tabanlı ve kapsayıcıdır.
    # PIL crop ise 0 tabanlı ve sağ-alt sınırı hariç tutar.
    left = math.floor(xmin)

    top = math.floor(ymin)

    if left >= 1:
        left -= 1

    if top >= 1:
        top -= 1

    right = math.ceil(xmax)
    bottom = math.ceil(ymax)

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

    return (
        left,
        top,
        right,
        bottom,
    )


def expand_box(
    box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    context_ratio: float,
    minimum_crop_size: int,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = box

    box_width = right - left
    box_height = bottom - top

    center_x = (
        left + right
    ) / 2.0

    center_y = (
        top + bottom
    ) / 2.0

    expanded_width = max(
        box_width * (
            1.0 + 2.0 * context_ratio
        ),
        float(minimum_crop_size),
    )

    expanded_height = max(
        box_height * (
            1.0 + 2.0 * context_ratio
        ),
        float(minimum_crop_size),
    )

    expanded_left = int(
        math.floor(
            center_x
            - expanded_width / 2.0
        )
    )

    expanded_top = int(
        math.floor(
            center_y
            - expanded_height / 2.0
        )
    )

    expanded_right = int(
        math.ceil(
            center_x
            + expanded_width / 2.0
        )
    )

    expanded_bottom = int(
        math.ceil(
            center_y
            + expanded_height / 2.0
        )
    )

    if expanded_left < 0:
        expanded_right -= expanded_left
        expanded_left = 0

    if expanded_top < 0:
        expanded_bottom -= expanded_top
        expanded_top = 0

    if expanded_right > image_width:
        difference = (
            expanded_right
            - image_width
        )

        expanded_left -= difference
        expanded_right = image_width

    if expanded_bottom > image_height:
        difference = (
            expanded_bottom
            - image_height
        )

        expanded_top -= difference
        expanded_bottom = image_height

    expanded_left = max(
        0,
        expanded_left,
    )

    expanded_top = max(
        0,
        expanded_top,
    )

    expanded_right = min(
        image_width,
        max(
            expanded_left + 1,
            expanded_right,
        ),
    )

    expanded_bottom = min(
        image_height,
        max(
            expanded_top + 1,
            expanded_bottom,
        ),
    )

    return (
        expanded_left,
        expanded_top,
        expanded_right,
        expanded_bottom,
    )


def save_qc_overlay(
    image: Image.Image,
    boxes: list[tuple[int, int, int, int]],
    destination: Path,
) -> None:
    overlay = image.convert(
        "RGB"
    ).copy()

    draw = ImageDraw.Draw(
        overlay
    )

    for object_index, box in enumerate(
        boxes,
        start=1,
    ):
        left, top, right, bottom = box

        draw.rectangle(
            [
                left,
                top,
                right - 1,
                bottom - 1,
            ],
            outline=(255, 0, 0),
            width=3,
        )

        draw.text(
            (
                left + 4,
                top + 4,
            ),
            str(object_index),
            fill=(255, 255, 0),
            stroke_width=1,
            stroke_fill=(0, 0, 0),
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    overlay.save(
        destination,
        quality=95,
    )


def write_report(
    summary: dict[str, Any],
) -> None:
    lines = [
        "# DARTIS Pozitif Verifier Crop Hazırlığı",
        "",
        "## Amaç",
        "",
        "DARTIS petrollü `ow` ve `oc` görüntülerindeki XML "
        "bounding box anotasyonlarını, ikinci aşama petrol "
        "doğrulama modeli için pozitif crop örneklerine çevirmek.",
        "",
        "## Önemli bilimsel karar",
        "",
        "XML kutuları piksel seviyesinde petrol maskesi değildir. "
        "Bu nedenle kutunun tamamı petrol maskesi yapılmamıştır. "
        "Kutular yalnızca pozitif aday bölgeleri crop etmek için "
        "kullanılmıştır.",
        "",
        "## Sonuç",
        "",
        f"- Manifest görüntüsü: {summary['manifest_images']}",
        f"- İşlenen görüntü: {summary['processed_images']}",
        f"- Başarılı görüntü: {summary['successful_images']}",
        f"- Hatalı görüntü: {summary['failed_images']}",
        f"- XML nesnesi: {summary['xml_objects_total']}",
        f"- Geçerli bounding box: {summary['valid_boxes']}",
        f"- Geçersiz bounding box: {summary['invalid_boxes']}",
        f"- Kaydedilen pozitif crop: {summary['saved_crops']}",
        f"- XML boyut uyuşmazlığı: {summary['xml_size_mismatch_count']}",
        "",
        "## Grup sonuçları",
        "",
        "| Grup | Görüntü | Nesne | Crop |",
        "|---|---:|---:|---:|",
    ]

    for group_name in (
        "ow",
        "oc",
    ):
        group = summary[
            "groups"
        ].get(
            group_name,
            {},
        )

        lines.append(
            f"| {group_name} "
            f"| {group.get('images', 0)} "
            f"| {group.get('objects', 0)} "
            f"| {group.get('crops', 0)} |"
        )

    lines.extend(
        [
            "",
            "## Sonraki aşama",
            "",
            "DARTIS `nc` ve `nw` petrolsüz görüntülerinden ve "
            "mevcut modelin yanlış alarm bölgelerinden negatif "
            "verifier crop örnekleri üretilecektir.",
            "",
        ]
    )

    REPORT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    if not POSITIVE_MANIFEST.exists():
        raise FileNotFoundError(
            f"Pozitif manifest bulunamadı: {POSITIVE_MANIFEST}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    QC_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_MANIFEST.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    for group_name in (
        "ow",
        "oc",
    ):
        (
            CROP_ROOT
            / group_name
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

    manifest = pd.read_csv(
        POSITIVE_MANIFEST,
        encoding="utf-8-sig",
    )

    if args.limit > 0:
        manifest = (
            manifest
            .head(args.limit)
            .copy()
        )

    print("=" * 78)
    print(
        "DARTIS POZİTİF VERIFIER CROP HAZIRLIĞI"
    )
    print("=" * 78)
    print("İşlenecek görüntü:", len(manifest))
    print("Context oranı:", args.context)
    print("Crop boyutu:", args.size)
    print()

    output_rows: list[
        dict[str, Any]
    ] = []

    failed_rows: list[
        dict[str, str]
    ] = []

    group_image_counts = Counter()
    group_object_counts = Counter()
    group_crop_counts = Counter()
    label_counts = Counter()

    xml_objects_total = 0
    valid_boxes = 0
    invalid_boxes = 0
    xml_size_mismatch_count = 0
    qc_saved = 0

    for position, row in enumerate(
        manifest.to_dict(
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
                "image_name",
                "",
            )
        )

        annotation_name = clean_text(
            row.get(
                "annotation_name",
                "",
            )
        )

        scene_id = clean_text(
            row.get(
                "scene_id",
                "",
            )
        )

        image_path = find_existing_file(
            RAW_ROOT,
            group_name,
            image_name,
            (
                ".jpg",
                ".jpeg",
                ".png",
                ".tif",
                ".tiff",
            ),
        )

        xml_path = find_existing_file(
            ANNOTATION_ROOT,
            group_name,
            annotation_name,
            (
                ".xml",
            ),
        )

        if image_path is None:
            failed_rows.append(
                {
                    "image_set": group_name,
                    "image_name": image_name,
                    "reason": "image_missing",
                }
            )

            print(
                f"[{position:04d}/{len(manifest):04d}] "
                f"HATA görüntü yok: {group_name}/{image_name}"
            )

            continue

        if xml_path is None:
            failed_rows.append(
                {
                    "image_set": group_name,
                    "image_name": image_name,
                    "reason": "xml_missing",
                }
            )

            print(
                f"[{position:04d}/{len(manifest):04d}] "
                f"HATA XML yok: {group_name}/{annotation_name}"
            )

            continue

        try:
            with Image.open(
                image_path
            ) as source_image:
                image = source_image.convert(
                    "L"
                )

            image_width, image_height = (
                image.size
            )

            annotation = (
                parse_xml_annotation(
                    xml_path
                )
            )

            objects = annotation[
                "objects"
            ]

            xml_objects_total += len(
                objects
            )

            group_image_counts[
                group_name
            ] += 1

            xml_width = annotation[
                "xml_width"
            ]

            xml_height = annotation[
                "xml_height"
            ]

            if (
                xml_width is not None
                and xml_height is not None
                and (
                    xml_width != image_width
                    or xml_height != image_height
                )
            ):
                xml_size_mismatch_count += 1

            qc_boxes: list[
                tuple[int, int, int, int]
            ] = []

            image_crop_count = 0

            for object_data in objects:
                object_index = int(
                    object_data[
                        "object_index"
                    ]
                )

                object_label = clean_text(
                    object_data[
                        "object_label"
                    ]
                )

                if not object_label:
                    object_label = "oil"

                try:
                    original_box = (
                        convert_voc_box(
                            object_data[
                                "bbox"
                            ],
                            image_width,
                            image_height,
                        )
                    )

                    left, top, right, bottom = (
                        original_box
                    )

                    box_width = (
                        right - left
                    )

                    box_height = (
                        bottom - top
                    )

                    if (
                        box_width
                        < args.minimum_box_size
                        or box_height
                        < args.minimum_box_size
                    ):
                        raise ValueError(
                            "Bounding box minimum boyuttan küçük."
                        )

                    crop_box = expand_box(
                        original_box,
                        image_width,
                        image_height,
                        args.context,
                        args.minimum_crop_size,
                    )

                    crop_left, crop_top, crop_right, crop_bottom = (
                        crop_box
                    )

                    crop = image.crop(
                        crop_box
                    )

                    crop = crop.resize(
                        (
                            args.size,
                            args.size,
                        ),
                        resample=Image.Resampling.BILINEAR,
                    )

                    crop_name = (
                        f"{Path(image_name).stem}"
                        f"__obj_{object_index:03d}.png"
                    )

                    crop_path = (
                        CROP_ROOT
                        / group_name
                        / crop_name
                    )

                    if (
                        not crop_path.exists()
                        or args.overwrite
                    ):
                        crop.save(
                            crop_path,
                            format="PNG",
                            optimize=True,
                        )

                    output_rows.append(
                        {
                            "source_dataset": "DARTIS",
                            "binary_label": 1,
                            "binary_label_name": "oil",
                            "image_set": group_name,
                            "scene_id": scene_id,
                            "source_image_name": image_name,
                            "source_image_path": str(
                                image_path
                            ),
                            "annotation_name": annotation_name,
                            "annotation_path": str(
                                xml_path
                            ),
                            "object_index": object_index,
                            "object_label": object_label,
                            "image_width": image_width,
                            "image_height": image_height,
                            "bbox_left": left,
                            "bbox_top": top,
                            "bbox_right": right,
                            "bbox_bottom": bottom,
                            "bbox_width": box_width,
                            "bbox_height": box_height,
                            "crop_left": crop_left,
                            "crop_top": crop_top,
                            "crop_right": crop_right,
                            "crop_bottom": crop_bottom,
                            "crop_width": (
                                crop_right
                                - crop_left
                            ),
                            "crop_height": (
                                crop_bottom
                                - crop_top
                            ),
                            "context_ratio": (
                                args.context
                            ),
                            "output_size": (
                                args.size
                            ),
                            "crop_name": crop_name,
                            "crop_path": str(
                                crop_path
                            ),
                            "split": "unassigned_v03",
                        }
                    )

                    valid_boxes += 1
                    image_crop_count += 1

                    group_object_counts[
                        group_name
                    ] += 1

                    group_crop_counts[
                        group_name
                    ] += 1

                    label_counts[
                        object_label
                    ] += 1

                    qc_boxes.append(
                        original_box
                    )

                except Exception as box_error:
                    invalid_boxes += 1

                    failed_rows.append(
                        {
                            "image_set": group_name,
                            "image_name": image_name,
                            "reason": (
                                f"invalid_box:"
                                f"{type(box_error).__name__}:"
                                f"{box_error}"
                            ),
                        }
                    )

            if (
                qc_boxes
                and qc_saved
                < args.qc_images
            ):
                qc_destination = (
                    QC_DIR
                    / group_name
                    / (
                        f"{Path(image_name).stem}"
                        f"__boxes.jpg"
                    )
                )

                save_qc_overlay(
                    image,
                    qc_boxes,
                    qc_destination,
                )

                qc_saved += 1

            print(
                f"[{position:04d}/{len(manifest):04d}] "
                f"OK {group_name}/{image_name} "
                f"| nesne={len(objects)} "
                f"| crop={image_crop_count}"
            )

        except Exception as error:
            failed_rows.append(
                {
                    "image_set": group_name,
                    "image_name": image_name,
                    "reason": (
                        f"{type(error).__name__}: "
                        f"{error}"
                    ),
                }
            )

            print(
                f"[{position:04d}/{len(manifest):04d}] "
                f"HATA {group_name}/{image_name}: "
                f"{type(error).__name__}: {error}"
            )

    output_dataframe = pd.DataFrame(
        output_rows
    )

    output_dataframe.to_csv(
        OUTPUT_MANIFEST,
        index=False,
        encoding="utf-8-sig",
    )

    successful_image_keys = {
        (
            row[
                "image_set"
            ],
            row[
                "source_image_name"
            ],
        )
        for row in output_rows
    }

    summary = {
        "manifest_images": int(
            len(manifest)
        ),
        "processed_images": int(
            len(manifest)
        ),
        "successful_images": int(
            len(
                successful_image_keys
            )
        ),
        "failed_images": int(
            len(manifest)
            - len(
                successful_image_keys
            )
        ),
        "xml_objects_total": int(
            xml_objects_total
        ),
        "valid_boxes": int(
            valid_boxes
        ),
        "invalid_boxes": int(
            invalid_boxes
        ),
        "saved_crops": int(
            len(output_dataframe)
        ),
        "xml_size_mismatch_count": int(
            xml_size_mismatch_count
        ),
        "qc_overlays_saved": int(
            qc_saved
        ),
        "context_ratio": float(
            args.context
        ),
        "output_size": int(
            args.size
        ),
        "groups": {
            group_name: {
                "images": int(
                    group_image_counts[
                        group_name
                    ]
                ),
                "objects": int(
                    group_object_counts[
                        group_name
                    ]
                ),
                "crops": int(
                    group_crop_counts[
                        group_name
                    ]
                ),
            }
            for group_name in (
                "ow",
                "oc",
            )
        },
        "object_labels": {
            str(key): int(value)
            for key, value in (
                label_counts.items()
            )
        },
        "failure_examples": (
            failed_rows[:30]
        ),
    }

    with SUMMARY_JSON.open(
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
    print("POZİTİF CROP SONUCU")
    print("=" * 78)

    print(
        "İşlenen görüntü:",
        summary[
            "processed_images"
        ],
    )

    print(
        "Başarılı görüntü:",
        summary[
            "successful_images"
        ],
    )

    print(
        "XML nesnesi:",
        summary[
            "xml_objects_total"
        ],
    )

    print(
        "Geçerli kutu:",
        summary[
            "valid_boxes"
        ],
    )

    print(
        "Geçersiz kutu:",
        summary[
            "invalid_boxes"
        ],
    )

    print(
        "Kaydedilen crop:",
        summary[
            "saved_crops"
        ],
    )

    print(
        "XML boyut uyuşmazlığı:",
        summary[
            "xml_size_mismatch_count"
        ],
    )

    print()
    print(
        "Manifest:",
        OUTPUT_MANIFEST.resolve(),
    )

    print(
        "QC:",
        QC_DIR.resolve(),
    )

    print(
        "Özet:",
        SUMMARY_JSON.resolve(),
    )

    print(
        "Rapor:",
        REPORT_PATH.resolve(),
    )


if __name__ == "__main__":
    main()
