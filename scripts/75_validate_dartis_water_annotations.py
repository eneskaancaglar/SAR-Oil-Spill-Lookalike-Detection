from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_QUEUE = (
    ROOT
    / "data"
    / "metadata"
    / "v06_dartis_water_active_batch.csv"
)

DEFAULT_STATUS = (
    ROOT
    / "data"
    / "metadata"
    / "v06_dartis_water_active_status.json"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "v06_dartis_water_annotation_audit"
)

AUDIT_CSV = (
    ROOT
    / "data"
    / "metadata"
    / "v06_dartis_water_annotation_audit.csv"
)

TRAINING_MANIFEST = (
    ROOT
    / "data"
    / "metadata"
    / "v06_dartis_water_active_training_manifest.csv"
)

CONTACT_SHEET = (
    OUTPUT_DIR
    / "annotation_audit_contact_sheet.jpg"
)

SUMMARY_PATH = OUTPUT_DIR / "summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "DARTIS aktif etiketleme paketindeki manuel kara-su "
            "maskelerini doğrular. Boyut, sınıf değerleri, tamamlanma "
            "durumu ve sınıf dağılımını denetler; eğitim manifesti "
            "ve görsel QA sayfası üretir."
        )
    )

    parser.add_argument(
        "--queue",
        type=Path,
        default=DEFAULT_QUEUE,
    )

    parser.add_argument(
        "--status",
        type=Path,
        default=DEFAULT_STATUS,
    )

    parser.add_argument(
        "--maximum-ignore-fraction",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--minimum-class-fraction",
        type=float,
        default=0.001,
        help=(
            "Su veya kara sınıfı bu oranın altındaysa uyarı üretir. "
            "Bu durum tek başına maskeyi eğitim dışı bırakmaz."
        ),
    )

    parser.add_argument(
        "--contact-width",
        type=int,
        default=1400,
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


def read_queue(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Etiketleme kuyruğu bulunamadı: {path}"
        )

    frame = pd.read_csv(
        path,
        encoding="utf-8-sig",
        low_memory=False,
    )

    required = {
        "sample_id",
        "coastal_group",
        "annotation_image_path",
        "manual_mask_path",
    }

    missing = required - set(frame.columns)

    if missing:
        raise RuntimeError(
            "Kuyrukta eksik sütunlar: "
            f"{sorted(missing)}"
        )

    if frame.empty:
        raise RuntimeError(
            "Etiketleme kuyruğu boş."
        )

    if frame["sample_id"].duplicated().any():
        duplicates = frame.loc[
            frame["sample_id"].duplicated(
                keep=False
            ),
            "sample_id",
        ].astype(str).tolist()

        raise RuntimeError(
            "Kuyrukta tekrarlı sample_id bulundu: "
            + ", ".join(
                duplicates[:10]
            )
        )

    return frame


def read_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "items": {},
            "last_index": 0,
        }

    data = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(data, dict):
        raise RuntimeError(
            "Status JSON nesne biçiminde değil."
        )

    items = data.get(
        "items",
        {},
    )

    if not isinstance(items, dict):
        raise RuntimeError(
            "Status JSON içindeki items alanı geçersiz."
        )

    return data


def quantize_preview_mask(
    array: np.ndarray,
) -> np.ndarray:
    output = np.zeros_like(
        array,
        dtype=np.uint8,
    )

    output[
        (array >= 64)
        & (array < 192)
    ] = 128

    output[array >= 192] = 255

    return output


def colorize_mask(
    mask: np.ndarray,
) -> np.ndarray:
    rgb = np.zeros(
        (
            mask.shape[0],
            mask.shape[1],
            3,
        ),
        dtype=np.uint8,
    )

    rgb[mask == 255] = (
        0,
        255,
        255,
    )

    rgb[mask == 128] = (
        255,
        0,
        255,
    )

    return rgb


def create_overlay(
    image: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    rgb = np.stack(
        [image, image, image],
        axis=2,
    ).astype(np.float32)

    overlay = rgb.copy()

    water = mask == 255
    ignore = mask == 128

    overlay[water] = (
        0,
        255,
        255,
    )

    overlay[ignore] = (
        255,
        0,
        255,
    )

    blended = (
        0.60 * rgb
        + 0.40 * overlay
    )

    return np.clip(
        blended,
        0,
        255,
    ).astype(np.uint8)


def audit_row(
    row: pd.Series,
    status_items: dict[str, Any],
    maximum_ignore_fraction: float,
    minimum_class_fraction: float,
) -> tuple[
    dict[str, Any],
    Image.Image | None,
]:
    sample_id = str(
        row["sample_id"]
    )

    state = status_items.get(
        sample_id,
        {},
    )

    saved = bool(
        state.get(
            "saved",
            False,
        )
    )

    completed = bool(
        state.get(
            "completed",
            False,
        )
    )

    image_path = resolve_path(
        row[
            "annotation_image_path"
        ]
    )

    mask_path = resolve_path(
        row[
            "manual_mask_path"
        ]
    )

    issues: list[str] = []
    warnings: list[str] = []

    image_exists = image_path.exists()
    mask_exists = mask_path.exists()

    if not image_exists:
        issues.append(
            "image_missing"
        )

    if not mask_exists:
        issues.append(
            "mask_missing"
        )

    if not saved:
        issues.append(
            "not_saved"
        )

    if not completed:
        issues.append(
            "not_completed"
        )

    image_shape = ""
    mask_shape = ""
    unique_values = ""
    invalid_value_count = 0
    land_fraction = 0.0
    water_fraction = 0.0
    ignore_fraction = 0.0
    valid_fraction = 0.0
    width = 0
    height = 0

    preview = None

    if image_exists and mask_exists:
        image = np.array(
            Image.open(
                image_path
            ).convert("L")
        )

        raw_mask = np.array(
            Image.open(
                mask_path
            ).convert("L")
        )

        image_shape = (
            f"{image.shape[0]}x"
            f"{image.shape[1]}"
        )

        mask_shape = (
            f"{raw_mask.shape[0]}x"
            f"{raw_mask.shape[1]}"
        )

        height, width = raw_mask.shape

        if image.shape != raw_mask.shape:
            issues.append(
                "shape_mismatch"
            )

        values, counts = np.unique(
            raw_mask,
            return_counts=True,
        )

        unique_values = "|".join(
            str(
                int(value)
            )
            for value in values
        )

        allowed = np.array(
            [0, 128, 255],
            dtype=np.uint8,
        )

        invalid = ~np.isin(
            raw_mask,
            allowed,
        )

        invalid_value_count = int(
            invalid.sum()
        )

        if invalid_value_count > 0:
            issues.append(
                "invalid_mask_values"
            )

        mask = quantize_preview_mask(
            raw_mask
        )

        pixel_count = max(
            int(mask.size),
            1,
        )

        land_fraction = float(
            (mask == 0).sum()
            / pixel_count
        )

        water_fraction = float(
            (mask == 255).sum()
            / pixel_count
        )

        ignore_fraction = float(
            (mask == 128).sum()
            / pixel_count
        )

        valid_fraction = float(
            1.0 - ignore_fraction
        )

        if (
            ignore_fraction
            > maximum_ignore_fraction
        ):
            issues.append(
                "too_much_ignore"
            )

        if (
            land_fraction
            < minimum_class_fraction
        ):
            warnings.append(
                "very_little_land"
            )

        if (
            water_fraction
            < minimum_class_fraction
        ):
            warnings.append(
                "very_little_water"
            )

        if (
            image.shape
            == raw_mask.shape
        ):
            original_rgb = np.stack(
                [
                    image,
                    image,
                    image,
                ],
                axis=2,
            )

            mask_rgb = colorize_mask(
                mask
            )

            overlay = create_overlay(
                image,
                mask,
            )

            canvas = np.concatenate(
                [
                    original_rgb,
                    mask_rgb,
                    overlay,
                ],
                axis=1,
            )

            preview = Image.fromarray(
                canvas
            )

    hard_pass = len(issues) == 0

    result = {
        "sample_id": sample_id,
        "coastal_group": str(
            row.get(
                "coastal_group",
                "",
            )
        ),
        "saved": saved,
        "completed": completed,
        "image_exists": image_exists,
        "mask_exists": mask_exists,
        "image_shape": image_shape,
        "mask_shape": mask_shape,
        "mask_width": width,
        "mask_height": height,
        "unique_values": (
            unique_values
        ),
        "invalid_value_count": (
            invalid_value_count
        ),
        "land_fraction": (
            land_fraction
        ),
        "water_fraction": (
            water_fraction
        ),
        "ignore_fraction": (
            ignore_fraction
        ),
        "valid_fraction": (
            valid_fraction
        ),
        "hard_pass": hard_pass,
        "issues": "|".join(
            issues
        ),
        "warnings": "|".join(
            warnings
        ),
        "annotation_image_path": (
            relative(
                image_path
            )
        ),
        "manual_mask_path": (
            relative(
                mask_path
            )
        ),
    }

    if "image_path" in row.index:
        result[
            "original_image_path"
        ] = str(
            row["image_path"]
        )

    return result, preview


def add_label_strip(
    image: Image.Image,
    text: str,
    width: int,
) -> Image.Image:
    ratio = width / image.width

    resized = image.resize(
        (
            width,
            max(
                1,
                int(
                    image.height
                    * ratio
                ),
            ),
        ),
        Image.Resampling.LANCZOS,
    )

    label_height = 28

    canvas = Image.new(
        "RGB",
        (
            width,
            resized.height
            + label_height,
        ),
        "black",
    )

    canvas.paste(
        resized,
        (0, label_height),
    )

    draw = ImageDraw.Draw(
        canvas
    )

    font = ImageFont.load_default()

    draw.text(
        (6, 8),
        text,
        fill="white",
        font=font,
    )

    return canvas


def create_contact_sheet(
    preview_entries: list[
        tuple[str, bool, Image.Image]
    ],
    output_path: Path,
    width: int,
) -> None:
    if not preview_entries:
        return

    rows = []

    for sample_id, hard_pass, preview in (
        preview_entries
    ):
        state = (
            "PASS"
            if hard_pass
            else "FAIL"
        )

        rows.append(
            add_label_strip(
                preview,
                f"{sample_id} | {state}",
                width,
            )
        )

    sheet = Image.new(
        "RGB",
        (
            width,
            sum(
                row.height
                for row in rows
            ),
        ),
        "black",
    )

    y = 0

    for row in rows:
        sheet.paste(
            row,
            (0, y),
        )
        y += row.height

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sheet.save(
        output_path,
        quality=90,
    )


def main() -> None:
    args = parse_args()

    queue_path = resolve_path(
        args.queue
    )

    status_path = resolve_path(
        args.status
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    AUDIT_CSV.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    queue = read_queue(
        queue_path
    )

    status = read_status(
        status_path
    )

    status_items = status.get(
        "items",
        {},
    )

    audit_rows: list[
        dict[str, Any]
    ] = []

    preview_entries: list[
        tuple[
            str,
            bool,
            Image.Image,
        ]
    ] = []

    for _, row in queue.iterrows():
        audit, preview = audit_row(
            row=row,
            status_items=status_items,
            maximum_ignore_fraction=(
                args.maximum_ignore_fraction
            ),
            minimum_class_fraction=(
                args.minimum_class_fraction
            ),
        )

        audit_rows.append(
            audit
        )

        if preview is not None:
            preview_entries.append(
                (
                    audit["sample_id"],
                    bool(
                        audit["hard_pass"]
                    ),
                    preview,
                )
            )

    audit_frame = pd.DataFrame(
        audit_rows
    )

    audit_frame.to_csv(
        AUDIT_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    training = audit_frame[
        audit_frame[
            "hard_pass"
        ].astype(bool)
    ].copy()

    training[
        "label_type"
    ] = "manual_land_water"

    training[
        "mask_semantics"
    ] = (
        "0=land|128=ignore|255=water"
    )

    training[
        "source_stage"
    ] = (
        "v06_dartis_water_active_batch"
    )

    training.to_csv(
        TRAINING_MANIFEST,
        index=False,
        encoding="utf-8-sig",
    )

    create_contact_sheet(
        preview_entries=(
            preview_entries
        ),
        output_path=(
            CONTACT_SHEET
        ),
        width=max(
            args.contact_width,
            600,
        ),
    )

    total = int(
        len(audit_frame)
    )

    completed_count = int(
        audit_frame[
            "completed"
        ].astype(bool).sum()
    )

    saved_count = int(
        audit_frame[
            "saved"
        ].astype(bool).sum()
    )

    passed_count = int(
        audit_frame[
            "hard_pass"
        ].astype(bool).sum()
    )

    failed_count = (
        total - passed_count
    )

    warning_count = int(
        audit_frame[
            "warnings"
        ].astype(str).ne("").sum()
    )

    issue_counts: dict[
        str,
        int,
    ] = {}

    for value in audit_frame[
        "issues"
    ].astype(str):
        if not value:
            continue

        for issue in value.split("|"):
            if issue:
                issue_counts[
                    issue
                ] = (
                    issue_counts.get(
                        issue,
                        0,
                    )
                    + 1
                )

    warning_counts: dict[
        str,
        int,
    ] = {}

    for value in audit_frame[
        "warnings"
    ].astype(str):
        if not value:
            continue

        for warning in value.split("|"):
            if warning:
                warning_counts[
                    warning
                ] = (
                    warning_counts.get(
                        warning,
                        0,
                    )
                    + 1
                )

    group_counts = (
        training.groupby(
            "coastal_group"
        )
        .size()
        .to_dict()
    )

    summary = {
        "stage": (
            "v06_dartis_water_annotation_audit"
        ),
        "queue_count": total,
        "saved_count": saved_count,
        "completed_count": (
            completed_count
        ),
        "hard_pass_count": (
            passed_count
        ),
        "hard_fail_count": (
            failed_count
        ),
        "warning_item_count": (
            warning_count
        ),
        "issue_counts": (
            issue_counts
        ),
        "warning_counts": (
            warning_counts
        ),
        "training_manifest_count": int(
            len(training)
        ),
        "training_group_counts": {
            str(key): int(value)
            for key, value
            in group_counts.items()
        },
        "criteria": {
            "allowed_mask_values": [
                0,
                128,
                255,
            ],
            "maximum_ignore_fraction": float(
                args.maximum_ignore_fraction
            ),
            "minimum_class_fraction_warning": float(
                args.minimum_class_fraction
            ),
            "requires_saved": True,
            "requires_completed": True,
            "requires_matching_shape": True,
        },
        "audit_csv": relative(
            AUDIT_CSV
        ),
        "training_manifest": (
            relative(
                TRAINING_MANIFEST
            )
        ),
        "contact_sheet": relative(
            CONTACT_SHEET
        ),
        "training_performed": False,
        "official_test_used": False,
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
        "DARTIS MANUEL MASKE DENETİMİ"
    )
    print("=" * 78)
    print(
        "Kuyruk:",
        total,
    )
    print(
        "Kaydedilmiş:",
        saved_count,
    )
    print(
        "Tamamlanmış:",
        completed_count,
    )
    print(
        "Hard PASS:",
        passed_count,
    )
    print(
        "Hard FAIL:",
        failed_count,
    )
    print(
        "Uyarılı görüntü:",
        warning_count,
    )
    print(
        "Eğitim manifesti:",
        len(training),
    )
    print(
        "Grup dağılımı:",
        {
            str(key): int(value)
            for key, value
            in group_counts.items()
        },
    )

    if issue_counts:
        print(
            "Hatalar:",
            issue_counts,
        )

    if warning_counts:
        print(
            "Uyarılar:",
            warning_counts,
        )

    print(
        "Audit CSV:",
        AUDIT_CSV.resolve(),
    )
    print(
        "Training manifest:",
        TRAINING_MANIFEST.resolve(),
    )
    print(
        "Görsel QA:",
        CONTACT_SHEET.resolve(),
    )
    print(
        "Özet:",
        SUMMARY_PATH.resolve(),
    )
    print()
    print(
        "Eğitim yapılmadı."
    )


if __name__ == "__main__":
    main()
