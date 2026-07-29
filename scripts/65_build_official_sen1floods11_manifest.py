from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = ROOT / "data" / "external" / "sen1floods11"

GROUPS = {
    "flood": {
        "image_dir": DATA_ROOT / "flood_sar",
        "mask_dir": DATA_ROOT / "flood_mask",
        "split_dir": DATA_ROOT / "flood_splits",
    },
    "permanent": {
        "image_dir": DATA_ROOT / "permanent_sar",
        "mask_dir": DATA_ROOT / "permanent_mask",
        "split_dir": DATA_ROOT / "permanent_splits",
    },
}

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "v06_sen1floods11_official_manifest"
)

MANIFEST_PATH = (
    ROOT
    / "data"
    / "metadata"
    / "v06_sen1floods11_official_pairs.csv"
)

DUPLICATE_REPORT_PATH = (
    OUTPUT_DIR
    / "duplicate_named_files.csv"
)

UNRESOLVED_REPORT_PATH = (
    OUTPUT_DIR
    / "unresolved_split_rows.csv"
)

SUMMARY_PATH = OUTPUT_DIR / "summary.json"

PREVIEW_DIR = OUTPUT_DIR / "alignment_previews"

CONTACT_SHEET_PATH = (
    OUTPUT_DIR
    / "alignment_contact_sheet.jpg"
)

DUPLICATE_SUFFIX_PATTERN = re.compile(
    r"\s+\(\d+\)$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sen1Floods11'in resmi split CSV dosyalarını kaynak "
            "kabul ederek SAR-mask çiftlerini kurar. Dosya adında "
            "(1) bulunan tarayıcı kopyalarını eğitimden dışlar."
        )
    )

    parser.add_argument(
        "--inspect-all",
        action="store_true",
        help=(
            "Tüm resmi çiftleri Rasterio ile açar; boyut, bant, "
            "etiket ve GeoTIFF metadata farklarını denetler."
        ),
    )

    parser.add_argument(
        "--preview-per-split",
        type=int,
        default=2,
    )

    return parser.parse_args()


def relative(path: Path) -> str:
    try:
        return str(
            path.resolve().relative_to(ROOT.resolve())
        ).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def is_duplicate_named(path: Path) -> bool:
    return bool(
        DUPLICATE_SUFFIX_PATTERN.search(
            path.stem
        )
    )


def find_rasters(
    directory: Path,
) -> list[Path]:
    paths: list[Path] = []

    if not directory.exists():
        return paths

    for pattern in ("*.tif", "*.tiff"):
        paths.extend(directory.rglob(pattern))

    return sorted(
        {
            path.resolve()
            for path in paths
            if path.is_file()
        }
    )


def build_file_index(
    paths: list[Path],
) -> tuple[
    dict[str, Path],
    list[dict[str, Any]],
]:
    index: dict[str, Path] = {}
    duplicate_rows: list[dict[str, Any]] = []

    for path in paths:
        if is_duplicate_named(path):
            duplicate_rows.append(
                {
                    "file_path": relative(path),
                    "file_name": path.name,
                    "size_bytes": int(
                        path.stat().st_size
                    ),
                    "reason": (
                        "Dosya adında tarayıcı kopyası "
                        "eki bulundu; resmi manifeste "
                        "alınmadı."
                    ),
                }
            )
            continue

        key = path.name.lower()

        if key in index:
            raise RuntimeError(
                "Aynı dosya adı birden fazla yerde "
                f"bulundu: {path.name}"
            )

        index[key] = path

    return index, duplicate_rows


def split_name_from_path(
    path: Path,
) -> str | None:
    name = path.stem.lower()

    if "train" in name:
        return "train"

    if (
        "valid" in name
        or re.search(
            r"(^|[_-])val([_-]|$)",
            name,
        )
    ):
        return "validation"

    if "test" in name:
        return "test"

    if (
        "bolivia" in name
        or "holdout" in name
    ):
        return "external_holdout"

    # permanent_water_data.csv gibi birleşik listeleri
    # tekrar çift üretmemesi için kullanmıyoruz.
    return None


def tif_tokens(
    row: list[str],
) -> list[str]:
    tokens: list[str] = []

    for cell in row:
        normalized = (
            str(cell)
            .strip()
            .strip("\"'")
            .replace("\\", "/")
        )

        if normalized.lower().endswith(
            (".tif", ".tiff")
        ):
            tokens.append(normalized)

    return tokens


def basename_key(value: str) -> str:
    return Path(
        value.replace("\\", "/")
    ).name.lower()


def classify_pair_tokens(
    tokens: list[str],
) -> tuple[str, str] | None:
    if len(tokens) < 2:
        return None

    image_candidates = []
    mask_candidates = []

    for token in tokens:
        lower = token.lower()

        if any(
            keyword in lower
            for keyword in (
                "s1hand",
                "s1perm",
                "sentinel",
            )
        ):
            image_candidates.append(token)

        if any(
            keyword in lower
            for keyword in (
                "labelhand",
                "jrcperm",
                "water_",
                "/water",
                "mask",
                "label",
            )
        ):
            mask_candidates.append(token)

    if (
        len(image_candidates) == 1
        and len(mask_candidates) == 1
    ):
        return (
            image_candidates[0],
            mask_candidates[0],
        )

    # Resmi CSV'lerde sıra genellikle image, mask.
    return tokens[0], tokens[1]


def sample_key(
    group_name: str,
    split_name: str,
    image_path: Path,
) -> str:
    digest = hashlib.sha256(
        (
            f"{group_name}|{split_name}|"
            f"{image_path.name.lower()}"
        ).encode("utf-8")
    ).hexdigest()[:12]

    return (
        f"{group_name}:"
        f"{split_name}:"
        f"{image_path.stem}:"
        f"{digest}"
    )


def affine_values(
    transform,
) -> np.ndarray:
    return np.array(
        [
            transform.a,
            transform.b,
            transform.c,
            transform.d,
            transform.e,
            transform.f,
        ],
        dtype=np.float64,
    )


def inspect_pair(
    image_path: Path,
    mask_path: Path,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "image_open_ok": False,
        "mask_open_ok": False,
        "same_shape": False,
        "expected_bands": False,
        "same_crs": False,
        "transform_exact": False,
        "transform_close_1e_9": False,
        "transform_close_1e_6": False,
        "transform_max_abs_difference": None,
        "image_width": None,
        "image_height": None,
        "image_bands": None,
        "mask_width": None,
        "mask_height": None,
        "mask_bands": None,
        "mask_nodata": None,
        "mask_unique_values": "",
        "mask_values_valid": False,
        "image_band1_min": None,
        "image_band1_max": None,
        "image_band2_min": None,
        "image_band2_max": None,
        "inspection_error": "",
    }

    try:
        with rasterio.open(
            image_path
        ) as image:
            result["image_open_ok"] = True
            result["image_width"] = int(
                image.width
            )
            result["image_height"] = int(
                image.height
            )
            result["image_bands"] = int(
                image.count
            )

            image_shape = (
                image.height,
                image.width,
            )
            image_crs = image.crs
            image_transform = image.transform

            for band_index in range(
                1,
                min(image.count, 2) + 1,
            ):
                band = image.read(
                    band_index,
                    masked=True,
                ).compressed()

                if band.size == 0:
                    continue

                result[
                    f"image_band{band_index}_min"
                ] = float(
                    np.nanmin(band)
                )
                result[
                    f"image_band{band_index}_max"
                ] = float(
                    np.nanmax(band)
                )

        with rasterio.open(
            mask_path
        ) as mask:
            result["mask_open_ok"] = True
            result["mask_width"] = int(
                mask.width
            )
            result["mask_height"] = int(
                mask.height
            )
            result["mask_bands"] = int(
                mask.count
            )

            mask_shape = (
                mask.height,
                mask.width,
            )
            mask_crs = mask.crs
            mask_transform = mask.transform
            result["mask_nodata"] = mask.nodata

            mask_values = mask.read(
                1,
                masked=True,
            ).compressed()

            unique_values = np.unique(
                mask_values
            )

            result[
                "mask_unique_values"
            ] = "|".join(
                str(float(value))
                for value in unique_values[:30]
            )

            valid_set = {
                -1.0,
                0.0,
                1.0,
                255.0,
            }

            result[
                "mask_values_valid"
            ] = all(
                float(value) in valid_set
                for value in unique_values
            )

        result["same_shape"] = (
            image_shape == mask_shape
        )

        result["expected_bands"] = (
            result["image_bands"] >= 1
            and result["mask_bands"] == 1
        )

        result["same_crs"] = (
            image_crs == mask_crs
        )

        image_affine = affine_values(
            image_transform
        )
        mask_affine = affine_values(
            mask_transform
        )

        difference = np.abs(
            image_affine - mask_affine
        )

        result[
            "transform_max_abs_difference"
        ] = float(
            difference.max()
        )

        result["transform_exact"] = bool(
            np.array_equal(
                image_affine,
                mask_affine,
            )
        )

        result[
            "transform_close_1e_9"
        ] = bool(
            np.allclose(
                image_affine,
                mask_affine,
                rtol=0.0,
                atol=1e-9,
            )
        )

        result[
            "transform_close_1e_6"
        ] = bool(
            np.allclose(
                image_affine,
                mask_affine,
                rtol=0.0,
                atol=1e-6,
            )
        )

    except Exception as error:
        result["inspection_error"] = (
            f"{type(error).__name__}: "
            f"{error}"
        )

    result["pixel_training_eligible"] = bool(
        result["image_open_ok"]
        and result["mask_open_ok"]
        and result["same_shape"]
        and result["expected_bands"]
        and result["mask_values_valid"]
    )

    return result


def normalize_sar_for_preview(
    array: np.ndarray,
) -> np.ndarray:
    array = array.astype(np.float32)

    finite = np.isfinite(array)

    if not finite.any():
        return np.zeros(
            array.shape,
            dtype=np.uint8,
        )

    values = array[finite]

    low = float(
        np.percentile(values, 2)
    )
    high = float(
        np.percentile(values, 98)
    )

    if high <= low:
        high = low + 1.0

    normalized = np.clip(
        (array - low)
        / (high - low),
        0.0,
        1.0,
    )

    normalized[
        ~finite
    ] = 0.0

    return (
        normalized * 255.0
    ).astype(np.uint8)


def create_preview(
    image_path: Path,
    mask_path: Path,
    output_path: Path,
) -> None:
    with rasterio.open(
        image_path
    ) as image:
        band_index = (
            2
            if image.count >= 2
            else 1
        )

        sar = image.read(
            band_index
        )

    with rasterio.open(
        mask_path
    ) as mask:
        label = mask.read(1)

    gray = normalize_sar_for_preview(
        sar
    )

    rgb = np.stack(
        [gray, gray, gray],
        axis=2,
    )

    water = label == 1
    invalid = (
        (label == -1)
        | (label == 255)
    )

    overlay = rgb.copy()

    overlay[water, 1] = 255
    overlay[water, 2] = 255

    overlay[invalid, 0] = 255
    overlay[invalid, 2] = 255

    mask_view = np.zeros_like(
        rgb
    )

    mask_view[water] = (
        255,
        255,
        255,
    )

    mask_view[invalid] = (
        255,
        0,
        255,
    )

    canvas = np.concatenate(
        [
            rgb,
            overlay,
            mask_view,
        ],
        axis=1,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Image.fromarray(
        canvas
    ).save(output_path)


def create_contact_sheet(
    preview_paths: list[Path],
) -> None:
    if not preview_paths:
        return

    columns = 2
    thumbnail_width = 768
    thumbnail_height = 256
    label_height = 24

    rows = math.ceil(
        len(preview_paths)
        / columns
    )

    canvas = Image.new(
        "RGB",
        (
            columns * thumbnail_width,
            rows
            * (
                thumbnail_height
                + label_height
            ),
        ),
        "black",
    )

    font = ImageFont.load_default()

    for index, path in enumerate(
        preview_paths
    ):
        image = Image.open(
            path
        ).convert("RGB")

        image.thumbnail(
            (
                thumbnail_width,
                thumbnail_height,
            ),
            Image.Resampling.LANCZOS,
        )

        cell = Image.new(
            "RGB",
            (
                thumbnail_width,
                thumbnail_height,
            ),
            "black",
        )

        cell.paste(
            image,
            (
                (
                    thumbnail_width
                    - image.width
                )
                // 2,
                (
                    thumbnail_height
                    - image.height
                )
                // 2,
            ),
        )

        column = index % columns
        row = index // columns

        x = column * thumbnail_width
        y = row * (
            thumbnail_height
            + label_height
        )

        canvas.paste(
            cell,
            (x, y),
        )

        draw = ImageDraw.Draw(
            canvas
        )

        draw.text(
            (
                x + 4,
                y
                + thumbnail_height
                + 5,
            ),
            path.stem,
            fill="white",
            font=font,
        )

    CONTACT_SHEET_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    canvas.save(
        CONTACT_SHEET_PATH,
        quality=90,
    )


def main() -> None:
    args = parse_args()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PREVIEW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_rows: list[dict[str, Any]] = []
    unresolved_rows: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []
    used_image_paths: dict[Path, str] = {}
    used_mask_paths: dict[Path, str] = {}
    split_file_summary: list[dict[str, Any]] = []

    for group_name, config in (
        GROUPS.items()
    ):
        image_paths = find_rasters(
            config["image_dir"]
        )
        mask_paths = find_rasters(
            config["mask_dir"]
        )

        (
            image_index,
            image_duplicates,
        ) = build_file_index(
            image_paths
        )

        (
            mask_index,
            mask_duplicates,
        ) = build_file_index(
            mask_paths
        )

        for duplicate in (
            image_duplicates
            + mask_duplicates
        ):
            duplicate["dataset_group"] = (
                group_name
            )
            duplicate_rows.append(
                duplicate
            )

        split_files = sorted(
            path
            for path in config[
                "split_dir"
            ].rglob("*")
            if path.is_file()
        )

        for split_file in split_files:
            split_name = (
                split_name_from_path(
                    split_file
                )
            )

            if split_name is None:
                split_file_summary.append(
                    {
                        "dataset_group": (
                            group_name
                        ),
                        "split_file": relative(
                            split_file
                        ),
                        "split": "ignored_aggregate",
                        "row_count": 0,
                        "resolved_count": 0,
                        "unresolved_count": 0,
                    }
                )
                continue

            row_count = 0
            resolved_count = 0
            unresolved_count = 0

            with split_file.open(
                "r",
                newline="",
                encoding="utf-8-sig",
                errors="ignore",
            ) as file:
                reader = csv.reader(
                    file
                )

                for csv_row_number, csv_row in enumerate(
                    reader,
                    start=1,
                ):
                    tokens = tif_tokens(
                        csv_row
                    )

                    pair_tokens = (
                        classify_pair_tokens(
                            tokens
                        )
                    )

                    if pair_tokens is None:
                        continue

                    row_count += 1

                    (
                        image_token,
                        mask_token,
                    ) = pair_tokens

                    image_path = image_index.get(
                        basename_key(
                            image_token
                        )
                    )

                    mask_path = mask_index.get(
                        basename_key(
                            mask_token
                        )
                    )

                    if (
                        image_path is None
                        or mask_path is None
                    ):
                        unresolved_count += 1

                        unresolved_rows.append(
                            {
                                "dataset_group": group_name,
                                "split": split_name,
                                "split_file": relative(
                                    split_file
                                ),
                                "csv_row_number": (
                                    csv_row_number
                                ),
                                "image_token": image_token,
                                "mask_token": mask_token,
                                "image_resolved": (
                                    image_path
                                    is not None
                                ),
                                "mask_resolved": (
                                    mask_path
                                    is not None
                                ),
                            }
                        )
                        continue

                    image_previous_split = (
                        used_image_paths.get(
                            image_path
                        )
                    )

                    mask_previous_split = (
                        used_mask_paths.get(
                            mask_path
                        )
                    )

                    split_conflict = bool(
                        (
                            image_previous_split
                            is not None
                            and image_previous_split
                            != split_name
                        )
                        or (
                            mask_previous_split
                            is not None
                            and mask_previous_split
                            != split_name
                        )
                    )

                    used_image_paths[
                        image_path
                    ] = split_name

                    used_mask_paths[
                        mask_path
                    ] = split_name

                    record = {
                        "sample_key": sample_key(
                            group_name,
                            split_name,
                            image_path,
                        ),
                        "dataset_group": (
                            group_name
                        ),
                        "split": split_name,
                        "split_file": relative(
                            split_file
                        ),
                        "split_csv_row": (
                            csv_row_number
                        ),
                        "image_path": relative(
                            image_path
                        ),
                        "mask_path": relative(
                            mask_path
                        ),
                        "split_conflict": (
                            split_conflict
                        ),
                    }

                    all_rows.append(record)
                    resolved_count += 1

            split_file_summary.append(
                {
                    "dataset_group": (
                        group_name
                    ),
                    "split_file": relative(
                        split_file
                    ),
                    "split": split_name,
                    "row_count": row_count,
                    "resolved_count": (
                        resolved_count
                    ),
                    "unresolved_count": (
                        unresolved_count
                    ),
                }
            )

    manifest = pd.DataFrame(
        all_rows
    )

    if manifest.empty:
        raise RuntimeError(
            "Resmi split CSV dosyalarından "
            "hiçbir çift çözülemedi."
        )

    manifest = manifest.drop_duplicates(
        subset=[
            "dataset_group",
            "split",
            "image_path",
            "mask_path",
        ]
    ).reset_index(drop=True)

    inspection_columns: set[str] = set()

    inspect_indices: set[int]

    if args.inspect_all:
        inspect_indices = set(
            manifest.index
        )
    else:
        inspect_indices = set()

        for (
            group_name,
            split_name,
        ), group in manifest.groupby(
            [
                "dataset_group",
                "split",
            ]
        ):
            inspect_indices.update(
                group.head(10).index
            )

    for index in sorted(
        inspect_indices
    ):
        row = manifest.loc[index]

        inspection = inspect_pair(
            ROOT / row["image_path"],
            ROOT / row["mask_path"],
        )

        for key, value in (
            inspection.items()
        ):
            manifest.loc[
                index,
                key,
            ] = value
            inspection_columns.add(key)

    for column in sorted(
        inspection_columns
    ):
        if column not in manifest:
            manifest[column] = ""

    manifest = manifest.sort_values(
        [
            "dataset_group",
            "split",
            "sample_key",
        ]
    ).reset_index(drop=True)

    manifest.to_csv(
        MANIFEST_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        duplicate_rows
    ).to_csv(
        DUPLICATE_REPORT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        unresolved_rows
    ).to_csv(
        UNRESOLVED_REPORT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    preview_paths: list[Path] = []

    for (
        group_name,
        split_name,
    ), group in manifest.groupby(
        [
            "dataset_group",
            "split",
        ]
    ):
        for _, row in group.head(
            max(
                args.preview_per_split,
                0,
            )
        ).iterrows():
            preview_path = (
                PREVIEW_DIR
                / (
                    f"{group_name}"
                    f"__{split_name}"
                    f"__{Path(row['image_path']).stem}.png"
                )
            )

            create_preview(
                ROOT / row["image_path"],
                ROOT / row["mask_path"],
                preview_path,
            )

            preview_paths.append(
                preview_path
            )

    create_contact_sheet(
        preview_paths
    )

    inspected = manifest[
        manifest[
            "image_open_ok"
        ].astype(str).ne("")
    ].copy()

    def count_false(
        column: str,
    ) -> int:
        return int(
            inspected[
                column
            ].astype(str).str.lower().eq(
                "false"
            ).sum()
        )

    split_counts = (
        manifest.groupby(
            [
                "dataset_group",
                "split",
            ]
        )
        .size()
        .reset_index(
            name="pair_count"
        )
        .to_dict(
            orient="records"
        )
    )

    summary = {
        "stage": (
            "v06_sen1floods11_official_manifest"
        ),
        "total_official_pairs": int(
            len(manifest)
        ),
        "split_counts": split_counts,
        "duplicate_named_file_count": int(
            len(duplicate_rows)
        ),
        "unresolved_split_row_count": int(
            len(unresolved_rows)
        ),
        "split_conflict_count": int(
            manifest[
                "split_conflict"
            ].astype(bool).sum()
        ),
        "inspected_pair_count": int(
            len(inspected)
        ),
        "open_failure_count": (
            count_false(
                "image_open_ok"
            )
            + count_false(
                "mask_open_ok"
            )
        ),
        "shape_failure_count": (
            count_false(
                "same_shape"
            )
        ),
        "band_failure_count": (
            count_false(
                "expected_bands"
            )
        ),
        "mask_value_failure_count": (
            count_false(
                "mask_values_valid"
            )
        ),
        "pixel_training_ineligible_count": (
            count_false(
                "pixel_training_eligible"
            )
        ),
        "crs_failure_count": (
            count_false(
                "same_crs"
            )
        ),
        "transform_exact_mismatch_count": (
            count_false(
                "transform_exact"
            )
        ),
        "transform_1e_9_mismatch_count": (
            count_false(
                "transform_close_1e_9"
            )
        ),
        "transform_1e_6_mismatch_count": (
            count_false(
                "transform_close_1e_6"
            )
        ),
        "official_pairing_note": (
            "Eğitim eşlemesi resmi split CSV satırlarına "
            "dayanır. Exact GeoTIFF transform eşitliği, "
            "piksel dizisiyle eğitim için zorunlu koşul "
            "olarak kullanılmaz."
        ),
        "training_performed": False,
        "locked_test_evaluated": False,
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
        "SEN1FLOODS11 OFFICIAL MANIFEST"
    )
    print("=" * 78)

    for entry in split_counts:
        print(
            f"{entry['dataset_group']:10s} "
            f"{entry['split']:18s}: "
            f"{entry['pair_count']}"
        )

    print()
    print(
        "Toplam resmi çift:",
        len(manifest),
    )
    print(
        "Dışlanan (1) kopyaları:",
        len(duplicate_rows),
    )
    print(
        "Çözülemeyen resmi satır:",
        len(unresolved_rows),
    )
    print(
        "Split çakışması:",
        int(
            manifest[
                "split_conflict"
            ].astype(bool).sum()
        ),
    )
    print(
        "İncelenen çift:",
        len(inspected),
    )
    print(
        "Piksel eğitime uygun olmayan:",
        summary[
            "pixel_training_ineligible_count"
        ],
    )
    print(
        "Exact transform uyuşmazlığı:",
        summary[
            "transform_exact_mismatch_count"
        ],
    )
    print(
        "1e-6 tolerans transform uyuşmazlığı:",
        summary[
            "transform_1e_6_mismatch_count"
        ],
    )
    print()
    print(
        "Manifest:",
        MANIFEST_PATH.resolve(),
    )
    print(
        "Özet:",
        SUMMARY_PATH.resolve(),
    )
    print(
        "Hızlı hizalama kontrolü:",
        CONTACT_SHEET_PATH.resolve(),
    )
    print()
    print(
        "Eğitim yapılmadı."
    )
    print(
        "Resmi test splitinde model değerlendirilmedi."
    )


if __name__ == "__main__":
    main()
