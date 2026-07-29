from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]

NEGATIVE_MANIFEST = (
    ROOT
    / "data"
    / "metadata"
    / "v07_canonical_hard_negative_candidates.csv"
)

NEGATIVE_SCENE_REPORT = (
    ROOT
    / "data"
    / "metadata"
    / "v07_canonical_hard_negative_scene_report.csv"
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
    / "v07_verifier_dataset_build"
)

COMBINED_MANIFEST = (
    ROOT
    / "data"
    / "metadata"
    / "v07_verifier_binary_dataset.csv"
)

FRESH_HOLDOUT_MANIFEST = (
    ROOT
    / "data"
    / "metadata"
    / "v07_fresh_end_to_end_holdout_scenes.csv"
)

DISCOVERY_AUDIT = (
    OUTPUT_DIR
    / "positive_manifest_discovery_audit.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "summary.json"
)

CONTACT_SHEET_PATH = (
    OUTPUT_DIR
    / "training_dataset_contact_sheet.jpg"
)

ALLOWED_POSITIVE_MANIFESTS = {
    "data/metadata/dartis_positive_verifier_crops.csv",
}

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
}

SCENE_PATTERNS = [
    re.compile(
        r"(nc-\d{4}-\d{2}-\d{6})",
        re.IGNORECASE,
    ),
    re.compile(
        r"(nw-\d{4}-\d{2}-\d{6})",
        re.IGNORECASE,
    ),
    re.compile(
        r"(oc-\d{4})",
        re.IGNORECASE,
    ),
    re.compile(
        r"(ow-\d{4})",
        re.IGNORECASE,
    ),
]

POSITIVE_VALUES = {
    "1",
    "true",
    "yes",
    "positive",
    "oil",
    "oil_candidate",
    "confirmed_oil",
    "confirmed oil",
    "petrol",
    "petrol_var",
    "oil spill",
    "oil_spill",
}

NEGATIVE_VALUES = {
    "0",
    "false",
    "no",
    "negative",
    "non_oil",
    "no_oil",
    "look_alike",
    "look-alike",
    "clean",
    "petrol_yok",
}

PATH_COLUMN_PRIORITY = [
    "crop_path",
    "candidate_crop_path",
    "patch_path",
    "roi_path",
    "image_path",
    "candidate_path",
    "path",
    "file_path",
]

LABEL_COLUMN_PRIORITY = [
    "ground_truth_label",
    "true_label",
    "target",
    "label",
    "class",
    "is_oil",
    "oil_label",
    "binary_label",
]

SPLIT_COLUMN_PRIORITY = [
    "split_role",
    "split",
    "subset",
    "partition",
    "fold_role",
]

SCENE_COLUMN_PRIORITY = [
    "sample_id",
    "scene_id",
    "source_scene_id",
    "image_id",
    "source_id",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mevcut doğrulanmış petrol verifier eğitim crop'larını "
            "bulur; yeni kanonik LOOK_ALIKE hard-negative crop'larıyla "
            "birleştirir. UNCERTAIN ayrı bir sahte sınıf olarak "
            "üretilmez; sonraki aşamada iki eşikli abstention bandı "
            "olarak kalibre edilir."
        )
    )

    parser.add_argument(
        "--holdout-per-group",
        type=int,
        default=20,
        help=(
            "Hiç açılmadan ayrılacak yeni nihai sahne sayısı: "
            "nc, nw, oc ve ow gruplarının her biri için."
        ),
    )

    parser.add_argument(
        "--minimum-positive-train",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--minimum-positive-calibration",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260729,
    )

    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(
        str(value).strip().replace(
            "\\",
            "/",
        )
    )

    if not path.is_absolute():
        path = ROOT / path

    return path.resolve()


def relative(path: Path) -> str:
    try:
        return str(
            path.resolve().relative_to(
                ROOT.resolve()
            )
        ).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def scene_id_from_text(
    value: Any,
) -> str | None:
    if value is None:
        return None

    text = str(value)

    for pattern in SCENE_PATTERNS:
        match = pattern.search(
            text
        )

        if match is not None:
            return match.group(1).lower()

    return None


def group_from_scene_id(
    scene_id: str | None,
) -> str | None:
    if scene_id is None:
        return None

    prefix = scene_id[:2].lower()

    if prefix in {
        "nc",
        "nw",
        "oc",
        "ow",
    }:
        return prefix

    return None


def deterministic_bucket(
    value: str,
    modulo: int = 100,
) -> int:
    digest = hashlib.sha256(
        value.encode("utf-8")
    ).digest()

    return int.from_bytes(
        digest[:8],
        "big",
    ) % modulo


def normalize_split(
    value: Any,
    scene_id: str,
) -> str:
    text = (
        ""
        if value is None
        else str(value).strip().lower()
    )

    if any(
        token in text
        for token in (
            "locked",
            "test",
            "external",
            "holdout",
        )
    ):
        return "excluded_existing_test"

    if any(
        token in text
        for token in (
            "calib",
            "valid",
            "val",
            "dev",
        )
    ):
        return "calibration"

    if "train" in text:
        return "train"

    bucket = deterministic_bucket(
        scene_id
    )

    return (
        "train"
        if bucket < 85
        else "calibration"
    )


def normalize_label(
    value: Any,
) -> int | None:
    if value is None:
        return None

    if isinstance(
        value,
        (bool, np.bool_),
    ):
        return int(value)

    if isinstance(
        value,
        (
            int,
            float,
            np.integer,
            np.floating,
        ),
    ):
        if isinstance(
            value,
            float,
        ) and np.isnan(value):
            return None

        return (
            1
            if float(value) >= 0.5
            else 0
        )

    text = str(value).strip().lower()

    if text in POSITIVE_VALUES:
        return 1

    if text in NEGATIVE_VALUES:
        return 0

    try:
        numeric = float(text)

        return (
            1
            if numeric >= 0.5
            else 0
        )
    except ValueError:
        return None


def choose_column(
    columns: list[str],
    priority: list[str],
) -> str | None:
    lower_lookup = {
        str(column).lower(): str(column)
        for column in columns
    }

    for candidate in priority:
        if candidate in lower_lookup:
            return lower_lookup[
                candidate
            ]

    return None


def find_path_column(
    columns: list[str],
) -> str | None:
    exact = choose_column(
        columns,
        PATH_COLUMN_PRIORITY,
    )

    if exact is not None:
        return exact

    for column in columns:
        lower = str(column).lower()

        if "mask" in lower:
            continue

        if any(
            token in lower
            for token in (
                "crop",
                "patch",
                "candidate",
                "roi",
            )
        ) and any(
            token in lower
            for token in (
                "path",
                "file",
                "image",
            )
        ):
            return str(column)

    return None


def find_label_column(
    columns: list[str],
) -> str | None:
    exact = choose_column(
        columns,
        LABEL_COLUMN_PRIORITY,
    )

    if exact is not None:
        return exact

    for column in columns:
        lower = str(column).lower()

        if (
            "ground" in lower
            and "label" in lower
        ):
            return str(column)

    return None


def manifest_score(
    csv_path: Path,
    path_column: str | None,
    label_column: str | None,
    split_column: str | None,
) -> int:
    name = csv_path.name.lower()
    score = 0

    if path_column is not None:
        lower = path_column.lower()

        score += (
            5
            if any(
                token in lower
                for token in (
                    "crop",
                    "patch",
                    "roi",
                    "candidate",
                )
            )
            else 2
        )

    if label_column is not None:
        score += 5

    if split_column is not None:
        score += 2

    if any(
        token in name
        for token in (
            "verifier",
            "candidate",
            "crop",
            "training",
            "dataset",
            "split",
        )
    ):
        score += 3

    if any(
        token in name
        for token in (
            "prediction",
            "result",
            "evaluation",
            "metric",
            "locked",
            "test",
        )
    ):
        score -= 5

    if csv_path.resolve() == (
        NEGATIVE_MANIFEST.resolve()
    ):
        score = -100

    return score


def validate_crop(
    path: Path,
) -> tuple[
    bool,
    dict[str, Any],
]:
    try:
        image = Image.open(
            path
        ).convert("L")

        array = np.asarray(
            image,
            dtype=np.uint8,
        )

        unique_count = int(
            np.unique(array).size
        )

        standard_deviation = float(
            array.std()
        )

        valid = (
            image.width >= 24
            and image.height >= 24
            and unique_count >= 16
            and standard_deviation >= 2.0
        )

        return (
            valid,
            {
                "width": image.width,
                "height": image.height,
                "unique_values": unique_count,
                "standard_deviation": standard_deviation,
            },
        )
    except Exception as error:
        return (
            False,
            {
                "error": repr(error),
            },
        )


def discover_positive_rows() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    search_roots = [
        ROOT / "data" / "metadata",
        ROOT / "outputs",
        ROOT / "reports",
    ]

    csv_paths: list[Path] = []

    for search_root in search_roots:
        if search_root.exists():
            csv_paths.extend(
                search_root.rglob(
                    "*.csv"
                )
            )

    positive_rows: list[
        dict[str, Any]
    ] = []

    audit_rows: list[
        dict[str, Any]
    ] = []

    for csv_path in sorted(
        set(
            path.resolve()
            for path in csv_paths
        )
    ):
        try:
            frame = pd.read_csv(
                csv_path,
                encoding="utf-8-sig",
                low_memory=False,
            )
        except Exception as error:
            audit_rows.append(
                {
                    "manifest": (
                        relative(csv_path)
                    ),
                    "status": "READ_ERROR",
                    "error": repr(error),
                }
            )
            continue

        columns = [
            str(column)
            for column in frame.columns
        ]

        path_column = find_path_column(
            columns
        )

        label_column = find_label_column(
            columns
        )

        split_column = choose_column(
            columns,
            SPLIT_COLUMN_PRIORITY,
        )

        scene_column = choose_column(
            columns,
            SCENE_COLUMN_PRIORITY,
        )

        score = manifest_score(
            csv_path,
            path_column,
            label_column,
            split_column,
        )

        audit = {
            "manifest": relative(
                csv_path
            ),
            "row_count": int(
                len(frame)
            ),
            "path_column": path_column,
            "label_column": label_column,
            "split_column": split_column,
            "scene_column": scene_column,
            "score": score,
            "positive_rows_found": 0,
            "valid_positive_crops": 0,
            "status": (
                "CANDIDATE"
                if score >= 7
                else "SKIPPED"
            ),
        }

        if (
            score < 7
            or path_column is None
            or label_column is None
        ):
            audit_rows.append(
                audit
            )
            continue

        for row in frame.to_dict(
            orient="records"
        ):
            label = normalize_label(
                row.get(
                    label_column
                )
            )

            if label != 1:
                continue

            audit[
                "positive_rows_found"
            ] += 1

            raw_path = row.get(
                path_column
            )

            if raw_path is None:
                continue

            try:
                crop_path = resolve_path(
                    raw_path
                )
            except Exception:
                continue

            if (
                not crop_path.exists()
                or crop_path.suffix.lower()
                not in IMAGE_EXTENSIONS
            ):
                continue

            scene_value = (
                row.get(
                    scene_column
                )
                if scene_column
                else None
            )

            scene_id = (
                scene_id_from_text(
                    scene_value
                )
                or scene_id_from_text(
                    crop_path
                )
            )

            if scene_id is None:
                continue

            group = group_from_scene_id(
                scene_id
            )

            if group not in {
                "oc",
                "ow",
            }:
                continue

            valid, diagnostics = (
                validate_crop(
                    crop_path
                )
            )

            if not valid:
                continue

            split_value = (
                row.get(
                    split_column
                )
                if split_column
                else None
            )

            split_role = normalize_split(
                split_value,
                scene_id,
            )

            positive_rows.append(
                {
                    "candidate_id": (
                        f"OIL::{scene_id}::"
                        f"{crop_path.stem}"
                    ),
                    "scene_id": scene_id,
                    "group": group,
                    "label": 1,
                    "class_name": (
                        "CONFIRMED_OIL"
                    ),
                    "split_role": (
                        split_role
                    ),
                    "crop_path": (
                        relative(
                            crop_path
                        )
                    ),
                    "source_manifest": (
                        relative(
                            csv_path
                        )
                    ),
                    "source_row_label": str(
                        row.get(
                            label_column
                        )
                    ),
                    **diagnostics,
                }
            )

            audit[
                "valid_positive_crops"
            ] += 1

        audit_rows.append(
            audit
        )

    positive_frame = pd.DataFrame(
        positive_rows
    )

    if not positive_frame.empty:
        positive_frame = (
            positive_frame.sort_values(
                [
                    "scene_id",
                    "crop_path",
                ]
            )
            .drop_duplicates(
                subset=[
                    "crop_path",
                ],
                keep="first",
            )
            .reset_index(
                drop=True
            )
        )

    audit_frame = pd.DataFrame(
        audit_rows
    )

    return (
        positive_frame,
        audit_frame,
    )


def load_negative_rows() -> pd.DataFrame:
    if not NEGATIVE_MANIFEST.exists():
        raise FileNotFoundError(
            f"Negatif manifest bulunamadı: {NEGATIVE_MANIFEST}"
        )

    frame = pd.read_csv(
        NEGATIVE_MANIFEST,
        encoding="utf-8-sig",
        low_memory=False,
    )

    required = {
        "candidate_id",
        "sample_id",
        "group",
        "split_role",
        "crop_path",
    }

    missing = required - set(
        frame.columns
    )

    if missing:
        raise RuntimeError(
            "Negatif manifestte eksik sütunlar: "
            f"{sorted(missing)}"
        )

    rows = []

    for row in frame.to_dict(
        orient="records"
    ):
        scene_id = (
            scene_id_from_text(
                row.get(
                    "sample_id"
                )
            )
            or scene_id_from_text(
                row.get(
                    "source_image_path"
                )
            )
            or scene_id_from_text(
                row.get(
                    "crop_path"
                )
            )
        )

        if scene_id is None:
            continue

        crop_path = resolve_path(
            row["crop_path"]
        )

        if not crop_path.exists():
            continue

        split_text = str(
            row.get(
                "split_role",
                "",
            )
        ).lower()

        if "train" in split_text:
            split_role = "train"
        elif any(
            token in split_text
            for token in (
                "calib",
                "val",
            )
        ):
            split_role = "calibration"
        else:
            split_role = (
                "excluded_seen_audit"
            )

        rows.append(
            {
                "candidate_id": (
                    f"LOOK::{row['candidate_id']}"
                ),
                "scene_id": scene_id,
                "group": str(
                    row["group"]
                ).lower(),
                "label": 0,
                "class_name": (
                    "LOOK_ALIKE"
                ),
                "split_role": split_role,
                "crop_path": relative(
                    crop_path
                ),
                "source_manifest": relative(
                    NEGATIVE_MANIFEST
                ),
                "source_row_label": (
                    "DARTIS_NO_OIL"
                ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    if result.empty:
        raise RuntimeError(
            "Kullanılabilir kanonik LOOK_ALIKE crop bulunamadı."
        )

    return (
        result.sort_values(
            [
                "scene_id",
                "crop_path",
            ]
        )
        .drop_duplicates(
            subset=[
                "crop_path",
            ],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )


def collect_reserved_scene_ids() -> set[str]:
    reserved: set[str] = set()

    metadata_root = (
        ROOT
        / "data"
        / "metadata"
    )

    if not metadata_root.exists():
        return reserved

    for csv_path in metadata_root.rglob(
        "*.csv"
    ):
        name = csv_path.name.lower()

        if not any(
            token in name
            for token in (
                "test",
                "locked",
                "external",
                "holdout",
                "validation",
                "val_",
            )
        ):
            continue

        try:
            frame = pd.read_csv(
                csv_path,
                encoding="utf-8-sig",
                low_memory=False,
            )
        except Exception:
            continue

        for column in frame.columns:
            for value in frame[
                column
            ].dropna():
                scene_id = scene_id_from_text(
                    value
                )

                if scene_id is not None:
                    reserved.add(
                        scene_id
                    )

    return reserved


def discover_raw_scenes() -> pd.DataFrame:
    rows = []

    for group in (
        "nc",
        "nw",
        "oc",
        "ow",
    ):
        group_dir = (
            DARTIS_RAW_ROOT
            / group
        )

        if not group_dir.exists():
            continue

        for path in sorted(
            group_dir.glob("*.jpg")
        ):
            scene_id = scene_id_from_text(
                path.stem
            )

            if scene_id is None:
                continue

            rows.append(
                {
                    "scene_id": scene_id,
                    "group": group,
                    "expected_scene_label": (
                        "NO_OIL"
                        if group
                        in {
                            "nc",
                            "nw",
                        }
                        else "OIL"
                    ),
                    "image_path": relative(
                        path
                    ),
                }
            )

    return pd.DataFrame(
        rows
    ).drop_duplicates(
        subset=[
            "scene_id",
        ],
        keep="first",
    )


def build_fresh_holdout(
    used_scene_ids: set[str],
    reserved_scene_ids: set[str],
    per_group: int,
) -> pd.DataFrame:
    raw = discover_raw_scenes()

    if raw.empty:
        raise RuntimeError(
            "DARTIS raw sahneleri bulunamadı."
        )

    excluded = (
        used_scene_ids
        | reserved_scene_ids
    )

    available = raw[
        ~raw[
            "scene_id"
        ].isin(
            excluded
        )
    ].copy()

    selected_parts = []

    for group in (
        "nc",
        "nw",
        "oc",
        "ow",
    ):
        group_frame = available[
            available[
                "group"
            ].eq(
                group
            )
        ].copy()

        group_frame[
            "selection_hash"
        ] = group_frame[
            "scene_id"
        ].map(
            lambda value: (
                deterministic_bucket(
                    f"fresh-holdout::{value}",
                    modulo=10**9,
                )
            )
        )

        group_frame = group_frame.sort_values(
            [
                "selection_hash",
                "scene_id",
            ]
        ).head(
            per_group
        )

        available_count = int(
            len(group_frame)
        )

        if available_count < per_group:
            print(
                f"UYARI: {group} için istenen {per_group} yeni holdout "
                f"sahnesinden yalnız {available_count} tane bulundu. "
                "Mevcut sahneler yeniden kullanılmayacak; bulunan kadar "
                "ayrılacak ve eksiklik rapora yazılacak."
            )

        group_frame = group_frame.head(
            min(
                per_group,
                available_count,
            )
        )

        selected_parts.append(
            group_frame
        )

    non_empty_parts = [
        part
        for part in selected_parts
        if not part.empty
    ]

    if non_empty_parts:
        holdout = pd.concat(
            non_empty_parts,
            ignore_index=True,
        )
    else:
        holdout = raw.iloc[
            0:0
        ].copy()

    holdout[
        "split_role"
    ] = (
        "fresh_locked_end_to_end"
    )

    holdout[
        "image_opened"
    ] = False

    holdout[
        "model_inference_run"
    ] = False

    return holdout[
        [
            "scene_id",
            "group",
            "expected_scene_label",
            "split_role",
            "image_path",
            "image_opened",
            "model_inference_run",
        ]
    ]


def enforce_positive_scene_split(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    if frame.empty:
        return frame

    result = frame.copy()

    split_by_scene: dict[str, str] = {}

    for scene_id, group in result.groupby(
        "scene_id"
    ):
        existing = set(
            group[
                "split_role"
            ].astype(str)
        )

        if "excluded_existing_test" in existing:
            split_role = (
                "excluded_existing_test"
            )
        else:
            bucket = deterministic_bucket(
                f"v07-verifier-scene-split::{scene_id}"
            )

            split_role = (
                "train"
                if bucket < 85
                else "calibration"
            )

        split_by_scene[
            str(scene_id)
        ] = split_role

    result[
        "split_role"
    ] = result[
        "scene_id"
    ].astype(str).map(
        split_by_scene
    )

    leakage = (
        result.groupby(
            "scene_id"
        )[
            "split_role"
        ]
        .nunique()
    )

    if int(
        (leakage > 1).sum()
    ) != 0:
        raise RuntimeError(
            "Pozitif crop'larda sahne bazlı split leakage oluştu."
        )

    return result


def enhance_image(
    image: Image.Image,
) -> Image.Image:
    array = np.asarray(
        image.convert("L"),
        dtype=np.float32,
    )

    low = float(
        np.percentile(array, 2)
    )

    high = float(
        np.percentile(array, 98)
    )

    if high <= low:
        enhanced = array
    else:
        enhanced = (
            np.clip(
                (array - low)
                / (high - low),
                0.0,
                1.0,
            )
            * 255.0
        )

    return Image.fromarray(
        enhanced.astype(np.uint8)
    )


def create_contact_sheet(
    frame: pd.DataFrame,
) -> None:
    visible = frame[
        frame[
            "split_role"
        ].isin(
            [
                "train",
                "calibration",
            ]
        )
    ].copy()

    if visible.empty:
        return

    samples = []

    for class_name in (
        "CONFIRMED_OIL",
        "LOOK_ALIKE",
    ):
        class_frame = visible[
            visible[
                "class_name"
            ].eq(
                class_name
            )
        ].copy()

        class_frame[
            "order_hash"
        ] = class_frame[
            "candidate_id"
        ].map(
            lambda value: (
                deterministic_bucket(
                    f"preview::{value}",
                    modulo=10**9,
                )
            )
        )

        samples.append(
            class_frame.sort_values(
                "order_hash"
            ).head(32)
        )

    selected = pd.concat(
        samples,
        ignore_index=True,
    )

    tile_width = 240
    image_height = 190
    header_height = 48
    tile_height = (
        image_height
        + header_height
    )

    columns = 4
    rows = int(
        np.ceil(
            len(selected)
            / columns
        )
    )

    sheet = Image.new(
        "RGB",
        (
            columns * tile_width,
            rows * tile_height,
        ),
        "black",
    )

    font = ImageFont.load_default()

    for index, row in enumerate(
        selected.itertuples()
    ):
        crop_path = resolve_path(
            row.crop_path
        )

        image = enhance_image(
            Image.open(
                crop_path
            )
        ).convert("RGB")

        image.thumbnail(
            (
                tile_width,
                image_height,
            ),
            Image.Resampling.LANCZOS,
        )

        tile = Image.new(
            "RGB",
            (
                tile_width,
                tile_height,
            ),
            "black",
        )

        offset_x = (
            tile_width
            - image.width
        ) // 2

        offset_y = (
            image_height
            - image.height
        ) // 2

        tile.paste(
            image,
            (
                offset_x,
                header_height
                + offset_y,
            ),
        )

        draw = ImageDraw.Draw(
            tile
        )

        draw.text(
            (5, 5),
            (
                f"{row.class_name} | "
                f"{row.split_role}"
            ),
            fill=(
                "white"
            ),
            font=font,
        )

        draw.text(
            (5, 24),
            str(
                row.scene_id
            )[:38],
            fill=(
                "white"
            ),
            font=font,
        )

        sheet.paste(
            tile,
            (
                (
                    index
                    % columns
                )
                * tile_width,
                (
                    index
                    // columns
                )
                * tile_height,
            ),
        )

    CONTACT_SHEET_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sheet.save(
        CONTACT_SHEET_PATH,
        quality=92,
    )


def main() -> None:
    args = parse_args()

    negative_frame = (
        load_negative_rows()
    )

    (
        positive_frame,
        audit_frame,
    ) = discover_positive_rows()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit_frame.to_csv(
        DISCOVERY_AUDIT,
        index=False,
        encoding="utf-8-sig",
    )

    if positive_frame.empty:
        raise RuntimeError(
            "Eski verifier eğitiminden doğrulanmış OIL crop "
            "bulunamadı. Audit dosyasını kontrol edin: "
            f"{DISCOVERY_AUDIT}"
        )

    discovered_positive_count = int(
        len(
            positive_frame
        )
    )

    positive_frame = positive_frame[
        positive_frame[
            "source_manifest"
        ].isin(
            ALLOWED_POSITIVE_MANIFESTS
        )
    ].copy()

    removed_non_verifier_positive_count = int(
        discovered_positive_count
        - len(
            positive_frame
        )
    )

    if positive_frame.empty:
        raise RuntimeError(
            "Whitelist sonrasında doğrulanmış verifier OIL crop "
            "kalmadı. İzin verilen kaynaklar: "
            f"{sorted(ALLOWED_POSITIVE_MANIFESTS)}"
        )

    positive_frame = (
        enforce_positive_scene_split(
            positive_frame
        )
    )

    print(
        "Pozitif kaynak whitelist:",
        sorted(
            ALLOWED_POSITIVE_MANIFESTS
        ),
    )

    print(
        "Verifier dışı pozitif kayıt çıkarıldı:",
        removed_non_verifier_positive_count,
    )

    positive_train_count = int(
        positive_frame[
            positive_frame[
                "split_role"
            ].eq(
                "train"
            )
        ].shape[0]
    )

    positive_calibration_count = int(
        positive_frame[
            positive_frame[
                "split_role"
            ].eq(
                "calibration"
            )
        ].shape[0]
    )

    if (
        positive_train_count
        < args.minimum_positive_train
    ):
        raise RuntimeError(
            "Yeterli pozitif train crop bulunamadı. "
            f"Beklenen en az {args.minimum_positive_train}, "
            f"bulunan {positive_train_count}. "
            f"Audit: {DISCOVERY_AUDIT}"
        )

    if (
        positive_calibration_count
        < args.minimum_positive_calibration
    ):
        raise RuntimeError(
            "Yeterli pozitif calibration crop bulunamadı. "
            f"Beklenen en az {args.minimum_positive_calibration}, "
            f"bulunan {positive_calibration_count}. "
            f"Audit: {DISCOVERY_AUDIT}"
        )

    combined = pd.concat(
        [
            positive_frame,
            negative_frame,
        ],
        ignore_index=True,
        sort=False,
    )

    combined = (
        combined.drop_duplicates(
            subset=[
                "candidate_id",
            ],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    COMBINED_MANIFEST.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined.to_csv(
        COMBINED_MANIFEST,
        index=False,
        encoding="utf-8-sig",
    )

    used_scene_ids = set(
        combined[
            "scene_id"
        ].dropna().astype(str)
    )

    if NEGATIVE_SCENE_REPORT.exists():
        scene_report = pd.read_csv(
            NEGATIVE_SCENE_REPORT,
            encoding="utf-8-sig",
            low_memory=False,
        )

        for value in scene_report.get(
            "sample_id",
            pd.Series(
                dtype=str
            ),
        ).dropna():
            scene_id = scene_id_from_text(
                value
            )

            if scene_id is not None:
                used_scene_ids.add(
                    scene_id
                )

    reserved_scene_ids = (
        collect_reserved_scene_ids()
    )

    holdout = build_fresh_holdout(
        used_scene_ids=used_scene_ids,
        reserved_scene_ids=(
            reserved_scene_ids
        ),
        per_group=(
            args.holdout_per_group
        ),
    )

    FRESH_HOLDOUT_MANIFEST.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    holdout.to_csv(
        FRESH_HOLDOUT_MANIFEST,
        index=False,
        encoding="utf-8-sig",
    )

    create_contact_sheet(
        combined
    )

    train_frame = combined[
        combined[
            "split_role"
        ].eq(
            "train"
        )
    ]

    calibration_frame = combined[
        combined[
            "split_role"
        ].eq(
            "calibration"
        )
    ]

    excluded_frame = combined[
        ~combined[
            "split_role"
        ].isin(
            [
                "train",
                "calibration",
            ]
        )
    ]

    summary = {
        "stage": (
            "v07_verifier_binary_dataset_build"
        ),
        "decision_design": {
            "trained_classes": [
                "LOOK_ALIKE",
                "CONFIRMED_OIL",
            ],
            "runtime_decisions": [
                "LOOK_ALIKE",
                "UNCERTAIN",
                "CONFIRMED_OIL",
            ],
            "uncertain_definition": (
                "UNCERTAIN yapay bir üçüncü eğitim sınıfı değildir. "
                "Calibration üzerinde seçilecek düşük ve yüksek "
                "olasılık eşikleri arasındaki abstention bandıdır."
            ),
        },
        "combined_candidate_count": int(
            len(combined)
        ),
        "train_count": int(
            len(train_frame)
        ),
        "calibration_count": int(
            len(calibration_frame)
        ),
        "excluded_seen_or_existing_test_count": int(
            len(excluded_frame)
        ),
        "train_class_counts": {
            str(key): int(value)
            for key, value in train_frame.groupby(
                "class_name"
            ).size().to_dict().items()
        },
        "calibration_class_counts": {
            str(key): int(value)
            for key, value in calibration_frame.groupby(
                "class_name"
            ).size().to_dict().items()
        },
        "all_class_counts": {
            str(key): int(value)
            for key, value in combined.groupby(
                "class_name"
            ).size().to_dict().items()
        },
        "positive_source_whitelist": sorted(
            ALLOWED_POSITIVE_MANIFESTS
        ),
        "discovered_positive_count_before_whitelist": int(
            discovered_positive_count
        ),
        "removed_non_verifier_positive_count": int(
            removed_non_verifier_positive_count
        ),
        "positive_source_manifest_counts": {
            str(key): int(value)
            for key, value in positive_frame.groupby(
                "source_manifest"
            ).size().to_dict().items()
        },
        "positive_scene_split_isolation": True,
        "fresh_holdout_requested_per_group": int(
            args.holdout_per_group
        ),
        "fresh_holdout_requested_total": int(
            args.holdout_per_group * 4
        ),
        "fresh_holdout_scene_count": int(
            len(holdout)
        ),
        "fresh_holdout_group_counts": {
            str(key): int(value)
            for key, value in holdout.groupby(
                "group"
            ).size().to_dict().items()
        },
        "fresh_holdout_complete": bool(
            len(holdout)
            == args.holdout_per_group * 4
            and all(
                int(
                    holdout.groupby(
                        "group"
                    ).size().to_dict().get(
                        group,
                        0,
                    )
                )
                == args.holdout_per_group
                for group in (
                    "nc",
                    "nw",
                    "oc",
                    "ow",
                )
            )
        ),
        "combined_manifest": relative(
            COMBINED_MANIFEST
        ),
        "fresh_holdout_manifest": relative(
            FRESH_HOLDOUT_MANIFEST
        ),
        "positive_discovery_audit": relative(
            DISCOVERY_AUDIT
        ),
        "contact_sheet": (
            relative(
                CONTACT_SHEET_PATH
            )
            if CONTACT_SHEET_PATH.exists()
            else None
        ),
        "fresh_holdout_opened": False,
        "fresh_holdout_inference_run": False,
        "training_performed": False,
        "scientific_note": (
            "88. aşamadaki locked_test adlı 25 negatif candidate "
            "contact sheet ve özet içinde görüldüğü için bağımsız final "
            "test olarak kullanılmayacaktır; excluded_seen_audit olarak "
            "ayrılmıştır. Nihai uçtan uca değerlendirme için dört DARTIS "
            "grubundan tamamen yeni sahneler yalnız dosya yolu düzeyinde "
            "rezerve edilmiştir. Bazı gruplarda yeterli tamamen yeni sahne "
            "kalmadıysa mevcut veya görülmüş sahneler sessizce yeniden "
            "kullanılmamış; eksik holdout sayısı rapora yazılmıştır. "
            "Holdout görüntüleri açılmamış ve inference yapılmamıştır. "
            "Pozitif verifier sınıfı yalnız "
            "dartis_positive_verifier_crops.csv kaynağından alınmış; "
            "v04 input-gate gibi tam-sahne destek veri setleri pozitif "
            "verifier crop olarak kullanılmamıştır. Pozitif train ve "
            "calibration ayrımı sahne bazında yapılmıştır."
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
        "v0.7 VERIFIER VERİ SETİ HAZIR"
    )
    print("=" * 78)

    print(
        "Train:",
        len(train_frame),
        train_frame.groupby(
            "class_name"
        ).size().to_dict(),
    )

    print(
        "Calibration:",
        len(calibration_frame),
        calibration_frame.groupby(
            "class_name"
        ).size().to_dict(),
    )

    print(
        "Hariç tutulan görülen/test kayıtları:",
        len(excluded_frame),
    )

    holdout_counts = holdout.groupby(
        "group"
    ).size().to_dict()

    holdout_complete = (
        len(holdout)
        == args.holdout_per_group * 4
        and all(
            int(
                holdout_counts.get(
                    group,
                    0,
                )
            )
            == args.holdout_per_group
            for group in (
                "nc",
                "nw",
                "oc",
                "ow",
            )
        )
    )

    print(
        "Yeni açılmamış holdout:",
        len(holdout),
        holdout_counts,
    )

    print(
        "Fresh holdout durumu:",
        (
            "COMPLETE"
            if holdout_complete
            else "INCOMPLETE - raporda sınırlılık olarak tutulacak"
        ),
    )

    print(
        "Manifest:",
        COMBINED_MANIFEST.resolve(),
    )

    print(
        "Fresh holdout:",
        FRESH_HOLDOUT_MANIFEST.resolve(),
    )

    print(
        "Audit:",
        DISCOVERY_AUDIT.resolve(),
    )

    print(
        "Özet:",
        SUMMARY_PATH.resolve(),
    )

    if CONTACT_SHEET_PATH.exists():
        print(
            "Görsel:",
            CONTACT_SHEET_PATH.resolve(),
        )


if __name__ == "__main__":
    main()
