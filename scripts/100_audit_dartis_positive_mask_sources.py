from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]

RAW_DARTIS_ROOT = (
    ROOT
    / "data"
    / "external"
    / "dartis"
    / "raw"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "v09_positive_mask_source_audit"
)

SOURCE_MASK_MAP = (
    ROOT
    / "data"
    / "metadata"
    / "v09_dartis_positive_source_mask_map.csv"
)

CANDIDATE_REPORT = (
    OUTPUT_DIR
    / "all_mask_candidates.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "summary.json"
)

CONTACT_SHEET_PATH = (
    OUTPUT_DIR
    / "positive_mask_pair_contact_sheet.jpg"
)

SCENE_PATTERN = re.compile(
    r"(?P<scene>(?:oc|ow)-\d{4})",
    re.IGNORECASE,
)

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
}

POSITIVE_PATH_TOKENS = {
    "oil",
    "spill",
    "mask",
    "masks",
    "label",
    "labels",
    "annotation",
    "annotations",
    "ground_truth",
    "groundtruth",
    "gt",
}

EXCLUDED_PATH_TOKENS = {
    "water",
    "land",
    "safe_water",
    "uncertain",
    "prelabel",
    "manual_masks",
    "flood",
    "sen1floods",
    "prediction",
    "predictions",
    "output",
    "outputs",
    "overlay",
    "candidate",
    "crop",
    "crops",
    "patch",
    "patches",
}

SEARCH_ROOTS = [
    ROOT / "data" / "external" / "dartis",
    ROOT / "data" / "raw",
    ROOT / "data" / "processed",
    ROOT / "data" / "annotations",
    ROOT / "data" / "masks",
    ROOT / "data" / "labels",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "DARTIS oc/ow pozitif sahneleri için gerçek petrol maskesi "
            "kaynaklarını güvenli biçimde bulur ve doğrular. Bu aşama "
            "eğitim yapmaz, crop üretmez ve yeni holdout'u açmaz."
        )
    )

    parser.add_argument(
        "--minimum-pair-coverage",
        type=float,
        default=0.95,
    )

    parser.add_argument(
        "--maximum-mask-positive-fraction",
        type=float,
        default=0.80,
    )

    parser.add_argument(
        "--minimum-mask-positive-pixels",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--preview-count",
        type=int,
        default=40,
    )

    return parser.parse_args()


def scene_id_from_text(value: Any) -> str | None:
    if value is None:
        return None

    match = SCENE_PATTERN.search(
        str(value)
    )

    if match is None:
        return None

    return match.group(
        "scene"
    ).lower()


def resolve_path(value: Any) -> Path | None:
    if value is None:
        return None

    text = str(value).strip()

    if not text or text.lower() == "nan":
        return None

    path = Path(
        text.replace(
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


def tokenize_path(path: Path) -> set[str]:
    text = str(
        path
    ).lower().replace(
        "\\",
        "/",
    )

    tokens = set(
        re.findall(
            r"[a-z0-9_]+",
            text,
        )
    )

    return tokens


def path_is_excluded(path: Path) -> bool:
    text = str(
        path
    ).lower().replace(
        "\\",
        "/",
    )

    return any(
        token in text
        for token in EXCLUDED_PATH_TOKENS
    )


def path_has_positive_hint(path: Path) -> bool:
    text = str(
        path
    ).lower().replace(
        "\\",
        "/",
    )

    return any(
        token in text
        for token in POSITIVE_PATH_TOKENS
    )


def discover_source_images() -> pd.DataFrame:
    rows = []

    for group in (
        "oc",
        "ow",
    ):
        group_dir = (
            RAW_DARTIS_ROOT
            / group
        )

        if not group_dir.exists():
            continue

        for path in sorted(
            group_dir.iterdir()
        ):
            if (
                not path.is_file()
                or path.suffix.lower()
                not in IMAGE_EXTENSIONS
            ):
                continue

            scene_id = scene_id_from_text(
                path.stem
            )

            if scene_id is None:
                continue

            rows.append(
                {
                    "scene_id": scene_id,
                    "group": group,
                    "source_image_path": relative(
                        path
                    ),
                }
            )

    frame = pd.DataFrame(
        rows
    )

    if frame.empty:
        raise RuntimeError(
            "DARTIS oc/ow ham kaynak görüntüleri bulunamadı: "
            f"{RAW_DARTIS_ROOT}"
        )

    duplicates = frame[
        "scene_id"
    ].duplicated(
        keep=False
    )

    if duplicates.any():
        duplicate_ids = sorted(
            frame.loc[
                duplicates,
                "scene_id",
            ].unique()
        )

        raise RuntimeError(
            "Aynı scene_id için birden fazla ham görüntü bulundu: "
            + ", ".join(
                duplicate_ids[:10]
            )
        )

    return frame.sort_values(
        "scene_id"
    ).reset_index(
        drop=True
    )


def discover_metadata_candidates() -> list[dict[str, Any]]:
    metadata_root = (
        ROOT
        / "data"
        / "metadata"
    )

    candidates: list[
        dict[str, Any]
    ] = []

    if not metadata_root.exists():
        return candidates

    for csv_path in metadata_root.rglob(
        "*.csv"
    ):
        if csv_path.resolve() == SOURCE_MASK_MAP.resolve():
            continue

        try:
            frame = pd.read_csv(
                csv_path,
                encoding="utf-8-sig",
                low_memory=False,
            )
        except Exception:
            continue

        mask_columns = [
            str(column)
            for column in frame.columns
            if (
                "mask" in str(
                    column
                ).lower()
                or "label_path"
                in str(
                    column
                ).lower()
                or "annotation_path"
                in str(
                    column
                ).lower()
            )
        ]

        if not mask_columns:
            continue

        scene_columns = [
            str(column)
            for column in frame.columns
            if any(
                token
                in str(
                    column
                ).lower()
                for token in (
                    "scene",
                    "sample",
                    "image",
                    "source",
                    "path",
                )
            )
        ]

        for row in frame.to_dict(
            orient="records"
        ):
            scene_id = None

            for column in scene_columns:
                scene_id = scene_id_from_text(
                    row.get(
                        column
                    )
                )

                if scene_id is not None:
                    break

            if scene_id is None:
                continue

            for mask_column in mask_columns:
                mask_path = resolve_path(
                    row.get(
                        mask_column
                    )
                )

                if (
                    mask_path is None
                    or not mask_path.exists()
                    or not mask_path.is_file()
                    or mask_path.suffix.lower()
                    not in IMAGE_EXTENSIONS
                ):
                    continue

                candidates.append(
                    {
                        "scene_id": scene_id,
                        "mask_path": relative(
                            mask_path
                        ),
                        "discovery_method": (
                            "metadata"
                        ),
                        "source_reference": relative(
                            csv_path
                        ),
                        "source_column": mask_column,
                    }
                )

    return candidates


def discover_filesystem_candidates(
    source_scene_ids: set[str],
) -> list[dict[str, Any]]:
    candidates: list[
        dict[str, Any]
    ] = []

    seen_paths: set[
        Path
    ] = set()

    for search_root in SEARCH_ROOTS:
        if not search_root.exists():
            continue

        for path in search_root.rglob(
            "*"
        ):
            if (
                not path.is_file()
                or path.suffix.lower()
                not in IMAGE_EXTENSIONS
            ):
                continue

            resolved = path.resolve()

            if resolved in seen_paths:
                continue

            seen_paths.add(
                resolved
            )

            scene_id = scene_id_from_text(
                path
            )

            if (
                scene_id is None
                or scene_id
                not in source_scene_ids
            ):
                continue

            if path_is_excluded(
                path
            ):
                continue

            if not path_has_positive_hint(
                path
            ):
                continue

            candidates.append(
                {
                    "scene_id": scene_id,
                    "mask_path": relative(
                        path
                    ),
                    "discovery_method": (
                        "filesystem"
                    ),
                    "source_reference": (
                        relative(
                            search_root
                        )
                    ),
                    "source_column": "",
                }
            )

    return candidates


def read_mask_array(path: Path) -> np.ndarray:
    image = Image.open(
        path
    )

    array = np.asarray(
        image
    )

    if array.ndim == 3:
        if array.shape[2] == 4:
            array = array[
                ...,
                :3,
            ]

        array = array.astype(
            np.float32
        ).max(
            axis=2
        )

    return np.asarray(
        array
    )


def normalize_mask(
    array: np.ndarray,
) -> np.ndarray:
    array = np.asarray(
        array
    )

    if array.dtype == np.bool_:
        return array.astype(
            np.uint8
        )

    finite = array[
        np.isfinite(
            array
        )
    ]

    if finite.size == 0:
        return np.zeros(
            array.shape,
            dtype=np.uint8,
        )

    unique = np.unique(
        finite
    )

    if unique.size <= 3:
        positive_value = float(
            unique.max()
        )

        if positive_value <= 0:
            return np.zeros(
                array.shape,
                dtype=np.uint8,
            )

        return (
            array
            == positive_value
        ).astype(
            np.uint8
        )

    high = float(
        np.percentile(
            finite,
            99,
        )
    )

    threshold = (
        0.5
        if high <= 1.0
        else 127.0
    )

    return (
        array
        > threshold
    ).astype(
        np.uint8
    )


def inspect_candidate(
    source_path: Path,
    mask_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    result: dict[
        str,
        Any,
    ] = {
        "readable": False,
        "size_match": False,
        "binary_like": False,
        "positive_pixels": 0,
        "positive_fraction": 0.0,
        "unique_value_count": 0,
        "valid_mask": False,
        "error": "",
    }

    try:
        source = Image.open(
            source_path
        ).convert("L")

        raw_mask = read_mask_array(
            mask_path
        )

        result[
            "readable"
        ] = True

        result[
            "source_width"
        ] = int(
            source.width
        )

        result[
            "source_height"
        ] = int(
            source.height
        )

        result[
            "mask_width"
        ] = int(
            raw_mask.shape[1]
        )

        result[
            "mask_height"
        ] = int(
            raw_mask.shape[0]
        )

        result[
            "size_match"
        ] = (
            raw_mask.shape[1]
            == source.width
            and raw_mask.shape[0]
            == source.height
        )

        finite = raw_mask[
            np.isfinite(
                raw_mask
            )
        ]

        unique_count = int(
            np.unique(
                finite
            ).size
        ) if finite.size else 0

        result[
            "unique_value_count"
        ] = unique_count

        result[
            "binary_like"
        ] = (
            unique_count <= 8
        )

        mask = normalize_mask(
            raw_mask
        )

        positive_pixels = int(
            mask.sum()
        )

        positive_fraction = float(
            positive_pixels
            / max(
                mask.size,
                1,
            )
        )

        result[
            "positive_pixels"
        ] = positive_pixels

        result[
            "positive_fraction"
        ] = positive_fraction

        result[
            "valid_mask"
        ] = bool(
            result[
                "size_match"
            ]
            and positive_pixels
            >= args.minimum_mask_positive_pixels
            and positive_fraction
            <= args.maximum_mask_positive_fraction
            and positive_fraction
            > 0.0
        )

    except Exception as error:
        result[
            "error"
        ] = repr(
            error
        )

    return result


def candidate_score(
    row: dict[str, Any],
) -> float:
    score = 0.0

    if row[
        "discovery_method"
    ] == "metadata":
        score += 100.0

    path = Path(
        row[
            "mask_path"
        ]
    )

    text = str(
        path
    ).lower()

    if "oil" in text:
        score += 25.0

    if "spill" in text:
        score += 20.0

    if "mask" in text:
        score += 15.0

    if "label" in text:
        score += 10.0

    if row.get(
        "size_match",
        False,
    ):
        score += 50.0

    if row.get(
        "binary_like",
        False,
    ):
        score += 10.0

    if row.get(
        "valid_mask",
        False,
    ):
        score += 1000.0

    positive_fraction = float(
        row.get(
            "positive_fraction",
            0.0,
        )
    )

    if (
        positive_fraction
        > 0.0
        and positive_fraction
        < 0.20
    ):
        score += 20.0

    return score


def enhance(image: Image.Image) -> Image.Image:
    array = np.asarray(
        image.convert("L"),
        dtype=np.float32,
    )

    low = float(
        np.percentile(
            array,
            2,
        )
    )

    high = float(
        np.percentile(
            array,
            98,
        )
    )

    if high <= low:
        result = array
    else:
        result = (
            np.clip(
                (
                    array
                    - low
                )
                / (
                    high
                    - low
                ),
                0.0,
                1.0,
            )
            * 255.0
        )

    return Image.fromarray(
        result.astype(
            np.uint8
        )
    )


def overlay_mask(
    source: Image.Image,
    mask: np.ndarray,
) -> Image.Image:
    base = np.asarray(
        source.convert("RGB"),
        dtype=np.uint8,
    ).copy()

    binary = (
        np.asarray(
            mask
        )
        > 0
    )

    red = np.zeros_like(
        base
    )

    red[
        ...,
        0,
    ] = 255

    base[
        binary
    ] = (
        0.45
        * base[
            binary
        ].astype(
            np.float32
        )
        + 0.55
        * red[
            binary
        ].astype(
            np.float32
        )
    ).astype(
        np.uint8
    )

    return Image.fromarray(
        base
    )


def create_contact_sheet(
    selected: pd.DataFrame,
    preview_count: int,
) -> None:
    valid = selected[
        selected[
            "pair_valid"
        ].eq(
            True
        )
    ].copy()

    if valid.empty:
        return

    valid[
        "preview_rank"
    ] = valid[
        "scene_id"
    ].map(
        lambda value: hash(
            value
        )
    )

    preview = (
        valid.sort_values(
            [
                "group",
                "preview_rank",
            ]
        )
        .groupby(
            "group",
            group_keys=False,
        )
        .head(
            max(
                preview_count
                // 2,
                1,
            )
        )
        .head(
            preview_count
        )
    )

    tile_width = 780
    image_height = 250
    header_height = 48
    tile_height = (
        image_height
        + header_height
    )

    rows = len(
        preview
    )

    sheet = Image.new(
        "RGB",
        (
            tile_width,
            rows
            * tile_height,
        ),
        "black",
    )

    font = ImageFont.load_default()

    for index, row in enumerate(
        preview.itertuples()
    ):
        source_path = resolve_path(
            row.source_image_path
        )

        mask_path = resolve_path(
            row.mask_path
        )

        if (
            source_path is None
            or mask_path is None
        ):
            continue

        source = Image.open(
            source_path
        ).convert("L")

        mask = normalize_mask(
            read_mask_array(
                mask_path
            )
        )

        source_view = enhance(
            source
        ).convert("RGB")

        mask_view = Image.fromarray(
            (
                mask
                * 255
            ).astype(
                np.uint8
            )
        ).convert("RGB")

        overlay_view = overlay_mask(
            source_view,
            mask,
        )

        panel_width = (
            tile_width
            // 3
        )

        tile = Image.new(
            "RGB",
            (
                tile_width,
                tile_height,
            ),
            "black",
        )

        for panel_index, image in enumerate(
            [
                source_view,
                mask_view,
                overlay_view,
            ]
        ):
            copy = image.copy()

            copy.thumbnail(
                (
                    panel_width,
                    image_height,
                ),
                Image.Resampling.LANCZOS,
            )

            x = (
                panel_index
                * panel_width
                + (
                    panel_width
                    - copy.width
                )
                // 2
            )

            y = (
                header_height
                + (
                    image_height
                    - copy.height
                )
                // 2
            )

            tile.paste(
                copy,
                (
                    x,
                    y,
                ),
            )

        draw = ImageDraw.Draw(
            tile
        )

        draw.text(
            (
                6,
                6,
            ),
            (
                f"{row.scene_id} | positive_fraction="
                f"{float(row.positive_fraction):.6f} | "
                f"{row.discovery_method}"
            ),
            fill="white",
            font=font,
        )

        draw.text(
            (
                6,
                25,
            ),
            "SOURCE SAR                    OIL MASK                     OVERLAY",
            fill="white",
            font=font,
        )

        sheet.paste(
            tile,
            (
                0,
                index
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

    sources = discover_source_images()

    source_scene_ids = set(
        sources[
            "scene_id"
        ]
    )

    candidates = (
        discover_metadata_candidates()
        + discover_filesystem_candidates(
            source_scene_ids
        )
    )

    candidate_frame = pd.DataFrame(
        candidates
    )

    if candidate_frame.empty:
        raise RuntimeError(
            "Hiç petrol maskesi adayı bulunamadı."
        )

    candidate_frame = (
        candidate_frame.drop_duplicates(
            subset=[
                "scene_id",
                "mask_path",
            ],
            keep="first",
        )
        .merge(
            sources,
            on="scene_id",
            how="inner",
            validate="many_to_one",
        )
    )

    inspected_rows = []

    for index, row in enumerate(
        candidate_frame.to_dict(
            orient="records"
        ),
        start=1,
    ):
        source_path = resolve_path(
            row[
                "source_image_path"
            ]
        )

        mask_path = resolve_path(
            row[
                "mask_path"
            ]
        )

        if (
            source_path is None
            or mask_path is None
        ):
            continue

        diagnostics = inspect_candidate(
            source_path,
            mask_path,
            args,
        )

        inspected_rows.append(
            {
                **row,
                **diagnostics,
            }
        )

        if (
            index % 250 == 0
            or index
            == len(
                candidate_frame
            )
        ):
            print(
                f"Mask candidate audit: "
                f"{index}/{len(candidate_frame)}"
            )

    inspected = pd.DataFrame(
        inspected_rows
    )

    inspected[
        "candidate_score"
    ] = inspected.apply(
        lambda row: candidate_score(
            row.to_dict()
        ),
        axis=1,
    )

    inspected = inspected.sort_values(
        [
            "scene_id",
            "candidate_score",
            "mask_path",
        ],
        ascending=[
            True,
            False,
            True,
        ],
    )

    selected = (
        inspected.groupby(
            "scene_id",
            as_index=False,
            group_keys=False,
        )
        .head(1)
        .copy()
    )

    selected[
        "pair_valid"
    ] = selected[
        "valid_mask"
    ].eq(
        True
    )

    source_mask_map = sources.merge(
        selected[
            [
                "scene_id",
                "mask_path",
                "discovery_method",
                "source_reference",
                "source_column",
                "readable",
                "size_match",
                "binary_like",
                "positive_pixels",
                "positive_fraction",
                "unique_value_count",
                "valid_mask",
                "pair_valid",
                "candidate_score",
                "error",
            ]
        ],
        on="scene_id",
        how="left",
        validate="one_to_one",
    )

    source_mask_map[
        "pair_valid"
    ] = source_mask_map[
        "pair_valid"
    ].fillna(
        False
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    inspected.to_csv(
        CANDIDATE_REPORT,
        index=False,
        encoding="utf-8-sig",
    )

    source_mask_map.to_csv(
        SOURCE_MASK_MAP,
        index=False,
        encoding="utf-8-sig",
    )

    pair_coverage = float(
        source_mask_map[
            "pair_valid"
        ].eq(
            True
        ).mean()
    )

    group_coverage = {}

    for group, group_frame in source_mask_map.groupby(
        "group"
    ):
        group_coverage[
            str(group)
        ] = {
            "scene_count": int(
                len(
                    group_frame
                )
            ),
            "valid_pair_count": int(
                group_frame[
                    "pair_valid"
                ].eq(
                    True
                ).sum()
            ),
            "coverage": float(
                group_frame[
                    "pair_valid"
                ].eq(
                    True
                ).mean()
            ),
        }

    build_passed = (
        pair_coverage
        >= args.minimum_pair_coverage
    )

    if source_mask_map[
        "pair_valid"
    ].any():
        create_contact_sheet(
            source_mask_map,
            preview_count=(
                args.preview_count
            ),
        )

    summary = {
        "stage": (
            "v09_positive_mask_source_audit"
        ),
        "source_scene_count": int(
            len(
                sources
            )
        ),
        "group_source_counts": {
            str(key): int(value)
            for key, value in sources.groupby(
                "group"
            ).size().to_dict().items()
        },
        "mask_candidate_count": int(
            len(
                inspected
            )
        ),
        "valid_source_mask_pair_count": int(
            source_mask_map[
                "pair_valid"
            ].eq(
                True
            ).sum()
        ),
        "pair_coverage": (
            pair_coverage
        ),
        "group_coverage": (
            group_coverage
        ),
        "minimum_pair_coverage": float(
            args.minimum_pair_coverage
        ),
        "build_passed": bool(
            build_passed
        ),
        "source_mask_map": relative(
            SOURCE_MASK_MAP
        ),
        "candidate_report": relative(
            CANDIDATE_REPORT
        ),
        "contact_sheet": (
            relative(
                CONTACT_SHEET_PATH
            )
            if CONTACT_SHEET_PATH.exists()
            else None
        ),
        "training_performed": False,
        "fresh_holdout_used": False,
        "scientific_note": (
            "Eski pozitif verifier crop'ları ham sahnelere güvenilir "
            "biçimde geri eşlenemediği için kullanılmamıştır. Bu aşama "
            "doğrudan DARTIS oc/ow ham sahneleri ile gerçek petrol maskesi "
            "eşlerini bulur. Water/land/prediction/crop yolları mask "
            "adayı olarak dışlanmıştır. Eğitim veya holdout inference "
            "yapılmamıştır."
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
        "v0.9 POZİTİF KAYNAK-MASKE AUDIT SONUCU"
    )
    print("=" * 78)

    print(
        "Audit:",
        (
            "PASS"
            if build_passed
            else "FAIL"
        ),
    )

    print(
        "Pozitif kaynak sahne:",
        len(
            sources
        ),
        sources.groupby(
            "group"
        ).size().to_dict(),
    )

    print(
        "Geçerli kaynak-mask çifti:",
        int(
            source_mask_map[
                "pair_valid"
            ].eq(
                True
            ).sum()
        ),
        "/",
        len(
            source_mask_map
        ),
    )

    print(
        "Coverage:",
        f"{pair_coverage:.4f}",
    )

    print(
        "Group coverage:",
        group_coverage,
    )

    print(
        "Eğitim yapıldı mı: False"
    )

    print(
        "Fresh holdout kullanıldı mı: False"
    )

    print(
        "Map:",
        SOURCE_MASK_MAP.resolve(),
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
