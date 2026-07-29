from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import runpy
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]

SAFE_PIPELINE_SCRIPT = (
    ROOT
    / "scripts"
    / "85_run_safe_water_masked_oil_pipeline.py"
)

OUTPUT_ROOT = (
    ROOT
    / "outputs"
    / "v07_hard_negative_mining"
)

CROP_DIR = (
    ROOT
    / "data"
    / "mined"
    / "v07_lookalike_candidates"
    / "crops"
)

MANIFEST_PATH = (
    ROOT
    / "data"
    / "metadata"
    / "v07_hard_negative_candidates.csv"
)

SCENE_REPORT_PATH = (
    ROOT
    / "data"
    / "metadata"
    / "v07_hard_negative_scene_report.csv"
)

SUMMARY_PATH = (
    OUTPUT_ROOT
    / "summary.json"
)

CONTACT_SHEET_PATH = (
    OUTPUT_ROOT
    / "top_hard_negatives_contact_sheet.jpg"
)

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
}

NEGATIVE_GROUPS = {
    "nc",
    "nw",
}

EXCLUDED_PARTS = {
    "masks",
    "mask",
    "manual_masks",
    "prelabel_water",
    "ensemble_vote_fraction",
    "review_overlay",
    "overlays",
    "overlay",
    "outputs",
    "crops",
    "checkpoints",
    ".venv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "DARTIS nc/nw petrolsüz sahnelerinde mevcut güvenli "
            "uçtan uca hattın petrol diye kabul ettiği adayları "
            "otomatik hard-negative/look-alike olarak çıkarır. "
            "Maske düzenlemez ve eğitim yapmaz."
        )
    )

    parser.add_argument(
        "--max-images-per-group",
        type=int,
        default=200,
        help=(
            "Pilot taramada nc ve nw gruplarının her birinden "
            "en fazla kaç görüntü işleneceği. 0 tümünü işler."
        ),
    )

    parser.add_argument(
        "--max-candidates-per-image",
        type=int,
        default=5,
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
        "--crop-padding-ratio",
        type=float,
        default=0.35,
    )

    parser.add_argument(
        "--minimum-crop-size",
        type=int,
        default=96,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260729,
    )

    parser.add_argument(
        "--resume",
        action="store_true",
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


def resolve_path(value: str | Path) -> Path:
    path = Path(str(value).replace("\\", "/"))

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


def sha256_file(path: Path) -> str:
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


def normalize_group(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip().lower()

    if text in NEGATIVE_GROUPS:
        return text

    for group in NEGATIVE_GROUPS:
        if text.startswith(
            f"{group}:"
        ):
            return group

        if text.startswith(
            f"{group}-"
        ):
            return group

        if text.startswith(
            f"{group}__"
        ):
            return group

    return None


def infer_group(
    path: Path,
    sample_id: Any = None,
    explicit_group: Any = None,
) -> str | None:
    group = normalize_group(
        explicit_group
    )

    if group is not None:
        return group

    group = normalize_group(
        sample_id
    )

    if group is not None:
        return group

    group = normalize_group(
        path.stem
    )

    if group is not None:
        return group

    for part in reversed(
        path.parts
    ):
        group = normalize_group(
            part
        )

        if group is not None:
            return group

    return None


def is_usable_image_path(
    path: Path,
) -> bool:
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        return False

    lower_parts = {
        part.lower()
        for part in path.parts
    }

    if lower_parts & EXCLUDED_PARTS:
        return False

    return path.exists()


def candidate_image_columns(
    columns: list[str],
) -> list[str]:
    preferred = [
        "annotation_image_path",
        "image_path",
        "source_image_path",
        "sar_image_path",
        "original_image_path",
        "image",
        "path",
    ]

    result = [
        column
        for column in preferred
        if column in columns
    ]

    for column in columns:
        lower = column.lower()

        if column in result:
            continue

        if "mask" in lower:
            continue

        if (
            "image" in lower
            and (
                "path" in lower
                or lower.endswith(
                    "image"
                )
            )
        ):
            result.append(
                column
            )

    return result


def discover_from_metadata() -> list[dict[str, str]]:
    metadata_root = (
        ROOT
        / "data"
        / "metadata"
    )

    records: list[
        dict[str, str]
    ] = []

    if not metadata_root.exists():
        return records

    for csv_path in sorted(
        metadata_root.rglob("*.csv")
    ):
        try:
            frame = pd.read_csv(
                csv_path,
                encoding="utf-8-sig",
                low_memory=False,
            )
        except Exception:
            continue

        if frame.empty:
            continue

        columns = [
            str(column)
            for column in frame.columns
        ]

        image_columns = (
            candidate_image_columns(
                columns
            )
        )

        if not image_columns:
            continue

        group_column = next(
            (
                column
                for column in (
                    "group",
                    "coastal_group",
                    "dartis_group",
                    "class",
                    "label_group",
                )
                if column in frame.columns
            ),
            None,
        )

        sample_column = next(
            (
                column
                for column in (
                    "sample_id",
                    "image_id",
                    "id",
                    "name",
                )
                if column in frame.columns
            ),
            None,
        )

        for row in frame.itertuples(
            index=False
        ):
            row_dict = row._asdict()

            explicit_group = (
                row_dict.get(
                    group_column
                )
                if group_column
                else None
            )

            sample_id = (
                row_dict.get(
                    sample_column
                )
                if sample_column
                else None
            )

            for image_column in image_columns:
                value = row_dict.get(
                    image_column
                )

                if value is None:
                    continue

                if isinstance(
                    value,
                    float,
                ) and np.isnan(value):
                    continue

                try:
                    path = resolve_path(
                        str(value)
                    )
                except Exception:
                    continue

                group = infer_group(
                    path,
                    sample_id=sample_id,
                    explicit_group=(
                        explicit_group
                    ),
                )

                if (
                    group is None
                    or not is_usable_image_path(
                        path
                    )
                ):
                    continue

                normalized_sample_id = (
                    str(sample_id)
                    if sample_id is not None
                    and str(sample_id).strip()
                    and str(sample_id).lower()
                    != "nan"
                    else (
                        f"{group}:{path.stem}"
                    )
                )

                records.append(
                    {
                        "sample_id": (
                            normalized_sample_id
                        ),
                        "group": group,
                        "image_path": str(
                            path
                        ),
                        "discovered_from": (
                            relative(csv_path)
                        ),
                    }
                )

                break

    return records


def discover_from_data_tree() -> list[dict[str, str]]:
    data_root = (
        ROOT
        / "data"
    )

    records: list[
        dict[str, str]
    ] = []

    if not data_root.exists():
        return records

    for path in data_root.rglob("*"):
        if (
            not path.is_file()
            or path.suffix.lower()
            not in IMAGE_EXTENSIONS
        ):
            continue

        group = infer_group(
            path
        )

        if (
            group is None
            or not is_usable_image_path(
                path
            )
        ):
            continue

        records.append(
            {
                "sample_id": (
                    f"{group}:{path.stem}"
                ),
                "group": group,
                "image_path": str(
                    path.resolve()
                ),
                "discovered_from": (
                    "data_tree_scan"
                ),
            }
        )

    return records


def deduplicate_records(
    records: list[dict[str, str]],
) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(
            columns=[
                "sample_id",
                "group",
                "image_path",
                "discovered_from",
            ]
        )

    frame = pd.DataFrame(
        records
    )

    frame["resolved_key"] = (
        frame["image_path"].map(
            lambda value: str(
                Path(value).resolve()
            ).lower()
        )
    )

    frame = frame.drop_duplicates(
        subset=[
            "resolved_key",
        ],
        keep="first",
    )

    frame = frame.drop(
        columns=[
            "resolved_key",
        ]
    )

    return frame.reset_index(
        drop=True
    )


def prioritize_and_sample(
    frame: pd.DataFrame,
    max_images_per_group: int,
    seed: int,
) -> pd.DataFrame:
    priority_ids = {
        "nc:nc-0140-01-000064",
        "nc:nc-0228-02-000068",
        "nc:nc-0038-00-000038",
    }

    rng = random.Random(
        seed
    )

    selected_parts = []

    for group in sorted(
        NEGATIVE_GROUPS
    ):
        group_frame = frame[
            frame["group"].eq(
                group
            )
        ].copy()

        if group_frame.empty:
            continue

        group_frame[
            "priority"
        ] = group_frame[
            "sample_id"
        ].astype(str).map(
            lambda value: (
                0
                if value
                in priority_ids
                else 1
            )
        )

        priority = group_frame[
            group_frame[
                "priority"
            ].eq(0)
        ]

        remainder = group_frame[
            group_frame[
                "priority"
            ].eq(1)
        ].copy()

        indices = list(
            remainder.index
        )

        rng.shuffle(
            indices
        )

        remainder = remainder.loc[
            indices
        ]

        ordered = pd.concat(
            [
                priority,
                remainder,
            ],
            ignore_index=True,
        )

        if max_images_per_group > 0:
            ordered = ordered.head(
                max_images_per_group
            )

        selected_parts.append(
            ordered.drop(
                columns=[
                    "priority",
                ],
                errors="ignore",
            )
        )

    if not selected_parts:
        return frame.iloc[
            0:0
        ].copy()

    result = pd.concat(
        selected_parts,
        ignore_index=True,
    )

    return result.reset_index(
        drop=True
    )


def load_existing_paths(
    path: Path,
) -> set[str]:
    if not path.exists():
        return set()

    try:
        frame = pd.read_csv(
            path,
            encoding="utf-8-sig",
            low_memory=False,
        )
    except Exception:
        return set()

    if "image_path" not in frame.columns:
        return set()

    return {
        str(
            resolve_path(value)
        ).lower()
        for value in frame[
            "image_path"
        ].dropna()
    }


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        return

    frame = pd.DataFrame(
        rows
    )

    frame.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
    )


def expand_crop_box(
    bbox: list[int],
    image_width: int,
    image_height: int,
    padding_ratio: float,
    minimum_size: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [
        int(value)
        for value in bbox
    ]

    width = max(
        x2 - x1,
        1,
    )

    height = max(
        y2 - y1,
        1,
    )

    target_width = max(
        int(
            round(
                width
                * (
                    1.0
                    + 2.0
                    * padding_ratio
                )
            )
        ),
        minimum_size,
    )

    target_height = max(
        int(
            round(
                height
                * (
                    1.0
                    + 2.0
                    * padding_ratio
                )
            )
        ),
        minimum_size,
    )

    center_x = (
        x1 + x2
    ) / 2.0

    center_y = (
        y1 + y2
    ) / 2.0

    crop_x1 = int(
        round(
            center_x
            - target_width
            / 2.0
        )
    )

    crop_y1 = int(
        round(
            center_y
            - target_height
            / 2.0
        )
    )

    crop_x2 = (
        crop_x1
        + target_width
    )

    crop_y2 = (
        crop_y1
        + target_height
    )

    if crop_x1 < 0:
        crop_x2 -= crop_x1
        crop_x1 = 0

    if crop_y1 < 0:
        crop_y2 -= crop_y1
        crop_y1 = 0

    if crop_x2 > image_width:
        shift = (
            crop_x2
            - image_width
        )
        crop_x1 = max(
            0,
            crop_x1 - shift,
        )
        crop_x2 = image_width

    if crop_y2 > image_height:
        shift = (
            crop_y2
            - image_height
        )
        crop_y1 = max(
            0,
            crop_y1 - shift,
        )
        crop_y2 = image_height

    return (
        crop_x1,
        crop_y1,
        crop_x2,
        crop_y2,
    )


def candidate_hard_score(
    candidate: dict[str, Any],
    probability_map: np.ndarray,
) -> float:
    bbox = candidate[
        "bbox"
    ]

    x1, y1, x2, y2 = [
        int(value)
        for value in bbox
    ]

    patch = probability_map[
        y1:y2,
        x1:x2,
    ]

    mean_probability = float(
        patch.mean()
        if patch.size
        else 0.0
    )

    peak_probability = float(
        patch.max()
        if patch.size
        else 0.0
    )

    verifier_probability = float(
        candidate[
            "selected_probability"
        ]
    )

    verifier_threshold = max(
        float(
            candidate[
                "selected_threshold"
            ]
        ),
        1e-6,
    )

    accepted_bonus = (
        2.0
        if candidate[
            "accepted"
        ]
        else 0.0
    )

    threshold_margin = (
        verifier_probability
        / verifier_threshold
    )

    area_score = min(
        np.log1p(
            int(
                candidate[
                    "area_pixels"
                ]
            )
        )
        / 10.0,
        1.0,
    )

    return float(
        accepted_bonus
        + 0.80
        * verifier_probability
        + 0.40
        * min(
            threshold_margin,
            3.0,
        )
        + 0.25
        * mean_probability
        + 0.15
        * peak_probability
        + 0.10
        * area_score
    )


def create_contact_sheet(
    manifest: pd.DataFrame,
    output_path: Path,
    maximum_items: int = 64,
) -> None:
    if manifest.empty:
        return

    selected = manifest.sort_values(
        [
            "hard_score",
            "verifier_probability",
        ],
        ascending=[
            False,
            False,
        ],
    ).head(
        maximum_items
    )

    tile_width = 240
    image_height = 190
    header_height = 50
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
            columns
            * tile_width,
            rows
            * tile_height,
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

        crop = Image.open(
            crop_path
        ).convert("L")

        crop.thumbnail(
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

        x = (
            tile_width
            - crop.width
        ) // 2

        y = (
            image_height
            - crop.height
        ) // 2

        tile.paste(
            crop.convert("RGB"),
            (
                x,
                header_height + y,
            ),
        )

        draw = ImageDraw.Draw(
            tile
        )

        line_1 = (
            f"{row.group} | "
            f"{row.sample_id}"
        )

        line_2 = (
            f"accepted={row.previously_accepted} "
            f"p={float(row.verifier_probability):.3f} "
            f"score={float(row.hard_score):.3f}"
        )

        draw.text(
            (5, 5),
            line_1[:38],
            fill="white",
            font=font,
        )

        draw.text(
            (5, 24),
            line_2[:45],
            fill="white",
            font=font,
        )

        sheet_x = (
            index
            % columns
        ) * tile_width

        sheet_y = (
            index
            // columns
        ) * tile_height

        sheet.paste(
            tile,
            (
                sheet_x,
                sheet_y,
            ),
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    sheet.save(
        output_path,
        quality=92,
    )


def main() -> None:
    args = parse_args()

    if not SAFE_PIPELINE_SCRIPT.exists():
        raise FileNotFoundError(
            "Güvenli uçtan uca script bulunamadı: "
            f"{SAFE_PIPELINE_SCRIPT}"
        )

    records = (
        discover_from_metadata()
        + discover_from_data_tree()
    )

    discovered = deduplicate_records(
        records
    )

    if discovered.empty:
        raise RuntimeError(
            "nc/nw DARTIS görüntüsü bulunamadı."
        )

    selected = prioritize_and_sample(
        discovered,
        max_images_per_group=(
            args.max_images_per_group
        ),
        seed=args.seed,
    )

    if selected.empty:
        raise RuntimeError(
            "İşlenecek negatif görüntü seçilemedi."
        )

    existing_scene_rows: list[
        dict[str, Any]
    ] = []

    existing_candidate_rows: list[
        dict[str, Any]
    ] = []

    processed_paths: set[str] = set()

    if args.resume:
        if SCENE_REPORT_PATH.exists():
            old_scene_frame = pd.read_csv(
                SCENE_REPORT_PATH,
                encoding="utf-8-sig",
                low_memory=False,
            )

            existing_scene_rows = (
                old_scene_frame.to_dict(
                    orient="records"
                )
            )

            if "image_path" in old_scene_frame.columns:
                processed_paths = {
                    str(
                        resolve_path(value)
                    ).lower()
                    for value in old_scene_frame[
                        "image_path"
                    ].dropna()
                }

        if MANIFEST_PATH.exists():
            old_candidate_frame = pd.read_csv(
                MANIFEST_PATH,
                encoding="utf-8-sig",
                low_memory=False,
            )

            existing_candidate_rows = (
                old_candidate_frame.to_dict(
                    orient="records"
                )
            )

    remaining = selected[
        ~selected[
            "image_path"
        ].map(
            lambda value: str(
                Path(value).resolve()
            ).lower()
            in processed_paths
        )
    ].copy()

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    CROP_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    safe_module = runpy.run_path(
        str(
            SAFE_PIPELINE_SCRIPT
        ),
        run_name=(
            "hard_negative_safe_pipeline"
        ),
    )

    print("=" * 78)
    print(
        "v0.7 HARD-NEGATIVE / LOOK-ALIKE MINING"
    )
    print("=" * 78)
    print(
        "Bulunan benzersiz nc/nw görüntüsü:",
        len(discovered),
    )

    print(
        "Bu çalışmada seçilen:",
        len(selected),
    )

    print(
        "Daha önce işlenen:",
        len(processed_paths),
    )

    print(
        "Bu çalışmada kalan:",
        len(remaining),
    )

    print(
        "Gruplar:",
        selected.groupby(
            "group"
        ).size().to_dict(),
    )

    print(
        "Maske düzenlenmeyecek."
    )

    print(
        "Eğitim yapılmayacak."
    )

    print()

    with torch.inference_mode():
        water_gate = safe_module[
            "load_water_gate"
        ](
            args.device
        )

        oil_pipeline = safe_module[
            "load_oil_pipeline"
        ](
            args.device
        )

    scene_rows = list(
        existing_scene_rows
    )

    candidate_rows = list(
        existing_candidate_rows
    )

    start_time = time.time()

    new_false_alarm_scenes = 0
    new_candidates = 0

    for position, row in enumerate(
        remaining.itertuples(),
        start=1,
    ):
        image_path = Path(
            row.image_path
        ).resolve()

        try:
            image = Image.open(
                image_path
            ).convert("L")

            with torch.inference_mode():
                water_result = safe_module[
                    "run_water_gate"
                ](
                    image,
                    water_gate,
                )

            scene_result: dict[
                str,
                Any,
            ] = {
                "sample_id": (
                    str(
                        row.sample_id
                    )
                ),
                "group": (
                    str(
                        row.group
                    )
                ),
                "image_path": (
                    relative(
                        image_path
                    )
                ),
                "image_sha256": (
                    sha256_file(
                        image_path
                    )
                ),
                "water_decision": (
                    water_result[
                        "decision"
                    ]
                ),
                "safe_water_fraction": float(
                    water_result[
                        "predicted_water_fraction"
                    ]
                ),
                "uncertain_fraction": float(
                    water_result[
                        "uncertain_fraction"
                    ]
                ),
                "oil_analysis_executed": False,
                "candidate_count": 0,
                "accepted_candidate_count": 0,
                "scene_false_alarm": False,
                "status": "OK",
                "error": "",
            }

            if water_result[
                "pipeline_allowed"
            ]:
                with torch.inference_mode():
                    oil_result = safe_module[
                        "run_oil_pipeline"
                    ](
                        image=image,
                        safe_water_mask=(
                            water_result[
                                "safe_water_mask"
                            ]
                        ),
                        pipeline=oil_pipeline,
                        segmentation_threshold=float(
                            args.segmentation_threshold
                        ),
                        minimum_component_ratio=float(
                            args.minimum_component_ratio
                        ),
                        minimum_component_pixels=int(
                            args.minimum_component_pixels
                        ),
                    )

                candidates = list(
                    oil_result[
                        "candidate_results"
                    ]
                )

                candidates = sorted(
                    candidates,
                    key=lambda item: (
                        bool(
                            item[
                                "accepted"
                            ]
                        ),
                        float(
                            item[
                                "selected_probability"
                            ]
                        ),
                        int(
                            item[
                                "area_pixels"
                            ]
                        ),
                    ),
                    reverse=True,
                )[
                    :args.max_candidates_per_image
                ]

                scene_result[
                    "oil_analysis_executed"
                ] = True

                scene_result[
                    "candidate_count"
                ] = int(
                    oil_result[
                        "candidate_count"
                    ]
                )

                scene_result[
                    "accepted_candidate_count"
                ] = int(
                    oil_result[
                        "accepted_candidate_count"
                    ]
                )

                scene_result[
                    "scene_false_alarm"
                ] = bool(
                    oil_result[
                        "oil_detected"
                    ]
                )

                if oil_result[
                    "oil_detected"
                ]:
                    new_false_alarm_scenes += 1

                for candidate in candidates:
                    hard_score = candidate_hard_score(
                        candidate,
                        oil_result[
                            "probability"
                        ],
                    )

                    crop_box = expand_crop_box(
                        candidate[
                            "bbox"
                        ],
                        image_width=(
                            image.width
                        ),
                        image_height=(
                            image.height
                        ),
                        padding_ratio=float(
                            args.crop_padding_ratio
                        ),
                        minimum_size=int(
                            args.minimum_crop_size
                        ),
                    )

                    crop = image.crop(
                        crop_box
                    )

                    crop_name = (
                        f"{row.group}__"
                        f"{Path(image_path).stem}__"
                        f"cand_{int(candidate['component_index']):02d}.png"
                    )

                    crop_path = (
                        CROP_DIR
                        / crop_name
                    )

                    crop.save(
                        crop_path
                    )

                    bbox = [
                        int(value)
                        for value in candidate[
                            "bbox"
                        ]
                    ]

                    x1, y1, x2, y2 = bbox

                    probability_patch = (
                        oil_result[
                            "probability"
                        ][
                            y1:y2,
                            x1:x2,
                        ]
                    )

                    candidate_rows.append(
                        {
                            "candidate_id": (
                                f"{row.group}:"
                                f"{Path(image_path).stem}:"
                                f"{int(candidate['component_index']):02d}"
                            ),
                            "label": (
                                "LOOK_ALIKE"
                            ),
                            "label_source": (
                                "DARTIS_NO_OIL_GROUP"
                            ),
                            "sample_id": (
                                str(
                                    row.sample_id
                                )
                            ),
                            "group": (
                                str(
                                    row.group
                                )
                            ),
                            "image_path": (
                                relative(
                                    image_path
                                )
                            ),
                            "crop_path": (
                                relative(
                                    crop_path
                                )
                            ),
                            "component_index": int(
                                candidate[
                                    "component_index"
                                ]
                            ),
                            "bbox_x1": x1,
                            "bbox_y1": y1,
                            "bbox_x2": x2,
                            "bbox_y2": y2,
                            "crop_x1": int(
                                crop_box[0]
                            ),
                            "crop_y1": int(
                                crop_box[1]
                            ),
                            "crop_x2": int(
                                crop_box[2]
                            ),
                            "crop_y2": int(
                                crop_box[3]
                            ),
                            "area_pixels": int(
                                candidate[
                                    "area_pixels"
                                ]
                            ),
                            "area_percent_total": float(
                                candidate[
                                    "area_percent_total"
                                ]
                            ),
                            "verifier_probability": float(
                                candidate[
                                    "selected_probability"
                                ]
                            ),
                            "verifier_threshold": float(
                                candidate[
                                    "selected_threshold"
                                ]
                            ),
                            "previously_accepted": bool(
                                candidate[
                                    "accepted"
                                ]
                            ),
                            "segmentation_mean_probability": float(
                                probability_patch.mean()
                                if probability_patch.size
                                else 0.0
                            ),
                            "segmentation_peak_probability": float(
                                probability_patch.max()
                                if probability_patch.size
                                else 0.0
                            ),
                            "safe_water_fraction": float(
                                water_result[
                                    "predicted_water_fraction"
                                ]
                            ),
                            "uncertain_fraction": float(
                                water_result[
                                    "uncertain_fraction"
                                ]
                            ),
                            "hard_score": float(
                                hard_score
                            ),
                            "split_role": (
                                "MINED_TRAINING_CANDIDATE"
                            ),
                        }
                    )

                    new_candidates += 1

            scene_rows.append(
                scene_result
            )

            print(
                f"[{position:04d}/{len(remaining):04d}] "
                f"{row.group} {Path(image_path).name} "
                f"| water={scene_result['water_decision']} "
                f"| candidates={scene_result['candidate_count']} "
                f"| accepted={scene_result['accepted_candidate_count']} "
                f"| false_alarm={scene_result['scene_false_alarm']}"
            )

        except Exception as error:
            scene_rows.append(
                {
                    "sample_id": (
                        str(
                            row.sample_id
                        )
                    ),
                    "group": (
                        str(
                            row.group
                        )
                    ),
                    "image_path": (
                        relative(
                            image_path
                        )
                    ),
                    "image_sha256": "",
                    "water_decision": (
                        "ERROR"
                    ),
                    "safe_water_fraction": (
                        np.nan
                    ),
                    "uncertain_fraction": (
                        np.nan
                    ),
                    "oil_analysis_executed": False,
                    "candidate_count": 0,
                    "accepted_candidate_count": 0,
                    "scene_false_alarm": False,
                    "status": "ERROR",
                    "error": repr(
                        error
                    ),
                }
            )

            print(
                f"[{position:04d}/{len(remaining):04d}] "
                f"ERROR {image_path.name}: {error}"
            )

        if (
            position % 10 == 0
            or position
            == len(remaining)
        ):
            write_csv(
                SCENE_REPORT_PATH,
                scene_rows,
            )

            write_csv(
                MANIFEST_PATH,
                candidate_rows,
            )

    scene_frame = pd.DataFrame(
        scene_rows
    )

    candidate_frame = pd.DataFrame(
        candidate_rows
    )

    if not candidate_frame.empty:
        candidate_frame = (
            candidate_frame.sort_values(
                [
                    "hard_score",
                    "verifier_probability",
                ],
                ascending=[
                    False,
                    False,
                ],
            )
            .drop_duplicates(
                subset=[
                    "candidate_id",
                ],
                keep="first",
            )
            .reset_index(
                drop=True
            )
        )

        candidate_frame.to_csv(
            MANIFEST_PATH,
            index=False,
            encoding="utf-8-sig",
        )

        create_contact_sheet(
            candidate_frame,
            CONTACT_SHEET_PATH,
        )

    if not scene_frame.empty:
        scene_frame = (
            scene_frame.drop_duplicates(
                subset=[
                    "image_path",
                ],
                keep="last",
            )
            .reset_index(
                drop=True
            )
        )

        scene_frame.to_csv(
            SCENE_REPORT_PATH,
            index=False,
            encoding="utf-8-sig",
        )

    successful = scene_frame[
        scene_frame[
            "status"
        ].eq(
            "OK"
        )
    ]

    analysed = successful[
        successful[
            "oil_analysis_executed"
        ].eq(
            True
        )
    ]

    false_alarm_scenes = analysed[
        analysed[
            "scene_false_alarm"
        ].eq(
            True
        )
    ]

    summary = {
        "stage": (
            "v07_hard_negative_lookalike_mining"
        ),
        "discovered_unique_negative_images": int(
            len(discovered)
        ),
        "selected_images": int(
            len(selected)
        ),
        "processed_scene_count": int(
            len(scene_frame)
        ),
        "successful_scene_count": int(
            len(successful)
        ),
        "water_accepted_scene_count": int(
            len(analysed)
        ),
        "false_alarm_scene_count": int(
            len(
                false_alarm_scenes
            )
        ),
        "false_alarm_scene_rate_among_water_accepted": float(
            len(
                false_alarm_scenes
            )
            / max(
                len(analysed),
                1,
            )
        ),
        "mined_candidate_count": int(
            len(
                candidate_frame
            )
        ),
        "previously_accepted_candidate_count": int(
            (
                candidate_frame[
                    "previously_accepted"
                ].eq(
                    True
                ).sum()
            )
            if not candidate_frame.empty
            else 0
        ),
        "group_scene_counts": {
            str(key): int(value)
            for key, value in (
                scene_frame.groupby(
                    "group"
                ).size().to_dict()
                if not scene_frame.empty
                else {}
            ).items()
        },
        "group_candidate_counts": {
            str(key): int(value)
            for key, value in (
                candidate_frame.groupby(
                    "group"
                ).size().to_dict()
                if not candidate_frame.empty
                else {}
            ).items()
        },
        "candidate_manifest": (
            relative(
                MANIFEST_PATH
            )
        ),
        "scene_report": (
            relative(
                SCENE_REPORT_PATH
            )
        ),
        "crop_directory": (
            relative(
                CROP_DIR
            )
        ),
        "contact_sheet": (
            relative(
                CONTACT_SHEET_PATH
            )
            if CONTACT_SHEET_PATH.exists()
            else None
        ),
        "parameters": {
            "max_images_per_group": int(
                args.max_images_per_group
            ),
            "max_candidates_per_image": int(
                args.max_candidates_per_image
            ),
            "segmentation_threshold": float(
                args.segmentation_threshold
            ),
            "minimum_component_ratio": float(
                args.minimum_component_ratio
            ),
            "minimum_component_pixels": int(
                args.minimum_component_pixels
            ),
            "seed": int(
                args.seed
            ),
        },
        "training_performed": False,
        "manual_annotation_required": False,
        "locked_water_test_used": False,
        "elapsed_seconds": float(
            time.time()
            - start_time
        ),
        "scientific_note": (
            "nc ve nw DARTIS grupları petrolsüz kabul edildiği için "
            "bu sahnelerden çıkarılan bütün petrol adayları LOOK_ALIKE "
            "hard-negative etiketi almıştır. Bu aşama modeli yeniden "
            "eğitmez; yalnız yeni verifier eğitimi için aday veri üretir."
        ),
    }

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print(
        "HARD-NEGATIVE MINING TAMAMLANDI"
    )
    print("=" * 78)

    print(
        "İşlenen sahne:",
        len(scene_frame),
    )

    print(
        "Su kapısından geçen:",
        len(analysed),
    )

    print(
        "Yanlış alarm üreten sahne:",
        len(false_alarm_scenes),
    )

    print(
        "Çıkarılan look-alike adayı:",
        len(candidate_frame),
    )

    print(
        "Eski verifier'ın kabul ettiği aday:",
        (
            int(
                candidate_frame[
                    "previously_accepted"
                ].eq(
                    True
                ).sum()
            )
            if not candidate_frame.empty
            else 0
        ),
    )

    print(
        "Manifest:",
        MANIFEST_PATH.resolve(),
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
