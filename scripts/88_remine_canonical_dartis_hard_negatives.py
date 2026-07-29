from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
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
    / "v07_canonical_hard_negative_mining"
)

CROP_DIR = (
    ROOT
    / "data"
    / "mined"
    / "v07_canonical_lookalike_candidates"
    / "crops"
)

CANDIDATE_MANIFEST = (
    ROOT
    / "data"
    / "metadata"
    / "v07_canonical_hard_negative_candidates.csv"
)

SCENE_REPORT = (
    ROOT
    / "data"
    / "metadata"
    / "v07_canonical_hard_negative_scene_report.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "summary.json"
)

CONTACT_SHEET_PATH = (
    OUTPUT_DIR
    / "top_canonical_hard_negatives.jpg"
)

CANONICAL_PATTERN = re.compile(
    r"(?P<group>nc|nw)-\d{4}-\d{2}-\d{6}",
    re.IGNORECASE,
)

PRIORITY_SCENES = {
    "nc-0140-01-000064",
    "nc-0228-02-000068",
    "nc-0038-00-000038",
}

RESERVED_MANIFESTS = (
    ROOT
    / "data"
    / "metadata"
    / "dartis_external_test_manifest.csv",
    ROOT
    / "data"
    / "metadata"
    / "dartis_hard_negative_val_manifest.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Yalnız data/external/dartis/raw/nc ve raw/nw altındaki "
            "kanonik DARTIS JPG görüntülerinden hard-negative çıkarır. "
            "Türetilmiş maske, patch, overlay ve annotation kopyalarını "
            "kesinlikle taramaz."
        )
    )

    parser.add_argument(
        "--max-images-per-group",
        type=int,
        default=150,
    )

    parser.add_argument(
        "--max-accepted-candidates-per-image",
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
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
    )

    return parser.parse_args()


def relative(path: Path) -> str:
    try:
        return str(
            path.resolve().relative_to(
                ROOT.resolve()
            )
        ).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def canonical_id_from_text(
    value: Any,
) -> str | None:
    if value is None:
        return None

    match = CANONICAL_PATTERN.search(
        str(value)
    )

    if match is None:
        return None

    return match.group(0).lower()


def deterministic_split(
    sample_id: str,
) -> str:
    if sample_id in PRIORITY_SCENES:
        return "train"

    digest = hashlib.sha256(
        sample_id.encode("utf-8")
    ).digest()

    bucket = int.from_bytes(
        digest[:4],
        "big",
    ) % 100

    if bucket < 70:
        return "train"

    if bucket < 85:
        return "calibration"

    return "locked_test"


def collect_reserved_ids() -> set[str]:
    reserved: set[str] = set()

    for csv_path in RESERVED_MANIFESTS:
        if not csv_path.exists():
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
            for value in frame[column].dropna():
                sample_id = canonical_id_from_text(
                    value
                )

                if sample_id is not None:
                    reserved.add(
                        sample_id
                    )

    water_split = (
        ROOT
        / "data"
        / "metadata"
        / "v06_dartis_water_locked_split.csv"
    )

    if water_split.exists():
        try:
            frame = pd.read_csv(
                water_split,
                encoding="utf-8-sig",
                low_memory=False,
            )

            if (
                "sample_id" in frame.columns
                and "split" in frame.columns
            ):
                locked = frame[
                    frame["split"]
                    .astype(str)
                    .str.contains(
                        "locked|test",
                        case=False,
                        regex=True,
                    )
                ]

                for value in locked[
                    "sample_id"
                ].dropna():
                    sample_id = canonical_id_from_text(
                        value
                    )

                    if sample_id is not None:
                        reserved.add(
                            sample_id
                        )
        except Exception:
            pass

    return reserved


def validate_canonical_image(
    path: Path,
) -> tuple[bool, dict[str, Any]]:
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

        percentile_span = float(
            np.percentile(array, 99)
            - np.percentile(array, 1)
        )

        standard_deviation = float(
            array.std()
        )

        valid = (
            image.width >= 256
            and image.height >= 256
            and unique_count >= 32
            and percentile_span >= 12.0
            and standard_deviation >= 4.0
        )

        return (
            valid,
            {
                "width": image.width,
                "height": image.height,
                "unique_values": unique_count,
                "percentile_span": percentile_span,
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


def discover_canonical_images(
    reserved_ids: set[str],
) -> tuple[
    pd.DataFrame,
    list[dict[str, Any]],
]:
    records = []
    rejected = []

    for group in ("nc", "nw"):
        group_dir = (
            RAW_DARTIS_ROOT
            / group
        )

        if not group_dir.exists():
            raise FileNotFoundError(
                f"Kanonik DARTIS klasörü bulunamadı: {group_dir}"
            )

        for path in sorted(
            group_dir.glob("*.jpg")
        ):
            sample_id = canonical_id_from_text(
                path.stem
            )

            if sample_id is None:
                rejected.append(
                    {
                        "path": relative(path),
                        "reason": (
                            "canonical_id_not_found"
                        ),
                    }
                )
                continue

            if sample_id in reserved_ids:
                rejected.append(
                    {
                        "path": relative(path),
                        "sample_id": sample_id,
                        "reason": (
                            "reserved_existing_test_or_validation"
                        ),
                    }
                )
                continue

            valid, diagnostics = (
                validate_canonical_image(
                    path
                )
            )

            if not valid:
                rejected.append(
                    {
                        "path": relative(path),
                        "sample_id": sample_id,
                        "reason": (
                            "not_grayscale_sar_like"
                        ),
                        **diagnostics,
                    }
                )
                continue

            records.append(
                {
                    "sample_id": sample_id,
                    "group": group,
                    "image_path": str(
                        path.resolve()
                    ),
                    "split_role": (
                        deterministic_split(
                            sample_id
                        )
                    ),
                    **diagnostics,
                }
            )

    frame = pd.DataFrame(
        records
    )

    if not frame.empty:
        frame = (
            frame.sort_values(
                [
                    "group",
                    "sample_id",
                ]
            )
            .drop_duplicates(
                subset=[
                    "sample_id",
                ],
                keep="first",
            )
            .reset_index(
                drop=True
            )
        )

    return frame, rejected


def select_scenes(
    frame: pd.DataFrame,
    max_per_group: int,
    seed: int,
) -> pd.DataFrame:
    rng = random.Random(
        seed
    )

    parts = []

    for group in ("nc", "nw"):
        group_frame = frame[
            frame["group"].eq(group)
        ].copy()

        priority = group_frame[
            group_frame[
                "sample_id"
            ].isin(
                PRIORITY_SCENES
            )
        ]

        remainder = group_frame[
            ~group_frame[
                "sample_id"
            ].isin(
                PRIORITY_SCENES
            )
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

        selected = pd.concat(
            [
                priority,
                remainder,
            ],
            ignore_index=True,
        )

        if max_per_group > 0:
            selected = selected.head(
                max_per_group
            )

        parts.append(
            selected
        )

    return pd.concat(
        parts,
        ignore_index=True,
    )


def expand_box(
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


def enhance_for_display(
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
        enhanced = np.clip(
            (array - low)
            / (high - low),
            0.0,
            1.0,
        ) * 255.0

    return Image.fromarray(
        enhanced.astype(np.uint8)
    )


def create_contact_sheet(
    manifest: pd.DataFrame,
) -> None:
    if manifest.empty:
        return

    selected = manifest.sort_values(
        [
            "verifier_probability",
            "area_pixels",
        ],
        ascending=[
            False,
            False,
        ],
    ).head(64)

    tile_width = 260
    image_height = 210
    header_height = 54
    tile_height = (
        image_height
        + header_height
    )
    columns = 4
    rows = int(
        math.ceil(
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
        crop_path = (
            ROOT
            / str(row.crop_path)
        )

        crop = Image.open(
            crop_path
        ).convert("L")

        preview = enhance_for_display(
            crop
        ).convert("RGB")

        draw = ImageDraw.Draw(
            preview
        )

        relative_bbox = [
            int(row.bbox_x1)
            - int(row.crop_x1),
            int(row.bbox_y1)
            - int(row.crop_y1),
            int(row.bbox_x2)
            - int(row.crop_x1),
            int(row.bbox_y2)
            - int(row.crop_y1),
        ]

        draw.rectangle(
            relative_bbox,
            outline=(255, 0, 0),
            width=3,
        )

        preview.thumbnail(
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
            - preview.width
        ) // 2

        offset_y = (
            image_height
            - preview.height
        ) // 2

        tile.paste(
            preview,
            (
                offset_x,
                header_height
                + offset_y,
            ),
        )

        text_draw = ImageDraw.Draw(
            tile
        )

        text_draw.text(
            (5, 5),
            (
                f"{row.sample_id} | "
                f"{row.split_role}"
            )[:43],
            fill="white",
            font=font,
        )

        text_draw.text(
            (5, 25),
            (
                f"p={float(row.verifier_probability):.4f} "
                f"area={int(row.area_pixels)}"
            ),
            fill="white",
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

    if not SAFE_PIPELINE_SCRIPT.exists():
        raise FileNotFoundError(
            f"85 scripti bulunamadı: {SAFE_PIPELINE_SCRIPT}"
        )

    reserved_ids = (
        collect_reserved_ids()
    )

    canonical_frame, rejected = (
        discover_canonical_images(
            reserved_ids
        )
    )

    if canonical_frame.empty:
        raise RuntimeError(
            "Kanonik DARTIS nc/nw görüntüsü bulunamadı."
        )

    canonical_counts = (
        canonical_frame.groupby(
            "group"
        ).size().to_dict()
    )

    selected = select_scenes(
        canonical_frame,
        max_per_group=(
            args.max_images_per_group
        ),
        seed=args.seed,
    )

    selected_counts = (
        selected.groupby(
            "group"
        ).size().to_dict()
    )

    OUTPUT_DIR.mkdir(
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
            "canonical_hard_negative_pipeline"
        ),
    )

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

    print("=" * 78)
    print(
        "v0.7 KANONİK DARTIS HARD-NEGATIVE MINING"
    )
    print("=" * 78)
    print(
        "Kaynak klasör:",
        RAW_DARTIS_ROOT,
    )
    print(
        "Kanonik uygun görüntü:",
        canonical_counts,
    )
    print(
        "Ayrılmış eski test/validation ID:",
        len(reserved_ids),
    )
    print(
        "Bu çalışmada seçilen:",
        selected_counts,
    )
    print(
        "Türetilmiş PNG/patch/mask taranmayacak."
    )
    print()

    scene_rows: list[
        dict[str, Any]
    ] = []

    candidate_rows: list[
        dict[str, Any]
    ] = []

    start_time = time.time()

    for index, row in enumerate(
        selected.itertuples(),
        start=1,
    ):
        image_path = Path(
            row.image_path
        )

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

        scene_row = {
            "sample_id": row.sample_id,
            "group": row.group,
            "split_role": (
                row.split_role
            ),
            "image_path": (
                relative(
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

            accepted_candidates = [
                candidate
                for candidate in oil_result[
                    "candidate_results"
                ]
                if candidate[
                    "accepted"
                ]
            ]

            accepted_candidates = sorted(
                accepted_candidates,
                key=lambda candidate: (
                    float(
                        candidate[
                            "selected_probability"
                        ]
                    ),
                    int(
                        candidate[
                            "area_pixels"
                        ]
                    ),
                ),
                reverse=True,
            )[
                :args.max_accepted_candidates_per_image
            ]

            scene_row[
                "oil_analysis_executed"
            ] = True

            scene_row[
                "candidate_count"
            ] = int(
                oil_result[
                    "candidate_count"
                ]
            )

            scene_row[
                "accepted_candidate_count"
            ] = int(
                oil_result[
                    "accepted_candidate_count"
                ]
            )

            scene_row[
                "scene_false_alarm"
            ] = bool(
                oil_result[
                    "oil_detected"
                ]
            )

            for candidate in accepted_candidates:
                crop_box = expand_box(
                    bbox=candidate[
                        "bbox"
                    ],
                    image_width=image.width,
                    image_height=image.height,
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
                    f"{row.sample_id}"
                    f"__candidate_"
                    f"{int(candidate['component_index']):02d}.png"
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

                candidate_rows.append(
                    {
                        "candidate_id": (
                            f"{row.sample_id}:"
                            f"{int(candidate['component_index']):02d}"
                        ),
                        "label": (
                            "LOOK_ALIKE"
                        ),
                        "label_source": (
                            "DARTIS_CANONICAL_NO_OIL_NC_NW"
                        ),
                        "sample_id": (
                            row.sample_id
                        ),
                        "group": row.group,
                        "split_role": (
                            row.split_role
                        ),
                        "source_image_path": (
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
                        "bbox_x1": bbox[0],
                        "bbox_y1": bbox[1],
                        "bbox_x2": bbox[2],
                        "bbox_y2": bbox[3],
                        "crop_x1": crop_box[0],
                        "crop_y1": crop_box[1],
                        "crop_x2": crop_box[2],
                        "crop_y2": crop_box[3],
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
                    }
                )

        scene_rows.append(
            scene_row
        )

        print(
            f"[{index:03d}/{len(selected):03d}] "
            f"{row.sample_id} "
            f"| water={scene_row['water_decision']} "
            f"| accepted={scene_row['accepted_candidate_count']} "
            f"| false_alarm={scene_row['scene_false_alarm']}"
        )

    scene_frame = pd.DataFrame(
        scene_rows
    )

    candidate_frame = pd.DataFrame(
        candidate_rows
    )

    SCENE_REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    scene_frame.to_csv(
        SCENE_REPORT,
        index=False,
        encoding="utf-8-sig",
    )

    CANDIDATE_MANIFEST.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not candidate_frame.empty:
        candidate_frame = (
            candidate_frame.drop_duplicates(
                subset=[
                    "candidate_id",
                ],
                keep="first",
            )
            .sort_values(
                [
                    "split_role",
                    "verifier_probability",
                ],
                ascending=[
                    True,
                    False,
                ],
            )
            .reset_index(
                drop=True
            )
        )

        candidate_frame.to_csv(
            CANDIDATE_MANIFEST,
            index=False,
            encoding="utf-8-sig",
        )

        create_contact_sheet(
            candidate_frame
        )
    else:
        pd.DataFrame(
            columns=[
                "candidate_id",
                "label",
                "sample_id",
                "group",
                "split_role",
                "source_image_path",
                "crop_path",
            ]
        ).to_csv(
            CANDIDATE_MANIFEST,
            index=False,
            encoding="utf-8-sig",
        )

    water_accepted = scene_frame[
        scene_frame[
            "oil_analysis_executed"
        ].eq(True)
    ]

    false_alarm_scenes = scene_frame[
        scene_frame[
            "scene_false_alarm"
        ].eq(True)
    ]

    summary = {
        "stage": (
            "v07_canonical_dartis_hard_negative_mining"
        ),
        "canonical_source_root": (
            relative(
                RAW_DARTIS_ROOT
            )
        ),
        "canonical_available_counts": {
            str(key): int(value)
            for key, value in canonical_counts.items()
        },
        "reserved_existing_test_or_validation_ids": int(
            len(reserved_ids)
        ),
        "selected_counts": {
            str(key): int(value)
            for key, value in selected_counts.items()
        },
        "processed_scene_count": int(
            len(scene_frame)
        ),
        "water_accepted_scene_count": int(
            len(water_accepted)
        ),
        "false_alarm_scene_count": int(
            len(false_alarm_scenes)
        ),
        "false_alarm_rate_among_water_accepted": float(
            len(false_alarm_scenes)
            / max(
                len(water_accepted),
                1,
            )
        ),
        "accepted_hard_negative_candidate_count": int(
            len(candidate_frame)
        ),
        "candidate_split_counts": {
            str(key): int(value)
            for key, value in (
                candidate_frame.groupby(
                    "split_role"
                ).size().to_dict()
                if not candidate_frame.empty
                else {}
            ).items()
        },
        "candidate_group_counts": {
            str(key): int(value)
            for key, value in (
                candidate_frame.groupby(
                    "group"
                ).size().to_dict()
                if not candidate_frame.empty
                else {}
            ).items()
        },
        "scene_report": (
            relative(
                SCENE_REPORT
            )
        ),
        "candidate_manifest": (
            relative(
                CANDIDATE_MANIFEST
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
        "rejected_source_file_count": int(
            len(rejected)
        ),
        "training_performed": False,
        "manual_annotation_required": False,
        "derived_images_scanned": False,
        "locked_water_test_used": False,
        "elapsed_seconds": float(
            time.time()
            - start_time
        ),
        "scientific_note": (
            "Yalnız kanonik DARTIS raw/nc ve raw/nw JPG görüntüleri "
            "kullanılmıştır. Türetilmiş hard/easy patch, mask, overlay, "
            "annotation kopyası ve önceki mining çıktıları taranmamıştır. "
            "Eski external test/validation listelerinde bulunan örnekler "
            "eğitim adaylarından ayrılmıştır."
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
        "KANONİK HARD-NEGATIVE MINING TAMAMLANDI"
    )
    print("=" * 78)
    print(
        "İşlenen kanonik sahne:",
        len(scene_frame),
    )
    print(
        "Su kapısından geçen:",
        len(water_accepted),
    )
    print(
        "Yanlış alarm sahnesi:",
        len(false_alarm_scenes),
    )
    print(
        "Kabul edilmiş gerçek hard-negative aday:",
        len(candidate_frame),
    )
    print(
        "Split:",
        (
            candidate_frame.groupby(
                "split_role"
            ).size().to_dict()
            if not candidate_frame.empty
            else {}
        ),
    )
    print(
        "Manifest:",
        CANDIDATE_MANIFEST.resolve(),
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
