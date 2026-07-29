from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import runpy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]

REGRESSION_SCRIPT = (
    ROOT
    / "scripts"
    / "92_test_v07_known_problem_scenes.py"
)

V07_RESULTS = (
    ROOT
    / "outputs"
    / "v07_fresh_negative_holdout"
    / "per_scene_results.csv"
)

V07_SUMMARY = (
    ROOT
    / "outputs"
    / "v07_fresh_negative_holdout"
    / "summary.json"
)

V07_CONSUMED_MARKER = (
    ROOT
    / "data"
    / "metadata"
    / "v07_fresh_negative_holdout_consumed.json"
)

V07_DATASET = (
    ROOT
    / "data"
    / "metadata"
    / "v07_verifier_binary_dataset.csv"
)

V07_CANONICAL_SCENE_REPORT = (
    ROOT
    / "data"
    / "metadata"
    / "v07_canonical_hard_negative_scene_report.csv"
)

V07_HOLDOUT_MANIFEST = (
    ROOT
    / "data"
    / "metadata"
    / "v07_fresh_end_to_end_holdout_scenes.csv"
)

V06_DEVELOPMENT_MANIFEST = (
    ROOT
    / "data"
    / "metadata"
    / "v06_dartis_water_combined_development_manifest.csv"
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
    / "v08_failure_hard_negative_build"
)

NEW_CROP_DIR = (
    ROOT
    / "data"
    / "mined"
    / "v08_holdout_failure_lookalikes"
    / "crops"
)

NEW_CANDIDATE_MANIFEST = (
    ROOT
    / "data"
    / "metadata"
    / "v08_holdout_failure_lookalikes.csv"
)

V08_DATASET = (
    ROOT
    / "data"
    / "metadata"
    / "v08_verifier_binary_dataset.csv"
)

V08_FRESH_NEGATIVE_HOLDOUT = (
    ROOT
    / "data"
    / "metadata"
    / "v08_fresh_negative_holdout_scenes.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "summary.json"
)

CONTACT_SHEET_PATH = (
    OUTPUT_DIR
    / "new_failure_lookalikes_contact_sheet.jpg"
)

SCENE_PATTERN = re.compile(
    r"(?P<group>nc|nw)-\d{4}-\d{2}-\d{6}",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tüketilmiş v0.7 negatif holdout'taki CONFIRMED_OIL "
            "yanlış alarmlarını yeni LOOK_ALIKE hard-negative crop'larına "
            "dönüştürür, v0.8 verifier veri setini hazırlar ve tamamen yeni "
            "nc/nw negatif holdout sahnelerini yalnız yol düzeyinde ayırır."
        )
    )

    parser.add_argument(
        "--holdout-per-group",
        type=int,
        default=20,
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
        "--device",
        choices=[
            "auto",
            "cuda",
            "cpu",
        ],
        default="auto",
    )

    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Gerekli dosya bulunamadı: {path}"
        )


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


def canonical_scene_id(value: Any) -> str | None:
    if value is None:
        return None

    match = SCENE_PATTERN.search(
        str(value)
    )

    if match is None:
        return None

    return match.group(0).lower()


def deterministic_rank(value: str) -> int:
    digest = hashlib.sha256(
        value.encode("utf-8")
    ).digest()

    return int.from_bytes(
        digest[:8],
        "big",
    )


def safe_name(value: str) -> str:
    return (
        value.replace(
            ":",
            "__",
        )
        .replace(
            "/",
            "_",
        )
        .replace(
            "\\",
            "_",
        )
    )


def enhance_image(image: Image.Image) -> Image.Image:
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
        enhanced = array
    else:
        enhanced = (
            np.clip(
                (
                    array - low
                )
                / (
                    high - low
                ),
                0.0,
                1.0,
            )
            * 255.0
        )

    return Image.fromarray(
        enhanced.astype(
            np.uint8
        )
    )


def load_v07_failure_rows() -> pd.DataFrame:
    for path in (
        V07_RESULTS,
        V07_SUMMARY,
        V07_CONSUMED_MARKER,
    ):
        require_file(path)

    summary = json.loads(
        V07_SUMMARY.read_text(
            encoding="utf-8-sig"
        )
    )

    if summary.get(
        "gate_passed",
        True,
    ):
        raise RuntimeError(
            "v0.7 holdout özeti FAIL değil; bu script yalnız başarısız "
            "holdout sonrası geliştirme için kullanılmalıdır."
        )

    frame = pd.read_csv(
        V07_RESULTS,
        encoding="utf-8-sig",
        low_memory=False,
    )

    required = {
        "scene_id",
        "group",
        "image_path",
        "false_alarm",
    }

    missing = required - set(
        frame.columns
    )

    if missing:
        raise RuntimeError(
            "v0.7 sahne sonuçlarında eksik sütunlar: "
            f"{sorted(missing)}"
        )

    false_alarm = frame[
        frame[
            "false_alarm"
        ].astype(str).str.lower().isin(
            {
                "true",
                "1",
                "yes",
            }
        )
    ].copy()

    if false_alarm.empty:
        raise RuntimeError(
            "v0.7 sonuçlarında confirmed-oil yanlış alarm sahnesi yok."
        )

    false_alarm[
        "resolved_image_path"
    ] = false_alarm[
        "image_path"
    ].map(
        resolve_path
    )

    for path in false_alarm[
        "resolved_image_path"
    ]:
        require_file(
            Path(path)
        )

    return false_alarm.reset_index(
        drop=True
    )


def load_runtime(
    device_name: str,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    require_file(
        REGRESSION_SCRIPT
    )

    regression = runpy.run_path(
        str(
            REGRESSION_SCRIPT
        ),
        run_name=(
            "v08_failure_mining_regression_module"
        ),
    )

    (
        safe_module,
        water_gate,
        oil_pipeline,
    ) = regression[
        "load_models"
    ](
        device_name
    )

    return (
        regression,
        safe_module,
        {
            "water_gate": water_gate,
            "oil_pipeline": oil_pipeline,
        },
    )


def extract_failure_candidates(
    failure_rows: pd.DataFrame,
    regression: dict[str, Any],
    safe_module: dict[str, Any],
    runtime: dict[str, Any],
    args: argparse.Namespace,
) -> pd.DataFrame:
    NEW_CROP_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidate_rows: list[
        dict[str, Any]
    ] = []

    for row in failure_rows.itertuples():
        scene_id = str(
            row.scene_id
        )

        group = str(
            row.group
        ).lower()

        image_path = Path(
            row.resolved_image_path
        )

        image = Image.open(
            image_path
        ).convert("L")

        with torch.inference_mode():
            water_result = safe_module[
                "run_water_gate"
            ](
                image,
                runtime[
                    "water_gate"
                ],
            )

        if not water_result[
            "pipeline_allowed"
        ]:
            raise RuntimeError(
                f"Önceki false-alarm sahnesi artık su kapısından geçmedi: "
                f"{scene_id}"
            )

        oil_result = regression[
            "run_oil_pipeline_v07"
        ](
            image=image,
            safe_water_mask=(
                water_result[
                    "safe_water_mask"
                ]
            ),
            pipeline=runtime[
                "oil_pipeline"
            ],
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

        confirmed = [
            candidate
            for candidate in oil_result[
                "candidate_results"
            ]
            if candidate[
                "decision"
            ]
            == "CONFIRMED_OIL"
        ]

        if not confirmed:
            raise RuntimeError(
                f"Önceki false-alarm sahnesinde CONFIRMED_OIL adayı "
                f"yeniden üretilemedi: {scene_id}"
            )

        for candidate in confirmed:
            crop_box = tuple(
                int(value)
                for value in candidate[
                    "crop_box"
                ]
            )

            crop = image.crop(
                crop_box
            )

            component_index = int(
                candidate[
                    "component_index"
                ]
            )

            crop_name = (
                f"{safe_name(scene_id)}"
                f"__confirmed_{component_index:02d}.png"
            )

            crop_path = (
                NEW_CROP_DIR
                / crop_name
            )

            crop.save(
                crop_path
            )

            candidate_rows.append(
                {
                    "candidate_id": (
                        f"V08_FAIL::{scene_id}::{component_index:02d}"
                    ),
                    "scene_id": scene_id,
                    "group": group,
                    "label": 0,
                    "class_name": (
                        "LOOK_ALIKE"
                    ),
                    "split_role": (
                        "train"
                    ),
                    "crop_path": relative(
                        crop_path
                    ),
                    "source_manifest": relative(
                        V07_RESULTS
                    ),
                    "source_row_label": (
                        "V07_FRESH_NEGATIVE_FALSE_ALARM"
                    ),
                    "source_image_path": relative(
                        image_path
                    ),
                    "component_index": (
                        component_index
                    ),
                    "bbox": json.dumps(
                        candidate[
                            "bbox"
                        ]
                    ),
                    "crop_box": json.dumps(
                        list(
                            crop_box
                        )
                    ),
                    "v07_probability": float(
                        candidate[
                            "calibrated_probability"
                        ]
                    ),
                    "v07_decision": (
                        candidate[
                            "decision"
                        ]
                    ),
                    "hard_negative_weight": (
                        4.0
                    ),
                }
            )

        print(
            f"{scene_id}: {len(confirmed)} yeni LOOK_ALIKE crop"
        )

    frame = pd.DataFrame(
        candidate_rows
    )

    if frame.empty:
        raise RuntimeError(
            "Yeni failure hard-negative crop üretilemedi."
        )

    frame = (
        frame.drop_duplicates(
            subset=[
                "candidate_id",
            ],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    NEW_CANDIDATE_MANIFEST.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
        NEW_CANDIDATE_MANIFEST,
        index=False,
        encoding="utf-8-sig",
    )

    return frame


def build_v08_dataset(
    new_negatives: pd.DataFrame,
) -> pd.DataFrame:
    require_file(
        V07_DATASET
    )

    old = pd.read_csv(
        V07_DATASET,
        encoding="utf-8-sig",
        low_memory=False,
    )

    required = {
        "candidate_id",
        "scene_id",
        "label",
        "class_name",
        "split_role",
        "crop_path",
    }

    missing = required - set(
        old.columns
    )

    if missing:
        raise RuntimeError(
            "v0.7 verifier manifestinde eksik sütunlar: "
            f"{sorted(missing)}"
        )

    old = old.copy()

    if "hard_negative_weight" not in old.columns:
        old[
            "hard_negative_weight"
        ] = 1.0

    new = new_negatives.copy()

    all_columns = sorted(
        set(
            old.columns
        )
        | set(
            new.columns
        )
    )

    old = old.reindex(
        columns=all_columns
    )

    new = new.reindex(
        columns=all_columns
    )

    combined = pd.concat(
        [
            old,
            new,
        ],
        ignore_index=True,
    )

    combined = (
        combined.drop_duplicates(
            subset=[
                "candidate_id",
            ],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    V08_DATASET.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    combined.to_csv(
        V08_DATASET,
        index=False,
        encoding="utf-8-sig",
    )

    return combined


def collect_used_negative_scene_ids(
    dataset: pd.DataFrame,
) -> set[str]:
    used: set[str] = set()

    negative_rows = dataset[
        dataset[
            "label"
        ].astype(int).eq(0)
    ]

    for value in negative_rows[
        "scene_id"
    ].dropna():
        scene_id = canonical_scene_id(
            value
        )

        if scene_id is not None:
            used.add(
                scene_id
            )

    for csv_path, columns in (
        (
            V07_CANONICAL_SCENE_REPORT,
            [
                "sample_id",
                "image_path",
            ],
        ),
        (
            V07_HOLDOUT_MANIFEST,
            [
                "scene_id",
                "image_path",
            ],
        ),
        (
            V06_DEVELOPMENT_MANIFEST,
            [
                "sample_id",
                "annotation_image_path",
            ],
        ),
    ):
        if not csv_path.exists():
            continue

        frame = pd.read_csv(
            csv_path,
            encoding="utf-8-sig",
            low_memory=False,
        )

        for column in columns:
            if column not in frame.columns:
                continue

            for value in frame[
                column
            ].dropna():
                scene_id = canonical_scene_id(
                    value
                )

                if scene_id is not None:
                    used.add(
                        scene_id
                    )

    return used


def build_fresh_negative_holdout(
    used_scene_ids: set[str],
    per_group: int,
) -> pd.DataFrame:
    rows = []

    for group in (
        "nc",
        "nw",
    ):
        group_dir = (
            RAW_DARTIS_ROOT
            / group
        )

        if not group_dir.exists():
            raise FileNotFoundError(
                f"Kanonik DARTIS klasörü bulunamadı: {group_dir}"
            )

        candidates = []

        for path in sorted(
            group_dir.glob(
                "*.jpg"
            )
        ):
            scene_id = canonical_scene_id(
                path.stem
            )

            if (
                scene_id is None
                or scene_id
                in used_scene_ids
            ):
                continue

            candidates.append(
                {
                    "scene_id": (
                        scene_id
                    ),
                    "group": group,
                    "expected_scene_label": (
                        "NO_OIL"
                    ),
                    "split_role": (
                        "v08_fresh_locked_negative"
                    ),
                    "image_path": relative(
                        path
                    ),
                    "selection_rank": (
                        deterministic_rank(
                            f"v08-fresh-negative::{scene_id}"
                        )
                    ),
                    "image_opened": False,
                    "model_inference_run": False,
                }
            )

        candidates = sorted(
            candidates,
            key=lambda item: (
                item[
                    "selection_rank"
                ],
                item[
                    "scene_id"
                ],
            ),
        )

        selected = candidates[
            :per_group
        ]

        if len(
            selected
        ) < per_group:
            raise RuntimeError(
                f"{group} için {per_group} yeni holdout sahnesi "
                f"bulunamadı. Bulunan: {len(selected)}"
            )

        rows.extend(
            selected
        )

    frame = pd.DataFrame(
        rows
    )

    frame = frame.drop(
        columns=[
            "selection_rank",
        ]
    )

    V08_FRESH_NEGATIVE_HOLDOUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
        V08_FRESH_NEGATIVE_HOLDOUT,
        index=False,
        encoding="utf-8-sig",
    )

    return frame


def create_contact_sheet(
    candidates: pd.DataFrame,
) -> None:
    if candidates.empty:
        return

    tile_width = 260
    image_height = 210
    header_height = 48
    tile_height = (
        image_height
        + header_height
    )

    columns = 4
    rows = int(
        math.ceil(
            len(candidates)
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
        candidates.itertuples()
    ):
        image = enhance_image(
            Image.open(
                resolve_path(
                    row.crop_path
                )
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

        x = (
            tile_width
            - image.width
        ) // 2

        y = (
            image_height
            - image.height
        ) // 2

        tile.paste(
            image,
            (
                x,
                header_height
                + y,
            ),
        )

        draw = ImageDraw.Draw(
            tile
        )

        draw.text(
            (
                5,
                5,
            ),
            (
                f"LOOK_ALIKE | {row.scene_id}"
            )[:42],
            fill="white",
            font=font,
        )

        draw.text(
            (
                5,
                24,
            ),
            (
                f"v07 p={float(row.v07_probability):.4f} "
                f"| component={int(row.component_index)}"
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

    failure_rows = (
        load_v07_failure_rows()
    )

    (
        regression,
        safe_module,
        runtime,
    ) = load_runtime(
        args.device
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 78)
    print(
        "v0.8 FAILURE HARD-NEGATIVE BUILD"
    )
    print("=" * 78)

    print(
        "v0.7 false-alarm scene:",
        len(
            failure_rows
        ),
    )

    print(
        "Tüketilmiş holdout yeniden test olarak kullanılmayacak."
    )

    print()

    new_negatives = (
        extract_failure_candidates(
            failure_rows=failure_rows,
            regression=regression,
            safe_module=safe_module,
            runtime=runtime,
            args=args,
        )
    )

    v08_dataset = (
        build_v08_dataset(
            new_negatives
        )
    )

    used_scene_ids = (
        collect_used_negative_scene_ids(
            v08_dataset
        )
    )

    fresh_holdout = (
        build_fresh_negative_holdout(
            used_scene_ids=(
                used_scene_ids
            ),
            per_group=(
                args.holdout_per_group
            ),
        )
    )

    create_contact_sheet(
        new_negatives
    )

    summary = {
        "stage": (
            "v08_failure_hard_negative_build"
        ),
        "v07_gate_result": (
            "FAIL"
        ),
        "v07_consumed_holdout_reused_as_test": (
            False
        ),
        "v07_false_alarm_scene_count": int(
            len(
                failure_rows
            )
        ),
        "new_failure_lookalike_candidate_count": int(
            len(
                new_negatives
            )
        ),
        "new_failure_scene_counts": {
            str(key): int(value)
            for key, value in new_negatives.groupby(
                "scene_id"
            ).size().to_dict().items()
        },
        "v08_dataset_count": int(
            len(
                v08_dataset
            )
        ),
        "v08_dataset_class_counts": {
            str(key): int(value)
            for key, value in v08_dataset.groupby(
                "class_name"
            ).size().to_dict().items()
        },
        "v08_dataset_split_counts": {
            str(key): int(value)
            for key, value in v08_dataset.groupby(
                "split_role"
            ).size().to_dict().items()
        },
        "v08_fresh_negative_holdout_count": int(
            len(
                fresh_holdout
            )
        ),
        "v08_fresh_negative_holdout_group_counts": {
            str(key): int(value)
            for key, value in fresh_holdout.groupby(
                "group"
            ).size().to_dict().items()
        },
        "v08_fresh_negative_holdout_opened": (
            False
        ),
        "v08_fresh_negative_holdout_inference_run": (
            False
        ),
        "new_candidate_manifest": relative(
            NEW_CANDIDATE_MANIFEST
        ),
        "v08_dataset_manifest": relative(
            V08_DATASET
        ),
        "v08_fresh_negative_holdout_manifest": relative(
            V08_FRESH_NEGATIVE_HOLDOUT
        ),
        "contact_sheet": relative(
            CONTACT_SHEET_PATH
        ),
        "training_performed": False,
        "scientific_note": (
            "v0.7 taze negatif holdout FAIL sonucu dondurulmuştur. "
            "Bu holdout'taki altı yanlış alarm sahnesi artık yalnız v0.8 "
            "geliştirme hard-negative verisidir ve tekrar bağımsız test "
            "olarak kullanılmayacaktır. v0.8 için farklı nc/nw sahnelerinden "
            "yeni açılmamış negatif holdout ayrılmıştır."
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

    print()
    print("=" * 78)
    print(
        "v0.8 VERİ HAZIRLAMA TAMAMLANDI"
    )
    print("=" * 78)

    print(
        "Yeni LOOK_ALIKE crop:",
        len(
            new_negatives
        ),
    )

    print(
        "v0.8 dataset:",
        len(
            v08_dataset
        ),
        v08_dataset.groupby(
            "class_name"
        ).size().to_dict(),
    )

    print(
        "Yeni açılmamış negatif holdout:",
        len(
            fresh_holdout
        ),
        fresh_holdout.groupby(
            "group"
        ).size().to_dict(),
    )

    print(
        "Dataset:",
        V08_DATASET.resolve(),
    )

    print(
        "Fresh holdout:",
        V08_FRESH_NEGATIVE_HOLDOUT.resolve(),
    )

    print(
        "Özet:",
        SUMMARY_PATH.resolve(),
    )

    print(
        "Görsel:",
        CONTACT_SHEET_PATH.resolve(),
    )


if __name__ == "__main__":
    main()
