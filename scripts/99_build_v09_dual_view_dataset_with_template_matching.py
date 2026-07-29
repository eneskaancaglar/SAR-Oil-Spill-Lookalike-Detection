from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

try:
    import cv2
except ImportError:
    cv2 = None


ROOT = Path(__file__).resolve().parents[1]

BASE_DATASET = (
    ROOT
    / "data"
    / "metadata"
    / "v08_verifier_binary_dataset.csv"
)

V08_FALSE_ALARM_RESULTS = (
    ROOT
    / "outputs"
    / "v08_fresh_negative_holdout"
    / "per_scene_results.csv"
)

V08_FALSE_ALARM_ROOT = (
    ROOT
    / "outputs"
    / "v08_fresh_negative_holdout"
    / "confirmed_false_alarms"
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
    / "v09_dual_view_dataset_build_v2"
)

LOCAL_CROP_DIR = (
    ROOT
    / "data"
    / "mined"
    / "v09_holdout_failure_candidates"
    / "local_crops"
)

CONTEXT_CROP_DIR = (
    ROOT
    / "data"
    / "mined"
    / "v09_context_crops_v2"
)

OUTPUT_DATASET = (
    ROOT
    / "data"
    / "metadata"
    / "v09_dual_view_verifier_dataset.csv"
)

NEW_FAILURE_MANIFEST = (
    ROOT
    / "data"
    / "metadata"
    / "v09_new_holdout_failure_lookalikes.csv"
)

CONTEXT_AUDIT = (
    OUTPUT_DIR
    / "context_recovery_audit.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "summary.json"
)

CONTACT_SHEET_PATH = (
    OUTPUT_DIR
    / "dual_view_contact_sheet.jpg"
)

FRESH_HOLDOUT_PATH = (
    ROOT
    / "data"
    / "metadata"
    / "v09_fresh_negative_holdout_scenes.csv"
)

USED_SCENE_MANIFESTS = [
    ROOT
    / "data"
    / "metadata"
    / "v07_canonical_hard_negative_scene_report.csv",
    ROOT
    / "data"
    / "metadata"
    / "v07_fresh_end_to_end_holdout_scenes.csv",
    ROOT
    / "data"
    / "metadata"
    / "v08_fresh_negative_holdout_scenes.csv",
    ROOT
    / "data"
    / "metadata"
    / "v06_dartis_water_combined_development_manifest.csv",
]

SCENE_PATTERN = re.compile(
    r"(?P<group>nc|nw)-\d{4}-\d{2}-\d{6}|"
    r"(?P<oil_group>oc|ow)-\d{4}",
    re.IGNORECASE,
)

PATH_COLUMNS = [
    "source_image_path",
    "original_image_path",
    "scene_image_path",
    "annotation_image_path",
    "image_path",
    "source_path",
]

CROP_PATH_COLUMNS = [
    "crop_path",
    "candidate_crop_path",
    "patch_path",
    "roi_path",
]

BOX_COLUMNS = [
    "crop_box",
    "context_box",
    "candidate_crop_box",
    "bbox",
    "bounding_box",
]

BOX_COLUMN_SETS = [
    ("crop_x1", "crop_y1", "crop_x2", "crop_y2"),
    ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"),
    ("x1", "y1", "x2", "y2"),
    ("xmin", "ymin", "xmax", "ymax"),
    ("left", "top", "right", "bottom"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "v0.9 dual-view verifier veri setini hazırlar. Her aday için "
            "mevcut dar crop ile kaynak SAR sahnesinden çıkarılmış daha geniş "
            "bir bağlam crop'u üretir. v0.8 bağımsız testteki üç yanlış "
            "CONFIRMED_OIL adayı LOOK_ALIKE geliştirme verisine eklenir. "
            "Model eğitilmez ve yeni holdout açılmaz."
        )
    )

    parser.add_argument(
        "--context-scale",
        type=float,
        default=3.0,
    )

    parser.add_argument(
        "--minimum-context-size",
        type=int,
        default=192,
    )

    parser.add_argument(
        "--minimum-overall-context-coverage",
        type=float,
        default=0.95,
    )

    parser.add_argument(
        "--minimum-positive-context-coverage",
        type=float,
        default=0.95,
    )

    parser.add_argument(
        "--minimum-negative-context-coverage",
        type=float,
        default=0.95,
    )

    parser.add_argument(
        "--fresh-holdout-per-group",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--minimum-template-match-score",
        type=float,
        default=0.97,
        help=(
            "Pozitif crop kaynak sahne içinde şablon eşleme ile "
            "bulunurken kabul edilen en düşük birleşik skor."
        ),
    )

    parser.add_argument(
        "--pilot-positive-count",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--minimum-pilot-match-rate",
        type=float,
        default=0.80,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260729,
    )

    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Gerekli dosya bulunamadı: {path}"
        )


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


def scene_id_from_text(value: Any) -> str | None:
    if value is None:
        return None

    match = SCENE_PATTERN.search(
        str(value)
    )

    if match is None:
        return None

    return match.group(0).lower()


def group_from_scene_id(scene_id: str) -> str:
    return scene_id[:2].lower()


def read_grayscale_cv(path: Path) -> np.ndarray:
    if cv2 is None:
        raise RuntimeError(
            "OpenCV bulunamadı. Proje venv içinde şu komutu çalıştırın: "
            '& ".\\.venv\\Scripts\\python.exe" -m pip install opencv-python'
        )

    resolved_path = path.resolve()

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Görüntü dosyası bulunamadı: {resolved_path}"
        )

    # cv2.imread() Windows'ta Türkçe/Unicode klasör adlarını
    # bazı sürümlerde yanlış kodlayabiliyor. Dosyayı NumPy ile
    # byte olarak okuyup cv2.imdecode() kullanmak Unicode-safe'tir.
    try:
        encoded_bytes = np.fromfile(
            resolved_path,
            dtype=np.uint8,
        )
    except OSError as error:
        raise RuntimeError(
            f"Görüntü byte olarak okunamadı: {resolved_path}"
        ) from error

    if encoded_bytes.size == 0:
        raise RuntimeError(
            f"Görüntü dosyası boş: {resolved_path}"
        )

    array = cv2.imdecode(
        encoded_bytes,
        cv2.IMREAD_GRAYSCALE,
    )

    if array is None:
        raise RuntimeError(
            f"Görüntü OpenCV ile decode edilemedi: {resolved_path}"
        )

    return array


def match_template_in_source(
    crop_path: Path,
    source_path: Path,
    minimum_score: float,
) -> tuple[
    tuple[int, int, int, int] | None,
    float,
    float,
]:
    source = read_grayscale_cv(
        source_path
    )

    crop = read_grayscale_cv(
        crop_path
    )

    source_height, source_width = (
        source.shape
    )

    crop_height, crop_width = (
        crop.shape
    )

    if (
        crop_height < 8
        or crop_width < 8
    ):
        return (
            None,
            0.0,
            1.0,
        )

    scales = [
        1.0,
        0.50,
        0.625,
        0.75,
        0.875,
        1.125,
        1.25,
        1.50,
        1.75,
        2.0,
    ]

    best_box = None
    best_score = -1.0
    best_scale = 1.0

    for scale in scales:
        target_width = int(
            round(
                crop_width
                * scale
            )
        )

        target_height = int(
            round(
                crop_height
                * scale
            )
        )

        if (
            target_width < 8
            or target_height < 8
            or target_width > source_width
            or target_height > source_height
        ):
            continue

        interpolation = (
            cv2.INTER_AREA
            if scale < 1.0
            else cv2.INTER_CUBIC
        )

        if scale == 1.0:
            template = crop
        else:
            template = cv2.resize(
                crop,
                (
                    target_width,
                    target_height,
                ),
                interpolation=interpolation,
            )

        if float(
            template.std()
        ) < 1e-6:
            result = cv2.matchTemplate(
                source,
                template,
                cv2.TM_SQDIFF_NORMED,
            )

            minimum_value, _, minimum_location, _ = (
                cv2.minMaxLoc(
                    result
                )
            )

            correlation_score = float(
                1.0
                - minimum_value
            )

            location = (
                minimum_location
            )
        else:
            result = cv2.matchTemplate(
                source,
                template,
                cv2.TM_CCOEFF_NORMED,
            )

            _, maximum_value, _, maximum_location = (
                cv2.minMaxLoc(
                    result
                )
            )

            correlation_score = float(
                maximum_value
            )

            location = (
                maximum_location
            )

        x1, y1 = (
            int(
                location[0]
            ),
            int(
                location[1]
            ),
        )

        x2 = (
            x1
            + target_width
        )

        y2 = (
            y1
            + target_height
        )

        matched_patch = source[
            y1:y2,
            x1:x2,
        ]

        if matched_patch.shape != template.shape:
            continue

        normalized_mae = float(
            np.mean(
                np.abs(
                    matched_patch.astype(
                        np.float32
                    )
                    - template.astype(
                        np.float32
                    )
                )
            )
            / 255.0
        )

        intensity_score = float(
            max(
                0.0,
                1.0
                - normalized_mae,
            )
        )

        combined_score = float(
            0.75
            * correlation_score
            + 0.25
            * intensity_score
        )

        if combined_score > best_score:
            best_score = (
                combined_score
            )

            best_scale = float(
                scale
            )

            best_box = (
                x1,
                y1,
                x2,
                y2,
            )

        if (
            scale == 1.0
            and combined_score
            >= 0.999
        ):
            break

    if (
        best_box is None
        or best_score
        < minimum_score
    ):
        return (
            None,
            float(
                best_score
            ),
            float(
                best_scale
            ),
        )

    return (
        best_box,
        float(
            best_score
        ),
        float(
            best_scale
        ),
    )


def run_positive_template_match_pilot(
    base: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    positive = base[
        base[
            "class_name"
        ].astype(str).eq(
            "CONFIRMED_OIL"
        )
    ].copy()

    positive[
        "pilot_rank"
    ] = positive[
        "candidate_id"
    ].astype(str).map(
        lambda value: (
            deterministic_rank(
                f"v09-template-pilot::{args.seed}::{value}"
            )
        )
    )

    pilot = positive.sort_values(
        "pilot_rank"
    ).head(
        int(
            args.pilot_positive_count
        )
    )

    rows = []

    for row in pilot.to_dict(
        orient="records"
    ):
        scene_id = str(
            row[
                "scene_id"
            ]
        )

        crop_path = resolve_path(
            row.get(
                "crop_path"
            )
        )

        source_path = locate_raw_scene(
            scene_id
        )

        box = None
        score = 0.0
        scale = 1.0
        error = ""

        try:
            if (
                crop_path is None
                or not crop_path.exists()
            ):
                error = "missing_crop"
            elif source_path is None:
                error = "missing_source"
            else:
                (
                    box,
                    score,
                    scale,
                ) = match_template_in_source(
                    crop_path=crop_path,
                    source_path=source_path,
                    minimum_score=float(
                        args.minimum_template_match_score
                    ),
                )

                if box is None:
                    error = "score_below_threshold"
        except Exception as exception:
            error = repr(
                exception
            )

        rows.append(
            {
                "candidate_id": str(
                    row[
                        "candidate_id"
                    ]
                ),
                "scene_id": scene_id,
                "crop_path": (
                    relative(
                        crop_path
                    )
                    if crop_path
                    is not None
                    else None
                ),
                "source_image_path": (
                    relative(
                        source_path
                    )
                    if source_path
                    is not None
                    else None
                ),
                "matched": bool(
                    box
                    is not None
                ),
                "match_score": float(
                    score
                ),
                "match_scale": float(
                    scale
                ),
                "candidate_box": (
                    json.dumps(
                        list(
                            box
                        )
                    )
                    if box
                    is not None
                    else None
                ),
                "error": error,
            }
        )

    result = pd.DataFrame(
        rows
    )

    pilot_path = (
        OUTPUT_DIR
        / "positive_template_match_pilot.csv"
    )

    pilot_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        pilot_path,
        index=False,
        encoding="utf-8-sig",
    )

    match_rate = float(
        result[
            "matched"
        ].eq(
            True
        ).mean()
    )

    print(
        "Positive template pilot:",
        f"{int(result['matched'].eq(True).sum())}/{len(result)}",
        f"rate={match_rate:.4f}",
    )

    if match_rate < float(
        args.minimum_pilot_match_rate
    ):
        raise RuntimeError(
            "Pozitif template-match pilot başarısız. "
            f"Match rate={match_rate:.4f}, gerekli="
            f"{float(args.minimum_pilot_match_rate):.4f}. "
            f"Audit: {pilot_path}"
        )

    return result


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


def deterministic_rank(value: str) -> int:
    digest = hashlib.sha256(
        value.encode("utf-8")
    ).digest()

    return int.from_bytes(
        digest[:8],
        "big",
    )


def parse_box_value(value: Any) -> tuple[int, int, int, int] | None:
    if value is None:
        return None

    if isinstance(
        value,
        (list, tuple, np.ndarray),
    ):
        parsed = list(value)
    else:
        text = str(value).strip()

        if not text or text.lower() == "nan":
            return None

        parsed = None

        for parser in (
            json.loads,
            ast.literal_eval,
        ):
            try:
                candidate = parser(text)

                if isinstance(
                    candidate,
                    (list, tuple),
                ):
                    parsed = list(
                        candidate
                    )

                    break
            except Exception:
                continue

        if parsed is None:
            numbers = re.findall(
                r"-?\d+(?:\.\d+)?",
                text,
            )

            if len(numbers) >= 4:
                parsed = numbers[:4]

    if parsed is None or len(parsed) < 4:
        return None

    try:
        box = tuple(
            int(
                round(
                    float(value)
                )
            )
            for value in parsed[:4]
        )
    except (
        TypeError,
        ValueError,
    ):
        return None

    x1, y1, x2, y2 = box

    if x2 <= x1 or y2 <= y1:
        return None

    return box


def box_from_row(row: dict[str, Any]) -> tuple[int, int, int, int] | None:
    for column in BOX_COLUMNS:
        if column in row:
            box = parse_box_value(
                row.get(column)
            )

            if box is not None:
                return box

    for columns in BOX_COLUMN_SETS:
        if not all(
            column in row
            for column in columns
        ):
            continue

        values = [
            row.get(column)
            for column in columns
        ]

        if any(
            value is None
            or str(value).lower()
            == "nan"
            for value in values
        ):
            continue

        try:
            box = tuple(
                int(
                    round(
                        float(value)
                    )
                )
                for value in values
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

        x1, y1, x2, y2 = box

        if x2 > x1 and y2 > y1:
            return box

    return None


def source_path_from_row(
    row: dict[str, Any],
) -> Path | None:
    for column in PATH_COLUMNS:
        if column not in row:
            continue

        path = resolve_path(
            row.get(column)
        )

        if path is not None and path.exists():
            return path

    return None


def locate_raw_scene(scene_id: str) -> Path | None:
    group = group_from_scene_id(
        scene_id
    )

    group_dir = (
        RAW_DARTIS_ROOT
        / group
    )

    if not group_dir.exists():
        return None

    exact = group_dir / f"{scene_id}.jpg"

    if exact.exists():
        return exact.resolve()

    matches = list(
        group_dir.glob(
            f"{scene_id}.*"
        )
    )

    if matches:
        return matches[0].resolve()

    return None


def normalized_path_key(value: Any) -> str:
    path = resolve_path(value)

    if path is None:
        return ""

    return str(path).lower()


def basename_key(value: Any) -> str:
    path = resolve_path(value)

    if path is None:
        return ""

    return path.name.lower()


class SourceManifestIndex:
    def __init__(self) -> None:
        self.cache: dict[
            Path,
            dict[str, Any],
        ] = {}

    def load(
        self,
        manifest_path: Path,
    ) -> dict[str, Any]:
        manifest_path = (
            manifest_path.resolve()
        )

        if manifest_path in self.cache:
            return self.cache[
                manifest_path
            ]

        if not manifest_path.exists():
            result = {
                "frame": pd.DataFrame(),
                "by_path": {},
                "by_basename_scene": {},
                "columns": [],
            }

            self.cache[
                manifest_path
            ] = result

            return result

        frame = pd.read_csv(
            manifest_path,
            encoding="utf-8-sig",
            low_memory=False,
        )

        rows = frame.to_dict(
            orient="records"
        )

        by_path: dict[
            str,
            dict[str, Any],
        ] = {}

        by_basename_scene: dict[
            tuple[str, str],
            list[dict[str, Any]],
        ] = {}

        crop_columns = [
            column
            for column in CROP_PATH_COLUMNS
            if column in frame.columns
        ]

        for row in rows:
            scene_id = (
                scene_id_from_text(
                    row.get(
                        "scene_id"
                    )
                )
                or scene_id_from_text(
                    row.get(
                        "sample_id"
                    )
                )
            )

            for column in crop_columns:
                value = row.get(
                    column
                )

                key = normalized_path_key(
                    value
                )

                if key:
                    by_path[
                        key
                    ] = row

                basename = basename_key(
                    value
                )

                if (
                    basename
                    and scene_id
                    is not None
                ):
                    by_basename_scene.setdefault(
                        (
                            basename,
                            scene_id,
                        ),
                        [],
                    ).append(
                        row
                    )

        result = {
            "frame": frame,
            "by_path": by_path,
            "by_basename_scene": (
                by_basename_scene
            ),
            "columns": list(
                frame.columns
            ),
        }

        self.cache[
            manifest_path
        ] = result

        return result

    def match(
        self,
        manifest_path: Path,
        dataset_row: dict[str, Any],
    ) -> dict[str, Any] | None:
        index = self.load(
            manifest_path
        )

        crop_path = dataset_row.get(
            "crop_path"
        )

        exact = index[
            "by_path"
        ].get(
            normalized_path_key(
                crop_path
            )
        )

        if exact is not None:
            return exact

        scene_id = str(
            dataset_row.get(
                "scene_id",
                "",
            )
        )

        candidates = index[
            "by_basename_scene"
        ].get(
            (
                basename_key(
                    crop_path
                ),
                scene_id,
            ),
            [],
        )

        if len(candidates) == 1:
            return candidates[0]

        return None


def clip_box(
    box: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = box

    x1 = max(
        0,
        min(
            width - 1,
            x1,
        ),
    )

    y1 = max(
        0,
        min(
            height - 1,
            y1,
        ),
    )

    x2 = max(
        x1 + 1,
        min(
            width,
            x2,
        ),
    )

    y2 = max(
        y1 + 1,
        min(
            height,
            y2,
        ),
    )

    if x2 <= x1 or y2 <= y1:
        return None

    return (
        x1,
        y1,
        x2,
        y2,
    )


def expand_box(
    box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    scale: float,
    minimum_size: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box

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
                width * scale
            )
        ),
        minimum_size,
    )

    target_height = max(
        int(
            round(
                height * scale
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

    context = (
        int(
            round(
                center_x
                - target_width
                / 2.0
            )
        ),
        int(
            round(
                center_y
                - target_height
                / 2.0
            )
        ),
        int(
            round(
                center_x
                + target_width
                / 2.0
            )
        ),
        int(
            round(
                center_y
                + target_height
                / 2.0
            )
        ),
    )

    clipped = clip_box(
        context,
        image_width,
        image_height,
    )

    if clipped is None:
        raise RuntimeError(
            "Bağlam kutusu geçersiz."
        )

    return clipped


def image_features(
    local_image: Image.Image,
    context_image: Image.Image,
    source_image: Image.Image,
    candidate_box: tuple[int, int, int, int],
) -> dict[str, float]:
    local = np.asarray(
        local_image.convert("L"),
        dtype=np.float32,
    )

    context = np.asarray(
        context_image.convert("L"),
        dtype=np.float32,
    )

    source = np.asarray(
        source_image.convert("L"),
        dtype=np.float32,
    )

    x1, y1, x2, y2 = (
        candidate_box
    )

    width = max(
        x2 - x1,
        1,
    )

    height = max(
        y2 - y1,
        1,
    )

    def edge_density(
        array: np.ndarray,
    ) -> float:
        if min(
            array.shape
        ) < 2:
            return 0.0

        gradient_x = np.abs(
            np.diff(
                array,
                axis=1,
            )
        )

        gradient_y = np.abs(
            np.diff(
                array,
                axis=0,
            )
        )

        values = np.concatenate(
            [
                gradient_x.reshape(
                    -1
                ),
                gradient_y.reshape(
                    -1
                ),
            ]
        )

        return float(
            (
                values
                >= np.percentile(
                    values,
                    85,
                )
            ).mean()
        )

    local_mean = float(
        local.mean()
    )

    context_mean = float(
        context.mean()
    )

    source_mean = float(
        source.mean()
    )

    local_std = float(
        local.std()
    )

    context_std = float(
        context.std()
    )

    source_std = float(
        source.std()
    )

    return {
        "candidate_width_pixels": float(
            width
        ),
        "candidate_height_pixels": float(
            height
        ),
        "candidate_aspect_ratio": float(
            width
            / height
        ),
        "candidate_area_box_pixels": float(
            width
            * height
        ),
        "candidate_center_x_normalized": float(
            (
                x1 + x2
            )
            / 2.0
            / source_image.width
        ),
        "candidate_center_y_normalized": float(
            (
                y1 + y2
            )
            / 2.0
            / source_image.height
        ),
        "candidate_width_normalized": float(
            width
            / source_image.width
        ),
        "candidate_height_normalized": float(
            height
            / source_image.height
        ),
        "local_mean": local_mean,
        "local_std": local_std,
        "local_p10": float(
            np.percentile(
                local,
                10,
            )
        ),
        "local_p90": float(
            np.percentile(
                local,
                90,
            )
        ),
        "local_edge_density": (
            edge_density(
                local
            )
        ),
        "context_mean": context_mean,
        "context_std": context_std,
        "context_edge_density": (
            edge_density(
                context
            )
        ),
        "source_mean": source_mean,
        "source_std": source_std,
        "local_minus_context_mean": float(
            local_mean
            - context_mean
        ),
        "local_to_context_std_ratio": float(
            local_std
            / max(
                context_std,
                1e-6,
            )
        ),
        "context_minus_source_mean": float(
            context_mean
            - source_mean
        ),
    }


def extract_new_v08_failures() -> pd.DataFrame:
    require_file(
        V08_FALSE_ALARM_RESULTS
    )

    results = pd.read_csv(
        V08_FALSE_ALARM_RESULTS,
        encoding="utf-8-sig",
        low_memory=False,
    )

    false_alarm = results[
        results[
            "false_alarm"
        ].astype(str).str.lower().isin(
            {
                "true",
                "1",
                "yes",
            }
        )
    ].copy()

    if len(false_alarm) != 2:
        raise RuntimeError(
            "v0.8 sonuçlarında iki false-alarm sahnesi bekleniyordu; "
            f"bulunan: {len(false_alarm)}"
        )

    LOCAL_CROP_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    sorted_scenes = sorted(
        false_alarm[
            "scene_id"
        ].astype(str)
    )

    split_by_scene = {
        sorted_scenes[0]: "train",
        sorted_scenes[1]: (
            "calibration"
        ),
    }

    for scene_row in false_alarm.itertuples():
        scene_id = str(
            scene_row.scene_id
        )

        group = str(
            scene_row.group
        ).lower()

        scene_dir = (
            V08_FALSE_ALARM_ROOT
            / scene_id
        )

        input_path = (
            scene_dir
            / "input.png"
        )

        decisions_path = (
            scene_dir
            / "candidate_decisions.csv"
        )

        require_file(
            input_path
        )

        require_file(
            decisions_path
        )

        source_image = Image.open(
            input_path
        ).convert("L")

        decisions = pd.read_csv(
            decisions_path,
            encoding="utf-8-sig",
            low_memory=False,
        )

        confirmed = decisions[
            decisions[
                "decision"
            ].astype(str).eq(
                "CONFIRMED_OIL"
            )
        ]

        for candidate in confirmed.to_dict(
            orient="records"
        ):
            component_index = int(
                candidate[
                    "component_index"
                ]
            )

            crop_box = parse_box_value(
                candidate.get(
                    "crop_box"
                )
            )

            bbox = parse_box_value(
                candidate.get(
                    "bbox"
                )
            )

            if crop_box is None:
                raise RuntimeError(
                    f"{scene_id} component {component_index}: crop_box yok."
                )

            crop_box = clip_box(
                crop_box,
                source_image.width,
                source_image.height,
            )

            if crop_box is None:
                raise RuntimeError(
                    f"{scene_id} component {component_index}: crop_box geçersiz."
                )

            local_crop = source_image.crop(
                crop_box
            )

            crop_name = (
                f"{safe_name(scene_id)}"
                f"__confirmed_{component_index:02d}.png"
            )

            local_path = (
                LOCAL_CROP_DIR
                / crop_name
            )

            local_crop.save(
                local_path
            )

            rows.append(
                {
                    "candidate_id": (
                        f"V09_FAIL::{scene_id}::{component_index:02d}"
                    ),
                    "scene_id": scene_id,
                    "group": group,
                    "label": 0,
                    "class_name": (
                        "LOOK_ALIKE"
                    ),
                    "split_role": (
                        split_by_scene[
                            scene_id
                        ]
                    ),
                    "crop_path": relative(
                        local_path
                    ),
                    "source_image_path": relative(
                        input_path
                    ),
                    "crop_box": json.dumps(
                        list(
                            crop_box
                        )
                    ),
                    "bbox": json.dumps(
                        list(
                            bbox
                            if bbox is not None
                            else crop_box
                        )
                    ),
                    "source_manifest": relative(
                        V08_FALSE_ALARM_RESULTS
                    ),
                    "source_row_label": (
                        "V08_FRESH_NEGATIVE_FALSE_ALARM"
                    ),
                    "hard_negative_weight": (
                        6.0
                    ),
                    "v08_probability": float(
                        candidate[
                            "calibrated_probability"
                        ]
                    ),
                }
            )

    frame = pd.DataFrame(
        rows
    )

    if len(frame) != 3:
        raise RuntimeError(
            "Üç yeni v0.8 false-alarm adayı bekleniyordu; "
            f"bulunan: {len(frame)}"
        )

    NEW_FAILURE_MANIFEST.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame.to_csv(
        NEW_FAILURE_MANIFEST,
        index=False,
        encoding="utf-8-sig",
    )

    return frame


def recover_context_metadata(
    row: dict[str, Any],
    source_index: SourceManifestIndex,
    crop_path: Path,
    minimum_template_match_score: float,
) -> tuple[
    Path | None,
    tuple[int, int, int, int] | None,
    str,
    float | None,
    float | None,
]:
    direct_source = source_path_from_row(
        row
    )

    direct_box = box_from_row(
        row
    )

    if (
        direct_source is not None
        and direct_box is not None
    ):
        return (
            direct_source,
            direct_box,
            "dataset_row",
            None,
            None,
        )

    source_manifest_value = row.get(
        "source_manifest"
    )

    source_manifest_path = resolve_path(
        source_manifest_value
    )

    matched_row = None

    if (
        source_manifest_path
        is not None
        and source_manifest_path.exists()
        and source_manifest_path.suffix.lower()
        == ".csv"
    ):
        matched_row = source_index.match(
            source_manifest_path,
            row,
        )

    matched_source = (
        source_path_from_row(
            matched_row
        )
        if matched_row
        is not None
        else None
    )

    matched_box = (
        box_from_row(
            matched_row
        )
        if matched_row
        is not None
        else None
    )

    source = (
        direct_source
        or matched_source
        or locate_raw_scene(
            str(
                row[
                    "scene_id"
                ]
            )
        )
    )

    box = (
        direct_box
        or matched_box
    )

    if (
        source is not None
        and box is not None
    ):
        method = (
            "source_manifest"
            if matched_row
            is not None
            else "dataset_box_raw_scene"
        )

        return (
            source,
            box,
            method,
            None,
            None,
        )

    if source is None:
        return (
            None,
            box,
            "missing_source_image",
            None,
            None,
        )

    (
        template_box,
        match_score,
        match_scale,
    ) = match_template_in_source(
        crop_path=crop_path,
        source_path=source,
        minimum_score=float(
            minimum_template_match_score
        ),
    )

    if template_box is not None:
        return (
            source,
            template_box,
            "template_match",
            float(
                match_score
            ),
            float(
                match_scale
            ),
        )

    return (
        source,
        None,
        "template_match_below_threshold",
        float(
            match_score
        ),
        float(
            match_scale
        ),
    )


def build_context_dataset(
    base: pd.DataFrame,
    new_failures: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    all_columns = sorted(
        set(
            base.columns
        )
        | set(
            new_failures.columns
        )
    )

    combined = pd.concat(
        [
            base.reindex(
                columns=all_columns
            ),
            new_failures.reindex(
                columns=all_columns
            ),
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

    source_index = (
        SourceManifestIndex()
    )

    CONTEXT_CROP_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_rows = []

    audit_rows = []

    for position, row in enumerate(
        combined.to_dict(
            orient="records"
        ),
        start=1,
    ):
        candidate_id = str(
            row[
                "candidate_id"
            ]
        )

        crop_path = resolve_path(
            row.get(
                "crop_path"
            )
        )

        audit = {
            "candidate_id": (
                candidate_id
            ),
            "scene_id": str(
                row.get(
                    "scene_id",
                    "",
                )
            ),
            "class_name": str(
                row.get(
                    "class_name",
                    "",
                )
            ),
            "split_role": str(
                row.get(
                    "split_role",
                    "",
                )
            ),
            "local_crop_exists": bool(
                crop_path is not None
                and crop_path.exists()
            ),
            "context_recovered": False,
            "recovery_method": "",
            "template_match_score": None,
            "template_match_scale": None,
            "error": "",
        }

        if (
            crop_path is None
            or not crop_path.exists()
        ):
            audit[
                "error"
            ] = (
                "missing_local_crop"
            )

            audit_rows.append(
                audit
            )

            continue

        try:
            (
                source_path,
                candidate_box,
                method,
                template_match_score,
                template_match_scale,
            ) = recover_context_metadata(
                row,
                source_index,
                crop_path=crop_path,
                minimum_template_match_score=float(
                    args.minimum_template_match_score
                ),
            )

            audit[
                "recovery_method"
            ] = method

            audit[
                "template_match_score"
            ] = template_match_score

            audit[
                "template_match_scale"
            ] = template_match_scale

            if (
                source_path is None
                or candidate_box is None
            ):
                audit[
                    "error"
                ] = method

                audit_rows.append(
                    audit
                )

                continue

            source_image = Image.open(
                source_path
            ).convert("L")

            clipped_box = clip_box(
                candidate_box,
                source_image.width,
                source_image.height,
            )

            if clipped_box is None:
                audit[
                    "error"
                ] = (
                    "candidate_box_outside_image"
                )

                audit_rows.append(
                    audit
                )

                continue

            context_box = expand_box(
                clipped_box,
                source_image.width,
                source_image.height,
                scale=float(
                    args.context_scale
                ),
                minimum_size=int(
                    args.minimum_context_size
                ),
            )

            context_crop = (
                source_image.crop(
                    context_box
                )
            )

            local_image = Image.open(
                crop_path
            ).convert("L")

            context_name = (
                hashlib.sha256(
                    candidate_id.encode(
                        "utf-8"
                    )
                ).hexdigest()[:20]
                + ".png"
            )

            context_path = (
                CONTEXT_CROP_DIR
                / context_name
            )

            context_crop.save(
                context_path
            )

            feature_values = image_features(
                local_image=local_image,
                context_image=context_crop,
                source_image=source_image,
                candidate_box=clipped_box,
            )

            output_row = dict(
                row
            )

            output_row.update(
                {
                    "crop_path": relative(
                        crop_path
                    ),
                    "source_image_path": (
                        relative(
                            source_path
                        )
                    ),
                    "candidate_box": (
                        json.dumps(
                            list(
                                clipped_box
                            )
                        )
                    ),
                    "context_box": (
                        json.dumps(
                            list(
                                context_box
                            )
                        )
                    ),
                    "context_crop_path": (
                        relative(
                            context_path
                        )
                    ),
                    "context_recovery_method": (
                        method
                    ),
                    "template_match_score": (
                        template_match_score
                    ),
                    "template_match_scale": (
                        template_match_scale
                    ),
                    **feature_values,
                }
            )

            output_rows.append(
                output_row
            )

            audit[
                "context_recovered"
            ] = True

            audit_rows.append(
                audit
            )

        except Exception as error:
            audit[
                "error"
            ] = repr(
                error
            )

            audit_rows.append(
                audit
            )

        if (
            position % 250 == 0
            or position
            == len(
                combined
            )
        ):
            print(
                f"Context recovery: "
                f"{position}/{len(combined)}"
            )

    return (
        pd.DataFrame(
            output_rows
        ),
        pd.DataFrame(
            audit_rows
        ),
    )


def coverage(
    audit: pd.DataFrame,
    class_name: str | None = None,
) -> float:
    subset = audit

    if class_name is not None:
        subset = subset[
            subset[
                "class_name"
            ].eq(
                class_name
            )
        ]

    if subset.empty:
        return 0.0

    return float(
        subset[
            "context_recovered"
        ].eq(
            True
        ).mean()
    )


def collect_used_scene_ids(
    dataset: pd.DataFrame,
) -> set[str]:
    used = set(
        dataset[
            "scene_id"
        ].dropna().astype(str)
    )

    for manifest_path in USED_SCENE_MANIFESTS:
        if not manifest_path.exists():
            continue

        frame = pd.read_csv(
            manifest_path,
            encoding="utf-8-sig",
            low_memory=False,
        )

        for column in frame.columns:
            if not any(
                token in column.lower()
                for token in (
                    "scene",
                    "sample",
                    "image",
                    "path",
                )
            ):
                continue

            for value in frame[
                column
            ].dropna():
                scene_id = scene_id_from_text(
                    value
                )

                if scene_id is not None:
                    used.add(
                        scene_id
                    )

    return used


def build_fresh_holdout(
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
                f"DARTIS raw klasörü bulunamadı: {group_dir}"
            )

        candidates = []

        for path in group_dir.glob(
            "*.jpg"
        ):
            scene_id = scene_id_from_text(
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
                    "scene_id": scene_id,
                    "group": group,
                    "expected_scene_label": (
                        "NO_OIL"
                    ),
                    "split_role": (
                        "v09_fresh_locked_negative"
                    ),
                    "image_path": relative(
                        path
                    ),
                    "image_opened": False,
                    "model_inference_run": False,
                    "selection_rank": (
                        deterministic_rank(
                            f"v09-fresh-negative::{scene_id}"
                        )
                    ),
                }
            )

        selected = sorted(
            candidates,
            key=lambda item: (
                item[
                    "selection_rank"
                ],
                item[
                    "scene_id"
                ],
            ),
        )[
            :per_group
        ]

        if len(selected) < per_group:
            raise RuntimeError(
                f"{group} için {per_group} yeni holdout sahnesi "
                f"bulunamadı. Bulunan: {len(selected)}"
            )

        rows.extend(
            selected
        )

    frame = pd.DataFrame(
        rows
    ).drop(
        columns=[
            "selection_rank",
        ]
    )

    frame.to_csv(
        FRESH_HOLDOUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    return frame


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
        result.astype(
            np.uint8
        )
    )


def create_contact_sheet(
    dataset: pd.DataFrame,
    seed: int,
) -> None:
    selected_parts = []

    for class_name in (
        "CONFIRMED_OIL",
        "LOOK_ALIKE",
    ):
        class_frame = dataset[
            dataset[
                "class_name"
            ].eq(
                class_name
            )
        ].copy()

        class_frame[
            "preview_rank"
        ] = class_frame[
            "candidate_id"
        ].astype(str).map(
            lambda value: (
                deterministic_rank(
                    f"preview::{seed}::{value}"
                )
            )
        )

        selected_parts.append(
            class_frame.sort_values(
                "preview_rank"
            ).head(
                24
            )
        )

    failures = dataset[
        dataset[
            "source_row_label"
        ].astype(str).eq(
            "V08_FRESH_NEGATIVE_FALSE_ALARM"
        )
    ]

    selected = pd.concat(
        [
            failures,
            *selected_parts,
        ],
        ignore_index=True,
    ).drop_duplicates(
        subset=[
            "candidate_id",
        ],
        keep="first",
    ).head(
        56
    )

    if selected.empty:
        return

    tile_width = 420
    image_height = 180
    header_height = 50
    tile_height = (
        image_height
        + header_height
    )

    columns = 2
    rows = int(
        math.ceil(
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
        local = enhance(
            Image.open(
                resolve_path(
                    row.crop_path
                )
            )
        ).convert("RGB")

        context = enhance(
            Image.open(
                resolve_path(
                    row.context_crop_path
                )
            )
        ).convert("RGB")

        local.thumbnail(
            (
                tile_width // 2,
                image_height,
            ),
            Image.Resampling.LANCZOS,
        )

        context.thumbnail(
            (
                tile_width // 2,
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

        tile.paste(
            local,
            (
                (
                    tile_width // 2
                    - local.width
                )
                // 2,
                header_height
                + (
                    image_height
                    - local.height
                )
                // 2,
            ),
        )

        tile.paste(
            context,
            (
                tile_width // 2
                + (
                    tile_width // 2
                    - context.width
                )
                // 2,
                header_height
                + (
                    image_height
                    - context.height
                )
                // 2,
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
                f"{row.class_name} | {row.split_role} | "
                f"{row.scene_id}"
            )[:68],
            fill="white",
            font=font,
        )

        draw.text(
            (
                5,
                25,
            ),
            "LOCAL CROP                         CONTEXT CROP",
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

    require_file(
        BASE_DATASET
    )

    base = pd.read_csv(
        BASE_DATASET,
        encoding="utf-8-sig",
        low_memory=False,
    )

    if cv2 is None:
        raise RuntimeError(
            "OpenCV bulunamadı. Proje venv içinde şu komutu çalıştırın: "
            '& ".\\.venv\\Scripts\\python.exe" -m pip install opencv-python'
        )

    run_positive_template_match_pilot(
        base=base,
        args=args,
    )

    new_failures = (
        extract_new_v08_failures()
    )

    (
        context_dataset,
        audit,
    ) = build_context_dataset(
        base=base,
        new_failures=(
            new_failures
        ),
        args=args,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit.to_csv(
        CONTEXT_AUDIT,
        index=False,
        encoding="utf-8-sig",
    )

    overall_coverage = coverage(
        audit
    )

    positive_coverage = coverage(
        audit,
        "CONFIRMED_OIL",
    )

    negative_coverage = coverage(
        audit,
        "LOOK_ALIKE",
    )

    checks = {
        "overall_context_coverage": (
            overall_coverage
            >= args.minimum_overall_context_coverage
        ),
        "positive_context_coverage": (
            positive_coverage
            >= args.minimum_positive_context_coverage
        ),
        "negative_context_coverage": (
            negative_coverage
            >= args.minimum_negative_context_coverage
        ),
        "new_failure_candidate_count": (
            len(
                new_failures
            )
            == 3
        ),
    }

    failed_checks = [
        name
        for name, passed
        in checks.items()
        if not passed
    ]

    build_passed = (
        len(
            failed_checks
        )
        == 0
    )

    fresh_holdout = None

    if build_passed:
        split_leakage = (
            context_dataset[
                context_dataset[
                    "split_role"
                ].isin(
                    [
                        "train",
                        "calibration",
                    ]
                )
            ]
            .groupby(
                "scene_id"
            )[
                "split_role"
            ]
            .nunique()
        )

        if int(
            (
                split_leakage
                > 1
            ).sum()
        ) != 0:
            raise RuntimeError(
                "v0.9 datasetinde sahne bazlı split leakage bulundu."
            )

        OUTPUT_DATASET.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        context_dataset.to_csv(
            OUTPUT_DATASET,
            index=False,
            encoding="utf-8-sig",
        )

        used_scene_ids = (
            collect_used_scene_ids(
                context_dataset
            )
        )

        fresh_holdout = (
            build_fresh_holdout(
                used_scene_ids=(
                    used_scene_ids
                ),
                per_group=(
                    args.fresh_holdout_per_group
                ),
            )
        )

        create_contact_sheet(
            context_dataset,
            seed=args.seed,
        )

    summary = {
        "stage": (
            "v09_dual_view_dataset_build"
        ),
        "base_candidate_count": int(
            len(
                base
            )
        ),
        "new_v08_failure_candidate_count": int(
            len(
                new_failures
            )
        ),
        "attempted_candidate_count": int(
            len(
                audit
            )
        ),
        "context_recovered_count": int(
            audit[
                "context_recovered"
            ].eq(
                True
            ).sum()
        ),
        "overall_context_coverage": (
            overall_coverage
        ),
        "positive_context_coverage": (
            positive_coverage
        ),
        "negative_context_coverage": (
            negative_coverage
        ),
        "build_passed": bool(
            build_passed
        ),
        "failed_checks": (
            failed_checks
        ),
        "criteria": {
            "minimum_overall_context_coverage": float(
                args.minimum_overall_context_coverage
            ),
            "minimum_positive_context_coverage": float(
                args.minimum_positive_context_coverage
            ),
            "minimum_negative_context_coverage": float(
                args.minimum_negative_context_coverage
            ),
            "minimum_template_match_score": float(
                args.minimum_template_match_score
            ),
            "minimum_pilot_match_rate": float(
                args.minimum_pilot_match_rate
            ),
        },
        "output_dataset": (
            relative(
                OUTPUT_DATASET
            )
            if OUTPUT_DATASET.exists()
            else None
        ),
        "context_audit": relative(
            CONTEXT_AUDIT
        ),
        "new_failure_manifest": relative(
            NEW_FAILURE_MANIFEST
        ),
        "context_crop_directory": relative(
            CONTEXT_CROP_DIR
        ),
        "fresh_holdout_manifest": (
            relative(
                FRESH_HOLDOUT_PATH
            )
            if FRESH_HOLDOUT_PATH.exists()
            else None
        ),
        "fresh_holdout_count": (
            int(
                len(
                    fresh_holdout
                )
            )
            if fresh_holdout
            is not None
            else 0
        ),
        "fresh_holdout_group_counts": (
            {
                str(key): int(value)
                for key, value
                in fresh_holdout.groupby(
                    "group"
                ).size().to_dict().items()
            }
            if fresh_holdout
            is not None
            else {}
        ),
        "contact_sheet": (
            relative(
                CONTACT_SHEET_PATH
            )
            if CONTACT_SHEET_PATH.exists()
            else None
        ),
        "training_performed": False,
        "fresh_v09_holdout_opened": False,
        "fresh_v09_holdout_inference_run": False,
        "scientific_note": (
            "v0.8 bağımsız negatif holdout sonucu FAIL olarak korunmuştur. "
            "Üç yanlış CONFIRMED_OIL adayı v0.9 geliştirme LOOK_ALIKE "
            "verisine eklenmiştir. Her aday için dar crop ve daha geniş "
            "bağlam crop'u hazırlanmıştır. Bağlam kurtarma kapsamı eşikleri "
            "sağlanmadan eğitim manifesti ve yeni holdout dondurulmaz."
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
        "v0.9 DUAL-VIEW VERİ SETİ SONUCU"
    )
    print("=" * 78)

    print(
        "Build:",
        (
            "PASS"
            if build_passed
            else "FAIL"
        ),
    )

    print(
        "Yeni v0.8 false-alarm adayı:",
        len(
            new_failures
        ),
    )

    print(
        "Context coverage overall:",
        f"{overall_coverage:.4f}",
    )

    print(
        "Context coverage positive:",
        f"{positive_coverage:.4f}",
    )

    print(
        "Context coverage negative:",
        f"{negative_coverage:.4f}",
    )

    print(
        "Failed:",
        failed_checks,
    )

    if build_passed:
        print(
            "v0.9 dataset:",
            len(
                context_dataset
            ),
            context_dataset.groupby(
                "class_name"
            ).size().to_dict(),
        )

        print(
            "Yeni açılmamış holdout:",
            len(
                fresh_holdout
            ),
            fresh_holdout.groupby(
                "group"
            ).size().to_dict(),
        )

        print(
            "Dataset:",
            OUTPUT_DATASET.resolve(),
        )

        print(
            "Görsel:",
            CONTACT_SHEET_PATH.resolve(),
        )

    print(
        "Audit:",
        CONTEXT_AUDIT.resolve(),
    )

    print(
        "Özet:",
        SUMMARY_PATH.resolve(),
    )


if __name__ == "__main__":
    main()
