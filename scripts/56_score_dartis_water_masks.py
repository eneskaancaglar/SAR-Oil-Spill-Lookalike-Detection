from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as functional
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    ROOT
    / "data"
    / "metadata"
    / "v06_dartis_water_masks_homography.csv"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "v06_dartis_water_mask_quality"
)

QUALITY_PATH = (
    OUTPUT_DIR
    / "mask_quality.csv"
)

MANUAL_REVIEW_PATH = (
    OUTPUT_DIR
    / "manual_review_candidates.csv"
)

STATUS_COUNTS_PATH = (
    OUTPUT_DIR
    / "status_counts.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "summary.json"
)

REVIEW_ROOT = (
    OUTPUT_DIR
    / "review"
)

REPORT_PATH = (
    ROOT
    / "reports"
    / "v06_dartis_water_mask_quality.md"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Homography tabanlı DARTIS kara maskelerinin "
            "SAR görüntüsündeki kıyı çizgisine ne kadar "
            "uyduğunu ölçer."
        )
    )

    parser.add_argument(
        "--search-radius",
        type=int,
        default=24,
        help=(
            "Mevcut maske sınırının çevresinde aranacak "
            "maksimum piksel kayması."
        ),
    )

    parser.add_argument(
        "--shift-step",
        type=int,
        default=4,
        help=(
            "Kayma aramasındaki piksel adımı."
        ),
    )

    parser.add_argument(
        "--maximum-boundary-samples",
        type=int,
        default=2500,
    )

    parser.add_argument(
        "--review-limit-per-status",
        type=int,
        default=120,
    )

    parser.add_argument(
        "--maximum-auto-shift",
        type=float,
        default=4.0,
    )

    parser.add_argument(
        "--maximum-auto-gain",
        type=float,
        default=1.06,
    )

    parser.add_argument(
        "--minimum-auto-contrast",
        type=float,
        default=1.10,
    )

    parser.add_argument(
        "--maximum-local-shift-spread",
        type=float,
        default=6.0,
    )

    return parser.parse_args()


def relative(path: Path) -> str:
    try:
        return str(
            path.resolve().relative_to(
                ROOT.resolve()
            )
        ).replace(
            "\\",
            "/",
        )

    except ValueError:
        return str(
            path.resolve()
        )


def resolve_project_path(
    value: Any,
) -> Path:
    path = Path(
        str(value).strip()
    )

    if not path.is_absolute():
        path = ROOT / path

    return path.resolve()


def load_grayscale(
    path: Path,
) -> np.ndarray:
    with Image.open(
        path
    ) as image:
        return np.asarray(
            image.convert(
                "L"
            ),
            dtype=np.uint8,
        )


def load_binary_mask(
    path: Path,
) -> np.ndarray:
    with Image.open(
        path
    ) as image:
        return (
            np.asarray(
                image.convert(
                    "L"
                )
            )
            > 127
        )


def dilate_mask(
    mask: np.ndarray,
    radius: int,
) -> np.ndarray:
    if radius <= 0:
        return mask.astype(
            bool,
            copy=True,
        )

    tensor = torch.from_numpy(
        mask.astype(
            np.float32
        )
    )[None, None]

    result = functional.max_pool2d(
        tensor,
        kernel_size=(
            radius * 2
            + 1
        ),
        stride=1,
        padding=radius,
    )

    return (
        result[
            0,
            0,
        ].numpy()
        > 0.5
    )


def erode_mask(
    mask: np.ndarray,
    radius: int,
) -> np.ndarray:
    if radius <= 0:
        return mask.astype(
            bool,
            copy=True,
        )

    inverse = (
        ~mask.astype(bool)
    )

    dilated_inverse = dilate_mask(
        inverse,
        radius,
    )

    return (
        ~dilated_inverse
    )


def mask_boundary(
    mask: np.ndarray,
) -> np.ndarray:
    dilated = dilate_mask(
        mask,
        1,
    )

    eroded = erode_mask(
        mask,
        1,
    )

    return (
        dilated
        & ~eroded
    )


def calculate_gradient(
    grayscale: np.ndarray,
) -> np.ndarray:
    image = (
        grayscale.astype(
            np.float32
        )
        / 255.0
    )

    tensor = torch.from_numpy(
        image
    )[None, None]

    # SAR speckle etkisini azaltmak i?in
    # ?nce hafif yumu?atma uygula.
    smoothed = functional.avg_pool2d(
        tensor,
        kernel_size=5,
        stride=1,
        padding=2,
    )

    # conv2d a??rl?k bi?imi:
    # [out_channel, in_channel, height, width]
    sobel_x = torch.tensor(
        [
            [
                -1.0,
                0.0,
                1.0,
            ],
            [
                -2.0,
                0.0,
                2.0,
            ],
            [
                -1.0,
                0.0,
                1.0,
            ],
        ],
        dtype=torch.float32,
    ).view(
        1,
        1,
        3,
        3,
    )

    sobel_y = torch.tensor(
        [
            [
                -1.0,
                -2.0,
                -1.0,
            ],
            [
                0.0,
                0.0,
                0.0,
            ],
            [
                1.0,
                2.0,
                1.0,
            ],
        ],
        dtype=torch.float32,
    ).view(
        1,
        1,
        3,
        3,
    )

    gradient_x = functional.conv2d(
        smoothed,
        sobel_x,
        padding=1,
    )

    gradient_y = functional.conv2d(
        smoothed,
        sobel_y,
        padding=1,
    )

    gradient = torch.sqrt(
        gradient_x.square()
        + gradient_y.square()
        + 1e-12
    )[
        0,
        0,
    ].numpy()

    normalization_value = float(
        np.quantile(
            gradient,
            0.995,
        )
    )

    if normalization_value <= 1e-12:
        return np.zeros_like(
            gradient,
            dtype=np.float32,
        )

    return np.clip(
        gradient
        / normalization_value,
        0.0,
        1.0,
    ).astype(
        np.float32
    )


def robust_score(
    values: np.ndarray,
) -> float:
    values = np.asarray(
        values,
        dtype=np.float32,
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    if values.size == 0:
        return 0.0

    if values.size < 10:
        return float(
            values.mean()
        )

    lower = float(
        np.quantile(
            values,
            0.20,
        )
    )

    upper = float(
        np.quantile(
            values,
            0.90,
        )
    )

    selected = values[
        (values >= lower)
        & (values <= upper)
    ]

    if selected.size == 0:
        selected = values

    return float(
        selected.mean()
    )


def sample_coordinates(
    boundary: np.ndarray,
    maximum_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    y_coordinates, x_coordinates = np.nonzero(
        boundary
    )

    if len(
        y_coordinates
    ) <= maximum_samples:
        return (
            y_coordinates,
            x_coordinates,
        )

    indices = np.linspace(
        0,
        len(y_coordinates) - 1,
        num=maximum_samples,
        dtype=int,
    )

    return (
        y_coordinates[
            indices
        ],
        x_coordinates[
            indices
        ],
    )


def score_shift(
    gradient: np.ndarray,
    y_coordinates: np.ndarray,
    x_coordinates: np.ndarray,
    dx: int,
    dy: int,
) -> float:
    height, width = gradient.shape

    shifted_y = (
        y_coordinates
        + dy
    )

    shifted_x = (
        x_coordinates
        + dx
    )

    valid = (
        (shifted_y >= 0)
        & (shifted_y < height)
        & (shifted_x >= 0)
        & (shifted_x < width)
    )

    if int(
        valid.sum()
    ) < 20:
        return 0.0

    values = gradient[
        shifted_y[
            valid
        ],
        shifted_x[
            valid
        ],
    ]

    return robust_score(
        values
    )


def find_best_shift(
    gradient: np.ndarray,
    y_coordinates: np.ndarray,
    x_coordinates: np.ndarray,
    radius: int,
    step: int,
) -> dict[str, float]:
    zero_score = score_shift(
        gradient,
        y_coordinates,
        x_coordinates,
        0,
        0,
    )

    best_score = zero_score
    best_dx = 0
    best_dy = 0

    shift_values = list(
        range(
            -radius,
            radius + 1,
            step,
        )
    )

    if 0 not in shift_values:
        shift_values.append(
            0
        )

    shift_values = sorted(
        set(
            shift_values
        )
    )

    for dy in shift_values:
        for dx in shift_values:
            score = score_shift(
                gradient,
                y_coordinates,
                x_coordinates,
                dx,
                dy,
            )

            if score > best_score:
                best_score = score
                best_dx = dx
                best_dy = dy

    shift_magnitude = math.sqrt(
        best_dx ** 2
        + best_dy ** 2
    )

    shift_gain = float(
        best_score
        / max(
            zero_score,
            1e-8,
        )
    )

    return {
        "zero_score": float(
            zero_score
        ),
        "best_score": float(
            best_score
        ),
        "best_dx": int(
            best_dx
        ),
        "best_dy": int(
            best_dy
        ),
        "shift_magnitude": float(
            shift_magnitude
        ),
        "shift_gain": float(
            shift_gain
        ),
    }


def calculate_local_shift_spread(
    gradient: np.ndarray,
    boundary: np.ndarray,
    radius: int,
    step: int,
    maximum_samples: int,
) -> tuple[
    float,
    int,
    str,
]:
    height, width = boundary.shape

    regions = [
        (
            "top_left",
            0,
            height // 2,
            0,
            width // 2,
        ),
        (
            "top_right",
            0,
            height // 2,
            width // 2,
            width,
        ),
        (
            "bottom_left",
            height // 2,
            height,
            0,
            width // 2,
        ),
        (
            "bottom_right",
            height // 2,
            height,
            width // 2,
            width,
        ),
    ]

    local_shifts = []

    for (
        region_name,
        y_start,
        y_end,
        x_start,
        x_end,
    ) in regions:
        region_boundary = np.zeros_like(
            boundary,
            dtype=bool,
        )

        region_boundary[
            y_start:y_end,
            x_start:x_end,
        ] = boundary[
            y_start:y_end,
            x_start:x_end,
        ]

        boundary_count = int(
            region_boundary.sum()
        )

        if boundary_count < 60:
            continue

        (
            region_y,
            region_x,
        ) = sample_coordinates(
            region_boundary,
            max(
                250,
                maximum_samples // 4,
            ),
        )

        result = find_best_shift(
            gradient,
            region_y,
            region_x,
            radius,
            step,
        )

        local_shifts.append(
            {
                "region": region_name,
                "dx": result[
                    "best_dx"
                ],
                "dy": result[
                    "best_dy"
                ],
            }
        )

    if len(local_shifts) <= 1:
        spread = 0.0

    else:
        maximum_distance = 0.0

        for first_index in range(
            len(local_shifts)
        ):
            for second_index in range(
                first_index + 1,
                len(local_shifts),
            ):
                first = local_shifts[
                    first_index
                ]

                second = local_shifts[
                    second_index
                ]

                distance = math.sqrt(
                    (
                        first["dx"]
                        - second["dx"]
                    ) ** 2
                    + (
                        first["dy"]
                        - second["dy"]
                    ) ** 2
                )

                maximum_distance = max(
                    maximum_distance,
                    distance,
                )

        spread = maximum_distance

    description = " | ".join(
        (
            f"{item['region']}:"
            f"dx={item['dx']},"
            f"dy={item['dy']}"
        )
        for item in local_shifts
    )

    return (
        float(spread),
        int(
            len(local_shifts)
        ),
        description,
    )


def shifted_boundary(
    boundary: np.ndarray,
    dx: int,
    dy: int,
) -> np.ndarray:
    output = np.zeros_like(
        boundary,
        dtype=bool,
    )

    height, width = boundary.shape

    y_coordinates, x_coordinates = (
        np.nonzero(
            boundary
        )
    )

    shifted_y = (
        y_coordinates
        + dy
    )

    shifted_x = (
        x_coordinates
        + dx
    )

    valid = (
        (shifted_y >= 0)
        & (shifted_y < height)
        & (shifted_x >= 0)
        & (shifted_x < width)
    )

    output[
        shifted_y[
            valid
        ],
        shifted_x[
            valid
        ],
    ] = True

    return output


def classify_record(
    *,
    shift_magnitude: float,
    shift_gain: float,
    boundary_contrast: float,
    local_shift_spread: float,
    safe_water_ratio: float,
    boundary_pixels: int,
    args: argparse.Namespace,
) -> tuple[
    str,
    str,
    str,
]:
    reasons = []
    severity = "LOW"

    if boundary_pixels < 50:
        return (
            "REJECT",
            "HIGH",
            "Kıyı sınırı oluşturmak için yeterli piksel yok.",
        )

    if (
        safe_water_ratio <= 0.001
        or safe_water_ratio >= 0.999
    ):
        return (
            "REJECT",
            "HIGH",
            "Maske tamamen veya neredeyse tamamen tek sınıf.",
        )

    if (
        safe_water_ratio < 0.03
        or safe_water_ratio > 0.97
    ):
        reasons.append(
            "Aşırı safe_water_ratio"
        )

    if (
        shift_magnitude
        > args.maximum_auto_shift
    ):
        reasons.append(
            "Mevcut sınırdan uzakta daha güçlü görüntü kenarı var"
        )

    if (
        shift_gain
        > args.maximum_auto_gain
    ):
        reasons.append(
            "Kaydırılan sınır mevcut sınırdan belirgin biçimde daha iyi"
        )

    if (
        boundary_contrast
        < args.minimum_auto_contrast
    ):
        reasons.append(
            "Maske sınırı görüntü kenarlarından yeterince ayrışmıyor"
        )

    if (
        local_shift_spread
        > args.maximum_local_shift_spread
    ):
        reasons.append(
            "Kıyının farklı bölümleri farklı yönlerde kayıyor"
        )

    if (
        shift_magnitude >= 12
        or shift_gain >= 1.18
        or local_shift_spread >= 14
    ):
        severity = "HIGH"

    elif reasons:
        severity = "MEDIUM"

    if reasons:
        return (
            "REVIEW_REQUIRED",
            severity,
            " | ".join(
                reasons
            ),
        )

    return (
        "AUTO_ACCEPT",
        "LOW",
        "Otomatik hizalama kontrolleri geçti.",
    )


def calculate_quality_score(
    *,
    shift_magnitude: float,
    shift_gain: float,
    boundary_contrast: float,
    local_shift_spread: float,
    safe_water_ratio: float,
) -> float:
    shift_score = math.exp(
        -shift_magnitude
        / 7.0
    )

    gain_score = math.exp(
        -max(
            shift_gain - 1.0,
            0.0,
        )
        * 7.0
    )

    contrast_score = float(
        np.clip(
            (
                boundary_contrast
                - 0.90
            )
            / 0.40,
            0.0,
            1.0,
        )
    )

    local_score = math.exp(
        -local_shift_spread
        / 10.0
    )

    ratio_score = (
        1.0
        if (
            0.03
            <= safe_water_ratio
            <= 0.97
        )
        else 0.50
    )

    quality = (
        shift_score * 0.30
        + gain_score * 0.25
        + contrast_score * 0.20
        + local_score * 0.20
        + ratio_score * 0.05
    )

    return float(
        np.clip(
            quality,
            0.0,
            1.0,
        )
    )


def apply_color(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    if not mask.any():
        return

    color_array = np.asarray(
        color,
        dtype=np.float32,
    )

    image[
        mask
    ] = (
        image[
            mask
        ]
        * (
            1.0
            - alpha
        )
        + color_array
        * alpha
    )


def create_review_image(
    *,
    grayscale: np.ndarray,
    land_mask: np.ndarray,
    gradient: np.ndarray,
    best_dx: int,
    best_dy: int,
    destination: Path,
) -> None:
    current_boundary = mask_boundary(
        land_mask
    )

    proposed_boundary = shifted_boundary(
        current_boundary,
        best_dx,
        best_dy,
    )

    original_rgb = np.stack(
        [
            grayscale,
            grayscale,
            grayscale,
        ],
        axis=-1,
    ).astype(
        np.float32
    )

    overlay = original_rgb.copy()

    apply_color(
        overlay,
        land_mask,
        (
            255,
            45,
            45,
        ),
        0.60,
    )

    apply_color(
        overlay,
        current_boundary,
        (
            255,
            215,
            30,
        ),
        0.95,
    )

    gradient_uint8 = (
        np.clip(
            gradient,
            0.0,
            1.0,
        )
        * 255.0
    ).astype(
        np.uint8
    )

    gradient_current = np.stack(
        [
            gradient_uint8,
            gradient_uint8,
            gradient_uint8,
        ],
        axis=-1,
    ).astype(
        np.float32
    )

    apply_color(
        gradient_current,
        current_boundary,
        (
            255,
            215,
            30,
        ),
        1.0,
    )

    gradient_shifted = np.stack(
        [
            gradient_uint8,
            gradient_uint8,
            gradient_uint8,
        ],
        axis=-1,
    ).astype(
        np.float32
    )

    apply_color(
        gradient_shifted,
        current_boundary,
        (
            255,
            215,
            30,
        ),
        0.90,
    )

    apply_color(
        gradient_shifted,
        proposed_boundary,
        (
            20,
            220,
            255,
        ),
        1.0,
    )

    combined = np.concatenate(
        [
            original_rgb,
            overlay,
            gradient_current,
            gradient_shifted,
        ],
        axis=1,
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Image.fromarray(
        np.clip(
            combined,
            0,
            255,
        ).astype(
            np.uint8
        )
    ).save(
        destination,
        format="PNG",
    )


def choose_review_rows(
    dataframe: pd.DataFrame,
    status: str,
    maximum_count: int,
) -> pd.DataFrame:
    subset = dataframe[
        dataframe[
            "quality_status"
        ].eq(
            status
        )
    ].copy()

    if subset.empty:
        return subset

    if status == "AUTO_ACCEPT":
        subset = subset.sort_values(
            [
                "quality_score",
                "sample_id",
            ],
            ascending=[
                False,
                True,
            ],
        )

    else:
        subset = subset.sort_values(
            [
                "quality_score",
                "shift_magnitude",
                "sample_id",
            ],
            ascending=[
                True,
                False,
                True,
            ],
        )

    return subset.head(
        maximum_count
    )


def main() -> None:
    args = parse_args()

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest bulunamadı: {MANIFEST_PATH}"
        )

    if args.search_radius < 0:
        raise ValueError(
            "--search-radius negatif olamaz."
        )

    if args.shift_step <= 0:
        raise ValueError(
            "--shift-step pozitif olmalıdır."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = pd.read_csv(
        MANIFEST_PATH,
        encoding="utf-8-sig",
        low_memory=False,
    )

    required_columns = {
        "sample_id",
        "subset",
        "surface_context",
        "image_path",
        "land_mask_path",
        "safe_water_mask_path",
        "generation_status",
    }

    missing_columns = (
        required_columns
        - set(
            dataframe.columns
        )
    )

    if missing_columns:
        raise RuntimeError(
            "Manifestte eksik sütunlar:\n"
            + "\n".join(
                sorted(
                    missing_columns
                )
            )
        )

    coastal = dataframe[
        dataframe[
            "surface_context"
        ].eq(
            "coast"
        )
        & dataframe[
            "generation_status"
        ].eq(
            "SUCCESS"
        )
    ].copy()

    if coastal.empty:
        raise RuntimeError(
            "Başarılı kıyılı görüntü bulunamadı."
        )

    print("=" * 78)
    print(
        "ROBUST BINARY OIL DETECTOR v0.6"
    )
    print(
        "DARTIS KARA-SU MASKESİ HİZALAMA KALİTESİ"
    )
    print("=" * 78)

    print(
        "Kıyılı görüntü:",
        len(coastal),
    )

    print(
        "Kayma aralığı:",
        f"±{args.search_radius}",
    )

    print(
        "Kayma adımı:",
        args.shift_step,
    )

    result_rows = []

    total = len(
        coastal
    )

    for record_number, row in enumerate(
        coastal.to_dict(
            orient="records"
        ),
        start=1,
    ):
        sample_id = str(
            row[
                "sample_id"
            ]
        )

        image_path = resolve_project_path(
            row[
                "image_path"
            ]
        )

        land_mask_path = (
            resolve_project_path(
                row[
                    "land_mask_path"
                ]
            )
        )

        safe_water_mask_path = (
            resolve_project_path(
                row[
                    "safe_water_mask_path"
                ]
            )
        )

        base_result = {
            **row,
            "quality_status": (
                "REJECT"
            ),
            "quality_severity": (
                "HIGH"
            ),
            "quality_reason": "",
        }

        try:
            for path in (
                image_path,
                land_mask_path,
                safe_water_mask_path,
            ):
                if not path.exists():
                    raise FileNotFoundError(
                        f"Dosya bulunamadı: {path}"
                    )

            grayscale = load_grayscale(
                image_path
            )

            land_mask = load_binary_mask(
                land_mask_path
            )

            safe_water_mask = load_binary_mask(
                safe_water_mask_path
            )

            if (
                grayscale.shape
                != land_mask.shape
                or grayscale.shape
                != safe_water_mask.shape
            ):
                raise RuntimeError(
                    "Görüntü ve maske boyutları eşleşmiyor."
                )

            boundary = mask_boundary(
                land_mask
            )

            boundary_pixels = int(
                boundary.sum()
            )

            total_pixels = int(
                land_mask.size
            )

            land_ratio = float(
                land_mask.sum()
                / total_pixels
            )

            safe_water_ratio = float(
                safe_water_mask.sum()
                / total_pixels
            )

            gradient = calculate_gradient(
                grayscale
            )

            (
                boundary_y,
                boundary_x,
            ) = sample_coordinates(
                boundary,
                args.maximum_boundary_samples,
            )

            shift_result = find_best_shift(
                gradient,
                boundary_y,
                boundary_x,
                args.search_radius,
                args.shift_step,
            )

            search_band = dilate_mask(
                boundary,
                args.search_radius,
            )

            reference_values = gradient[
                search_band
            ]

            reference_score = max(
                robust_score(
                    reference_values
                ),
                1e-8,
            )

            boundary_contrast = float(
                shift_result[
                    "best_score"
                ]
                / reference_score
            )

            (
                local_shift_spread,
                local_region_count,
                local_shift_description,
            ) = calculate_local_shift_spread(
                gradient,
                boundary,
                args.search_radius,
                args.shift_step,
                args.maximum_boundary_samples,
            )

            (
                quality_status,
                quality_severity,
                quality_reason,
            ) = classify_record(
                shift_magnitude=(
                    shift_result[
                        "shift_magnitude"
                    ]
                ),
                shift_gain=(
                    shift_result[
                        "shift_gain"
                    ]
                ),
                boundary_contrast=(
                    boundary_contrast
                ),
                local_shift_spread=(
                    local_shift_spread
                ),
                safe_water_ratio=(
                    safe_water_ratio
                ),
                boundary_pixels=(
                    boundary_pixels
                ),
                args=args,
            )

            quality_score = (
                calculate_quality_score(
                    shift_magnitude=(
                        shift_result[
                            "shift_magnitude"
                        ]
                    ),
                    shift_gain=(
                        shift_result[
                            "shift_gain"
                        ]
                    ),
                    boundary_contrast=(
                        boundary_contrast
                    ),
                    local_shift_spread=(
                        local_shift_spread
                    ),
                    safe_water_ratio=(
                        safe_water_ratio
                    ),
                )
            )

            result_rows.append(
                {
                    **row,
                    "image_path": relative(
                        image_path
                    ),
                    "land_mask_path": relative(
                        land_mask_path
                    ),
                    "safe_water_mask_path": (
                        relative(
                            safe_water_mask_path
                        )
                    ),
                    "boundary_pixels": (
                        boundary_pixels
                    ),
                    "land_ratio_checked": (
                        land_ratio
                    ),
                    "safe_water_ratio_checked": (
                        safe_water_ratio
                    ),
                    "zero_shift_score": (
                        shift_result[
                            "zero_score"
                        ]
                    ),
                    "best_shift_score": (
                        shift_result[
                            "best_score"
                        ]
                    ),
                    "best_dx": (
                        shift_result[
                            "best_dx"
                        ]
                    ),
                    "best_dy": (
                        shift_result[
                            "best_dy"
                        ]
                    ),
                    "shift_magnitude": (
                        shift_result[
                            "shift_magnitude"
                        ]
                    ),
                    "shift_gain": (
                        shift_result[
                            "shift_gain"
                        ]
                    ),
                    "boundary_contrast": (
                        boundary_contrast
                    ),
                    "local_shift_spread": (
                        local_shift_spread
                    ),
                    "local_region_count": (
                        local_region_count
                    ),
                    "local_shift_description": (
                        local_shift_description
                    ),
                    "quality_score": (
                        quality_score
                    ),
                    "quality_status": (
                        quality_status
                    ),
                    "quality_severity": (
                        quality_severity
                    ),
                    "quality_reason": (
                        quality_reason
                    ),
                    "manual_reviewed": False,
                    "manual_decision": "",
                    "eligible_for_water_model": (
                        quality_status
                        == "AUTO_ACCEPT"
                    ),
                }
            )

        except Exception as error:
            result_rows.append(
                {
                    **base_result,
                    "quality_reason": (
                        f"{type(error).__name__}: "
                        f"{error}"
                    ),
                    "quality_score": 0.0,
                    "manual_reviewed": False,
                    "manual_decision": "",
                    "eligible_for_water_model": False,
                }
            )

        if (
            record_number % 50 == 0
            or record_number == total
        ):
            print(
                f"{record_number}/{total}"
            )

    result = pd.DataFrame(
        result_rows
    )

    result = result.sort_values(
        [
            "quality_status",
            "quality_score",
            "sample_id",
        ],
        ascending=[
            True,
            True,
            True,
        ],
    ).reset_index(
        drop=True
    )

    required_metric_defaults = {
        "shift_magnitude": np.nan,
        "shift_gain": np.nan,
        "best_dx": np.nan,
        "best_dy": np.nan,
        "boundary_contrast": np.nan,
        "local_shift_spread": np.nan,
        "quality_score": 0.0,
    }

    for (
        metric_column,
        default_value,
    ) in required_metric_defaults.items():
        if metric_column not in result.columns:
            result[
                metric_column
            ] = default_value

    result.to_csv(
        QUALITY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    manual_review = result[
        result[
            "quality_status"
        ].isin(
            [
                "REVIEW_REQUIRED",
                "REJECT",
            ]
        )
    ].copy()

    manual_review = manual_review.sort_values(
        [
            "quality_severity",
            "quality_score",
            "shift_magnitude",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    )

    manual_review.to_csv(
        MANUAL_REVIEW_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    status_counts = (
        result
        .groupby(
            [
                "quality_status",
                "quality_severity",
            ],
            dropna=False,
        )
        .size()
        .reset_index(
            name="sample_count"
        )
        .sort_values(
            [
                "quality_status",
                "quality_severity",
            ]
        )
    )

    status_counts.to_csv(
        STATUS_COUNTS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    if REVIEW_ROOT.exists():
        shutil.rmtree(
            REVIEW_ROOT
        )

    REVIEW_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    review_index_rows = []

    for status in (
        "AUTO_ACCEPT",
        "REVIEW_REQUIRED",
        "REJECT",
    ):
        review_rows = choose_review_rows(
            result,
            status,
            args.review_limit_per_status,
        )

        for review_number, review_row in enumerate(
            review_rows.to_dict(
                orient="records"
            ),
            start=1,
        ):
            try:
                image_path = (
                    resolve_project_path(
                        review_row[
                            "image_path"
                        ]
                    )
                )

                land_mask_path = (
                    resolve_project_path(
                        review_row[
                            "land_mask_path"
                        ]
                    )
                )

                grayscale = load_grayscale(
                    image_path
                )

                land_mask = load_binary_mask(
                    land_mask_path
                )

                gradient = calculate_gradient(
                    grayscale
                )

                destination = (
                    REVIEW_ROOT
                    / status
                    / (
                        f"{review_number:03d}"
                        f"__q_{float(review_row['quality_score']):.3f}"
                        f"__shift_{float(review_row.get('shift_magnitude', 0)):.1f}"
                        f"__dx_{int(float(review_row.get('best_dx', 0)))}"
                        f"__dy_{int(float(review_row.get('best_dy', 0)))}"
                        f"__{review_row['sample_id'].replace(':', '_')}"
                        f".png"
                    )
                )

                create_review_image(
                    grayscale=grayscale,
                    land_mask=land_mask,
                    gradient=gradient,
                    best_dx=int(
                        float(
                            review_row.get(
                                "best_dx",
                                0,
                            )
                        )
                    ),
                    best_dy=int(
                        float(
                            review_row.get(
                                "best_dy",
                                0,
                            )
                        )
                    ),
                    destination=destination,
                )

                review_index_rows.append(
                    {
                        "quality_status": status,
                        "review_number": (
                            review_number
                        ),
                        "sample_id": (
                            review_row[
                                "sample_id"
                            ]
                        ),
                        "quality_score": (
                            review_row[
                                "quality_score"
                            ]
                        ),
                        "shift_magnitude": (
                            review_row.get(
                                "shift_magnitude",
                                np.nan,
                            )
                        ),
                        "best_dx": (
                            review_row.get(
                                "best_dx",
                                np.nan,
                            )
                        ),
                        "best_dy": (
                            review_row.get(
                                "best_dy",
                                np.nan,
                            )
                        ),
                        "quality_reason": (
                            review_row[
                                "quality_reason"
                            ]
                        ),
                        "review_image_path": (
                            relative(
                                destination
                            )
                        ),
                    }
                )

            except Exception:
                continue

    pd.DataFrame(
        review_index_rows
    ).to_csv(
        OUTPUT_DIR
        / "review_index.csv",
        index=False,
        encoding="utf-8-sig",
    )

    auto_accept_count = int(
        result[
            "quality_status"
        ].eq(
            "AUTO_ACCEPT"
        ).sum()
    )

    review_required_count = int(
        result[
            "quality_status"
        ].eq(
            "REVIEW_REQUIRED"
        ).sum()
    )

    reject_count = int(
        result[
            "quality_status"
        ].eq(
            "REJECT"
        ).sum()
    )

    summary = {
        "stage": (
            "v06_dartis_water_mask_quality"
        ),
        "coastal_sample_count": int(
            len(result)
        ),
        "auto_accept_count": (
            auto_accept_count
        ),
        "review_required_count": (
            review_required_count
        ),
        "reject_count": (
            reject_count
        ),
        "auto_accept_rate": float(
            auto_accept_count
            / len(result)
        ),
        "search_radius": int(
            args.search_radius
        ),
        "shift_step": int(
            args.shift_step
        ),
        "maximum_auto_shift": float(
            args.maximum_auto_shift
        ),
        "maximum_auto_gain": float(
            args.maximum_auto_gain
        ),
        "minimum_auto_contrast": float(
            args.minimum_auto_contrast
        ),
        "maximum_local_shift_spread": (
            float(
                args.maximum_local_shift_spread
            )
        ),
        "masks_modified": False,
        "training_performed": False,
        "locked_test_used": False,
        "quality_scores_are_ground_truth": False,
        "manual_review_required": True,
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

    report = f"""# v0.6 DARTIS Kara–Su Maskesi Kalite Denetimi

## Amaç

Homography ile üretilmiş kıyı maskelerinin SAR görüntüsündeki gerçek
kıyı sınırına ne kadar uyduğunu ölçmek.

## Sonuç

- Kıyılı örnek: {len(result)}
- AUTO_ACCEPT: {auto_accept_count}
- REVIEW_REQUIRED: {review_required_count}
- REJECT: {reject_count}
- Otomatik kabul oranı: %{auto_accept_count / len(result) * 100:.2f}

## Kullanılan işaretler

- Mevcut sınırın görüntü gradientiyle uyumu
- Yakındaki daha iyi global kayma
- Farklı kıyı bölümlerinin kayma uyuşmazlığı
- Aşırı kara veya güvenli-su oranı

## Bilimsel sınırlama

Bu puanlar insan anotasyonu değildir ve maskenin doğru olduğunu
kanıtlamaz. AUTO_ACCEPT klasörü de örnekleme yöntemiyle görsel olarak
denetlenmelidir.

REVIEW_REQUIRED ve REJECT örnekleri water-mask eğitimine otomatik olarak
girmeyecektir.

Bu aşamada maskeler değiştirilmemiş, model eğitilmemiş ve kilitli test
kullanılmamıştır.
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print(
        "v0.6 MASKE KALİTE DENETİM SONUCU"
    )
    print("=" * 78)

    print(
        "Kıyılı örnek:",
        len(result),
    )

    print(
        "AUTO_ACCEPT:",
        auto_accept_count,
    )

    print(
        "REVIEW_REQUIRED:",
        review_required_count,
    )

    print(
        "REJECT:",
        reject_count,
    )

    print(
        "Otomatik kabul oranı:",
        f"%{auto_accept_count / len(result) * 100:.2f}",
    )

    print()
    print(
        "Durum dağılımı:"
    )

    print(
        status_counts.to_string(
            index=False
        )
    )

    print()
    print(
        "Kalite CSV:",
        QUALITY_PATH.resolve(),
    )

    print(
        "Elle inceleme adayları:",
        MANUAL_REVIEW_PATH.resolve(),
    )

    print(
        "İnceleme klasörü:",
        REVIEW_ROOT.resolve(),
    )

    print(
        "Özet:",
        SUMMARY_PATH.resolve(),
    )

    print(
        "Rapor:",
        REPORT_PATH.resolve(),
    )

    print()
    print(
        "ÖNEMLİ: Bu aşamada hiçbir maske "
        "değiştirilmedi ve model eğitilmedi."
    )


if __name__ == "__main__":
    main()
