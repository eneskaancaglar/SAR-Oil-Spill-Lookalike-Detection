from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio.features
import torch
import torch.nn.functional as functional
from PIL import Image
from rasterio.control import GroundControlPoint
from rasterio.transform import GCPTransformer
from shapely.geometry import box
from shapely.ops import transform as transform_geometry


ROOT = Path(__file__).resolve().parents[1]

MAPPING_PATH = (
    ROOT
    / "data"
    / "metadata"
    / "v06_dartis_table_mapping.csv"
)

OSM_ROOT = (
    ROOT
    / "data"
    / "external"
    / "osm_land_polygons"
)

MASK_ROOT = (
    ROOT
    / "data"
    / "derived"
    / "v06_water_masks"
)

LAND_MASK_ROOT = (
    MASK_ROOT
    / "land"
)

SAFE_WATER_ROOT = (
    MASK_ROOT
    / "safe_water"
)

OUTPUT_MANIFEST = (
    ROOT
    / "data"
    / "metadata"
    / "v06_dartis_water_masks.csv"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "v06_dartis_water_masks"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "summary.json"
)

GROUP_COUNTS_PATH = (
    OUTPUT_DIR
    / "group_counts.csv"
)

FAILURES_PATH = (
    OUTPUT_DIR
    / "failures.csv"
)

REVIEW_DIR = (
    OUTPUT_DIR
    / "review"
)

REVIEW_INDEX_PATH = (
    OUTPUT_DIR
    / "review_index.csv"
)

REPORT_PATH = (
    ROOT
    / "reports"
    / "v06_dartis_water_masks.md"
)

COORDINATE_COLUMNS = [
    "patch_ul_lon",
    "patch_ul_lat",
    "patch_ur_lon",
    "patch_ur_lat",
    "patch_br_lon",
    "patch_br_lat",
    "patch_bl_lon",
    "patch_bl_lat",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "DARTIS görüntüleri için kara ve "
            "kıyı-tamponlu güvenli su maskeleri üretir."
        )
    )

    parser.add_argument(
        "--coast-buffer-pixels",
        type=int,
        default=8,
        help=(
            "Kara sınırının su tarafında analiz dışı "
            "bırakılacak yaklaşık piksel tamponu."
        ),
    )

    parser.add_argument(
        "--review-count",
        type=int,
        default=36,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
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


def locate_land_shapefile() -> Path:
    candidates = sorted(
        OSM_ROOT.rglob(
            "land_polygons.shp"
        )
    )

    if not candidates:
        raise FileNotFoundError(
            "land_polygons.shp bulunamadı. "
            f"Aranan klasör: {OSM_ROOT}"
        )

    if len(candidates) > 1:
        preferred = [
            path
            for path in candidates
            if "split" in str(path).lower()
        ]

        if len(preferred) == 1:
            return preferred[0]

        raise RuntimeError(
            "Birden fazla land_polygons.shp bulundu:\n"
            + "\n".join(
                str(path)
                for path in candidates
            )
        )

    return candidates[0]


def validate_mapping(
    dataframe: pd.DataFrame,
) -> None:
    required_columns = {
        "subset",
        "surface_context",
        "oil_label",
        "mapping_status",
        "local_image_path",
        *COORDINATE_COLUMNS,
    }

    missing = (
        required_columns
        - set(dataframe.columns)
    )

    if missing:
        raise RuntimeError(
            "Mapping CSV eksik sütunlar:\n"
            + "\n".join(
                sorted(missing)
            )
        )

    if not dataframe[
        "mapping_status"
    ].eq(
        "MATCHED"
    ).all():
        raise RuntimeError(
            "Mapping içinde MATCHED olmayan kayıt var."
        )

    if len(dataframe) != 3655:
        raise RuntimeError(
            "Beklenen görüntü sayısı 3655, "
            f"bulunan {len(dataframe)}."
        )

    coastal = dataframe[
        dataframe[
            "surface_context"
        ].eq(
            "coast"
        )
    ]

    if coastal[
        COORDINATE_COLUMNS
    ].isna().any().any():
        raise RuntimeError(
            "Kıyılı görüntülerde eksik koordinat var."
        )


def dilate_boolean_mask(
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

    kernel_size = (
        radius * 2
        + 1
    )

    dilated = functional.max_pool2d(
        tensor,
        kernel_size=kernel_size,
        stride=1,
        padding=radius,
    )

    return (
        dilated[
            0,
            0,
        ].numpy()
        > 0.5
    )


def safe_geometry(
    geometry,
):
    if geometry is None:
        return None

    if geometry.is_empty:
        return None

    if geometry.is_valid:
        return geometry

    try:
        fixed = geometry.buffer(
            0
        )

        if fixed.is_empty:
            return None

        return fixed

    except Exception:
        return None


def create_geospatial_land_mask(
    *,
    row: pd.Series,
    image_width: int,
    image_height: int,
    land_geodataframe: gpd.GeoDataFrame,
) -> tuple[np.ndarray, int]:
    longitude_values = [
        float(
            row[
                "patch_ul_lon"
            ]
        ),
        float(
            row[
                "patch_ur_lon"
            ]
        ),
        float(
            row[
                "patch_br_lon"
            ]
        ),
        float(
            row[
                "patch_bl_lon"
            ]
        ),
    ]

    latitude_values = [
        float(
            row[
                "patch_ul_lat"
            ]
        ),
        float(
            row[
                "patch_ur_lat"
            ]
        ),
        float(
            row[
                "patch_br_lat"
            ]
        ),
        float(
            row[
                "patch_bl_lat"
            ]
        ),
    ]

    minimum_longitude = min(
        longitude_values
    )

    maximum_longitude = max(
        longitude_values
    )

    minimum_latitude = min(
        latitude_values
    )

    maximum_latitude = max(
        latitude_values
    )

    patch_box = box(
        minimum_longitude,
        minimum_latitude,
        maximum_longitude,
        maximum_latitude,
    )

    candidate_indices = (
        land_geodataframe
        .sindex
        .query(
            patch_box,
            predicate="intersects",
        )
    )

    if len(
        candidate_indices
    ) == 0:
        return (
            np.zeros(
                (
                    image_height,
                    image_width,
                ),
                dtype=np.uint8,
            ),
            0,
        )

    candidate_geometries = (
        land_geodataframe
        .iloc[
            np.asarray(
                candidate_indices,
                dtype=int,
            )
        ]
        .geometry
    )

    clipped_geometries = []

    for geometry in candidate_geometries:
        geometry = safe_geometry(
            geometry
        )

        if geometry is None:
            continue

        try:
            clipped = geometry.intersection(
                patch_box
            )

        except Exception:
            continue

        clipped = safe_geometry(
            clipped
        )

        if clipped is not None:
            clipped_geometries.append(
                clipped
            )

    if not clipped_geometries:
        return (
            np.zeros(
                (
                    image_height,
                    image_width,
                ),
                dtype=np.uint8,
            ),
            0,
        )

    ground_control_points = [
        GroundControlPoint(
            -0.5,
            -0.5,
            float(
                row[
                    "patch_ul_lon"
                ]
            ),
            float(
                row[
                    "patch_ul_lat"
                ]
            ),
        ),
        GroundControlPoint(
            -0.5,
            image_width - 0.5,
            float(
                row[
                    "patch_ur_lon"
                ]
            ),
            float(
                row[
                    "patch_ur_lat"
                ]
            ),
        ),
        GroundControlPoint(
            image_height - 0.5,
            -0.5,
            float(
                row[
                    "patch_bl_lon"
                ]
            ),
            float(
                row[
                    "patch_bl_lat"
                ]
            ),
        ),
        GroundControlPoint(
            image_height - 0.5,
            image_width - 0.5,
            float(
                row[
                    "patch_br_lon"
                ]
            ),
            float(
                row[
                    "patch_br_lat"
                ]
            ),
        ),
    ]

    pixel_geometries = []

    with GCPTransformer(
        ground_control_points,
        tps=True,
    ) as transformer:

        def longitude_latitude_to_pixel(
            x,
            y,
            z=None,
        ):
            rows, columns = (
                transformer.rowcol(
                    x,
                    y,
                    op=lambda value: value,
                )
            )

            return (
                columns,
                rows,
            )

        for geometry in clipped_geometries:
            try:
                pixel_geometry = (
                    transform_geometry(
                        longitude_latitude_to_pixel,
                        geometry,
                    )
                )

            except Exception:
                continue

            pixel_geometry = safe_geometry(
                pixel_geometry
            )

            if pixel_geometry is not None:
                pixel_geometries.append(
                    pixel_geometry
                )

    if not pixel_geometries:
        return (
            np.zeros(
                (
                    image_height,
                    image_width,
                ),
                dtype=np.uint8,
            ),
            len(
                clipped_geometries
            ),
        )

    land_mask = rasterio.features.rasterize(
        (
            (
                geometry,
                1,
            )
            for geometry
            in pixel_geometries
        ),
        out_shape=(
            image_height,
            image_width,
        ),
        fill=0,
        all_touched=True,
        dtype=np.uint8,
    )

    return (
        land_mask,
        len(
            pixel_geometries
        ),
    )


def save_binary_mask(
    mask: np.ndarray,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Image.fromarray(
        (
            mask.astype(
                np.uint8
            )
            * 255
        ),
        mode="L",
    ).save(
        path,
        format="PNG",
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


def create_review_image(
    *,
    image_path: Path,
    land_mask_path: Path,
    safe_water_path: Path,
    destination: Path,
) -> None:
    with Image.open(
        image_path
    ) as image:
        grayscale = np.asarray(
            image.convert(
                "L"
            ),
            dtype=np.uint8,
        )

    land = load_binary_mask(
        land_mask_path
    )

    safe_water = load_binary_mask(
        safe_water_path
    )

    excluded_buffer = (
        ~safe_water
        & ~land
    )

    overlay = np.stack(
        [
            grayscale,
            grayscale,
            grayscale,
        ],
        axis=-1,
    ).astype(
        np.float32
    )

    overlay[
        land
    ] = (
        overlay[
            land
        ]
        * 0.35
        + np.array(
            [
                255,
                40,
                40,
            ],
            dtype=np.float32,
        )
        * 0.65
    )

    overlay[
        excluded_buffer
    ] = (
        overlay[
            excluded_buffer
        ]
        * 0.35
        + np.array(
            [
                255,
                210,
                40,
            ],
            dtype=np.float32,
        )
        * 0.65
    )

    overlay = np.clip(
        overlay,
        0,
        255,
    ).astype(
        np.uint8
    )

    water_only = grayscale.copy()

    water_only[
        ~safe_water
    ] = 0

    water_rgb = np.stack(
        [
            water_only,
            water_only,
            water_only,
        ],
        axis=-1,
    )

    original_rgb = np.stack(
        [
            grayscale,
            grayscale,
            grayscale,
        ],
        axis=-1,
    )

    combined = np.concatenate(
        [
            original_rgb,
            overlay,
            water_rgb,
        ],
        axis=1,
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    Image.fromarray(
        combined
    ).save(
        destination,
        format="PNG",
    )


def choose_review_records(
    dataframe: pd.DataFrame,
    review_count: int,
) -> pd.DataFrame:
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
        return coastal

    coastal = coastal.sort_values(
        [
            "land_ratio",
            "sample_id",
        ]
    ).reset_index(
        drop=True
    )

    count = min(
        review_count,
        len(coastal),
    )

    indices = np.linspace(
        0,
        len(coastal) - 1,
        num=count,
        dtype=int,
    )

    return coastal.iloc[
        np.unique(
            indices
        )
    ].copy()


def main() -> None:
    args = parse_args()

    if args.coast_buffer_pixels < 0:
        raise ValueError(
            "--coast-buffer-pixels negatif olamaz."
        )

    if not MAPPING_PATH.exists():
        raise FileNotFoundError(
            f"Mapping bulunamadı: {MAPPING_PATH}"
        )

    shapefile_path = (
        locate_land_shapefile()
    )

    dataframe = pd.read_csv(
        MAPPING_PATH,
        encoding="utf-8-sig",
        low_memory=False,
    )

    validate_mapping(
        dataframe
    )

    for column in COORDINATE_COLUMNS:
        dataframe[column] = (
            pd.to_numeric(
                dataframe[column],
                errors="coerce",
            )
        )

    coastal = dataframe[
        dataframe[
            "surface_context"
        ].eq(
            "coast"
        )
    ]

    longitude_columns = [
        column
        for column
        in COORDINATE_COLUMNS
        if column.endswith(
            "_lon"
        )
    ]

    latitude_columns = [
        column
        for column
        in COORDINATE_COLUMNS
        if column.endswith(
            "_lat"
        )
    ]

    overall_minimum_longitude = float(
        coastal[
            longitude_columns
        ].min().min()
    )

    overall_maximum_longitude = float(
        coastal[
            longitude_columns
        ].max().max()
    )

    overall_minimum_latitude = float(
        coastal[
            latitude_columns
        ].min().min()
    )

    overall_maximum_latitude = float(
        coastal[
            latitude_columns
        ].max().max()
    )

    geographical_margin = 0.2

    read_bbox = (
        overall_minimum_longitude
        - geographical_margin,
        overall_minimum_latitude
        - geographical_margin,
        overall_maximum_longitude
        + geographical_margin,
        overall_maximum_latitude
        + geographical_margin,
    )

    print("=" * 78)
    print(
        "ROBUST BINARY OIL DETECTOR v0.6"
    )
    print(
        "DARTIS KARA VE GÜVENLİ SU MASKELERİ"
    )
    print("=" * 78)

    print(
        "Shapefile:",
        shapefile_path,
    )

    print(
        "Kıyı tamponu:",
        args.coast_buffer_pixels,
        "piksel",
    )

    print(
        "OSM okuma bbox:",
        read_bbox,
    )

    print(
        "Yalnız Doğu Akdeniz poligonları okunuyor..."
    )

    land_geodataframe = gpd.read_file(
        shapefile_path,
        bbox=read_bbox,
        engine="pyogrio",
    )

    if land_geodataframe.empty:
        raise RuntimeError(
            "OSM kara poligonu sorgusu boş döndü."
        )

    if land_geodataframe.crs is None:
        raise RuntimeError(
            "OSM shapefile CRS bilgisi bulunamadı."
        )

    if (
        land_geodataframe.crs
        .to_epsg()
        != 4326
    ):
        land_geodataframe = (
            land_geodataframe
            .to_crs(
                epsg=4326
            )
        )

    LAND_MASK_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    SAFE_WATER_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_DIR.mkdir(
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

    result_rows = []
    failure_rows = []

    total = len(
        dataframe
    )

    for row_number, (
        _,
        row,
    ) in enumerate(
        dataframe.iterrows(),
        start=1,
    ):
        subset = str(
            row[
                "subset"
            ]
        ).strip().lower()

        image_path = resolve_project_path(
            row[
                "local_image_path"
            ]
        )

        sample_id = (
            f"{subset}:"
            f"{image_path.stem}"
        )

        land_mask_path = (
            LAND_MASK_ROOT
            / subset
            / (
                image_path.stem
                + ".png"
            )
        )

        safe_water_path = (
            SAFE_WATER_ROOT
            / subset
            / (
                image_path.stem
                + ".png"
            )
        )

        try:
            if not image_path.exists():
                raise FileNotFoundError(
                    f"Görüntü bulunamadı: {image_path}"
                )

            with Image.open(
                image_path
            ) as image:
                grayscale = image.convert(
                    "L"
                )

                image_width, image_height = (
                    grayscale.size
                )

            existing_masks = (
                land_mask_path.exists()
                and safe_water_path.exists()
            )

            if (
                existing_masks
                and not args.overwrite
            ):
                land_mask = (
                    load_binary_mask(
                        land_mask_path
                    )
                )

                safe_water_mask = (
                    load_binary_mask(
                        safe_water_path
                    )
                )

                generation_mode = (
                    "RESUMED_EXISTING"
                )

                polygon_count = -1

            elif str(
                row[
                    "surface_context"
                ]
            ) == "water":
                land_mask = np.zeros(
                    (
                        image_height,
                        image_width,
                    ),
                    dtype=bool,
                )

                safe_water_mask = np.ones(
                    (
                        image_height,
                        image_width,
                    ),
                    dtype=bool,
                )

                save_binary_mask(
                    land_mask,
                    land_mask_path,
                )

                save_binary_mask(
                    safe_water_mask,
                    safe_water_path,
                )

                generation_mode = (
                    "OPEN_WATER_SEMANTIC"
                )

                polygon_count = 0

            else:
                (
                    raw_land_mask,
                    polygon_count,
                ) = create_geospatial_land_mask(
                    row=row,
                    image_width=image_width,
                    image_height=image_height,
                    land_geodataframe=(
                        land_geodataframe
                    ),
                )

                land_mask = (
                    raw_land_mask
                    > 0
                )

                excluded_mask = (
                    dilate_boolean_mask(
                        land_mask,
                        args.coast_buffer_pixels,
                    )
                )

                safe_water_mask = (
                    ~excluded_mask
                )

                save_binary_mask(
                    land_mask,
                    land_mask_path,
                )

                save_binary_mask(
                    safe_water_mask,
                    safe_water_path,
                )

                generation_mode = (
                    "OSM_GEOGRAPHIC_MASK"
                )

            total_pixels = (
                image_width
                * image_height
            )

            land_pixels = int(
                land_mask.sum()
            )

            safe_water_pixels = int(
                safe_water_mask.sum()
            )

            excluded_pixels = int(
                total_pixels
                - safe_water_pixels
            )

            result_rows.append(
                {
                    "sample_id": sample_id,
                    "subset": subset,
                    "oil_label": int(
                        row[
                            "oil_label"
                        ]
                    ),
                    "surface_context": str(
                        row[
                            "surface_context"
                        ]
                    ),
                    "image_path": relative(
                        image_path
                    ),
                    "image_width": int(
                        image_width
                    ),
                    "image_height": int(
                        image_height
                    ),
                    "land_mask_path": relative(
                        land_mask_path
                    ),
                    "safe_water_mask_path": (
                        relative(
                            safe_water_path
                        )
                    ),
                    "mask_source": (
                        generation_mode
                    ),
                    "coast_buffer_pixels": int(
                        args.coast_buffer_pixels
                    ),
                    "polygon_count": int(
                        polygon_count
                    ),
                    "land_pixels": (
                        land_pixels
                    ),
                    "land_ratio": float(
                        land_pixels
                        / total_pixels
                    ),
                    "safe_water_pixels": (
                        safe_water_pixels
                    ),
                    "safe_water_ratio": float(
                        safe_water_pixels
                        / total_pixels
                    ),
                    "excluded_pixels": (
                        excluded_pixels
                    ),
                    "excluded_ratio": float(
                        excluded_pixels
                        / total_pixels
                    ),
                    "generation_status": (
                        "SUCCESS"
                    ),
                    "eligible_for_training": bool(
                        safe_water_pixels > 0
                    ),
                    "locked_test_used": False,
                }
            )

        except Exception as error:
            failure_rows.append(
                {
                    "sample_id": sample_id,
                    "subset": subset,
                    "image_path": str(
                        image_path
                    ),
                    "error_type": type(
                        error
                    ).__name__,
                    "error": str(
                        error
                    ),
                }
            )

            result_rows.append(
                {
                    "sample_id": sample_id,
                    "subset": subset,
                    "oil_label": int(
                        row[
                            "oil_label"
                        ]
                    ),
                    "surface_context": str(
                        row[
                            "surface_context"
                        ]
                    ),
                    "image_path": relative(
                        image_path
                    ),
                    "land_mask_path": relative(
                        land_mask_path
                    ),
                    "safe_water_mask_path": (
                        relative(
                            safe_water_path
                        )
                    ),
                    "generation_status": (
                        "FAILED"
                    ),
                    "eligible_for_training": False,
                    "locked_test_used": False,
                }
            )

        if (
            row_number % 250 == 0
            or row_number == total
        ):
            print(
                f"{row_number}/{total} "
                f"| başarılı="
                f"{row_number - len(failure_rows)} "
                f"| hatalı="
                f"{len(failure_rows)}"
            )

    result = pd.DataFrame(
        result_rows
    )

    failures = pd.DataFrame(
        failure_rows,
        columns=[
            "sample_id",
            "subset",
            "image_path",
            "error_type",
            "error",
        ],
    )

    result.to_csv(
        OUTPUT_MANIFEST,
        index=False,
        encoding="utf-8-sig",
    )

    failures.to_csv(
        FAILURES_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    successful = result[
        result[
            "generation_status"
        ].eq(
            "SUCCESS"
        )
    ].copy()

    group_counts = (
        successful
        .groupby(
            [
                "subset",
                "surface_context",
                "oil_label",
                "mask_source",
            ],
            dropna=False,
        )
        .agg(
            image_count=(
                "sample_id",
                "count",
            ),
            mean_land_ratio=(
                "land_ratio",
                "mean",
            ),
            mean_safe_water_ratio=(
                "safe_water_ratio",
                "mean",
            ),
            minimum_safe_water_ratio=(
                "safe_water_ratio",
                "min",
            ),
            maximum_safe_water_ratio=(
                "safe_water_ratio",
                "max",
            ),
        )
        .reset_index()
        .sort_values(
            "subset"
        )
    )

    group_counts.to_csv(
        GROUP_COUNTS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    if REVIEW_DIR.exists():
        shutil.rmtree(
            REVIEW_DIR
        )

    REVIEW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    review_records = (
        choose_review_records(
            successful,
            args.review_count,
        )
    )

    review_index_rows = []

    for review_number, row in enumerate(
        review_records.to_dict(
            orient="records"
        ),
        start=1,
    ):
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

        safe_water_path = (
            resolve_project_path(
                row[
                    "safe_water_mask_path"
                ]
            )
        )

        destination = (
            REVIEW_DIR
            / (
                f"{review_number:03d}"
                f"__{row['subset']}"
                f"__land_{row['land_ratio']:.3f}"
                f"__safe_{row['safe_water_ratio']:.3f}"
                f"__{image_path.stem}.png"
            )
        )

        create_review_image(
            image_path=image_path,
            land_mask_path=(
                land_mask_path
            ),
            safe_water_path=(
                safe_water_path
            ),
            destination=destination,
        )

        review_index_rows.append(
            {
                "review_number": (
                    review_number
                ),
                "sample_id": row[
                    "sample_id"
                ],
                "subset": row[
                    "subset"
                ],
                "land_ratio": row[
                    "land_ratio"
                ],
                "safe_water_ratio": row[
                    "safe_water_ratio"
                ],
                "review_image": relative(
                    destination
                ),
            }
        )

    pd.DataFrame(
        review_index_rows
    ).to_csv(
        REVIEW_INDEX_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    coastal_successful = successful[
        successful[
            "surface_context"
        ].eq(
            "coast"
        )
    ]

    zero_land_coastal = int(
        coastal_successful[
            "land_pixels"
        ].eq(
            0
        ).sum()
    )

    fully_blocked = int(
        successful[
            "safe_water_pixels"
        ].eq(
            0
        ).sum()
    )

    summary = {
        "stage": (
            "v06_dartis_water_masks"
        ),
        "total_records": int(
            len(result)
        ),
        "successful_records": int(
            len(successful)
        ),
        "failed_records": int(
            len(failures)
        ),
        "coastal_records": int(
            len(coastal_successful)
        ),
        "open_water_records": int(
            successful[
                "surface_context"
            ].eq(
                "water"
            ).sum()
        ),
        "zero_land_coastal_records": (
            zero_land_coastal
        ),
        "fully_blocked_records": (
            fully_blocked
        ),
        "coast_buffer_pixels": int(
            args.coast_buffer_pixels
        ),
        "review_image_count": int(
            len(review_index_rows)
        ),
        "shapefile": relative(
            shapefile_path
        ),
        "shapefile_source": (
            "OpenStreetMap-derived land polygons"
        ),
        "shapefile_license": (
            "Open Database License (ODbL)"
        ),
        "original_images_modified": False,
        "training_performed": False,
        "locked_test_used": False,
        "visual_review_required": True,
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

    report = f"""# v0.6 DARTIS Kara ve Güvenli Su Maskeleri

## İşlem

- `oc` ve `nc`: WGS84 köşe koordinatları ve OpenStreetMap kara
  poligonları kullanılarak kara maskesi üretildi.
- `ow` ve `nw`: DARTIS açık-su grup anlamına göre bütün görüntü
  su kabul edildi.
- Kara maskesi `{args.coast_buffer_pixels}` piksel genişletilerek
  kıyı tamponu oluşturuldu.
- Petrol/look-alike analizi yalnız `safe_water_mask=1` alanında
  yapılacaktır.

## Sonuç

- Toplam kayıt: {len(result)}
- Başarılı: {len(successful)}
- Hatalı: {len(failures)}
- Kıyılı kayıt: {len(coastal_successful)}
- Açık-su kayıt: {int(successful['surface_context'].eq('water').sum())}
- Kara pikseli bulunmayan kıyılı kayıt: {zero_land_coastal}
- Tamamen engellenen kayıt: {fully_blocked}

## Maske anlamı

- Kara maskesi: `255=kara`, `0=su`
- Güvenli su maskesi: `255=analiz edilebilir su`,
  `0=kara veya kıyı tamponu`

## Bilimsel durum

Bu maskeler görüntüden öğrenilmiş gerçek insan anotasyonları değil,
DARTIS köşe koordinatları ile OpenStreetMap kara poligonlarından üretilmiş
coğrafi denetim maskeleridir.

Model eğitiminden önce inceleme görüntülerinin elle kontrol edilmesi
zorunludur.

OpenStreetMap verisi ODbL kapsamında kullanılmıştır.

Kilitli test kullanılmamış ve model eğitilmemiştir.
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print(
        "v0.6 SU MASKESİ ÜRETİM SONUCU"
    )
    print("=" * 78)

    print(
        "Başarılı:",
        len(successful),
    )

    print(
        "Hatalı:",
        len(failures),
    )

    print(
        "Kıyılı:",
        len(coastal_successful),
    )

    print(
        "Açık su:",
        int(
            successful[
                "surface_context"
            ].eq(
                "water"
            ).sum()
        ),
    )

    print(
        "Kara bulunmayan kıyılı:",
        zero_land_coastal,
    )

    print(
        "Tamamen engellenen:",
        fully_blocked,
    )

    print()
    print(
        "Grup sonuçları:"
    )

    print(
        group_counts.to_string(
            index=False
        )
    )

    print()
    print(
        "Manifest:",
        OUTPUT_MANIFEST.resolve(),
    )

    print(
        "İnceleme klasörü:",
        REVIEW_DIR.resolve(),
    )

    print(
        "Hatalar:",
        FAILURES_PATH.resolve(),
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
        "ÖNEMLİ: Model eğitiminden önce "
        "review klasöründeki maskeleri görsel olarak kontrol et."
    )


if __name__ == "__main__":
    main()
