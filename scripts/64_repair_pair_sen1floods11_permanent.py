from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = ROOT / "data" / "external" / "sen1floods11"

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "v06_sen1floods11_pair_audit_v2"
)

MANIFEST_PATH = (
    ROOT
    / "data"
    / "metadata"
    / "v06_sen1floods11_water_pairs_v2.csv"
)

SUMMARY_PATH = OUTPUT_DIR / "summary.json"

REPORT_PATH = (
    ROOT
    / "reports"
    / "v06_sen1floods11_pair_audit_v2.md"
)

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

KNOWN_SUFFIXES = (
    "_s1hand",
    "_labelhand",
    "_s1perm",
    "_jrcperm",
)

KNOWN_PREFIXES = (
    "sentinel_",
    "sentinel1_",
    "sentinel-1_",
    "s1_",
    "jrc_",
    "jrcwater_",
    "label_",
    "mask_",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sen1Floods11 SAR ve su maskelerini, permanent-water "
            "dosyalarındaki sentinel_/jrc_ isim farkını düzelterek "
            "eşler ve tüm çiftleri denetler."
        )
    )

    parser.add_argument(
        "--inspect-all",
        action="store_true",
    )

    return parser.parse_args()


def relative(path: Path) -> str:
    try:
        return str(
            path.resolve().relative_to(ROOT.resolve())
        ).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def find_rasters(directory: Path) -> list[Path]:
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


def clean_stem(value: str) -> str:
    name = Path(
        value.replace("\\", "/")
    ).name.lower().strip()

    for extension in (
        ".tiff",
        ".tif",
        ".png",
        ".jpg",
        ".jpeg",
    ):
        if name.endswith(extension):
            name = name[: -len(extension)]

    for suffix in KNOWN_SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)]

    changed = True

    while changed:
        changed = False

        for prefix in KNOWN_PREFIXES:
            if name.startswith(prefix):
                name = name[len(prefix):]
                changed = True
                break

    name = re.sub(
        r"_+",
        "_",
        name,
    ).strip("_")

    return name


def coordinate_signature(value: str) -> str:
    cleaned = clean_stem(value)

    numbers = re.findall(
        r"-?\d+(?:\.\d+)?",
        cleaned,
    )

    if len(numbers) < 2:
        return ""

    try:
        latitude = round(
            float(numbers[-2]),
            8,
        )

        longitude = round(
            float(numbers[-1]),
            8,
        )
    except ValueError:
        return ""

    return (
        f"{latitude:.8f}|"
        f"{longitude:.8f}"
    )


def build_unique_index(
    paths: list[Path],
    key_function,
) -> tuple[dict[str, Path], set[str]]:
    grouped: dict[str, list[Path]] = (
        defaultdict(list)
    )

    for path in paths:
        key = key_function(
            path.name
        )

        if key:
            grouped[key].append(path)

    unique = {
        key: values[0]
        for key, values in grouped.items()
        if len(values) == 1
    }

    ambiguous = {
        key
        for key, values in grouped.items()
        if len(values) > 1
    }

    return unique, ambiguous


def pair_paths(
    image_paths: list[Path],
    mask_paths: list[Path],
) -> tuple[
    list[tuple[str, Path, Path, str]],
    list[Path],
    list[Path],
    dict,
]:
    image_stem, image_stem_ambiguous = (
        build_unique_index(
            image_paths,
            clean_stem,
        )
    )

    mask_stem, mask_stem_ambiguous = (
        build_unique_index(
            mask_paths,
            clean_stem,
        )
    )

    matched_images: set[Path] = set()
    matched_masks: set[Path] = set()

    pairs: list[
        tuple[str, Path, Path, str]
    ] = []

    common_stems = sorted(
        set(image_stem)
        & set(mask_stem)
    )

    for key in common_stems:
        image_path = image_stem[key]
        mask_path = mask_stem[key]

        pairs.append(
            (
                key,
                image_path,
                mask_path,
                "normalized_stem",
            )
        )

        matched_images.add(image_path)
        matched_masks.add(mask_path)

    remaining_images = [
        path
        for path in image_paths
        if path not in matched_images
    ]

    remaining_masks = [
        path
        for path in mask_paths
        if path not in matched_masks
    ]

    image_coordinates, image_coord_ambiguous = (
        build_unique_index(
            remaining_images,
            coordinate_signature,
        )
    )

    mask_coordinates, mask_coord_ambiguous = (
        build_unique_index(
            remaining_masks,
            coordinate_signature,
        )
    )

    common_coordinates = sorted(
        set(image_coordinates)
        & set(mask_coordinates)
    )

    for signature in common_coordinates:
        image_path = image_coordinates[
            signature
        ]

        mask_path = mask_coordinates[
            signature
        ]

        key = clean_stem(
            image_path.name
        )

        pairs.append(
            (
                key,
                image_path,
                mask_path,
                "coordinate_fallback",
            )
        )

        matched_images.add(image_path)
        matched_masks.add(mask_path)

    image_only = [
        path
        for path in image_paths
        if path not in matched_images
    ]

    mask_only = [
        path
        for path in mask_paths
        if path not in matched_masks
    ]

    diagnostics = {
        "normalized_stem_pairs": sum(
            1
            for pair in pairs
            if pair[3]
            == "normalized_stem"
        ),
        "coordinate_fallback_pairs": sum(
            1
            for pair in pairs
            if pair[3]
            == "coordinate_fallback"
        ),
        "image_stem_ambiguous_count": len(
            image_stem_ambiguous
        ),
        "mask_stem_ambiguous_count": len(
            mask_stem_ambiguous
        ),
        "image_coordinate_ambiguous_count": len(
            image_coord_ambiguous
        ),
        "mask_coordinate_ambiguous_count": len(
            mask_coord_ambiguous
        ),
    }

    return (
        sorted(
            pairs,
            key=lambda item: item[0],
        ),
        image_only,
        mask_only,
        diagnostics,
    )


def split_name_from_path(
    path: Path,
) -> str:
    name = path.stem.lower()

    if "train" in name:
        return "train"

    if (
        "valid" in name
        or "val" in name
    ):
        return "validation"

    if "test" in name:
        return "test"

    return "unknown"


def read_split_mapping(
    split_directory: Path,
) -> dict[str, str]:
    mapping: dict[str, str] = {}

    if not split_directory.exists():
        return mapping

    for path in sorted(
        split_directory.rglob("*")
    ):
        if not path.is_file():
            continue

        split_name = split_name_from_path(
            path
        )

        if split_name == "unknown":
            continue

        text = path.read_text(
            encoding="utf-8-sig",
            errors="ignore",
        )

        tokens = re.split(
            r"[\s,;]+",
            text,
        )

        for token in tokens:
            key = clean_stem(token)

            if not key:
                continue

            current = mapping.get(key)

            if (
                current is not None
                and current != split_name
            ):
                mapping[key] = "conflict"
            else:
                mapping[key] = split_name

    return mapping


def inspect_pair(
    image_path: Path,
    mask_path: Path,
) -> dict:
    result = {
        "image_open_ok": False,
        "mask_open_ok": False,
        "same_shape": False,
        "same_crs": False,
        "same_transform": False,
        "image_width": None,
        "image_height": None,
        "image_bands": None,
        "mask_width": None,
        "mask_height": None,
        "mask_bands": None,
        "mask_min": None,
        "mask_max": None,
        "mask_unique_preview": "",
        "valid_pixel_ratio": None,
        "inspection_error": "",
    }

    try:
        with rasterio.open(
            image_path
        ) as image:
            result[
                "image_open_ok"
            ] = True

            result[
                "image_width"
            ] = int(image.width)

            result[
                "image_height"
            ] = int(image.height)

            result[
                "image_bands"
            ] = int(image.count)

            image_shape = (
                image.height,
                image.width,
            )

            image_crs = image.crs
            image_transform = (
                image.transform
            )

        with rasterio.open(
            mask_path
        ) as mask:
            result[
                "mask_open_ok"
            ] = True

            result[
                "mask_width"
            ] = int(mask.width)

            result[
                "mask_height"
            ] = int(mask.height)

            result[
                "mask_bands"
            ] = int(mask.count)

            mask_shape = (
                mask.height,
                mask.width,
            )

            mask_crs = mask.crs
            mask_transform = (
                mask.transform
            )

            array = mask.read(
                1,
                masked=True,
            )

            values = array.compressed()

            if values.size > 0:
                result[
                    "mask_min"
                ] = float(
                    np.nanmin(values)
                )

                result[
                    "mask_max"
                ] = float(
                    np.nanmax(values)
                )

                unique_values = np.unique(
                    values
                )

                result[
                    "mask_unique_preview"
                ] = "|".join(
                    str(float(value))
                    for value
                    in unique_values[:20]
                )

                result[
                    "valid_pixel_ratio"
                ] = float(
                    values.size
                    / array.size
                )

        result["same_shape"] = (
            image_shape == mask_shape
        )

        result["same_crs"] = (
            image_crs == mask_crs
        )

        result[
            "same_transform"
        ] = (
            image_transform
            == mask_transform
        )

    except Exception as error:
        result[
            "inspection_error"
        ] = (
            f"{type(error).__name__}: "
            f"{error}"
        )

    return result


def main() -> None:
    args = parse_args()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows: list[dict] = []
    group_summaries: dict = {}

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
            pairs,
            image_only,
            mask_only,
            diagnostics,
        ) = pair_paths(
            image_paths,
            mask_paths,
        )

        split_mapping = read_split_mapping(
            config["split_dir"]
        )

        inspect_count = (
            len(pairs)
            if args.inspect_all
            else min(
                20,
                len(pairs),
            )
        )

        for pair_index, (
            sample_key,
            image_path,
            mask_path,
            pairing_method,
        ) in enumerate(pairs):
            inspection = {}

            if pair_index < inspect_count:
                inspection = inspect_pair(
                    image_path,
                    mask_path,
                )

            split = split_mapping.get(
                sample_key,
                "unspecified",
            )

            if split == "unspecified":
                coordinate_key = (
                    coordinate_signature(
                        image_path.name
                    )
                )

                split = split_mapping.get(
                    coordinate_key,
                    "unspecified",
                )

            row = {
                "sample_key": sample_key,
                "dataset_group": group_name,
                "split": split,
                "pairing_method": (
                    pairing_method
                ),
                "image_path": relative(
                    image_path
                ),
                "mask_path": relative(
                    mask_path
                ),
            }

            row.update(inspection)
            rows.append(row)

        split_counts = {
            split_name: sum(
                1
                for row in rows
                if row[
                    "dataset_group"
                ]
                == group_name
                and row["split"]
                == split_name
            )
            for split_name in (
                "train",
                "validation",
                "test",
                "unspecified",
                "conflict",
            )
        }

        group_summaries[group_name] = {
            "image_count": len(
                image_paths
            ),
            "mask_count": len(
                mask_paths
            ),
            "paired_count": len(
                pairs
            ),
            "image_only_count": len(
                image_only
            ),
            "mask_only_count": len(
                mask_only
            ),
            "split_counts": split_counts,
            "pairing_diagnostics": (
                diagnostics
            ),
            "image_only_examples": [
                path.name
                for path in image_only[:10]
            ],
            "mask_only_examples": [
                path.name
                for path in mask_only[:10]
            ],
        }

    manifest = pd.DataFrame(rows)

    if not manifest.empty:
        manifest = manifest.sort_values(
            [
                "dataset_group",
                "split",
                "sample_key",
            ]
        )

    manifest.to_csv(
        MANIFEST_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    inspected = manifest[
        manifest[
            "image_open_ok"
        ].astype(str).ne("")
    ].copy()

    open_failures = int(
        (
            inspected[
                "image_open_ok"
            ].astype(str).str.lower().eq(
                "false"
            )
            |
            inspected[
                "mask_open_ok"
            ].astype(str).str.lower().eq(
                "false"
            )
        ).sum()
    )

    shape_failures = int(
        inspected[
            "same_shape"
        ].astype(str).str.lower().eq(
            "false"
        ).sum()
    )

    crs_failures = int(
        inspected[
            "same_crs"
        ].astype(str).str.lower().eq(
            "false"
        ).sum()
    )

    transform_failures = int(
        inspected[
            "same_transform"
        ].astype(str).str.lower().eq(
            "false"
        ).sum()
    )

    summary = {
        "stage": (
            "v06_sen1floods11_pair_audit_v2"
        ),
        "inspect_all": bool(
            args.inspect_all
        ),
        "total_pairs": int(
            len(manifest)
        ),
        "group_summaries": (
            group_summaries
        ),
        "inspected_pair_count": int(
            len(inspected)
        ),
        "open_failure_count": (
            open_failures
        ),
        "shape_failure_count": (
            shape_failures
        ),
        "crs_failure_count": (
            crs_failures
        ),
        "transform_failure_count": (
            transform_failures
        ),
        "training_performed": False,
        "locked_test_used": False,
    }

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    report_lines = [
        "# v0.6 Sen1Floods11 SAR–Su Eşleme Denetimi v2",
        "",
        (
            f"- Toplam eşleşmiş çift: "
            f"{len(manifest)}"
        ),
        (
            f"- İncelenen çift: "
            f"{len(inspected)}"
        ),
        (
            f"- Açılmayan çift: "
            f"{open_failures}"
        ),
        (
            f"- Boyut uyuşmazlığı: "
            f"{shape_failures}"
        ),
        (
            f"- CRS uyuşmazlığı: "
            f"{crs_failures}"
        ),
        (
            f"- Transform uyuşmazlığı: "
            f"{transform_failures}"
        ),
        "",
    ]

    for group_name, group_summary in (
        group_summaries.items()
    ):
        report_lines.extend(
            [
                f"## {group_name}",
                "",
                (
                    f"- SAR: "
                    f"{group_summary['image_count']}"
                ),
                (
                    f"- Maske: "
                    f"{group_summary['mask_count']}"
                ),
                (
                    f"- Eşleşen: "
                    f"{group_summary['paired_count']}"
                ),
                (
                    f"- Maskesiz SAR: "
                    f"{group_summary['image_only_count']}"
                ),
                (
                    f"- Görüntüsüz maske: "
                    f"{group_summary['mask_only_count']}"
                ),
                (
                    f"- Split: "
                    f"{group_summary['split_counts']}"
                ),
                (
                    f"- Eşleme: "
                    f"{group_summary['pairing_diagnostics']}"
                ),
                "",
            ]
        )

    REPORT_PATH.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    print("=" * 78)
    print(
        "SEN1FLOODS11 PAIR AUDIT v2"
    )
    print("=" * 78)

    for group_name, group_summary in (
        group_summaries.items()
    ):
        print()
        print(group_name.upper())

        print(
            "  SAR:",
            group_summary[
                "image_count"
            ],
        )

        print(
            "  Maske:",
            group_summary[
                "mask_count"
            ],
        )

        print(
            "  Eşleşen:",
            group_summary[
                "paired_count"
            ],
        )

        print(
            "  Maskesiz SAR:",
            group_summary[
                "image_only_count"
            ],
        )

        print(
            "  Görüntüsüz maske:",
            group_summary[
                "mask_only_count"
            ],
        )

        print(
            "  Split:",
            group_summary[
                "split_counts"
            ],
        )

        print(
            "  Eşleme yöntemi:",
            group_summary[
                "pairing_diagnostics"
            ],
        )

        if (
            group_summary[
                "image_only_examples"
            ]
        ):
            print(
                "  İlk eşleşmeyen SAR:",
                group_summary[
                    "image_only_examples"
                ][:3],
            )

        if (
            group_summary[
                "mask_only_examples"
            ]
        ):
            print(
                "  İlk eşleşmeyen maske:",
                group_summary[
                    "mask_only_examples"
                ][:3],
            )

    print()
    print(
        "Toplam eşleşmiş çift:",
        len(manifest),
    )

    print(
        "İncelenen çift:",
        len(inspected),
    )

    print(
        "Açılmayan çift:",
        open_failures,
    )

    print(
        "Boyut uyuşmazlığı:",
        shape_failures,
    )

    print(
        "CRS uyuşmazlığı:",
        crs_failures,
    )

    print(
        "Transform uyuşmazlığı:",
        transform_failures,
    )

    print(
        "Manifest:",
        MANIFEST_PATH.resolve(),
    )

    print(
        "Özet:",
        SUMMARY_PATH.resolve(),
    )

    print()
    print(
        "Eğitim yapılmadı."
    )

    print(
        "Kilitli test kullanılmadı."
    )


if __name__ == "__main__":
    main()
