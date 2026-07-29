from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import rasterio


ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = (
    ROOT
    / "data"
    / "external"
    / "sen1floods11"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "v06_sen1floods11_pair_audit"
)

MANIFEST_PATH = (
    ROOT
    / "data"
    / "metadata"
    / "v06_sen1floods11_water_pairs.csv"
)

SUMMARY_PATH = OUTPUT_DIR / "summary.json"
REPORT_PATH = (
    ROOT
    / "reports"
    / "v06_sen1floods11_pair_audit.md"
)

GROUPS = {
    "flood": {
        "image_dir": DATA_ROOT / "flood_sar",
        "mask_dir": DATA_ROOT / "flood_mask",
        "split_dir": DATA_ROOT / "flood_splits",
        "image_suffix": "_S1Hand",
        "mask_suffix": "_LabelHand",
    },
    "permanent": {
        "image_dir": DATA_ROOT / "permanent_sar",
        "mask_dir": DATA_ROOT / "permanent_mask",
        "split_dir": DATA_ROOT / "permanent_splits",
        "image_suffix": "_S1Perm",
        "mask_suffix": "_JRCPerm",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "İndirilen Sen1Floods11 SAR görüntülerini ve hizalı "
            "su maskelerini eşler, splitleri okur ve eğitim "
            "manifesti oluşturur."
        )
    )

    parser.add_argument(
        "--inspect-all",
        action="store_true",
        help=(
            "Bütün görüntü ve maskeleri rasterio ile açıp "
            "boyut/bant/değer kontrolü yapar."
        ),
    )

    return parser.parse_args()


def relative(path: Path) -> str:
    try:
        return str(
            path.resolve().relative_to(ROOT.resolve())
        ).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def normalized_key(
    path: Path,
    removable_suffix: str,
) -> str:
    stem = path.stem

    if stem.endswith(removable_suffix):
        stem = stem[: -len(removable_suffix)]

    return stem.strip()


def find_rasters(directory: Path) -> list[Path]:
    if not directory.exists():
        return []

    rasters: list[Path] = []

    for pattern in ("*.tif", "*.tiff"):
        rasters.extend(directory.rglob(pattern))

    return sorted(
        {
            path.resolve()
            for path in rasters
            if path.is_file()
        }
    )


def split_name_from_filename(path: Path) -> str:
    name = path.stem.lower()

    if "train" in name:
        return "train"

    if "valid" in name or "val" in name:
        return "validation"

    if "test" in name:
        return "test"

    return "unknown"


def clean_split_token(token: str) -> str:
    token = token.strip().strip(
        "\"'[](){}"
    )

    token = token.replace("\\", "/")
    token = Path(token).name

    for extension in (
        ".tif",
        ".tiff",
        ".png",
        ".jpg",
        ".jpeg",
    ):
        if token.lower().endswith(extension):
            token = token[: -len(extension)]

    for suffix in (
        "_S1Hand",
        "_LabelHand",
        "_S1Perm",
        "_JRCPerm",
    ):
        if token.endswith(suffix):
            token = token[: -len(suffix)]

    return token.strip()


def read_split_mapping(
    split_dir: Path,
) -> dict[str, str]:
    mapping: dict[str, str] = {}

    if not split_dir.exists():
        return mapping

    files = [
        path
        for path in split_dir.rglob("*")
        if path.is_file()
    ]

    for path in sorted(files):
        split_name = split_name_from_filename(
            path
        )

        if split_name == "unknown":
            continue

        try:
            text = path.read_text(
                encoding="utf-8-sig",
                errors="ignore",
            )
        except Exception:
            continue

        tokens = re.split(
            r"[\s,;]+",
            text,
        )

        for token in tokens:
            key = clean_split_token(token)

            if not key:
                continue

            existing = mapping.get(key)

            if (
                existing is not None
                and existing != split_name
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
        "error": "",
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

            mask_array = mask.read(
                1,
                masked=True,
            )

            mask_shape = (
                mask.height,
                mask.width,
            )

            compressed = (
                mask_array.compressed()
            )

            if compressed.size > 0:
                result["mask_min"] = float(
                    np.nanmin(compressed)
                )
                result["mask_max"] = float(
                    np.nanmax(compressed)
                )

                unique_values = np.unique(
                    compressed
                )

                result[
                    "mask_unique_preview"
                ] = "|".join(
                    str(float(value))
                    for value in unique_values[:20]
                )

                result[
                    "valid_pixel_ratio"
                ] = float(
                    compressed.size
                    / mask_array.size
                )

        result["same_shape"] = (
            image_shape == mask_shape
        )

    except Exception as error:
        result["error"] = (
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

    records = []
    group_summaries = {}

    for group_name, config in GROUPS.items():
        image_paths = find_rasters(
            config["image_dir"]
        )

        mask_paths = find_rasters(
            config["mask_dir"]
        )

        image_map = {
            normalized_key(
                path,
                config["image_suffix"],
            ): path
            for path in image_paths
        }

        mask_map = {
            normalized_key(
                path,
                config["mask_suffix"],
            ): path
            for path in mask_paths
        }

        split_mapping = read_split_mapping(
            config["split_dir"]
        )

        common_keys = sorted(
            set(image_map)
            & set(mask_map)
        )

        image_only = sorted(
            set(image_map)
            - set(mask_map)
        )

        mask_only = sorted(
            set(mask_map)
            - set(image_map)
        )

        inspect_keys: Iterable[str]

        if args.inspect_all:
            inspect_keys = common_keys
        else:
            inspect_keys = common_keys[:20]

        inspected = {}

        for key in inspect_keys:
            inspected[key] = inspect_pair(
                image_map[key],
                mask_map[key],
            )

        for key in common_keys:
            split = split_mapping.get(
                key,
                "unspecified",
            )

            inspection = inspected.get(
                key,
                {},
            )

            records.append(
                {
                    "sample_key": key,
                    "dataset_group": group_name,
                    "split": split,
                    "image_path": relative(
                        image_map[key]
                    ),
                    "mask_path": relative(
                        mask_map[key]
                    ),
                    "image_open_ok": inspection.get(
                        "image_open_ok",
                        "",
                    ),
                    "mask_open_ok": inspection.get(
                        "mask_open_ok",
                        "",
                    ),
                    "same_shape": inspection.get(
                        "same_shape",
                        "",
                    ),
                    "image_width": inspection.get(
                        "image_width",
                        "",
                    ),
                    "image_height": inspection.get(
                        "image_height",
                        "",
                    ),
                    "image_bands": inspection.get(
                        "image_bands",
                        "",
                    ),
                    "mask_width": inspection.get(
                        "mask_width",
                        "",
                    ),
                    "mask_height": inspection.get(
                        "mask_height",
                        "",
                    ),
                    "mask_bands": inspection.get(
                        "mask_bands",
                        "",
                    ),
                    "mask_min": inspection.get(
                        "mask_min",
                        "",
                    ),
                    "mask_max": inspection.get(
                        "mask_max",
                        "",
                    ),
                    "mask_unique_preview": inspection.get(
                        "mask_unique_preview",
                        "",
                    ),
                    "valid_pixel_ratio": inspection.get(
                        "valid_pixel_ratio",
                        "",
                    ),
                    "inspection_error": inspection.get(
                        "error",
                        "",
                    ),
                }
            )

        group_summaries[group_name] = {
            "image_count": len(image_paths),
            "mask_count": len(mask_paths),
            "paired_count": len(common_keys),
            "image_only_count": len(
                image_only
            ),
            "mask_only_count": len(
                mask_only
            ),
            "split_mapping_count": len(
                split_mapping
            ),
            "split_counts": {
                split_name: sum(
                    1
                    for key in common_keys
                    if split_mapping.get(
                        key,
                        "unspecified",
                    )
                    == split_name
                )
                for split_name in (
                    "train",
                    "validation",
                    "test",
                    "unspecified",
                    "conflict",
                )
            },
            "image_only_examples": image_only[
                :10
            ],
            "mask_only_examples": mask_only[
                :10
            ],
        }

    manifest = pd.DataFrame(
        records
    ).sort_values(
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

    inspected_rows = manifest[
        manifest[
            "same_shape"
        ].astype(str).ne("")
    ].copy()

    shape_failures = int(
        inspected_rows[
            "same_shape"
        ].astype(str).str.lower().eq(
            "false"
        ).sum()
    )

    open_failures = int(
        (
            inspected_rows[
                "image_open_ok"
            ].astype(str).str.lower().eq(
                "false"
            )
            |
            inspected_rows[
                "mask_open_ok"
            ].astype(str).str.lower().eq(
                "false"
            )
        ).sum()
    )

    summary = {
        "stage": (
            "v06_sen1floods11_pair_audit"
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
            len(inspected_rows)
        ),
        "shape_failure_count": (
            shape_failures
        ),
        "open_failure_count": (
            open_failures
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

    lines = [
        "# v0.6 Sen1Floods11 SAR–Su Eşleme Denetimi",
        "",
        f"- Toplam eşleşmiş çift: {len(manifest)}",
        f"- Rasterio ile incelenen çift: {len(inspected_rows)}",
        f"- Boyut uyuşmazlığı: {shape_failures}",
        f"- Açılmayan çift: {open_failures}",
        "",
    ]

    for group_name, group_summary in (
        group_summaries.items()
    ):
        lines.extend(
            [
                f"## {group_name}",
                "",
                f"- SAR: {group_summary['image_count']}",
                f"- Maske: {group_summary['mask_count']}",
                f"- Eşleşen: {group_summary['paired_count']}",
                f"- Maskesiz SAR: {group_summary['image_only_count']}",
                f"- Görüntüsüz maske: {group_summary['mask_only_count']}",
                f"- Split dağılımı: {group_summary['split_counts']}",
                "",
            ]
        )

    REPORT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("=" * 78)
    print(
        "SEN1FLOODS11 PAIR AUDIT"
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
            "  Split:",
            group_summary[
                "split_counts"
            ],
        )

    print()
    print(
        "Toplam eşleşmiş çift:",
        len(manifest),
    )
    print(
        "İncelenen çift:",
        len(inspected_rows),
    )
    print(
        "Boyut uyuşmazlığı:",
        shape_failures,
    )
    print(
        "Açılmayan çift:",
        open_failures,
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
