from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

DARTIS_RAW_ROOT = (
    ROOT
    / "data"
    / "external"
    / "dartis"
    / "raw"
)

POSITIVE_MANIFEST = (
    ROOT
    / "data"
    / "metadata"
    / "dartis_positive_oil_manifest.csv"
)

NEGATIVE_MANIFEST = (
    ROOT
    / "data"
    / "metadata"
    / "dartis_no_oil_scene_split.csv"
)

OUTPUT_MANIFEST = (
    ROOT
    / "data"
    / "metadata"
    / "v04_input_gate_dataset.csv"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "v04_input_gate_dataset"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "dataset_summary.json"
)

REPORT_PATH = (
    ROOT
    / "reports"
    / "v04_input_gate_dataset.md"
)

SUPPORTED_DARTIS_GROUPS = {
    "ow",
    "oc",
    "nw",
    "nc",
}

IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
}

UC_MERCED_CLASSES = {
    "agricultural",
    "airplane",
    "baseballdiamond",
    "beach",
    "buildings",
    "chaparral",
    "denseresidential",
    "forest",
    "freeway",
    "golfcourse",
    "harbor",
    "intersection",
    "mediumresidential",
    "mobilehomepark",
    "overpass",
    "parkinglot",
    "river",
    "runway",
    "sparseresidential",
    "storagetanks",
    "tenniscourt",
}

# Bu kategoriler eğitimde hiç görülmeyecek.
# Özellikle airplane, mevcut hatayı bağımsız test edecek.
OPTICAL_TEST_LOCKED_CLASSES = {
    "airplane",
    "harbor",
    "river",
    "runway",
}

OPTICAL_CALIBRATION_CLASSES = {
    "parkinglot",
    "storagetanks",
}

OPTICAL_VALIDATION_CLASSES = {
    "buildings",
    "freeway",
    "overpass",
}

SKIP_DIRECTORY_NAMES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "outputs",
    "checkpoints",
    "wandb",
    "patches",
    "patch",
    "noisy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "DARTIS deniz SAR görüntülerini desteklenen, "
            "UC Merced optik görüntülerini desteklenmeyen "
            "olarak etiketleyip v0.4 input-gate manifesti oluşturur."
        )
    )

    parser.add_argument(
        "--sar-cam-root",
        type=Path,
        required=True,
        help=(
            "UC Merced verisinin bulunduğu SAR-CAM "
            "projesinin ana klasörü."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    parser.add_argument(
        "--minimum-supported",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--minimum-unsupported",
        type=int,
        default=500,
    )

    return parser.parse_args()


def clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if text.lower() in {
        "",
        "nan",
        "none",
        "<na>",
    }:
        return ""

    return text


def normalize_token(value: str) -> str:
    return re.sub(
        r"[^a-z0-9]",
        "",
        value.lower(),
    )


NORMALIZED_CLASS_MAP = {
    normalize_token(class_name): class_name
    for class_name in UC_MERCED_CLASSES
}


def stable_bucket(
    group_identifier: str,
    seed: int,
) -> int:
    text = (
        f"{seed}|{group_identifier}"
    ).encode("utf-8")

    digest = hashlib.sha256(
        text
    ).hexdigest()

    return int(
        digest[:12],
        16,
    ) % 100


def split_from_bucket(
    bucket: int,
) -> str:
    if bucket < 70:
        return "train"

    if bucket < 80:
        return "validation"

    if bucket < 90:
        return "calibration"

    return "test_locked"


def file_sample_id(
    source_dataset: str,
    image_path: Path,
) -> str:
    text = (
        f"{source_dataset}|"
        f"{image_path.resolve()}"
    ).encode("utf-8")

    return hashlib.sha256(
        text
    ).hexdigest()[:24]


def relative_or_absolute(
    path: Path,
) -> str:
    try:
        return str(
            path.resolve().relative_to(
                ROOT.resolve()
            )
        )

    except ValueError:
        return str(
            path.resolve()
        )


def find_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    normalized_columns = {
        normalize_token(str(column)): str(column)
        for column in dataframe.columns
    }

    for candidate in candidates:
        normalized = normalize_token(
            candidate
        )

        if normalized in normalized_columns:
            return normalized_columns[
                normalized
            ]

    return None


def build_scene_mapping(
    manifest_path: Path,
    image_candidates: list[str],
    scene_candidates: list[str],
    group_candidates: list[str],
) -> dict[
    tuple[str, str],
    str,
]:
    if not manifest_path.exists():
        return {}

    dataframe = pd.read_csv(
        manifest_path,
        encoding="utf-8-sig",
        low_memory=False,
    )

    image_column = find_column(
        dataframe,
        image_candidates,
    )

    scene_column = find_column(
        dataframe,
        scene_candidates,
    )

    group_column = find_column(
        dataframe,
        group_candidates,
    )

    if (
        image_column is None
        or scene_column is None
    ):
        return {}

    mapping = {}

    for row in dataframe.to_dict(
        orient="records"
    ):
        image_name = clean_text(
            row.get(
                image_column,
                "",
            )
        )

        scene_id = clean_text(
            row.get(
                scene_column,
                "",
            )
        )

        group_name = clean_text(
            row.get(
                group_column,
                "",
            )
        ).lower() if group_column else ""

        if not image_name or not scene_id:
            continue

        mapping[
            (
                group_name,
                Path(image_name).name.lower(),
            )
        ] = scene_id

        mapping[
            (
                "",
                Path(image_name).name.lower(),
            )
        ] = scene_id

    return mapping


def fallback_dartis_scene_id(
    group_name: str,
    image_path: Path,
) -> str:
    stem = image_path.stem
    parts = stem.split("-")

    # nc-0281-04-000002 -> nc-0281-04
    # nw-0505-03-000002 -> nw-0505-03
    if (
        group_name in {"nc", "nw"}
        and len(parts) >= 4
        and parts[-1].isdigit()
    ):
        return "-".join(
            parts[:-1]
        )

    return stem


def inspect_image(
    image_path: Path,
) -> tuple[int, int, str] | None:
    try:
        with Image.open(
            image_path
        ) as image:
            width, height = image.size
            mode = str(image.mode)

        if width < 32 or height < 32:
            return None

        return (
            int(width),
            int(height),
            mode,
        )

    except Exception:
        return None


def collect_dartis_rows(
    seed: int,
) -> list[dict[str, Any]]:
    if not DARTIS_RAW_ROOT.exists():
        raise FileNotFoundError(
            "DARTIS raw klasörü bulunamadı: "
            f"{DARTIS_RAW_ROOT}"
        )

    positive_mapping = build_scene_mapping(
        POSITIVE_MANIFEST,
        image_candidates=[
            "source_image_name",
            "image_name",
            "filename",
        ],
        scene_candidates=[
            "scene_id",
            "source_scene_id",
            "id_3",
        ],
        group_candidates=[
            "image_set",
            "source_group",
            "group",
        ],
    )

    negative_mapping = build_scene_mapping(
        NEGATIVE_MANIFEST,
        image_candidates=[
            "image_name",
            "sample_name",
            "source_image_name",
            "filename",
        ],
        scene_candidates=[
            "scene_id",
            "id_3",
            "source_scene_id",
        ],
        group_candidates=[
            "image_set",
            "source_group",
            "group",
        ],
    )

    scene_mapping = {
        **positive_mapping,
        **negative_mapping,
    }

    rows = []

    for group_name in sorted(
        SUPPORTED_DARTIS_GROUPS
    ):
        group_directory = (
            DARTIS_RAW_ROOT
            / group_name
        )

        if not group_directory.exists():
            print(
                "UYARI: DARTIS grubu bulunamadı:",
                group_directory,
            )

            continue

        image_paths = sorted(
            path
            for path in group_directory.rglob("*")
            if (
                path.is_file()
                and path.suffix.lower()
                in IMAGE_SUFFIXES
            )
        )

        for image_path in image_paths:
            inspected = inspect_image(
                image_path
            )

            if inspected is None:
                continue

            width, height, mode = inspected

            image_name = (
                image_path.name.lower()
            )

            scene_id = (
                scene_mapping.get(
                    (
                        group_name,
                        image_name,
                    )
                )
                or scene_mapping.get(
                    (
                        "",
                        image_name,
                    )
                )
                or fallback_dartis_scene_id(
                    group_name,
                    image_path,
                )
            )

            split_group_id = (
                f"dartis:{group_name}:{scene_id}"
            )

            split = split_from_bucket(
                stable_bucket(
                    split_group_id,
                    seed,
                )
            )

            rows.append(
                {
                    "sample_id": file_sample_id(
                        "dartis",
                        image_path,
                    ),
                    "binary_label": 1,
                    "binary_label_name": (
                        "supported_sea_sar"
                    ),
                    "split": split,
                    "source_dataset": (
                        "dartis"
                    ),
                    "source_group": (
                        group_name
                    ),
                    "scene_id": (
                        scene_id
                    ),
                    "split_group_id": (
                        split_group_id
                    ),
                    "image_path": (
                        relative_or_absolute(
                            image_path
                        )
                    ),
                    "width": width,
                    "height": height,
                    "image_mode": mode,
                    "input_policy": (
                        "accept_for_oil_pipeline"
                    ),
                }
            )

    return rows


def identify_uc_merced_class(
    path: Path,
) -> str | None:
    for part in reversed(
        path.parts
    ):
        normalized = normalize_token(
            part
        )

        if normalized in NORMALIZED_CLASS_MAP:
            return NORMALIZED_CLASS_MAP[
                normalized
            ]

    return None


def optical_path_score(
    path: Path,
) -> int:
    normalized_parts = {
        normalize_token(part)
        for part in path.parts
    }

    score = 0

    if "raw" in normalized_parts:
        score += 100

    if "original" in normalized_parts:
        score += 100

    if (
        "ucmerced"
        in "".join(
            normalized_parts
        )
    ):
        score += 40

    if "processed" in normalized_parts:
        score -= 20

    if "clean" in normalized_parts:
        score -= 10

    if "grayscale" in normalized_parts:
        score -= 20

    return score


def discover_uc_merced_candidates(
    sar_cam_root: Path,
) -> dict[
    tuple[str, str],
    Path,
]:
    if not sar_cam_root.exists():
        raise FileNotFoundError(
            "SAR-CAM klasörü bulunamadı: "
            f"{sar_cam_root}"
        )

    selected: dict[
        tuple[str, str],
        tuple[int, Path],
    ] = {}

    for current_root, directory_names, filenames in os.walk(
        sar_cam_root
    ):
        directory_names[:] = [
            directory_name
            for directory_name
            in directory_names
            if (
                normalize_token(
                    directory_name
                )
                not in {
                    normalize_token(
                        name
                    )
                    for name
                    in SKIP_DIRECTORY_NAMES
                }
                and "patch"
                not in normalize_token(
                    directory_name
                )
            )
        ]

        current_path = Path(
            current_root
        )

        for filename in filenames:
            image_path = (
                current_path
                / filename
            )

            if (
                image_path.suffix.lower()
                not in IMAGE_SUFFIXES
            ):
                continue

            class_name = (
                identify_uc_merced_class(
                    image_path
                )
            )

            if class_name is None:
                continue

            # Aynı UC Merced görüntüsünün raw, clean,
            # train veya test kopyaları bulunabilir.
            # Sınıf + dosya gövdesine göre tek kopya seçilir.
            key = (
                class_name,
                normalize_token(
                    image_path.stem
                ),
            )

            score = optical_path_score(
                image_path
            )

            previous = selected.get(
                key
            )

            if (
                previous is None
                or score > previous[0]
                or (
                    score == previous[0]
                    and len(str(image_path))
                    < len(
                        str(previous[1])
                    )
                )
            ):
                selected[key] = (
                    score,
                    image_path,
                )

    return {
        key: value[1]
        for key, value
        in selected.items()
    }


def optical_split(
    class_name: str,
) -> str:
    if (
        class_name
        in OPTICAL_TEST_LOCKED_CLASSES
    ):
        return "test_locked"

    if (
        class_name
        in OPTICAL_CALIBRATION_CLASSES
    ):
        return "calibration"

    if (
        class_name
        in OPTICAL_VALIDATION_CLASSES
    ):
        return "validation"

    return "train"


def collect_optical_rows(
    sar_cam_root: Path,
) -> list[dict[str, Any]]:
    candidates = (
        discover_uc_merced_candidates(
            sar_cam_root
        )
    )

    if not candidates:
        raise RuntimeError(
            "SAR-CAM klasöründe UC Merced sınıf "
            "klasörleri bulunamadı."
        )

    rows = []

    for (
        class_name,
        image_stem,
    ), image_path in sorted(
        candidates.items()
    ):
        inspected = inspect_image(
            image_path
        )

        if inspected is None:
            continue

        width, height, mode = inspected

        split = optical_split(
            class_name
        )

        # Optik sınıfların tamamı aynı split'te kalır.
        # Böylece testte görülmemiş optik kategoriler bulunur.
        split_group_id = (
            f"ucmerced-class:{class_name}"
        )

        rows.append(
            {
                "sample_id": file_sample_id(
                    "uc_merced",
                    image_path,
                ),
                "binary_label": 0,
                "binary_label_name": (
                    "unsupported_input"
                ),
                "split": split,
                "source_dataset": (
                    "uc_merced"
                ),
                "source_group": (
                    class_name
                ),
                "scene_id": (
                    f"{class_name}:{image_stem}"
                ),
                "split_group_id": (
                    split_group_id
                ),
                "image_path": (
                    relative_or_absolute(
                        image_path
                    )
                ),
                "width": width,
                "height": height,
                "image_mode": mode,
                "input_policy": (
                    "reject_before_oil_pipeline"
                ),
            }
        )

    return rows


def check_split_leakage(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    leakage = (
        dataframe
        .groupby(
            "split_group_id"
        )["split"]
        .nunique()
    )

    leaking_groups = leakage[
        leakage > 1
    ].index

    return dataframe[
        dataframe[
            "split_group_id"
        ].isin(
            leaking_groups
        )
    ].copy()


def nested_counts(
    dataframe: pd.DataFrame,
    columns: list[str],
) -> dict[str, Any]:
    grouped = (
        dataframe
        .groupby(
            columns
        )
        .size()
    )

    result: dict[str, Any] = {}

    for index, count in grouped.items():
        if not isinstance(
            index,
            tuple,
        ):
            index = (
                index,
            )

        target = result

        for key in index[:-1]:
            target = target.setdefault(
                str(key),
                {},
            )

        target[
            str(index[-1])
        ] = int(
            count
        )

    return result


def write_report(
    dataframe: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    lines = [
        "# v0.4 Input Gate Veri Seti",
        "",
        "## Amaç",
        "",
        "Petrol analizinden önce yüklenen görüntünün "
        "desteklenen veri alanında olup olmadığını belirlemek.",
        "",
        "Desteklenen giriş:",
        "",
        "- DARTIS benzeri deniz veya kıyı içeren SAR görüntüsü",
        "",
        "Desteklenmeyen giriş:",
        "",
        "- Optik uydu görüntüsü",
        "- Hava fotoğrafı",
        "- Uçak, bina, tarım, yol ve benzeri optik sahneler",
        "",
        "## Etiketler",
        "",
        "- `1`: supported_sea_sar",
        "- `0`: unsupported_input",
        "",
        "## Split protokolü",
        "",
        "- DARTIS görüntüleri sahne kimliğine göre bölündü.",
        "- Aynı DARTIS sahnesi birden fazla split'e girmedi.",
        "- UC Merced kategorileri sınıf bazında bölündü.",
        "- `airplane`, `harbor`, `river` ve `runway` "
        "yalnız kilitli testte tutuldu.",
        "",
        "## Dağılım",
        "",
        "| Split | Desteklenen | Desteklenmeyen | Toplam |",
        "|---|---:|---:|---:|",
    ]

    for split_name in (
        "train",
        "validation",
        "calibration",
        "test_locked",
    ):
        subset = dataframe[
            dataframe["split"].eq(
                split_name
            )
        ]

        supported = int(
            subset[
                "binary_label"
            ].eq(1).sum()
        )

        unsupported = int(
            subset[
                "binary_label"
            ].eq(0).sum()
        )

        lines.append(
            f"| {split_name} "
            f"| {supported} "
            f"| {unsupported} "
            f"| {len(subset)} |"
        )

    lines.extend(
        [
            "",
            f"- Toplam örnek: {len(dataframe)}",
            f"- Desteklenen deniz SAR: "
            f"{int(dataframe['binary_label'].eq(1).sum())}",
            f"- Desteklenmeyen optik: "
            f"{int(dataframe['binary_label'].eq(0).sum())}",
            f"- Split leakage: "
            f"{summary['split_leakage_rows']}",
            "",
            "## Önemli sınır",
            "",
            "Bu veri seti evrensel bir `SAR / optik` sınıflandırıcısı "
            "oluşturmaz. Amaç, uygulamanın eğitim alanına uymayan "
            "girdileri muhafazakâr biçimde reddetmesidir.",
            "",
            "Model kararsız kaldığında görüntü petrol analizine "
            "gönderilmeyecektir.",
            "",
        ]
    )

    REPORT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    sar_cam_root = (
        args.sar_cam_root
        .expanduser()
        .resolve()
    )

    OUTPUT_MANIFEST.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 78)
    print(
        "ROBUST BINARY OIL DETECTOR v0.4"
    )
    print(
        "INPUT GATE VERİ SETİ HAZIRLIĞI"
    )
    print("=" * 78)

    print(
        "DARTIS root:",
        DARTIS_RAW_ROOT,
    )

    print(
        "SAR-CAM root:",
        sar_cam_root,
    )

    print()
    print(
        "DARTIS deniz SAR görüntüleri taranıyor..."
    )

    supported_rows = (
        collect_dartis_rows(
            args.seed
        )
    )

    print(
        "Desteklenen SAR görüntüsü:",
        len(supported_rows),
    )

    print()
    print(
        "UC Merced optik görüntüleri taranıyor..."
    )

    unsupported_rows = (
        collect_optical_rows(
            sar_cam_root
        )
    )

    print(
        "Desteklenmeyen optik görüntü:",
        len(unsupported_rows),
    )

    if (
        len(supported_rows)
        < args.minimum_supported
    ):
        raise RuntimeError(
            "Yeterli DARTIS görüntüsü bulunamadı. "
            f"Bulunan: {len(supported_rows)}"
        )

    if (
        len(unsupported_rows)
        < args.minimum_unsupported
    ):
        raise RuntimeError(
            "Yeterli UC Merced görüntüsü bulunamadı. "
            f"Bulunan: {len(unsupported_rows)}"
        )

    dataframe = pd.DataFrame(
        supported_rows
        + unsupported_rows
    )

    dataframe = (
        dataframe
        .drop_duplicates(
            subset=[
                "source_dataset",
                "image_path",
            ]
        )
        .sort_values(
            [
                "split",
                "binary_label",
                "source_dataset",
                "source_group",
                "scene_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    leakage = check_split_leakage(
        dataframe
    )

    if not leakage.empty:
        print(
            leakage[
                [
                    "split_group_id",
                    "split",
                    "image_path",
                ]
            ]
            .head(20)
            .to_string(
                index=False
            )
        )

        raise RuntimeError(
            "Split leakage tespit edildi."
        )

    dataframe.to_csv(
        OUTPUT_MANIFEST,
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "total_samples": int(
            len(dataframe)
        ),
        "supported_samples": int(
            dataframe[
                "binary_label"
            ].eq(1).sum()
        ),
        "unsupported_samples": int(
            dataframe[
                "binary_label"
            ].eq(0).sum()
        ),
        "unique_split_groups": int(
            dataframe[
                "split_group_id"
            ].nunique()
        ),
        "split_leakage_rows": int(
            len(leakage)
        ),
        "counts_by_split_and_label": (
            nested_counts(
                dataframe,
                [
                    "split",
                    "binary_label_name",
                ],
            )
        ),
        "counts_by_dataset_and_group": (
            nested_counts(
                dataframe,
                [
                    "source_dataset",
                    "source_group",
                ],
            )
        ),
        "locked_optical_classes": sorted(
            OPTICAL_TEST_LOCKED_CLASSES
        ),
        "validation_optical_classes": sorted(
            OPTICAL_VALIDATION_CLASSES
        ),
        "calibration_optical_classes": sorted(
            OPTICAL_CALIBRATION_CLASSES
        ),
        "manifest_path": str(
            OUTPUT_MANIFEST
        ),
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

    write_report(
        dataframe,
        summary,
    )

    print()
    print("=" * 78)
    print(
        "INPUT GATE VERİ SETİ SONUCU"
    )
    print("=" * 78)

    print(
        "Toplam örnek:",
        len(dataframe),
    )

    print(
        "Desteklenen deniz SAR:",
        int(
            dataframe[
                "binary_label"
            ].eq(1).sum()
        ),
    )

    print(
        "Desteklenmeyen optik:",
        int(
            dataframe[
                "binary_label"
            ].eq(0).sum()
        ),
    )

    print(
        "Split leakage:",
        len(leakage),
    )

    print()

    for split_name in (
        "train",
        "validation",
        "calibration",
        "test_locked",
    ):
        subset = dataframe[
            dataframe[
                "split"
            ].eq(
                split_name
            )
        ]

        supported = int(
            subset[
                "binary_label"
            ].eq(1).sum()
        )

        unsupported = int(
            subset[
                "binary_label"
            ].eq(0).sum()
        )

        print(
            f"{split_name:12s}: "
            f"{len(subset):5d} "
            f"| SAR={supported:5d} "
            f"| unsupported={unsupported:5d}"
        )

    print()
    print(
        "Kilitli optik sınıflar:",
        ", ".join(
            sorted(
                OPTICAL_TEST_LOCKED_CLASSES
            )
        ),
    )

    print()
    print(
        "Manifest:",
        OUTPUT_MANIFEST.resolve(),
    )

    print(
        "Özet:",
        SUMMARY_PATH.resolve(),
    )

    print(
        "Rapor:",
        REPORT_PATH.resolve(),
    )


if __name__ == "__main__":
    main()
