from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"

OUTPUT_DIR = ROOT / "outputs" / "v06_label_audit"
REPORT_PATH = ROOT / "reports" / "v06_label_audit.md"

CSV_AUDIT_PATH = OUTPUT_DIR / "metadata_csv_audit.csv"
MASK_AUDIT_PATH = OUTPUT_DIR / "mask_directory_audit.csv"
DARTIS_AUDIT_PATH = OUTPUT_DIR / "dartis_group_audit.csv"
PAIR_AUDIT_PATH = OUTPUT_DIR / "image_mask_pair_audit.csv"
SEMANTIC_AUDIT_PATH = OUTPUT_DIR / "semantic_directory_audit.csv"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
}

RELEVANT_COLUMN_KEYWORDS = {
    "oil",
    "spill",
    "look",
    "slick",
    "mask",
    "label",
    "class",
    "target",
    "water",
    "sea",
    "ocean",
    "land",
    "coast",
    "shore",
}

SEMANTIC_DIRECTORY_KEYWORDS = {
    "mask",
    "label",
    "oil",
    "spill",
    "look",
    "slick",
    "water",
    "sea",
    "ocean",
    "land",
    "coast",
    "shore",
}


def is_image(path: Path) -> bool:
    return (
        path.is_file()
        and path.suffix.lower()
        in IMAGE_EXTENSIONS
    )


def relative(path: Path) -> str:
    try:
        return str(
            path.resolve().relative_to(
                ROOT.resolve()
            )
        ).replace("\\", "/")

    except ValueError:
        return str(path.resolve())


def count_images(path: Path) -> int:
    if not path.exists():
        return 0

    return sum(
        1
        for file_path in path.rglob("*")
        if is_image(file_path)
    )


def image_stems(path: Path) -> set[str]:
    if not path.exists():
        return set()

    return {
        file_path.stem.lower()
        for file_path in path.rglob("*")
        if is_image(file_path)
    }


def audit_metadata_csvs() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    metadata_root = DATA_ROOT / "metadata"

    if not metadata_root.exists():
        return pd.DataFrame(rows)

    for csv_path in sorted(
        metadata_root.rglob("*.csv")
    ):
        try:
            dataframe = pd.read_csv(
                csv_path,
                encoding="utf-8-sig",
                nrows=5,
                low_memory=False,
            )

            columns = [
                str(column)
                for column in dataframe.columns
            ]

            relevant_columns = [
                column
                for column in columns
                if any(
                    keyword in column.lower()
                    for keyword
                    in RELEVANT_COLUMN_KEYWORDS
                )
            ]

            status = "OK"
            error = ""

        except Exception as exc:
            columns = []
            relevant_columns = []
            status = "ERROR"
            error = (
                f"{type(exc).__name__}: {exc}"
            )

        rows.append(
            {
                "csv_path": relative(csv_path),
                "status": status,
                "column_count": len(columns),
                "columns": " | ".join(columns),
                "relevant_columns": " | ".join(
                    relevant_columns
                ),
                "error": error,
            }
        )

    return pd.DataFrame(rows)


def find_semantic_directories() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    if not DATA_ROOT.exists():
        return pd.DataFrame(rows)

    for current_root, directory_names, _ in os.walk(
        DATA_ROOT
    ):
        current_path = Path(current_root)

        for directory_name in directory_names:
            lowered = directory_name.lower()

            matched = sorted(
                keyword
                for keyword
                in SEMANTIC_DIRECTORY_KEYWORDS
                if keyword in lowered
            )

            if not matched:
                continue

            directory_path = (
                current_path
                / directory_name
            )

            rows.append(
                {
                    "directory_path": relative(
                        directory_path
                    ),
                    "directory_name": directory_name,
                    "matched_keywords": " | ".join(
                        matched
                    ),
                    "image_count": count_images(
                        directory_path
                    ),
                }
            )

    return pd.DataFrame(rows)


def sample_mask_directory(
    directory: Path,
    maximum_samples: int = 30,
) -> dict[str, Any]:
    image_paths = [
        path
        for path in directory.rglob("*")
        if is_image(path)
    ]

    image_paths = sorted(
        image_paths
    )[:maximum_samples]

    modes: set[str] = set()
    shapes: set[str] = set()
    unique_values: set[int] = set()
    readable = 0
    failures = 0

    for image_path in image_paths:
        try:
            with Image.open(image_path) as image:
                array = np.asarray(image)

                modes.add(
                    str(image.mode)
                )

                shapes.add(
                    "x".join(
                        str(value)
                        for value in array.shape
                    )
                )

                values = np.unique(array)

                if len(values) <= 64:
                    unique_values.update(
                        int(value)
                        for value in values
                    )

                readable += 1

        except Exception:
            failures += 1

    sorted_values = sorted(
        unique_values
    )

    binary_like = (
        bool(sorted_values)
        and len(sorted_values) <= 3
        and set(sorted_values).issubset(
            {
                0,
                1,
                255,
            }
        )
    )

    return {
        "sampled": len(image_paths),
        "readable": readable,
        "failures": failures,
        "modes": " | ".join(sorted(modes)),
        "shapes": " | ".join(sorted(shapes)),
        "unique_values": " | ".join(
            str(value)
            for value in sorted_values[:64]
        ),
        "binary_like": binary_like,
    }


def audit_mask_directories(
    semantic_directories: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    if semantic_directories.empty:
        return pd.DataFrame(rows)

    mask_rows = semantic_directories[
        semantic_directories[
            "directory_name"
        ].str.lower().str.contains(
            "mask|label",
            regex=True,
        )
    ]

    for row in mask_rows.to_dict(
        orient="records"
    ):
        directory = ROOT / row[
            "directory_path"
        ]

        statistics = sample_mask_directory(
            directory
        )

        rows.append(
            {
                "directory_path": row[
                    "directory_path"
                ],
                "image_count": int(
                    row["image_count"]
                ),
                **statistics,
            }
        )

    return pd.DataFrame(rows)


def audit_image_mask_pairs() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    if not DATA_ROOT.exists():
        return pd.DataFrame(rows)

    image_directories = [
        path
        for path in DATA_ROOT.rglob("*")
        if path.is_dir()
        and path.name.lower()
        in {
            "image",
            "images",
        }
    ]

    for image_directory in image_directories:
        sibling_candidates = [
            image_directory.parent / "mask",
            image_directory.parent / "masks",
            image_directory.parent / "label",
            image_directory.parent / "labels",
        ]

        for mask_directory in sibling_candidates:
            if not mask_directory.exists():
                continue

            images = image_stems(
                image_directory
            )

            masks = image_stems(
                mask_directory
            )

            rows.append(
                {
                    "image_directory": relative(
                        image_directory
                    ),
                    "mask_directory": relative(
                        mask_directory
                    ),
                    "image_count": len(images),
                    "mask_count": len(masks),
                    "paired_stem_count": len(
                        images & masks
                    ),
                    "images_without_mask": len(
                        images - masks
                    ),
                    "masks_without_image": len(
                        masks - images
                    ),
                }
            )

    return pd.DataFrame(rows)


def locate_dartis_roots() -> list[Path]:
    roots: list[Path] = []

    if not DATA_ROOT.exists():
        return roots

    for path in DATA_ROOT.rglob("*"):
        if (
            path.is_dir()
            and path.name.lower() == "dartis"
        ):
            roots.append(path)

    direct_candidate = (
        DATA_ROOT
        / "external"
        / "dartis"
    )

    if (
        direct_candidate.exists()
        and direct_candidate not in roots
    ):
        roots.append(
            direct_candidate
        )

    return sorted(
        set(roots)
    )


def audit_dartis_groups() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for dartis_root in locate_dartis_roots():
        content_root = dartis_root

        raw_candidate = (
            dartis_root
            / "raw"
        )

        if raw_candidate.exists():
            content_root = raw_candidate

        group_directories = [
            path
            for path in content_root.iterdir()
            if path.is_dir()
        ]

        for group_directory in sorted(
            group_directories
        ):
            rows.append(
                {
                    "dartis_root": relative(
                        dartis_root
                    ),
                    "content_root": relative(
                        content_root
                    ),
                    "group_name": (
                        group_directory.name
                    ),
                    "image_count": count_images(
                        group_directory
                    ),
                    "potential_role": (
                        "potential_non_oil_or_lookalike"
                        if group_directory.name
                        .lower()
                        .startswith("n")
                        else
                        "potential_oil"
                        if group_directory.name
                        .lower()
                        .startswith("o")
                        else
                        "unknown"
                    ),
                    "role_confirmed": False,
                }
            )

    return pd.DataFrame(rows)


def contains_keyword(
    values: list[str],
    keywords: set[str],
) -> bool:
    combined = " ".join(
        value.lower()
        for value in values
    )

    return any(
        keyword in combined
        for keyword in keywords
    )


def main() -> None:
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
        "ROBUST BINARY OIL DETECTOR v0.6"
    )
    print(
        "WATER / OIL / LOOK-ALIKE ETİKET DENETİMİ"
    )
    print("=" * 78)

    csv_audit = audit_metadata_csvs()
    semantic_audit = (
        find_semantic_directories()
    )
    mask_audit = audit_mask_directories(
        semantic_audit
    )
    pair_audit = audit_image_mask_pairs()
    dartis_audit = audit_dartis_groups()

    csv_audit.to_csv(
        CSV_AUDIT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    semantic_audit.to_csv(
        SEMANTIC_AUDIT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    mask_audit.to_csv(
        MASK_AUDIT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    pair_audit.to_csv(
        PAIR_AUDIT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    dartis_audit.to_csv(
        DARTIS_AUDIT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    csv_relevant_values = (
        csv_audit[
            "relevant_columns"
        ].dropna().astype(str).tolist()
        if not csv_audit.empty
        else []
    )

    semantic_values = (
        semantic_audit[
            "directory_path"
        ].dropna().astype(str).tolist()
        if not semantic_audit.empty
        else []
    )

    explicit_water_land_evidence = (
        contains_keyword(
            csv_relevant_values,
            {
                "water",
                "sea",
                "ocean",
                "land",
                "coast",
                "shore",
            },
        )
        or contains_keyword(
            semantic_values,
            {
                "water",
                "sea",
                "ocean",
                "land",
                "coast",
                "shore",
            },
        )
    )

    explicit_lookalike_evidence = (
        contains_keyword(
            csv_relevant_values,
            {
                "look",
                "slick",
            },
        )
        or contains_keyword(
            semantic_values,
            {
                "lookalike",
                "look-alike",
                "slick",
            },
        )
    )

    binary_mask_candidates = 0

    if (
        not mask_audit.empty
        and "binary_like"
        in mask_audit.columns
    ):
        binary_mask_candidates = int(
            mask_audit[
                "binary_like"
            ].astype(bool).sum()
        )

    paired_image_mask_count = 0

    if (
        not pair_audit.empty
        and "paired_stem_count"
        in pair_audit.columns
    ):
        paired_image_mask_count = int(
            pair_audit[
                "paired_stem_count"
            ].sum()
        )

    summary = {
        "audit_version": "v06",
        "data_modified": False,
        "training_performed": False,
        "locked_test_used": False,
        "metadata_csv_count": int(
            len(csv_audit)
        ),
        "semantic_directory_count": int(
            len(semantic_audit)
        ),
        "mask_directory_count": int(
            len(mask_audit)
        ),
        "binary_mask_candidate_directories": (
            binary_mask_candidates
        ),
        "paired_image_mask_count": (
            paired_image_mask_count
        ),
        "dartis_group_count": int(
            len(dartis_audit)
        ),
        "explicit_water_land_evidence": bool(
            explicit_water_land_evidence
        ),
        "explicit_lookalike_evidence": bool(
            explicit_lookalike_evidence
        ),
        "important_note": (
            "DARTIS grup adlarından yapılan potential_role "
            "işaretleri yalnız olasılıktır; veri seti "
            "dokümantasyonu doğrulanmadan gerçek etiket "
            "olarak kullanılmayacaktır."
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

    lines = [
        "# v0.6 Water / Oil / Look-Alike Etiket Denetimi",
        "",
        "## Amaç",
        "",
        "Mevcut verilerde aşağıdaki bilgilerin bulunup "
        "bulunmadığını doğrulamak:",
        "",
        "- Piksel seviyesinde petrol maskesi",
        "- Piksel seviyesinde kara–su maskesi",
        "- Görüntü veya aday seviyesinde look-alike etiketi",
        "",
        "## Genel sonuç",
        "",
        f"- Metadata CSV: {len(csv_audit)}",
        f"- Semantik klasör: {len(semantic_audit)}",
        f"- Maske klasörü: {len(mask_audit)}",
        f"- Binary maske adayı klasör: "
        f"{binary_mask_candidates}",
        f"- Eşleşen görüntü–maske çifti: "
        f"{paired_image_mask_count}",
        f"- DARTIS grup sayısı: {len(dartis_audit)}",
        f"- Açık kara–su etiketi kanıtı: "
        f"{explicit_water_land_evidence}",
        f"- Açık look-alike etiketi kanıtı: "
        f"{explicit_lookalike_evidence}",
        "",
        "## Bilimsel uyarı",
        "",
        "Klasör adlarının `ow`, `oc`, `nw`, `nc` olması "
        "tek başına etiket anlamlarını doğrulamaz. "
        "Dokümantasyon veya manifest kanıtı olmadan bu "
        "gruplar model sınıfı olarak kabul edilmeyecektir.",
        "",
        "Bu denetim sırasında hiçbir veri değiştirilmemiş, "
        "model eğitilmemiş ve kilitli test kullanılmamıştır.",
    ]

    REPORT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print()
    print("Metadata CSV:", len(csv_audit))
    print(
        "Semantik klasör:",
        len(semantic_audit),
    )
    print(
        "Maske klasörü:",
        len(mask_audit),
    )
    print(
        "Binary maske adayı klasör:",
        binary_mask_candidates,
    )
    print(
        "Eşleşen görüntü-maskeler:",
        paired_image_mask_count,
    )
    print(
        "DARTIS grup sayısı:",
        len(dartis_audit),
    )
    print(
        "Açık kara-su etiketi kanıtı:",
        explicit_water_land_evidence,
    )
    print(
        "Açık look-alike etiketi kanıtı:",
        explicit_lookalike_evidence,
    )

    if not dartis_audit.empty:
        print()
        print("DARTIS grupları:")
        print(
            dartis_audit[
                [
                    "group_name",
                    "image_count",
                    "potential_role",
                ]
            ].to_string(
                index=False
            )
        )

    if not pair_audit.empty:
        print()
        print("Görüntü-maske eşleşmeleri:")
        print(
            pair_audit.to_string(
                index=False
            )
        )

    print()
    print(
        "Özet:",
        SUMMARY_PATH.resolve(),
    )
    print(
        "Rapor:",
        REPORT_PATH.resolve(),
    )
    print(
        "CSV denetimi:",
        CSV_AUDIT_PATH.resolve(),
    )
    print(
        "Maske denetimi:",
        MASK_AUDIT_PATH.resolve(),
    )
    print()
    print(
        "Kilitli test kullanımı: YOK"
    )


if __name__ == "__main__":
    main()
