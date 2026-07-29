from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]

FINAL_MANIFEST = (
    ROOT
    / "data"
    / "metadata"
    / "v06_dartis_water_masks_final.csv"
)

FALLBACK_MANIFEST = (
    ROOT
    / "data"
    / "metadata"
    / "v06_dartis_water_masks_homography.csv"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "v06_coastal_annotation_batch"
)

IMAGE_DIR = OUTPUT_DIR / "images"
PRIOR_DIR = OUTPUT_DIR / "prior_reference_only"
CONTACT_SHEET_DIR = OUTPUT_DIR / "contact_sheets"

BATCH_MANIFEST = OUTPUT_DIR / "annotation_manifest.csv"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
README_PATH = OUTPUT_DIR / "README.txt"
ZIP_PATH = OUTPUT_DIR / "v06_coastal_annotation_batch.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Elle doğrulanmamış DARTIS kıyı görüntülerinden "
            "çeşitli bir anotasyon paketi hazırlar."
        )
    )
    parser.add_argument(
        "--total",
        type=int,
        default=120,
    )
    parser.add_argument(
        "--maximum-per-scene",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--contact-sheet-columns",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--contact-sheet-rows",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--thumbnail-size",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )
    return parser.parse_args()


def relative(path: Path) -> str:
    try:
        return str(
            path.resolve().relative_to(ROOT.resolve())
        ).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def resolve_path(value: Any) -> Path:
    path = Path(str(value).strip())
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def stable_key(value: str, seed: int) -> str:
    return hashlib.sha256(
        f"{seed}|{value}".encode("utf-8")
    ).hexdigest()


def parse_scene_family(sample_id: str) -> str:
    value = str(sample_id).split(":", 1)[-1]
    parts = value.split("-")

    if len(parts) >= 3:
        return "-".join(parts[:3])

    return value


def safe_boolean(
    series: pd.Series,
) -> pd.Series:
    if series.dtype == bool:
        return series

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(
            {
                "true",
                "1",
                "yes",
                "evet",
            }
        )
    )


def choose_source_manifest() -> Path:
    if FINAL_MANIFEST.exists():
        return FINAL_MANIFEST

    if FALLBACK_MANIFEST.exists():
        return FALLBACK_MANIFEST

    raise FileNotFoundError(
        "Ne final ne de homography manifesti bulundu.\n"
        f"- {FINAL_MANIFEST}\n"
        f"- {FALLBACK_MANIFEST}"
    )


def prepare_candidates(
    dataframe: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    required = {
        "sample_id",
        "subset",
        "surface_context",
        "image_path",
    }

    missing = required - set(dataframe.columns)

    if missing:
        raise RuntimeError(
            f"Manifestte eksik sütunlar: {sorted(missing)}"
        )

    candidates = dataframe[
        dataframe["surface_context"].eq("coast")
    ].copy()

    if "manual_verified" in candidates.columns:
        candidates = candidates[
            ~safe_boolean(
                candidates["manual_verified"]
            )
        ].copy()

    elif "final_mask_source" in candidates.columns:
        candidates = candidates[
            ~candidates[
                "final_mask_source"
            ].astype(str).eq(
                "MANUAL_CORRECTION"
            )
        ].copy()

    ratio_columns = [
        "land_ratio_checked",
        "land_ratio",
        "prior_land_ratio",
        "refined_land_ratio",
    ]

    selected_ratio_column = None

    for column in ratio_columns:
        if column in candidates.columns:
            selected_ratio_column = column
            break

    if selected_ratio_column is None:
        candidates["selection_land_ratio"] = 0.5

    else:
        candidates["selection_land_ratio"] = (
            pd.to_numeric(
                candidates[
                    selected_ratio_column
                ],
                errors="coerce",
            )
            .fillna(0.5)
            .clip(0.0, 1.0)
        )

    candidates["ratio_bin"] = pd.cut(
        candidates["selection_land_ratio"],
        bins=np.linspace(0.0, 1.0, 11),
        labels=False,
        include_lowest=True,
    ).fillna(5).astype(int)

    candidates["scene_family"] = candidates[
        "sample_id"
    ].map(parse_scene_family)

    candidates["_stable"] = candidates[
        "sample_id"
    ].map(
        lambda value: stable_key(
            str(value),
            seed,
        )
    )

    return candidates.sort_values(
        [
            "subset",
            "ratio_bin",
            "_stable",
        ]
    ).reset_index(drop=True)


def select_diverse_rows(
    candidates: pd.DataFrame,
    total: int,
    maximum_per_scene: int,
) -> pd.DataFrame:
    if total <= 0:
        raise ValueError(
            "--total pozitif olmalıdır."
        )

    if maximum_per_scene <= 0:
        raise ValueError(
            "--maximum-per-scene pozitif olmalıdır."
        )

    subsets = [
        subset
        for subset in ("oc", "nc")
        if subset in set(
            candidates["subset"].astype(str)
        )
    ]

    if not subsets:
        raise RuntimeError(
            "oc veya nc kıyı grubu bulunamadı."
        )

    base_quota = total // len(subsets)
    remainder = total % len(subsets)

    subset_quotas = {
        subset: (
            base_quota
            + (
                1
                if index < remainder
                else 0
            )
        )
        for index, subset in enumerate(subsets)
    }

    selected_indices: list[int] = []
    scene_counts: dict[str, int] = {}

    for subset in subsets:
        subset_rows = candidates[
            candidates["subset"].eq(subset)
        ].copy()

        groups = {
            int(bin_value): group.reset_index()
            for bin_value, group
            in subset_rows.groupby(
                "ratio_bin",
                dropna=False,
            )
        }

        positions = {
            bin_value: 0
            for bin_value in groups
        }

        quota = subset_quotas[subset]

        while (
            sum(
                1
                for index in selected_indices
                if candidates.loc[
                    index,
                    "subset",
                ] == subset
            )
            < quota
        ):
            progressed = False

            for bin_value in sorted(groups):
                group = groups[bin_value]

                while positions[bin_value] < len(group):
                    row = group.iloc[
                        positions[bin_value]
                    ]
                    positions[bin_value] += 1

                    original_index = int(
                        row["index"]
                    )
                    scene_family = str(
                        row["scene_family"]
                    )

                    if (
                        scene_counts.get(
                            scene_family,
                            0,
                        )
                        >= maximum_per_scene
                    ):
                        continue

                    if original_index in selected_indices:
                        continue

                    selected_indices.append(
                        original_index
                    )
                    scene_counts[
                        scene_family
                    ] = (
                        scene_counts.get(
                            scene_family,
                            0,
                        )
                        + 1
                    )
                    progressed = True
                    break

                current_count = sum(
                    1
                    for index in selected_indices
                    if candidates.loc[
                        index,
                        "subset",
                    ] == subset
                )

                if current_count >= quota:
                    break

            if not progressed:
                break

    if len(selected_indices) < total:
        remaining = candidates[
            ~candidates.index.isin(
                selected_indices
            )
        ].sort_values("_stable")

        for original_index, row in remaining.iterrows():
            scene_family = str(
                row["scene_family"]
            )

            if (
                scene_counts.get(
                    scene_family,
                    0,
                )
                >= maximum_per_scene
            ):
                continue

            selected_indices.append(
                int(original_index)
            )
            scene_counts[
                scene_family
            ] = (
                scene_counts.get(
                    scene_family,
                    0,
                )
                + 1
            )

            if len(selected_indices) >= total:
                break

    selected = candidates.loc[
        selected_indices
    ].copy()

    selected = selected.sort_values(
        [
            "subset",
            "ratio_bin",
            "_stable",
        ]
    ).reset_index(drop=True)

    selected["batch_index"] = np.arange(
        1,
        len(selected) + 1,
    )

    return selected


def find_prior_path(row: pd.Series) -> Path | None:
    columns = [
        "final_land_mask_path",
        "land_mask_path",
        "refined_land_mask_path",
    ]

    for column in columns:
        if column not in row.index:
            continue

        value = str(
            row.get(column, "")
        ).strip()

        if not value or value.lower() in {
            "nan",
            "none",
        }:
            continue

        path = resolve_path(value)

        if path.exists():
            return path

    return None


def create_contact_sheets(
    selected: pd.DataFrame,
    columns: int,
    rows: int,
    thumbnail_size: int,
) -> list[Path]:
    if columns <= 0 or rows <= 0:
        raise ValueError(
            "Contact sheet satır/sütun değerleri pozitif olmalıdır."
        )

    per_sheet = columns * rows
    label_height = 42
    cell_width = thumbnail_size
    cell_height = thumbnail_size + label_height

    font = ImageFont.load_default()
    output_paths = []

    for sheet_number, start in enumerate(
        range(0, len(selected), per_sheet),
        start=1,
    ):
        subset = selected.iloc[
            start:start + per_sheet
        ]

        canvas = Image.new(
            "RGB",
            (
                columns * cell_width,
                rows * cell_height,
            ),
            "black",
        )

        draw = ImageDraw.Draw(canvas)

        for local_index, row in enumerate(
            subset.to_dict(
                orient="records"
            )
        ):
            image_path = resolve_path(
                row["packaged_image_path"]
            )

            with Image.open(image_path) as image:
                image = image.convert("L")
                image.thumbnail(
                    (
                        thumbnail_size,
                        thumbnail_size,
                    ),
                    Image.Resampling.LANCZOS,
                )

                rgb = Image.new(
                    "RGB",
                    (
                        thumbnail_size,
                        thumbnail_size,
                    ),
                    "black",
                )

                offset = (
                    (
                        thumbnail_size
                        - image.width
                    ) // 2,
                    (
                        thumbnail_size
                        - image.height
                    ) // 2,
                )

                rgb.paste(
                    image.convert("RGB"),
                    offset,
                )

            column = local_index % columns
            row_index = local_index // columns

            x = column * cell_width
            y = row_index * cell_height

            canvas.paste(
                rgb,
                (x, y),
            )

            label = (
                f"{int(row['batch_index']):03d} "
                f"{row['sample_id']} "
                f"land≈{float(row['selection_land_ratio']):.2f}"
            )

            draw.rectangle(
                (
                    x,
                    y + thumbnail_size,
                    x + cell_width,
                    y + cell_height,
                ),
                fill=(
                    20,
                    20,
                    20,
                ),
            )

            draw.text(
                (
                    x + 5,
                    y + thumbnail_size + 7,
                ),
                label,
                fill="white",
                font=font,
            )

        path = (
            CONTACT_SHEET_DIR
            / f"contact_sheet_{sheet_number:02d}.jpg"
        )

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        canvas.save(
            path,
            quality=90,
        )

        output_paths.append(path)

    return output_paths


def build_zip() -> None:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    with zipfile.ZipFile(
        ZIP_PATH,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for path in sorted(
            OUTPUT_DIR.rglob("*")
        ):
            if (
                not path.is_file()
                or path == ZIP_PATH
            ):
                continue

            archive.write(
                path,
                path.relative_to(
                    OUTPUT_DIR
                ),
            )


def main() -> None:
    args = parse_args()

    source_manifest = (
        choose_source_manifest()
    )

    if OUTPUT_DIR.exists():
        shutil.rmtree(
            OUTPUT_DIR
        )

    IMAGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    PRIOR_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    CONTACT_SHEET_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = pd.read_csv(
        source_manifest,
        encoding="utf-8-sig",
        low_memory=False,
    )

    candidates = prepare_candidates(
        dataframe,
        args.seed,
    )

    selected = select_diverse_rows(
        candidates,
        args.total,
        args.maximum_per_scene,
    )

    if selected.empty:
        raise RuntimeError(
            "Anotasyon için görüntü seçilemedi."
        )

    output_rows = []

    for row in selected.to_dict(
        orient="records"
    ):
        image_path = resolve_path(
            row["image_path"]
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Görüntü bulunamadı: {image_path}"
            )

        safe_id = str(
            row["sample_id"]
        ).replace(":", "__")

        destination = (
            IMAGE_DIR
            / str(row["subset"])
            / (
                f"{int(row['batch_index']):03d}"
                f"__{safe_id}"
                f"{image_path.suffix.lower()}"
            )
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(
            image_path,
            destination,
        )

        prior_path = find_prior_path(
            pd.Series(row)
        )

        packaged_prior_path = ""

        if prior_path is not None:
            prior_destination = (
                PRIOR_DIR
                / str(row["subset"])
                / (
                    f"{int(row['batch_index']):03d}"
                    f"__{safe_id}.png"
                )
            )

            prior_destination.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with Image.open(
                prior_path
            ) as image:
                image.convert("L").save(
                    prior_destination
                )

            packaged_prior_path = relative(
                prior_destination
            )

        output_rows.append(
            {
                "batch_index": int(
                    row["batch_index"]
                ),
                "sample_id": row["sample_id"],
                "subset": row["subset"],
                "scene_family": (
                    row["scene_family"]
                ),
                "selection_land_ratio": float(
                    row["selection_land_ratio"]
                ),
                "ratio_bin": int(
                    row["ratio_bin"]
                ),
                "original_image_path": relative(
                    image_path
                ),
                "packaged_image_path": relative(
                    destination
                ),
                "prior_reference_path": (
                    packaged_prior_path
                ),
                "annotation_status": "PENDING",
                "land_mask_path": "",
                "safe_water_mask_path": "",
                "important_note": (
                    "Prior yalnız referanstır; "
                    "doğru etiket kabul edilmemelidir."
                ),
            }
        )

    batch = pd.DataFrame(
        output_rows
    )

    batch.to_csv(
        BATCH_MANIFEST,
        index=False,
        encoding="utf-8-sig",
    )

    contact_sheets = create_contact_sheets(
        batch,
        args.contact_sheet_columns,
        args.contact_sheet_rows,
        args.thumbnail_size,
    )

    subset_counts = (
        batch.groupby(
            [
                "subset",
                "ratio_bin",
            ],
            dropna=False,
        )
        .size()
        .reset_index(
            name="sample_count"
        )
        .sort_values(
            [
                "subset",
                "ratio_bin",
            ]
        )
    )

    summary = {
        "stage": (
            "v06_coastal_annotation_batch"
        ),
        "source_manifest": relative(
            source_manifest
        ),
        "available_unverified_coastal": int(
            len(candidates)
        ),
        "selected_count": int(
            len(batch)
        ),
        "subset_counts": (
            batch[
                "subset"
            ].value_counts()
            .sort_index()
            .to_dict()
        ),
        "ratio_bin_counts": (
            subset_counts.to_dict(
                orient="records"
            )
        ),
        "scene_family_count": int(
            batch[
                "scene_family"
            ].nunique()
        ),
        "maximum_per_scene": int(
            args.maximum_per_scene
        ),
        "contact_sheet_count": int(
            len(contact_sheets)
        ),
        "zip_path": relative(
            ZIP_PATH
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

    README_PATH.write_text(
        """v0.6 KIYI ANOTASYON PAKETİ

Bu ZIP'in içeriği:
- images/: doğru kara-su etiketi hazırlanacak SAR görüntüleri
- prior_reference_only/: yalnız yaklaşık referans maskeleri
- contact_sheets/: hızlı inceleme görselleri
- annotation_manifest.csv: görüntü listesi ve eşleme bilgisi
- summary.json: seçim özeti

ÖNEMLİ:
prior_reference_only altındaki maskeler doğru etiket değildir.
Yalnız yardımcı referans olarak tutulmuştur.

Sonraki işlem:
ZIP dosyasını ChatGPT konuşmasına yükleyin. Doğru kara ve
güvenli-su maskeleri hazırlanıp geri verilecektir.
""",
        encoding="utf-8",
    )

    build_zip()

    print("=" * 78)
    print(
        "v0.6 KIYI ANOTASYON PAKETİ HAZIR"
    )
    print("=" * 78)
    print(
        "Kaynak manifest:",
        source_manifest.resolve(),
    )
    print(
        "Mevcut doğrulanmamış kıyılı:",
        len(candidates),
    )
    print(
        "Seçilen:",
        len(batch),
    )
    print(
        "Sahne ailesi:",
        batch[
            "scene_family"
        ].nunique(),
    )
    print()
    print(
        "Alt grup dağılımı:"
    )
    print(
        subset_counts.to_string(
            index=False
        )
    )
    print()
    print(
        "ZIP:",
        ZIP_PATH.resolve(),
    )
    print(
        "Boyut:",
        f"{ZIP_PATH.stat().st_size / (1024 ** 2):.2f} MB",
    )
    print()
    print(
        "Bu ZIP'i sohbete yükleyin."
    )


if __name__ == "__main__":
    main()
