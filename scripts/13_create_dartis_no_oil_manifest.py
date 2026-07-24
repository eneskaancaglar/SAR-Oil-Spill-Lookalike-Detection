from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CATALOG_PATH = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "dartis"
    / "metadata"
    / "dartis_catalog_raw.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "dartis"
    / "metadata"
    / "dartis_no_oil_manifest.csv"
)

NO_OIL_GROUPS = {
    "nc": "no_oil_coast",
    "nw": "no_oil_water",
}

EXPECTED_NO_OIL_IMAGE_COUNT = 2290


def main() -> None:
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(
            f"Katalog bulunamadi: {CATALOG_PATH}"
        )

    dataframe = pd.read_csv(
        CATALOG_PATH,
        encoding="utf-8-sig",
    )

    required_columns = {
        "Image set",
        "IMAGE",
        "Width",
        "Height",
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        raise RuntimeError(
            "Katalogda gerekli sutunlar bulunamadi: "
            f"{sorted(missing_columns)}"
        )

    # PANGAEA indirmesinde kullanılacak gerçek satır numarasını korur.
    dataframe = (
        dataframe
        .reset_index()
        .rename(columns={"index": "catalog_index"})
    )

    dataframe["Image set"] = (
        dataframe["Image set"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    no_oil_rows = dataframe[
        dataframe["Image set"].isin(NO_OIL_GROUPS)
    ].copy()

    no_oil_rows = no_oil_rows[
        no_oil_rows["IMAGE"].notna()
    ].copy()

    no_oil_rows["IMAGE"] = (
        no_oil_rows["IMAGE"]
        .astype(str)
        .str.strip()
    )

    raw_row_count = len(no_oil_rows)
    unique_image_count = no_oil_rows["IMAGE"].nunique()
    repeated_row_count = raw_row_count - unique_image_count

    # Aynı görüntünün farklı satırlarda farklı boyutlarla
    # tanımlanıp tanımlanmadığını kontrol eder.
    dimension_variants = (
        no_oil_rows
        .groupby("IMAGE")[["Width", "Height"]]
        .nunique()
    )

    inconsistent_dimensions = dimension_variants[
        (dimension_variants["Width"] > 1)
        | (dimension_variants["Height"] > 1)
    ]

    if not inconsistent_dimensions.empty:
        raise RuntimeError(
            "Ayni görüntü için farklı boyut bilgileri bulundu: "
            f"{inconsistent_dimensions.head(10).index.tolist()}"
        )

    # Aynı IMAGE birden fazla kez bulunursa ilk katalog satırını tutar.
    manifest = (
        no_oil_rows
        .sort_values("catalog_index")
        .drop_duplicates(subset=["IMAGE"], keep="first")
        .copy()
    )

    manifest["source_dataset"] = "DARTIS_2019"
    manifest["image_set"] = manifest["Image set"]
    manifest["category"] = (
        manifest["image_set"].map(NO_OIL_GROUPS)
    )
    manifest["image_name"] = manifest["IMAGE"]
    manifest["expected_has_oil"] = 0

    manifest["Width"] = (
        pd.to_numeric(
            manifest["Width"],
            errors="raise",
        )
        .astype(int)
    )

    manifest["Height"] = (
        pd.to_numeric(
            manifest["Height"],
            errors="raise",
        )
        .astype(int)
    )

    manifest = manifest[
        [
            "source_dataset",
            "catalog_index",
            "image_set",
            "category",
            "image_name",
            "Width",
            "Height",
            "expected_has_oil",
        ]
    ].rename(
        columns={
            "Width": "width",
            "Height": "height",
        }
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    group_counts = (
        manifest["image_set"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    size_counts = (
        manifest
        .groupby(["width", "height"])
        .size()
        .sort_values(ascending=False)
        .to_dict()
    )

    print("=" * 70)
    print("DARTIS NO-OIL MANIFEST OZETI")
    print("=" * 70)

    print(f"Filtrelenmis katalog satiri: {raw_row_count}")
    print(f"Tekrarlanan satir:           {repeated_row_count}")
    print(f"Benzersiz no-oil goruntu:    {len(manifest)}")

    print()
    print("Image set dagilimi:", group_counts)
    print("Goruntu boyutlari: ", size_counts)

    if len(manifest) != EXPECTED_NO_OIL_IMAGE_COUNT:
        print()
        print(
            "UYARI: Beklenen no-oil goruntu sayisi "
            f"{EXPECTED_NO_OIL_IMAGE_COUNT}, "
            f"bulunan {len(manifest)}."
        )
    else:
        print()
        print(
            "BASARILI: Beklenen 2290 benzersiz "
            "no-oil goruntu bulundu."
        )

    print()
    print("Manifest:", OUTPUT_PATH.resolve())


if __name__ == "__main__":
    main()
