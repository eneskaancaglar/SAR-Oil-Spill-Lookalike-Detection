from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from pangaeapy import PanDataSet


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CACHE_DIR = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "dartis"
    / "cache"
)

METADATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "dartis"
    / "metadata"
)

OUTPUT_CSV = METADATA_DIR / "dartis_catalog_raw.csv"

DATASET_ID = 980773


def get_dataframe(dataset: PanDataSet) -> pd.DataFrame:
    """
    PanDataSet içinden pandas DataFrame nesnesini güvenli biçimde alır.
    """
    data: Any = getattr(dataset, "data", None)

    if isinstance(data, pd.DataFrame):
        return data

    if isinstance(data, list):
        dataframes = [
            item
            for item in data
            if isinstance(item, pd.DataFrame)
        ]

        if len(dataframes) == 1:
            return dataframes[0]

        if len(dataframes) > 1:
            return pd.concat(
                dataframes,
                ignore_index=True,
            )

    raise RuntimeError(
        "PANGAEA verisi pandas DataFrame olarak okunamadi. "
        f"Alinan veri tipi: {type(data)}"
    )


def print_relevant_columns(dataframe: pd.DataFrame) -> None:
    """
    Görüntü, alt küme ve binary dosya bilgisi taşıyabilecek
    sütunların örnek değerlerini gösterir.
    """
    keywords = (
        "image",
        "binary",
        "identification",
        "tag",
        "patch",
        "set",
    )

    print()
    print("=" * 75)
    print("ILGILI SUTUNLAR VE ORNEK DEGERLER")
    print("=" * 75)

    for column in dataframe.columns:
        column_text = str(column)
        lower_name = column_text.lower()

        if not any(
            keyword in lower_name
            for keyword in keywords
        ):
            continue

        print()
        print(f"Sutun: {column_text!r}")

        sample_values = (
            dataframe[column]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .head(10)
            .tolist()
        )

        for value in sample_values:
            print(f"  - {value}")


def main() -> None:
    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    METADATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 75)
    print("DARTIS PANGAEA KATALOG INCELEMESI")
    print("=" * 75)
    print("Dataset ID:", DATASET_ID)
    print("Cache:", CACHE_DIR.resolve())

    dataset = PanDataSet(
        id=DATASET_ID,
        enable_cache=True,
        cachedir=str(CACHE_DIR),
        include_data=True,
    )

    dataframe = get_dataframe(dataset)

    # Sütun adlarını yazılabilir metne dönüştürüyoruz.
    dataframe.columns = [
        str(column)
        for column in dataframe.columns
    ]

    dataframe.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("Satir sayisi:", len(dataframe))
    print("Sutun sayisi:", len(dataframe.columns))

    print()
    print("=" * 75)
    print("BUTUN SUTUNLAR")
    print("=" * 75)

    for index, column in enumerate(
        dataframe.columns,
        start=1,
    ):
        print(f"{index:02d}. {column!r}")

    print()
    print("=" * 75)
    print("ILK UC SATIR")
    print("=" * 75)

    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        240,
        "display.max_colwidth",
        100,
    ):
        print(dataframe.head(3).to_string())

    print_relevant_columns(dataframe)

    print()
    print("=" * 75)
    print("KATALOG KAYDEDILDI")
    print("=" * 75)
    print("Dosya:", OUTPUT_CSV.resolve())


if __name__ == "__main__":
    main()
