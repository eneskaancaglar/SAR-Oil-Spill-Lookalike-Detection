from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import pandas as pd
from PIL import Image
from pangaeapy import PanDataSet


ROOT = Path(__file__).resolve().parents[1]

DATASET_ID = 980773

MANIFEST_PATH = (
    ROOT
    / "data"
    / "metadata"
    / "dartis_positive_oil_manifest.csv"
)

CACHE_DIR = (
    ROOT
    / "data"
    / "external"
    / "dartis"
    / "cache"
    / "positive"
)

RAW_ROOT = (
    ROOT
    / "data"
    / "external"
    / "dartis"
    / "raw"
)

ANNOTATION_ROOT = (
    ROOT
    / "data"
    / "external"
    / "dartis"
    / "annotations"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "v03_dartis_positive_download"
)

STATUS_CSV = (
    OUTPUT_DIR
    / "positive_download_status.csv"
)

SUMMARY_JSON = (
    OUTPUT_DIR
    / "positive_download_summary.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "DARTIS ow/oc görüntülerini ve XML anotasyonlarını "
            "PANGAEA üzerinden indirir."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "İşlenecek en fazla görüntü sayısı. "
            "0 bütün manifesti işler."
        ),
    )

    parser.add_argument(
        "--retries",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
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


def extract_filename(value: Any) -> str:
    text = clean_text(value)

    if not text:
        return ""

    parsed = urlparse(text)
    path_text = parsed.path or text

    return Path(path_text).name


def find_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> str:
    column_map = {
        str(column).strip().lower(): str(column)
        for column in dataframe.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower()

        if key in column_map:
            return column_map[key]

    raise RuntimeError(
        f"Gerekli sütun bulunamadı: {candidates}. "
        f"Mevcut sütunlar: {list(dataframe.columns)}"
    )


def create_dataset(
    token: str,
) -> PanDataSet:
    common_arguments = {
        "enable_cache": True,
        "auth_token": token,
    }

    errors = []

    for cache_argument in (
        "cache_dir",
        "cachedir",
    ):
        try:
            return PanDataSet(
                DATASET_ID,
                **common_arguments,
                **{
                    cache_argument:
                    str(CACHE_DIR)
                },
            )
        except TypeError as error:
            errors.append(
                f"{cache_argument}: {error}"
            )

    raise RuntimeError(
        "PanDataSet başlatılamadı. "
        + " | ".join(errors)
    )


def flatten_download_result(
    value: Any,
) -> Iterable[Path]:
    if value is None:
        return

    if isinstance(
        value,
        (str, Path),
    ):
        yield Path(value)
        return

    if isinstance(
        value,
        dict,
    ):
        for nested_value in value.values():
            yield from flatten_download_result(
                nested_value
            )
        return

    if isinstance(
        value,
        Iterable,
    ):
        for nested_value in value:
            yield from flatten_download_result(
                nested_value
            )


def find_downloaded_file(
    download_result: Any,
    expected_filename: str,
) -> Path | None:
    expected_filename_lower = (
        expected_filename.lower()
    )

    for candidate in flatten_download_result(
        download_result
    ):
        possible_paths = [
            candidate,
            CACHE_DIR / candidate,
        ]

        for possible_path in possible_paths:
            if (
                possible_path.exists()
                and possible_path.is_file()
                and possible_path.name.lower()
                == expected_filename_lower
            ):
                return possible_path.resolve()

    matching_files = [
        path
        for path in CACHE_DIR.rglob(
            expected_filename
        )
        if path.is_file()
    ]

    if not matching_files:
        matching_files = [
            path
            for path in CACHE_DIR.rglob("*")
            if (
                path.is_file()
                and path.name.lower()
                == expected_filename_lower
            )
        ]

    if not matching_files:
        return None

    matching_files.sort(
        key=lambda path:
        path.stat().st_mtime,
        reverse=True,
    )

    return matching_files[0].resolve()


def validate_image(
    path: Path,
) -> tuple[bool, str]:
    try:
        with Image.open(path) as image:
            image.verify()

        with Image.open(path) as image:
            width, height = image.size

        if width <= 0 or height <= 0:
            return False, "Geçersiz görüntü boyutu."

        return True, f"{width}x{height}"

    except Exception as error:
        return False, str(error)


def validate_xml(
    path: Path,
) -> tuple[bool, str]:
    try:
        root = ET.parse(path).getroot()

        object_count = len(
            root.findall(".//object")
        )

        return (
            True,
            f"object_count={object_count}",
        )

    except Exception as error:
        return False, str(error)


def copy_downloaded_file(
    source: Path,
    destination: Path,
) -> None:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = destination.with_suffix(
        destination.suffix + ".part"
    )

    shutil.copy2(
        source,
        temporary_path,
    )

    temporary_path.replace(
        destination
    )


def download_column(
    dataset: PanDataSet,
    row_index: int,
    column_name: str,
    expected_filename: str,
    destination: Path,
    retries: int,
    sleep_seconds: float,
    overwrite: bool,
    validator,
) -> tuple[
    bool,
    str,
]:
    if (
        destination.exists()
        and not overwrite
    ):
        valid, validation_message = (
            validator(destination)
        )

        if valid:
            return (
                True,
                "already_exists:"
                + validation_message,
            )

        destination.unlink(
            missing_ok=True
        )

    last_error = ""

    for attempt in range(
        1,
        retries + 1,
    ):
        try:
            result = dataset.download(
                indices=[row_index],
                columns=[column_name],
            )

            downloaded_path = (
                find_downloaded_file(
                    result,
                    expected_filename,
                )
            )

            if downloaded_path is None:
                raise FileNotFoundError(
                    "İndirilen dosya cache içinde "
                    f"bulunamadı: {expected_filename}"
                )

            copy_downloaded_file(
                downloaded_path,
                destination,
            )

            valid, validation_message = (
                validator(destination)
            )

            if not valid:
                destination.unlink(
                    missing_ok=True
                )

                raise RuntimeError(
                    "Dosya doğrulaması başarısız: "
                    + validation_message
                )

            return (
                True,
                validation_message,
            )

        except Exception as error:
            last_error = (
                f"Deneme {attempt}/{retries}: "
                f"{type(error).__name__}: {error}"
            )

            if attempt < retries:
                time.sleep(
                    sleep_seconds * attempt
                )

    return False, last_error


def build_row_mapping(
    dataset: PanDataSet,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    catalog = pd.DataFrame(
        dataset.data
    ).copy()

    group_column = find_column(
        catalog,
        [
            "Image set",
            "image_set",
        ],
    )

    image_column = find_column(
        catalog,
        [
            "IMAGE",
            "image",
            "image_name",
        ],
    )

    binary_column = find_column(
        catalog,
        [
            "Binary",
            "binary",
        ],
    )

    catalog[
        "pangaea_row_index"
    ] = range(len(catalog))

    catalog[
        "mapping_image_set"
    ] = (
        catalog[group_column]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    catalog[
        "mapping_image_name"
    ] = catalog[
        image_column
    ].map(extract_filename)

    catalog[
        "mapping_annotation_name"
    ] = catalog[
        binary_column
    ].map(extract_filename)

    catalog = catalog[
        catalog[
            "mapping_image_set"
        ].isin(
            ["ow", "oc"]
        )
    ].copy()

    mapping = (
        catalog[
            [
                "mapping_image_set",
                "mapping_image_name",
                "mapping_annotation_name",
                "pangaea_row_index",
            ]
        ]
        .drop_duplicates(
            subset=[
                "mapping_image_set",
                "mapping_image_name",
            ],
            keep="first",
        )
    )

    manifest = manifest.copy()

    manifest[
        "mapping_image_set"
    ] = (
        manifest["image_set"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    manifest[
        "mapping_image_name"
    ] = manifest[
        "image_name"
    ].map(extract_filename)

    merged = manifest.merge(
        mapping,
        on=[
            "mapping_image_set",
            "mapping_image_name",
        ],
        how="left",
        validate="one_to_one",
    )

    missing_mapping = merged[
        "pangaea_row_index"
    ].isna()

    if missing_mapping.any():
        examples = merged.loc[
            missing_mapping,
            [
                "image_set",
                "image_name",
            ],
        ].head(10)

        raise RuntimeError(
            f"{int(missing_mapping.sum())} görüntü "
            "PANGAEA tablosuyla eşleşmedi.\n"
            + examples.to_string(index=False)
        )

    annotation_mismatch = (
        merged[
            "mapping_annotation_name"
        ].astype(str)
        != merged[
            "annotation_name"
        ].astype(str)
    )

    if annotation_mismatch.any():
        examples = merged.loc[
            annotation_mismatch,
            [
                "image_name",
                "annotation_name",
                "mapping_annotation_name",
            ],
        ].head(10)

        raise RuntimeError(
            "Manifest XML adları ile PANGAEA XML "
            "adları arasında uyuşmazlık var.\n"
            + examples.to_string(index=False)
        )

    merged[
        "pangaea_row_index"
    ] = merged[
        "pangaea_row_index"
    ].astype(int)

    merged.attrs[
        "image_column"
    ] = image_column

    merged.attrs[
        "binary_column"
    ] = binary_column

    return merged


def summarize_status(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    summary = {
        "processed_rows": int(
            len(dataframe)
        ),
        "image_success": int(
            dataframe[
                "image_success"
            ].sum()
        ),
        "annotation_success": int(
            dataframe[
                "annotation_success"
            ].sum()
        ),
        "pair_success": int(
            dataframe[
                "pair_success"
            ].sum()
        ),
        "pair_failed": int(
            (
                ~dataframe[
                    "pair_success"
                ]
            ).sum()
        ),
        "groups": {},
    }

    for group_name in (
        "ow",
        "oc",
    ):
        subset = dataframe[
            dataframe["image_set"]
            == group_name
        ]

        summary["groups"][
            group_name
        ] = {
            "processed": int(
                len(subset)
            ),
            "image_success": int(
                subset[
                    "image_success"
                ].sum()
            ),
            "annotation_success": int(
                subset[
                    "annotation_success"
                ].sum()
            ),
            "pair_success": int(
                subset[
                    "pair_success"
                ].sum()
            ),
        }

    return summary


def main() -> None:
    args = parse_args()

    token = os.environ.get(
        "PANGAEA_TOKEN",
        "",
    ).strip()

    if not token:
        raise RuntimeError(
            "PANGAEA_TOKEN ortam değişkeni bulunamadı. "
            "Tokenı PowerShell ortam değişkenine yükle."
        )

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Pozitif manifest bulunamadı: "
            f"{MANIFEST_PATH}"
        )

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for group_name in (
        "ow",
        "oc",
    ):
        (
            RAW_ROOT / group_name
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            ANNOTATION_ROOT / group_name
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

    manifest = pd.read_csv(
        MANIFEST_PATH,
        encoding="utf-8-sig",
    )

    print("=" * 78)
    print(
        "DARTIS POZİTİF GÖRÜNTÜ VE XML İNDİRİCİ"
    )
    print("=" * 78)
    print("PANGAEA dataset:", DATASET_ID)
    print("Manifest görüntüsü:", len(manifest))
    print("Cache:", CACHE_DIR)
    print()

    dataset = create_dataset(
        token
    )

    mapped_manifest = build_row_mapping(
        dataset,
        manifest,
    )

    image_column = mapped_manifest.attrs[
        "image_column"
    ]

    binary_column = mapped_manifest.attrs[
        "binary_column"
    ]

    print(
        "PANGAEA IMAGE sütunu:",
        image_column,
    )

    print(
        "PANGAEA Binary sütunu:",
        binary_column,
    )

    print(
        "Eşleşen görüntü:",
        len(mapped_manifest),
    )

    if args.limit > 0:
        mapped_manifest = (
            mapped_manifest
            .head(args.limit)
            .copy()
        )

    print(
        "Bu çalışmada işlenecek:",
        len(mapped_manifest),
    )

    status_rows = []

    for position, row in enumerate(
        mapped_manifest.itertuples(
            index=False
        ),
        start=1,
    ):
        group_name = str(
            row.image_set
        )

        image_name = str(
            row.image_name
        )

        annotation_name = str(
            row.annotation_name
        )

        row_index = int(
            row.pangaea_row_index
        )

        image_destination = (
            RAW_ROOT
            / group_name
            / image_name
        )

        annotation_destination = (
            ANNOTATION_ROOT
            / group_name
            / annotation_name
        )

        image_success, image_message = (
            download_column(
                dataset=dataset,
                row_index=row_index,
                column_name=image_column,
                expected_filename=image_name,
                destination=image_destination,
                retries=args.retries,
                sleep_seconds=args.sleep,
                overwrite=args.overwrite,
                validator=validate_image,
            )
        )

        annotation_success = False
        annotation_message = (
            "image_failed"
        )

        if image_success:
            (
                annotation_success,
                annotation_message,
            ) = download_column(
                dataset=dataset,
                row_index=row_index,
                column_name=binary_column,
                expected_filename=(
                    annotation_name
                ),
                destination=(
                    annotation_destination
                ),
                retries=args.retries,
                sleep_seconds=args.sleep,
                overwrite=args.overwrite,
                validator=validate_xml,
            )

        pair_success = (
            image_success
            and annotation_success
        )

        status_rows.append(
            {
                "image_set": group_name,
                "image_name": image_name,
                "annotation_name": (
                    annotation_name
                ),
                "scene_id": str(
                    row.scene_id
                ),
                "pangaea_row_index": (
                    row_index
                ),
                "image_success": (
                    image_success
                ),
                "annotation_success": (
                    annotation_success
                ),
                "pair_success": (
                    pair_success
                ),
                "image_message": (
                    image_message
                ),
                "annotation_message": (
                    annotation_message
                ),
                "local_image_path": str(
                    image_destination
                ),
                "local_annotation_path": str(
                    annotation_destination
                ),
            }
        )

        status_dataframe = pd.DataFrame(
            status_rows
        )

        status_dataframe.to_csv(
            STATUS_CSV,
            index=False,
            encoding="utf-8-sig",
        )

        result_text = (
            "OK"
            if pair_success
            else "HATA"
        )

        print(
            f"[{position:04d}/"
            f"{len(mapped_manifest):04d}] "
            f"{result_text} "
            f"{group_name}/{image_name}"
        )

        time.sleep(
            args.sleep
        )

    status_dataframe = pd.DataFrame(
        status_rows
    )

    summary = summarize_status(
        status_dataframe
    )

    with SUMMARY_JSON.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 78)
    print("İNDİRME SONUCU")
    print("=" * 78)

    print(
        "İşlenen:",
        summary["processed_rows"],
    )

    print(
        "Başarılı görüntü:",
        summary["image_success"],
    )

    print(
        "Başarılı XML:",
        summary["annotation_success"],
    )

    print(
        "Başarılı çift:",
        summary["pair_success"],
    )

    print(
        "Başarısız çift:",
        summary["pair_failed"],
    )

    for group_name in (
        "ow",
        "oc",
    ):
        group = summary[
            "groups"
        ][group_name]

        print(
            f"{group_name}: "
            f"{group['pair_success']}/"
            f"{group['processed']}"
        )

    print()
    print("Status:", STATUS_CSV.resolve())
    print("Özet:  ", SUMMARY_JSON.resolve())


if __name__ == "__main__":
    main()
