from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DARTIS_ROOT = (
    ROOT
    / "data"
    / "external"
    / "dartis"
)

RAW_ROOT = (
    DARTIS_ROOT
    / "raw"
)

TABLE_PATH = (
    DARTIS_ROOT
    / "DARTIS_2019.tab"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "v06_dartis_table_mapping"
)

MAPPING_PATH = (
    ROOT
    / "data"
    / "metadata"
    / "v06_dartis_table_mapping.csv"
)

UNMATCHED_PATH = (
    OUTPUT_DIR
    / "unmatched_local_images.csv"
)

SUBSET_COUNTS_PATH = (
    OUTPUT_DIR
    / "subset_counts.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "summary.json"
)

REPORT_PATH = (
    ROOT
    / "reports"
    / "v06_dartis_table_mapping.md"
)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
}

REQUIRED_COLUMNS = {
    "subset",
    "jpg_file",
    "tag",
    "patch_name",
    "patch_width",
    "patch_height",
    "patch_ul_lon",
    "patch_ul_lat",
    "patch_ur_lon",
    "patch_ur_lat",
    "patch_br_lon",
    "patch_br_lat",
    "patch_bl_lon",
    "patch_bl_lat",
}

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


def relative(path: Path) -> str:
    try:
        return str(
            path.resolve().relative_to(
                ROOT.resolve()
            )
        ).replace("\\", "/")

    except ValueError:
        return str(path.resolve())


def clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""

    value = str(value).strip()

    if value.lower() in {
        "",
        "nan",
        "none",
        "<na>",
    }:
        return ""

    return value


def read_text_with_fallback(
    path: Path,
) -> str:
    errors = []

    for encoding in (
        "utf-8-sig",
        "utf-8",
        "latin-1",
    ):
        try:
            return path.read_text(
                encoding=encoding
            )

        except UnicodeDecodeError as error:
            errors.append(
                f"{encoding}: {error}"
            )

    raise RuntimeError(
        "DARTIS tablosu okunamadı:\n"
        + "\n".join(errors)
    )


def parse_heading(
    heading: str,
) -> str:
    matches = re.findall(
        r"\((.*?)\)",
        heading,
    )

    if matches:
        column_name = matches[-1]
    else:
        column_name = heading

    if ";" in column_name:
        column_name = (
            column_name
            .split(";", 1)[0]
        )

    return (
        column_name
        .strip()
        .replace(" ", "_")
    )


def read_pangaea_table(
    path: Path,
) -> pd.DataFrame:
    text = read_text_with_fallback(
        path
    )

    rows = []
    headings = None

    inside_comment = False

    for raw_line in text.splitlines():
        stripped = raw_line.strip()

        if not stripped:
            continue

        if "/*" in stripped:
            inside_comment = True

        if inside_comment:
            if "*/" in stripped:
                inside_comment = False

            continue

        fields = raw_line.split("\t")

        if headings is None:
            headings = [
                parse_heading(field)
                for field in fields
            ]

            continue

        if len(fields) < len(headings):
            fields.extend(
                [""] * (
                    len(headings)
                    - len(fields)
                )
            )

        if len(fields) > len(headings):
            fields = fields[
                :len(headings)
            ]

        rows.append(
            fields
        )

    if headings is None:
        raise RuntimeError(
            "PANGAEA tablo başlığı bulunamadı."
        )

    dataframe = pd.DataFrame(
        rows,
        columns=headings,
    )

    return dataframe


def collect_local_images() -> dict[
    str,
    list[Path],
]:
    result = {}

    for subset in (
        "oc",
        "ow",
        "nc",
        "nw",
    ):
        directory = (
            RAW_ROOT
            / subset
        )

        if not directory.exists():
            raise FileNotFoundError(
                f"Grup klasörü bulunamadı: "
                f"{directory}"
            )

        result[subset] = sorted(
            path
            for path in directory.rglob("*")
            if path.is_file()
            and path.suffix.lower()
            in IMAGE_EXTENSIONS
        )

    return result


def build_indexes(
    image_paths: list[Path],
) -> tuple[
    dict[str, list[Path]],
    dict[str, list[Path]],
]:
    name_index: dict[
        str,
        list[Path],
    ] = {}

    stem_index: dict[
        str,
        list[Path],
    ] = {}

    for path in image_paths:
        name_index.setdefault(
            path.name.lower(),
            [],
        ).append(path)

        stem_index.setdefault(
            path.stem.lower(),
            [],
        ).append(path)

    return (
        name_index,
        stem_index,
    )


def candidate_keys(
    row: pd.Series,
) -> tuple[list[str], list[str]]:
    filename_keys = []
    stem_keys = []

    for column in (
        "jpg_file",
        "tag",
        "patch_name",
    ):
        value = clean_text(
            row.get(column, "")
        )

        if not value:
            continue

        name = Path(value).name

        filename_keys.append(
            name.lower()
        )

        if Path(name).suffix:
            stem_keys.append(
                Path(name).stem.lower()
            )

        else:
            stem_keys.append(
                name.lower()
            )

            for suffix in (
                ".jpg",
                ".jpeg",
                ".png",
                ".tif",
                ".tiff",
            ):
                filename_keys.append(
                    (
                        name
                        + suffix
                    ).lower()
                )

    return (
        list(dict.fromkeys(filename_keys)),
        list(dict.fromkeys(stem_keys)),
    )


def find_local_match(
    row: pd.Series,
    name_index: dict[
        str,
        list[Path],
    ],
    stem_index: dict[
        str,
        list[Path],
    ],
) -> tuple[
    str,
    str,
    str,
]:
    filename_keys, stem_keys = (
        candidate_keys(row)
    )

    matches = []

    for key in filename_keys:
        for path in name_index.get(
            key,
            [],
        ):
            matches.append(
                (
                    path,
                    f"filename:{key}",
                )
            )

    if not matches:
        for key in stem_keys:
            for path in stem_index.get(
                key,
                [],
            ):
                matches.append(
                    (
                        path,
                        f"stem:{key}",
                    )
                )

    unique_matches = {}

    for path, match_key in matches:
        unique_matches[
            str(path.resolve())
        ] = (
            path,
            match_key,
        )

    matches = list(
        unique_matches.values()
    )

    if len(matches) == 1:
        path, match_key = matches[0]

        return (
            "MATCHED",
            relative(path),
            match_key,
        )

    if len(matches) > 1:
        return (
            "AMBIGUOUS",
            "",
            " | ".join(
                relative(path)
                for path, _
                in matches
            ),
        )

    return (
        "UNMATCHED",
        "",
        "",
    )


def main() -> None:
    for required_path in (
        TABLE_PATH,
        RAW_ROOT,
    ):
        if not required_path.exists():
            raise FileNotFoundError(
                f"Gerekli dosya/klasör yok: "
                f"{required_path}"
            )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MAPPING_PATH.parent.mkdir(
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
        "DARTIS TABLO VE YEREL GÖRÜNTÜ EŞLEŞTİRMESİ"
    )
    print("=" * 78)

    dataframe = read_pangaea_table(
        TABLE_PATH
    )

    missing_columns = (
        REQUIRED_COLUMNS
        - set(dataframe.columns)
    )

    if missing_columns:
        raise RuntimeError(
            "DARTIS tablosunda eksik sütunlar:\n"
            + "\n".join(
                sorted(missing_columns)
            )
            + "\n\nBulunan sütunlar:\n"
            + "\n".join(
                str(column)
                for column
                in dataframe.columns
            )
        )

    dataframe["subset"] = (
        dataframe["subset"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    dataframe = dataframe[
        dataframe["subset"].isin(
            {
                "oc",
                "ow",
                "nc",
                "nw",
            }
        )
    ].copy()

    for column in (
        COORDINATE_COLUMNS
        + [
            "patch_width",
            "patch_height",
        ]
    ):
        dataframe[column] = (
            pd.to_numeric(
                dataframe[column],
                errors="coerce",
            )
        )

    # Aynı görüntü birden fazla nesne satırında
    # bulunuyorsa yalnız tek görüntü kaydı tut.
    unique_images = (
        dataframe
        .sort_values(
            [
                "subset",
                "jpg_file",
            ]
        )
        .drop_duplicates(
            subset=[
                "subset",
                "jpg_file",
            ],
            keep="first",
        )
        .reset_index(drop=True)
    )

    local_images = collect_local_images()

    indexes = {
        subset: build_indexes(paths)
        for subset, paths
        in local_images.items()
    }

    mapping_rows = []

    for row in unique_images.itertuples(
        index=False
    ):
        row_series = pd.Series(
            row._asdict()
        )

        subset = clean_text(
            row_series["subset"]
        ).lower()

        (
            name_index,
            stem_index,
        ) = indexes[subset]

        (
            mapping_status,
            local_image_path,
            mapping_key,
        ) = find_local_match(
            row_series,
            name_index,
            stem_index,
        )

        coordinate_complete = bool(
            row_series[
                COORDINATE_COLUMNS
            ].notna().all()
        )

        mapping_rows.append(
            {
                "subset": subset,
                "jpg_file": clean_text(
                    row_series["jpg_file"]
                ),
                "tag": clean_text(
                    row_series["tag"]
                ),
                "patch_name": clean_text(
                    row_series["patch_name"]
                ),
                "patch_width": (
                    row_series[
                        "patch_width"
                    ]
                ),
                "patch_height": (
                    row_series[
                        "patch_height"
                    ]
                ),
                "patch_ul_lon": (
                    row_series[
                        "patch_ul_lon"
                    ]
                ),
                "patch_ul_lat": (
                    row_series[
                        "patch_ul_lat"
                    ]
                ),
                "patch_ur_lon": (
                    row_series[
                        "patch_ur_lon"
                    ]
                ),
                "patch_ur_lat": (
                    row_series[
                        "patch_ur_lat"
                    ]
                ),
                "patch_br_lon": (
                    row_series[
                        "patch_br_lon"
                    ]
                ),
                "patch_br_lat": (
                    row_series[
                        "patch_br_lat"
                    ]
                ),
                "patch_bl_lon": (
                    row_series[
                        "patch_bl_lon"
                    ]
                ),
                "patch_bl_lat": (
                    row_series[
                        "patch_bl_lat"
                    ]
                ),
                "coordinate_complete": (
                    coordinate_complete
                ),
                "surface_context": (
                    "coast"
                    if subset.endswith("c")
                    else "water"
                ),
                "oil_label": (
                    1
                    if subset.startswith("o")
                    else 0
                ),
                "mapping_status": (
                    mapping_status
                ),
                "mapping_key": (
                    mapping_key
                ),
                "local_image_path": (
                    local_image_path
                ),
            }
        )

    mapping = pd.DataFrame(
        mapping_rows
    )

    mapping.to_csv(
        MAPPING_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    matched_paths = set(
        mapping.loc[
            mapping[
                "mapping_status"
            ].eq("MATCHED"),
            "local_image_path",
        ].astype(str)
    )

    unmatched_local_rows = []

    for subset, paths in local_images.items():
        for path in paths:
            path_text = relative(path)

            if path_text not in matched_paths:
                unmatched_local_rows.append(
                    {
                        "subset": subset,
                        "local_image_path": (
                            path_text
                        ),
                    }
                )

    unmatched_local = pd.DataFrame(
        unmatched_local_rows,
        columns=[
            "subset",
            "local_image_path",
        ],
    )

    unmatched_local.to_csv(
        UNMATCHED_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    subset_counts = (
        mapping
        .groupby(
            [
                "subset",
                "surface_context",
                "oil_label",
                "mapping_status",
                "coordinate_complete",
            ],
            dropna=False,
        )
        .size()
        .reset_index(
            name="image_count"
        )
        .sort_values(
            [
                "subset",
                "mapping_status",
            ]
        )
    )

    subset_counts.to_csv(
        SUBSET_COUNTS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    matched_count = int(
        mapping[
            "mapping_status"
        ].eq(
            "MATCHED"
        ).sum()
    )

    ambiguous_count = int(
        mapping[
            "mapping_status"
        ].eq(
            "AMBIGUOUS"
        ).sum()
    )

    unmatched_table_count = int(
        mapping[
            "mapping_status"
        ].eq(
            "UNMATCHED"
        ).sum()
    )

    coastal = mapping[
        mapping[
            "surface_context"
        ].eq(
            "coast"
        )
    ]

    coastal_coordinate_complete = int(
        coastal[
            "coordinate_complete"
        ].sum()
    )

    local_total = sum(
        len(paths)
        for paths
        in local_images.values()
    )

    ready_for_landmask = bool(
        matched_count == len(mapping)
        and ambiguous_count == 0
        and unmatched_table_count == 0
        and len(unmatched_local) == 0
        and coastal_coordinate_complete
        == len(coastal)
    )

    summary = {
        "stage": (
            "v06_dartis_table_mapping"
        ),
        "raw_table_rows": int(
            len(dataframe)
        ),
        "unique_table_images": int(
            len(mapping)
        ),
        "local_image_count": int(
            local_total
        ),
        "matched_table_images": (
            matched_count
        ),
        "ambiguous_table_images": (
            ambiguous_count
        ),
        "unmatched_table_images": (
            unmatched_table_count
        ),
        "unmatched_local_images": int(
            len(unmatched_local)
        ),
        "coastal_images": int(
            len(coastal)
        ),
        "coastal_coordinate_complete": (
            coastal_coordinate_complete
        ),
        "ready_for_landmask_generation": (
            ready_for_landmask
        ),
        "locked_test_used": False,
        "training_performed": False,
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

    report = f"""# v0.6 DARTIS Tablo Eşleştirme Raporu

## Sonuç

- Ham tablo satırı: {len(dataframe)}
- Benzersiz tablo görüntüsü: {len(mapping)}
- Yerel görüntü: {local_total}
- Eşleşen tablo görüntüsü: {matched_count}
- Belirsiz eşleşme: {ambiguous_count}
- Eşleşmeyen tablo görüntüsü: {unmatched_table_count}
- Eşleşmeyen yerel görüntü: {len(unmatched_local)}
- Kıyılı görüntü: {len(coastal)}
- Koordinatı tam kıyılı görüntü: {coastal_coordinate_complete}

## Kara maskesine hazır

`{ready_for_landmask}`

Bu aşamada model eğitilmemiş ve kilitli test kullanılmamıştır.
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )

    print()
    print(
        "Ham tablo satırı:",
        len(dataframe),
    )

    print(
        "Benzersiz tablo görüntüsü:",
        len(mapping),
    )

    print(
        "Yerel görüntü:",
        local_total,
    )

    print(
        "Eşleşen:",
        matched_count,
    )

    print(
        "Belirsiz:",
        ambiguous_count,
    )

    print(
        "Eşleşmeyen tablo görüntüsü:",
        unmatched_table_count,
    )

    print(
        "Eşleşmeyen yerel görüntü:",
        len(unmatched_local),
    )

    print(
        "Kıyılı görüntü:",
        len(coastal),
    )

    print(
        "Koordinatı tam kıyılı görüntü:",
        coastal_coordinate_complete,
    )

    print()
    print(
        "Kara maskesi üretimine hazır:",
        ready_for_landmask,
    )

    print()
    print(
        "Alt-grup sonuçları:"
    )

    print(
        subset_counts.to_string(
            index=False
        )
    )

    print()
    print(
        "Mapping:",
        MAPPING_PATH.resolve(),
    )

    print(
        "Eşleşmeyen yereller:",
        UNMATCHED_PATH.resolve(),
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
        "Kilitli test kullanımı: YOK"
    )


if __name__ == "__main__":
    main()
