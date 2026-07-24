from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Iterator

import pandas as pd
from PIL import Image
from pangaeapy import PanDataSet


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "dartis"
    / "metadata"
    / "dartis_no_oil_manifest.csv"
)

DEFAULT_RAW_DIR = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "dartis"
    / "raw"
)

DEFAULT_CACHE_DIR = (
    PROJECT_ROOT
    / "data"
    / "external"
    / "dartis"
    / "cache"
)

DATASET_ID = 980773


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DARTIS nc ve nw görüntülerini indirir."
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
    )

    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
    )

    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help=(
            "0 tum bekleyen dosyalar; "
            "3 ilk test icin kullanilabilir."
        ),
    )

    return parser.parse_args()


def create_chunks(
    records: list[dict[str, object]],
    chunk_size: int,
) -> Iterator[list[dict[str, object]]]:
    for start in range(0, len(records), chunk_size):
        yield records[start : start + chunk_size]


def validate_image(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)

    if path.stat().st_size == 0:
        raise RuntimeError(f"Bos dosya indirildi: {path}")

    with Image.open(path) as image:
        image.verify()


def find_downloaded_file(
    expected_name: str,
    returned_paths: list[Path],
    cache_dir: Path,
) -> Path | None:
    for path in returned_paths:
        if path.name == expected_name and path.exists():
            return path

    matches = list(cache_dir.rglob(expected_name))

    if matches:
        return matches[0]

    return None


def main() -> None:
    args = parse_args()

    if args.chunk_size <= 0:
        raise ValueError("chunk-size sifirdan buyuk olmali.")

    if not args.manifest.exists():
        raise FileNotFoundError(
            f"Manifest bulunamadi: {args.manifest}"
        )

    dataframe = pd.read_csv(
        args.manifest,
        encoding="utf-8-sig",
    )

    required_columns = {
        "catalog_index",
        "image_set",
        "image_name",
    }

    missing = required_columns - set(dataframe.columns)

    if missing:
        raise RuntimeError(
            f"Manifest sutunlari eksik: {sorted(missing)}"
        )

    args.raw_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = dataframe.to_dict("records")

    pending_records: list[dict[str, object]] = []

    existing_count = 0

    for record in records:
        image_set = str(record["image_set"])
        image_name = str(record["image_name"])

        destination = (
            args.raw_dir
            / image_set
            / image_name
        )

        if destination.exists():
            try:
                validate_image(destination)
                existing_count += 1
                continue
            except Exception:
                destination.unlink(missing_ok=True)

        pending_records.append(record)

    if args.max_files > 0:
        pending_records = pending_records[: args.max_files]

    token = os.getenv("PANGAEA_TOKEN", "").strip()

    # Resmî dokümana göre birkaç dosya token olmadan
    # test edilebilir. Tam indirme için token istiyoruz.
    if not token and (
        args.max_files == 0
        or args.max_files > 3
    ):
        raise RuntimeError(
            "Tam indirme icin PANGAEA_TOKEN bulunamadi. "
            "Once PANGAEA profilinden bearer token alip "
            "PowerShell ortam degiskenine kaydet."
        )

    dataset_kwargs = {
        "id": DATASET_ID,
        "enable_cache": True,
        "cachedir": str(args.cache_dir),
        "include_data": True,
    }

    if token:
        dataset_kwargs["auth_token"] = token

    dataset = PanDataSet(**dataset_kwargs)

    print("=" * 75)
    print("DARTIS NO-OIL INDIRME")
    print("=" * 75)
    print("Manifest goruntu sayisi:", len(records))
    print("Onceden mevcut:         ", existing_count)
    print("Bu calismada indirilecek:", len(pending_records))
    print("Token mevcut:           ", bool(token))

    downloaded_count = 0

    for chunk_number, chunk in enumerate(
        create_chunks(
            pending_records,
            args.chunk_size,
        ),
        start=1,
    ):
        indices = [
            int(record["catalog_index"])
            for record in chunk
        ]

        print()
        print(
            f"Chunk {chunk_number}: "
            f"{len(indices)} dosya isteniyor..."
        )

        downloaded_paths_raw = dataset.download(
            indices=indices,
            columns=["IMAGE"],
        )

        returned_paths = [
            Path(path)
            for path in downloaded_paths_raw
        ]

        for record in chunk:
            image_set = str(record["image_set"])
            image_name = str(record["image_name"])

            source_path = find_downloaded_file(
                expected_name=image_name,
                returned_paths=returned_paths,
                cache_dir=args.cache_dir,
            )

            if source_path is None:
                raise RuntimeError(
                    "Indirilen dosya bulunamadi: "
                    f"{image_name}"
                )

            destination_dir = (
                args.raw_dir
                / image_set
            )

            destination_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            destination = (
                destination_dir
                / image_name
            )

            if source_path.resolve() != destination.resolve():
                shutil.copy2(
                    source_path,
                    destination,
                )

            validate_image(destination)
            downloaded_count += 1

        print(
            f"Tamamlanan yeni dosya: {downloaded_count}/"
            f"{len(pending_records)}"
        )

    total_jpg_count = len(
        list(args.raw_dir.rglob("*.jpg"))
    )

    print()
    print("=" * 75)
    print("INDIRME TAMAMLANDI")
    print("=" * 75)
    print("Bu calismada indirilen:", downloaded_count)
    print("Raw klasorundeki JPG:  ", total_jpg_count)
    print("Raw klasoru:           ", args.raw_dir.resolve())


if __name__ == "__main__":
    main()
