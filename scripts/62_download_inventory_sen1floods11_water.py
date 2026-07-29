from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

from google.cloud import storage


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
    / "v06_sen1floods11_inventory"
)

MANIFEST_PATH = (
    ROOT
    / "data"
    / "metadata"
    / "v06_sen1floods11_water_inventory.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "summary.json"
)

REPORT_PATH = (
    ROOT
    / "reports"
    / "v06_sen1floods11_water_inventory.md"
)

BUCKET_NAME = "sen1floods11"

PREFIXES = {
    "flood_sar": (
        "v1.1/data/flood_events/"
        "HandLabeled/S1Hand/"
    ),
    "flood_mask": (
        "v1.1/data/flood_events/"
        "HandLabeled/LabelHand/"
    ),
    "permanent_sar": (
        "v1.1/data/perm_water/S1Perm/"
    ),
    "permanent_mask": (
        "v1.1/data/perm_water/JRCPerm/"
    ),
    "flood_splits": (
        "v1.1/splits/flood_handlabeled/"
    ),
    "permanent_splits": (
        "v1.1/splits/perm_water/"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sen1Floods11'in yalnız SAR su-segmentasyonu "
            "için gereken alt kümelerini anonim GCS "
            "erişimiyle indirir ve envanter çıkarır."
        )
    )

    parser.add_argument(
        "--maximum-download-gb",
        type=float,
        default=8.0,
    )

    parser.add_argument(
        "--inventory-only",
        action="store_true",
    )

    return parser.parse_args()


def relative(path: Path) -> str:
    try:
        return str(
            path.resolve().relative_to(
                ROOT.resolve()
            )
        ).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def suffix_without_layer(
    name: str,
) -> str:
    stem = Path(name).stem

    layer_suffixes = (
        "_S1Hand",
        "_LabelHand",
        "_S1Perm",
        "_JRCPerm",
    )

    for suffix in layer_suffixes:
        if stem.endswith(suffix):
            return stem[: -len(suffix)]

    return stem


def list_blobs(
    bucket: storage.Bucket,
    prefix: str,
) -> list[storage.Blob]:
    return [
        blob
        for blob in bucket.list_blobs(
            prefix=prefix
        )
        if not blob.name.endswith("/")
    ]


def total_bytes(
    blobs: Iterable[storage.Blob],
) -> int:
    return sum(
        int(blob.size or 0)
        for blob in blobs
    )


def download_blob(
    blob: storage.Blob,
    destination: Path,
) -> str:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    expected_size = int(
        blob.size or 0
    )

    if (
        destination.exists()
        and expected_size > 0
        and destination.stat().st_size
        == expected_size
    ):
        return "EXISTING"

    temporary = destination.with_suffix(
        destination.suffix + ".part"
    )

    blob.download_to_filename(
        str(temporary)
    )

    if (
        expected_size > 0
        and temporary.stat().st_size
        != expected_size
    ):
        raise RuntimeError(
            "İndirme boyutu uyuşmuyor: "
            f"{blob.name}"
        )

    temporary.replace(
        destination
    )

    return "DOWNLOADED"


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

    print("=" * 78)
    print(
        "ROBUST BINARY OIL DETECTOR v0.6"
    )
    print(
        "SEN1FLOODS11 WATER-SEGMENTATION DATA"
    )
    print("=" * 78)

    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(
        BUCKET_NAME
    )

    blobs_by_group = {}

    for group_name, prefix in PREFIXES.items():
        print(
            f"Listeleniyor: {group_name}"
        )

        blobs = list_blobs(
            bucket,
            prefix,
        )

        blobs_by_group[
            group_name
        ] = blobs

        print(
            f"  Dosya: {len(blobs)}"
        )

    all_blobs = [
        blob
        for blobs in blobs_by_group.values()
        for blob in blobs
    ]

    download_bytes = total_bytes(
        all_blobs
    )

    download_gb = (
        download_bytes
        / (1024 ** 3)
    )

    print()
    print(
        "Toplam uzak boyut:",
        f"{download_gb:.2f} GB",
    )

    if download_gb > args.maximum_download_gb:
        raise RuntimeError(
            "İndirme güvenlik sınırını aşıyor. "
            f"Bulunan={download_gb:.2f} GB, "
            f"sınır={args.maximum_download_gb:.2f} GB"
        )

    records = []

    downloaded = 0
    existing = 0

    for group_name, blobs in blobs_by_group.items():
        prefix = PREFIXES[
            group_name
        ]

        for file_number, blob in enumerate(
            blobs,
            start=1,
        ):
            relative_blob_name = blob.name[
                len(prefix):
            ]

            destination = (
                DATA_ROOT
                / group_name
                / relative_blob_name
            )

            status = "INVENTORY_ONLY"

            if not args.inventory_only:
                status = download_blob(
                    blob,
                    destination,
                )

                if status == "DOWNLOADED":
                    downloaded += 1
                else:
                    existing += 1

            records.append(
                {
                    "group_name": group_name,
                    "blob_name": blob.name,
                    "local_path": relative(
                        destination
                    ),
                    "size_bytes": int(
                        blob.size or 0
                    ),
                    "download_status": status,
                    "sample_key": (
                        suffix_without_layer(
                            relative_blob_name
                        )
                        if destination.suffix.lower()
                        in {
                            ".tif",
                            ".tiff",
                        }
                        else ""
                    ),
                }
            )

            if (
                file_number % 100 == 0
                or file_number == len(blobs)
            ):
                print(
                    f"{group_name}: "
                    f"{file_number}/{len(blobs)}"
                )

    with MANIFEST_PATH.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "group_name",
                "blob_name",
                "local_path",
                "size_bytes",
                "download_status",
                "sample_key",
            ],
        )

        writer.writeheader()
        writer.writerows(
            records
        )

    group_counts = {
        group_name: len(blobs)
        for group_name, blobs
        in blobs_by_group.items()
    }

    summary = {
        "stage": (
            "v06_sen1floods11_water_inventory"
        ),
        "bucket": BUCKET_NAME,
        "group_counts": group_counts,
        "total_files": len(records),
        "total_remote_bytes": int(
            download_bytes
        ),
        "total_remote_gb": float(
            download_gb
        ),
        "downloaded_files": int(
            downloaded
        ),
        "existing_files": int(
            existing
        ),
        "inventory_only": bool(
            args.inventory_only
        ),
        "locked_test_used": False,
        "training_performed": False,
    }

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    REPORT_PATH.write_text(
        f"""# v0.6 Sen1Floods11 Su Segmentasyonu Veri Envanteri

## Amaç

Koordinat tabanlı ve kaymış DARTIS kara maskelerini eğitim etiketi
olarak kullanmak yerine, piksel seviyesinde hizalı Sentinel-1 SAR ve
su maskesi çiftlerini indirmek.

## İndirilen alt kümeler

- Flood hand-labeled Sentinel-1
- Flood hand-labeled water masks
- Permanent-water Sentinel-1
- Permanent-water masks
- Resmî train/validation/test split dosyaları

## Sonuç

- Toplam dosya: {len(records)}
- Toplam uzak boyut: {download_gb:.2f} GB
- Yeni indirilen: {downloaded}
- Zaten mevcut: {existing}

Bu aşamada model eğitilmedi ve kilitli test kullanılmadı.
""",
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print(
        "SEN1FLOODS11 ENVANTER SONUCU"
    )
    print("=" * 78)

    for group_name, count in group_counts.items():
        print(
            f"{group_name:18s}: {count}"
        )

    print(
        "Toplam uzak boyut:",
        f"{download_gb:.2f} GB",
    )

    print(
        "Yeni indirilen:",
        downloaded,
    )

    print(
        "Zaten mevcut:",
        existing,
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
        "Kilitli test kullanımı: YOK"
    )


if __name__ == "__main__":
    main()
