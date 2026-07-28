from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = (
    ROOT
    / "data"
    / "external"
    / "eurosat"
)

ZIP_PATH = (
    DATASET_DIR
    / "EuroSAT_RGB.zip"
)

EXTRACT_ROOT = (
    DATASET_DIR
    / "raw"
)

EXTRACT_MARKER = (
    DATASET_DIR
    / ".rgb_extracted.json"
)

MANIFEST_PATH = (
    ROOT
    / "data"
    / "metadata"
    / "v05_eurosat_inventory.csv"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "v05_eurosat_inventory"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "summary.json"
)

CLASS_COUNTS_PATH = (
    OUTPUT_DIR
    / "class_counts.csv"
)

REPORT_PATH = (
    ROOT
    / "reports"
    / "v05_eurosat_inventory.md"
)

DOWNLOAD_URL = (
    "https://zenodo.org/api/records/7711810/"
    "files/EuroSAT_RGB.zip/content"
)

EXPECTED_MD5 = (
    "f46e308c4d50d4bf32fedad2d3d62f3b"
)

EXPECTED_IMAGE_COUNT = 27000

EXPECTED_CLASSES = {
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
}

IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
}

HIGH_RISK_CLASSES = {
    "River",
    "SeaLake",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "EuroSAT RGB veri setini resmî Zenodo kaynağından "
            "indirir, doğrular, çıkarır ve v0.5 input-gate "
            "hard-negative envanterini oluşturur."
        )
    )

    parser.add_argument(
        "--force-download",
        action="store_true",
        help=(
            "Var olan ZIP dosyasını silerek yeniden indirir."
        ),
    )

    parser.add_argument(
        "--force-extract",
        action="store_true",
        help=(
            "Var olan çıkarılmış klasörü silerek yeniden çıkarır."
        ),
    )

    return parser.parse_args()


def calculate_md5(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    digest = hashlib.md5()

    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def human_size(
    size_bytes: int,
) -> str:
    value = float(size_bytes)

    for unit in (
        "B",
        "KB",
        "MB",
        "GB",
    ):
        if value < 1024.0:
            return f"{value:.2f} {unit}"

        value /= 1024.0

    return f"{value:.2f} TB"


def download_file(
    url: str,
    destination: Path,
) -> None:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = destination.with_suffix(
        destination.suffix + ".part"
    )

    if temporary_path.exists():
        temporary_path.unlink()

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "SAR-Oil-Spill-Segmentation/"
                "v0.5-data-preparation"
            )
        },
    )

    print("İndirme başlatılıyor...")

    with urllib.request.urlopen(
        request,
        timeout=120,
    ) as response:
        total_size_text = response.headers.get(
            "Content-Length",
            "",
        )

        total_size = (
            int(total_size_text)
            if total_size_text.isdigit()
            else 0
        )

        downloaded = 0
        last_printed_percent = -1

        with temporary_path.open("wb") as output:
            while True:
                chunk = response.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                output.write(chunk)
                downloaded += len(chunk)

                if total_size > 0:
                    percent = int(
                        downloaded
                        * 100
                        / total_size
                    )

                    if (
                        percent >= last_printed_percent + 5
                        or percent == 100
                    ):
                        print(
                            f"  %{percent:3d} "
                            f"({human_size(downloaded)} "
                            f"/ {human_size(total_size)})"
                        )

                        last_printed_percent = percent

                else:
                    print(
                        "\r  İndirilen: "
                        f"{human_size(downloaded)}",
                        end="",
                        flush=True,
                    )

    print()

    temporary_path.replace(
        destination
    )


def safe_extract_zip(
    zip_path: Path,
    destination: Path,
) -> None:
    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination_resolved = (
        destination.resolve()
    )

    with zipfile.ZipFile(
        zip_path,
        "r",
    ) as archive:
        members = archive.infolist()

        print(
            "ZIP içindeki öğe sayısı:",
            len(members),
        )

        for member in members:
            target_path = (
                destination
                / member.filename
            ).resolve()

            try:
                target_path.relative_to(
                    destination_resolved
                )

            except ValueError as error:
                raise RuntimeError(
                    "Güvensiz ZIP yolu tespit edildi: "
                    f"{member.filename}"
                ) from error

        archive.extractall(
            destination
        )


def discover_images() -> list[Path]:
    if not EXTRACT_ROOT.exists():
        return []

    return sorted(
        path
        for path in EXTRACT_ROOT.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in IMAGE_SUFFIXES
        )
    )


def identify_class(
    image_path: Path,
) -> str | None:
    for parent in image_path.parents:
        if parent.name in EXPECTED_CLASSES:
            return parent.name

    return None


def inspect_image(
    image_path: Path,
) -> tuple[
    int,
    int,
    str,
] | None:
    try:
        with Image.open(
            image_path
        ) as image:
            width, height = image.size
            mode = str(image.mode)

            image.verify()

        return (
            int(width),
            int(height),
            mode,
        )

    except Exception as error:
        print(
            "UYARI: Görüntü okunamadı:",
            image_path,
            type(error).__name__,
            error,
        )

        return None


def create_sample_id(
    relative_path: str,
) -> str:
    return hashlib.sha256(
        (
            "eurosat_rgb|"
            + relative_path.lower()
        ).encode("utf-8")
    ).hexdigest()[:24]


def relative_to_root(
    path: Path,
) -> str:
    return str(
        path.resolve().relative_to(
            ROOT.resolve()
        )
    )


def development_weight(
    class_name: str,
) -> float:
    if class_name in HIGH_RISK_CLASSES:
        return 5.0

    if class_name in {
        "Industrial",
        "Residential",
        "Highway",
    }:
        return 2.0

    return 1.0


def risk_type(
    class_name: str,
) -> str:
    if class_name in HIGH_RISK_CLASSES:
        return (
            "optical_water_hard_negative"
        )

    if class_name in {
        "Industrial",
        "Residential",
        "Highway",
    }:
        return (
            "optical_structural_negative"
        )

    return "optical_land_negative"


def build_inventory(
    image_paths: list[Path],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for index, image_path in enumerate(
        image_paths,
        start=1,
    ):
        class_name = identify_class(
            image_path
        )

        if class_name is None:
            continue

        inspected = inspect_image(
            image_path
        )

        if inspected is None:
            continue

        width, height, mode = inspected

        relative_path = relative_to_root(
            image_path
        )

        rows.append(
            {
                "sample_id": create_sample_id(
                    relative_path
                ),
                "source_dataset": "eurosat_rgb",
                "source_group": class_name,
                "binary_label": 0,
                "binary_label_name": (
                    "unsupported_input"
                ),
                "image_path": relative_path,
                "width": width,
                "height": height,
                "image_mode": mode,
                "input_policy": (
                    "reject_before_oil_pipeline"
                ),
                "development_role": (
                    "v05_hard_negative_pool"
                ),
                "difficulty_type": risk_type(
                    class_name
                ),
                "hard_negative_weight": (
                    development_weight(
                        class_name
                    )
                ),
                "eligible_for_v05_locked_test": (
                    False
                ),
            }
        )

        if (
            index % 2500 == 0
            or index == len(image_paths)
        ):
            print(
                f"Envanter: {index}/{len(image_paths)}"
            )

    return rows


def write_manifest(
    rows: list[dict[str, Any]],
) -> None:
    MANIFEST_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        raise RuntimeError(
            "Manifest için görüntü bulunamadı."
        )

    fieldnames = list(
        rows[0].keys()
    )

    with MANIFEST_PATH.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def write_class_counts(
    counts: Counter[str],
) -> None:
    with CLASS_COUNTS_PATH.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "source_group",
                "image_count",
                "development_weight",
                "difficulty_type",
            ],
        )

        writer.writeheader()

        for class_name in sorted(
            counts
        ):
            writer.writerow(
                {
                    "source_group": class_name,
                    "image_count": int(
                        counts[class_name]
                    ),
                    "development_weight": (
                        development_weight(
                            class_name
                        )
                    ),
                    "difficulty_type": (
                        risk_type(
                            class_name
                        )
                    ),
                }
            )


def write_report(
    summary: dict[str, Any],
) -> None:
    class_counts = summary[
        "class_counts"
    ]

    lines = [
        "# v0.5 EuroSAT RGB Envanteri",
        "",
        "## Amaç",
        "",
        "Input-gate modeline optik uydu, optik su, nehir, "
        "göl ve kara görüntülerini reddetmeyi öğretmek.",
        "",
        "Bu veri setindeki bütün görüntüler:",
        "",
        "- `binary_label = 0`",
        "- `unsupported_input`",
        "- Petrol analizinden önce reddedilecek",
        "",
        "## Kritik hard-negative sınıflar",
        "",
        "- River",
        "- SeaLake",
        "",
        "Bu iki sınıf mevcut modelin nehir ve su yüzeyi "
        "yanlış kabullerini azaltmak için daha yüksek "
        "eğitim ağırlığı alacaktır.",
        "",
        "## Sınıf dağılımı",
        "",
        "| Sınıf | Görüntü | Ağırlık | Tür |",
        "|---|---:|---:|---|",
    ]

    for class_name in sorted(
        class_counts
    ):
        lines.append(
            f"| {class_name} "
            f"| {class_counts[class_name]} "
            f"| {development_weight(class_name):.1f} "
            f"| {risk_type(class_name)} |"
        )

    lines.extend(
        [
            "",
            "## Doğrulama",
            "",
            f"- Toplam görüntü: "
            f"{summary['total_images']}",
            f"- Beklenen görüntü: "
            f"{summary['expected_images']}",
            f"- Sınıf sayısı: "
            f"{summary['class_count']}",
            f"- ZIP MD5: "
            f"`{summary['zip_md5']}`",
            f"- Checksum doğrulandı: "
            f"{summary['checksum_valid']}",
            "",
            "## Bilimsel kullanım",
            "",
            "EuroSAT görüntüleri v0.5 geliştirme verisidir. "
            "Yeni v0.5 kilitli test sonucu olarak "
            "kullanılmayacaktır.",
            "",
            "Optik görüntülerin yanında gerçek deniz olmayan "
            "SAR görüntüleri de ayrıca negatif sınıfa "
            "eklenecektir.",
            "",
        ]
    )

    REPORT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    DATASET_DIR.mkdir(
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
        "ROBUST BINARY OIL DETECTOR v0.5"
    )
    print(
        "EUROSAT RGB DOWNLOAD VE ENVANTER"
    )
    print("=" * 78)

    if args.force_download:
        if ZIP_PATH.exists():
            ZIP_PATH.unlink()

        if EXTRACT_MARKER.exists():
            EXTRACT_MARKER.unlink()

    if ZIP_PATH.exists():
        print(
            "ZIP zaten mevcut:",
            ZIP_PATH,
        )

    else:
        download_file(
            DOWNLOAD_URL,
            ZIP_PATH,
        )

    print(
        "ZIP boyutu:",
        human_size(
            ZIP_PATH.stat().st_size
        ),
    )

    print(
        "MD5 hesaplanıyor..."
    )

    zip_md5 = calculate_md5(
        ZIP_PATH
    )

    checksum_valid = (
        zip_md5.lower()
        == EXPECTED_MD5.lower()
    )

    print(
        "Hesaplanan MD5:",
        zip_md5,
    )

    print(
        "Beklenen MD5:  ",
        EXPECTED_MD5,
    )

    if not checksum_valid:
        raise RuntimeError(
            "EuroSAT ZIP checksum doğrulaması başarısız. "
            "Bozuk veya farklı bir dosya indirilmiş olabilir."
        )

    print(
        "Checksum: BAŞARILI"
    )

    if args.force_extract:
        if EXTRACT_ROOT.exists():
            shutil.rmtree(
                EXTRACT_ROOT
            )

        if EXTRACT_MARKER.exists():
            EXTRACT_MARKER.unlink()

    marker_valid = False

    if EXTRACT_MARKER.exists():
        try:
            with EXTRACT_MARKER.open(
                "r",
                encoding="utf-8",
            ) as file:
                marker = json.load(file)

            marker_valid = (
                marker.get("zip_md5")
                == EXPECTED_MD5
                and EXTRACT_ROOT.exists()
            )

        except Exception:
            marker_valid = False

    if marker_valid:
        print(
            "Çıkarılmış veri zaten mevcut."
        )

    else:
        if EXTRACT_ROOT.exists():
            shutil.rmtree(
                EXTRACT_ROOT
            )

        print(
            "ZIP çıkarılıyor..."
        )

        safe_extract_zip(
            ZIP_PATH,
            EXTRACT_ROOT,
        )

        with EXTRACT_MARKER.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                {
                    "zip_md5": zip_md5,
                    "source": DOWNLOAD_URL,
                },
                file,
                indent=2,
            )

    print(
        "Görüntüler taranıyor..."
    )

    image_paths = discover_images()

    print(
        "Bulunan görüntü:",
        len(image_paths),
    )

    rows = build_inventory(
        image_paths
    )

    class_counts = Counter(
        str(row["source_group"])
        for row in rows
    )

    discovered_classes = set(
        class_counts
    )

    missing_classes = (
        EXPECTED_CLASSES
        - discovered_classes
    )

    unexpected_classes = (
        discovered_classes
        - EXPECTED_CLASSES
    )

    if missing_classes:
        raise RuntimeError(
            "Eksik EuroSAT sınıfları: "
            f"{sorted(missing_classes)}"
        )

    if unexpected_classes:
        raise RuntimeError(
            "Beklenmeyen sınıflar: "
            f"{sorted(unexpected_classes)}"
        )

    if len(rows) != EXPECTED_IMAGE_COUNT:
        raise RuntimeError(
            "EuroSAT görüntü sayısı beklenenden farklı. "
            f"Beklenen={EXPECTED_IMAGE_COUNT}, "
            f"bulunan={len(rows)}"
        )

    write_manifest(
        rows
    )

    write_class_counts(
        class_counts
    )

    summary = {
        "dataset": "EuroSAT_RGB",
        "source": "official_zenodo_record_7711810",
        "zip_path": str(
            ZIP_PATH.resolve()
        ),
        "zip_size_bytes": int(
            ZIP_PATH.stat().st_size
        ),
        "zip_md5": zip_md5,
        "expected_md5": EXPECTED_MD5,
        "checksum_valid": bool(
            checksum_valid
        ),
        "total_images": int(
            len(rows)
        ),
        "expected_images": (
            EXPECTED_IMAGE_COUNT
        ),
        "class_count": int(
            len(class_counts)
        ),
        "class_counts": {
            class_name: int(count)
            for class_name, count
            in sorted(
                class_counts.items()
            )
        },
        "high_risk_classes": sorted(
            HIGH_RISK_CLASSES
        ),
        "development_only": True,
        "eligible_for_v05_locked_test": False,
        "manifest_path": str(
            MANIFEST_PATH.resolve()
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
        summary
    )

    print()
    print("=" * 78)
    print(
        "EUROSAT ENVANTER SONUCU"
    )
    print("=" * 78)

    print(
        "Toplam görüntü:",
        len(rows),
    )

    print(
        "Sınıf sayısı:",
        len(class_counts),
    )

    print(
        "River:",
        class_counts.get(
            "River",
            0,
        ),
    )

    print(
        "SeaLake:",
        class_counts.get(
            "SeaLake",
            0,
        ),
    )

    print()
    print(
        "Manifest:",
        MANIFEST_PATH.resolve(),
    )

    print(
        "Sınıf dağılımı:",
        CLASS_COUNTS_PATH.resolve(),
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
        "Bu veri yalnız v0.5 geliştirmesinde "
        "kullanılacaktır."
    )


if __name__ == "__main__":
    main()
