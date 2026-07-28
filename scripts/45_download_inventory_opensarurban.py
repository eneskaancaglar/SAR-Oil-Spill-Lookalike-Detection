from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import shutil
import urllib.error
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = (
    ROOT
    / "data"
    / "external"
    / "opensarurban"
)

ARCHIVE_PATH = (
    DATASET_ROOT
    / "OpenSARUrbanDataset.zip"
)

CURATED_ROOT = (
    DATASET_ROOT
    / "curated_grayscale"
)

MANIFEST_PATH = (
    ROOT
    / "data"
    / "metadata"
    / "v05_opensarurban_inventory.csv"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "v05_opensarurban_inventory"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "summary.json"
)

GROUP_COUNTS_PATH = (
    OUTPUT_DIR
    / "group_counts.csv"
)

ARCHIVE_CONTENTS_PATH = (
    OUTPUT_DIR
    / "archive_contents.csv"
)

REPORT_PATH = (
    ROOT
    / "reports"
    / "v05_opensarurban_inventory.md"
)

DOWNLOAD_URL = (
    "https://opensar.sjtu.edu.cn/"
    "openSAR/OpenSARUrbanDataset.zip"
)

IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
}

FORMAT_KEYWORDS = {
    "gray",
    "grey",
    "grayscale",
    "greyscale",
    "uint8",
    "visual",
    "enhance",
    "enhanced",
}

EXCLUDED_FORMAT_KEYWORDS = {
    "pseudo",
    "pseudocolor",
    "pseudo-color",
    "rgb",
    "jet",
    "colormap",
    "colour",
    "color",
}

GENERIC_DIRECTORY_NAMES = {
    "",
    "data",
    "dataset",
    "opensarurban",
    "opensarurbandataset",
    "image",
    "images",
    "patch",
    "patches",
    "gray",
    "grey",
    "grayscale",
    "greyscale",
    "uint8",
    "visual",
    "visualized",
    "enhance",
    "enhanced",
    "vv",
    "vh",
    "original",
    "calibrated",
    "calibration",
    "pseudocolor",
    "pseudo",
    "rgb",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "OpenSARUrban verisini indirir ve deniz olmayan "
            "gerçek SAR hard-negative havuzu oluşturur."
        )
    )

    parser.add_argument(
        "--maximum-total",
        type=int,
        default=8000,
        help=(
            "Çıkarılacak en yüksek gri tonlu SAR "
            "görüntüsü sayısı."
        ),
    )

    parser.add_argument(
        "--minimum-total",
        type=int,
        default=1000,
        help=(
            "Başarılı sayılmak için gerekli en düşük "
            "görüntü sayısı."
        ),
    )

    parser.add_argument(
        "--maximum-per-group",
        type=int,
        default=1500,
    )

    parser.add_argument(
        "--preferred-polarization",
        type=str,
        default="VV",
        choices=[
            "VV",
            "VH",
            "ANY",
        ],
    )

    parser.add_argument(
        "--maximum-download-gb",
        type=float,
        default=20.0,
        help=(
            "Sunucu dosya boyutunu bildirirse bu sınırın "
            "üzerindeki indirmeyi iptal eder."
        ),
    )

    parser.add_argument(
        "--force-download",
        action="store_true",
    )

    parser.add_argument(
        "--force-curate",
        action="store_true",
    )

    return parser.parse_args()


def human_size(size_bytes: int) -> str:
    value = float(size_bytes)

    for unit in (
        "B",
        "KB",
        "MB",
        "GB",
        "TB",
    ):
        if value < 1024.0:
            return f"{value:.2f} {unit}"

        value /= 1024.0

    return f"{value:.2f} PB"


def calculate_sha256(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(
                chunk_size
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def probe_remote_size(
    url: str,
) -> int | None:
    request = urllib.request.Request(
        url,
        method="HEAD",
        headers={
            "User-Agent": (
                "Mozilla/5.0 "
                "SAR-Oil-Spill-Segmentation-v0.5"
            )
        },
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=60,
        ) as response:
            content_length = response.headers.get(
                "Content-Length"
            )

            if (
                content_length
                and content_length.isdigit()
            ):
                return int(
                    content_length
                )

    except Exception as error:
        print(
            "UYARI: Uzak dosya boyutu "
            "önceden öğrenilemedi:",
            type(error).__name__,
        )

    return None


def download_with_resume(
    url: str,
    destination: Path,
    maximum_download_gb: float,
) -> None:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    remote_size = probe_remote_size(
        url
    )

    maximum_bytes = int(
        maximum_download_gb
        * 1024 ** 3
    )

    if remote_size is not None:
        print(
            "Uzak arşiv boyutu:",
            human_size(
                remote_size
            ),
        )

        if remote_size > maximum_bytes:
            raise RuntimeError(
                "Arşiv güvenlik sınırından büyük. "
                f"Boyut={human_size(remote_size)}, "
                f"sınır={maximum_download_gb:.1f} GB"
            )

    temporary_path = destination.with_suffix(
        destination.suffix + ".part"
    )

    existing_size = (
        temporary_path.stat().st_size
        if temporary_path.exists()
        else 0
    )

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "SAR-Oil-Spill-Segmentation-v0.5"
        )
    }

    if existing_size > 0:
        headers["Range"] = (
            f"bytes={existing_size}-"
        )

        print(
            "Yarım kalan indirme bulundu:",
            human_size(existing_size),
        )

    request = urllib.request.Request(
        url,
        headers=headers,
    )

    try:
        response = urllib.request.urlopen(
            request,
            timeout=180,
        )

    except urllib.error.HTTPError as error:
        raise RuntimeError(
            "OpenSARUrban indirilemedi. "
            f"HTTP {error.code}: {error.reason}"
        ) from error

    status = getattr(
        response,
        "status",
        200,
    )

    if existing_size > 0 and status == 206:
        mode = "ab"
        downloaded = existing_size

    else:
        mode = "wb"
        downloaded = 0
        existing_size = 0

    content_length_text = (
        response.headers.get(
            "Content-Length",
            ""
        )
    )

    response_size = (
        int(content_length_text)
        if content_length_text.isdigit()
        else 0
    )

    total_size = (
        downloaded + response_size
        if status == 206
        else response_size
    )

    if (
        total_size > 0
        and total_size > maximum_bytes
    ):
        response.close()

        raise RuntimeError(
            "İndirme sırasında arşivin güvenlik "
            "sınırından büyük olduğu görüldü. "
            f"Boyut={human_size(total_size)}"
        )

    print(
        "İndirme başlatılıyor..."
    )

    last_printed_percent = -1

    with response:
        with temporary_path.open(
            mode
        ) as output:
            while True:
                chunk = response.read(
                    1024 * 1024
                )

                if not chunk:
                    break

                output.write(chunk)
                downloaded += len(chunk)

                if downloaded > maximum_bytes:
                    raise RuntimeError(
                        "İndirme belirlenen maksimum "
                        "boyutu aştı."
                    )

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

                        last_printed_percent = (
                            percent
                        )

                elif (
                    downloaded
                    // (100 * 1024 * 1024)
                    != (
                        downloaded
                        - len(chunk)
                    )
                    // (100 * 1024 * 1024)
                ):
                    print(
                        "  İndirilen:",
                        human_size(downloaded),
                    )

    final_size = temporary_path.stat().st_size

    if total_size > 0 and final_size < total_size:
        raise RuntimeError(
            "?ndirme erken sonland?. "
            f"Beklenen={human_size(total_size)}, "
            f"indirilen={human_size(final_size)}. "
            ".part dosyas? korundu. Ayn? komutu yeniden ?al??t?r."
        )

    if remote_size is not None and final_size < remote_size:
        raise RuntimeError(
            "?ndirme uzak dosya boyutuna ula?mad?. "
            f"Beklenen={human_size(remote_size)}, "
            f"indirilen={human_size(final_size)}. "
            ".part dosyas? korundu."
        )

    temporary_path.replace(
        destination
    )


def normalize_token(value: str) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "",
        value.lower(),
    )


def detect_polarization(
    member_name: str,
) -> str:
    normalized = member_name.replace(
        "\\",
        "/",
    ).lower()

    tokens = re.split(
        r"[^a-z0-9]+",
        normalized,
    )

    if "vv" in tokens:
        return "VV"

    if "vh" in tokens:
        return "VH"

    return "UNKNOWN"


def format_score(
    member_name: str,
    preferred_polarization: str,
) -> int:
    lowered = member_name.lower()

    score = 0

    if any(
        keyword in lowered
        for keyword in FORMAT_KEYWORDS
    ):
        score += 20

    if any(
        keyword in lowered
        for keyword
        in EXCLUDED_FORMAT_KEYWORDS
    ):
        score -= 50

    if "original" in lowered:
        score -= 10

    if "calibrat" in lowered:
        score -= 5

    polarization = detect_polarization(
        member_name
    )

    if (
        preferred_polarization != "ANY"
        and polarization
        == preferred_polarization
    ):
        score += 10

    elif (
        preferred_polarization != "ANY"
        and polarization
        not in {
            preferred_polarization,
            "UNKNOWN",
        }
    ):
        score -= 3

    suffix = PurePosixPath(
        member_name
    ).suffix.lower()

    if suffix in {
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
    }:
        score += 4

    return score


def infer_group(
    member_name: str,
) -> str:
    path = PurePosixPath(
        member_name.replace(
            "\\",
            "/",
        )
    )

    parent_parts = list(
        path.parts[:-1]
    )

    format_index = None

    for index, part in enumerate(
        parent_parts
    ):
        normalized = normalize_token(
            part
        )

        if any(
            normalize_token(keyword)
            in normalized
            for keyword
            in (
                FORMAT_KEYWORDS
                | EXCLUDED_FORMAT_KEYWORDS
            )
        ):
            format_index = index
            break

    candidates = (
        parent_parts[:format_index]
        if format_index is not None
        else parent_parts
    )

    for part in reversed(
        candidates
    ):
        normalized = normalize_token(
            part
        )

        if (
            normalized
            and normalized
            not in {
                normalize_token(value)
                for value
                in GENERIC_DIRECTORY_NAMES
            }
        ):
            return re.sub(
                r"[^a-zA-Z0-9_-]+",
                "_",
                part,
            )[:80]

    stem = path.stem

    prefix = re.split(
        r"[_\-\d]+",
        stem,
        maxsplit=1,
    )[0]

    if prefix:
        return re.sub(
            r"[^a-zA-Z0-9_-]+",
            "_",
            prefix,
        )[:80]

    return "urban_sar"


def canonical_patch_key(
    member_name: str,
    group_name: str,
) -> str:
    stem = PurePosixPath(
        member_name
    ).stem.lower()

    stem = re.sub(
        r"(^|[_-])(vv|vh)(?=$|[_-])",
        "_",
        stem,
    )

    stem = re.sub(
        r"[^a-z0-9]+",
        "_",
        stem,
    ).strip("_")

    return (
        f"{normalize_token(group_name)}"
        f"|{stem}"
    )


def stable_rank(
    value: str,
) -> str:
    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def select_archive_members(
    archive: zipfile.ZipFile,
    maximum_total: int,
    maximum_per_group: int,
    preferred_polarization: str,
) -> tuple[
    list[zipfile.ZipInfo],
    list[dict[str, Any]],
]:
    image_members = [
        info
        for info in archive.infolist()
        if (
            not info.is_dir()
            and PurePosixPath(
                info.filename
            ).suffix.lower()
            in IMAGE_SUFFIXES
        )
    ]

    if not image_members:
        raise RuntimeError(
            "ZIP içinde desteklenen görüntü "
            "uzantısı bulunamadı."
        )

    best_by_patch: dict[
        str,
        tuple[int, zipfile.ZipInfo, str],
    ] = {}

    content_rows = []

    for info in image_members:
        group_name = infer_group(
            info.filename
        )

        score = format_score(
            info.filename,
            preferred_polarization,
        )

        polarization = detect_polarization(
            info.filename
        )

        patch_key = canonical_patch_key(
            info.filename,
            group_name,
        )

        content_rows.append(
            {
                "archive_member": (
                    info.filename
                ),
                "compressed_size": int(
                    info.compress_size
                ),
                "uncompressed_size": int(
                    info.file_size
                ),
                "source_group": (
                    group_name
                ),
                "polarization": (
                    polarization
                ),
                "selection_score": int(
                    score
                ),
                "canonical_patch_key": (
                    patch_key
                ),
            }
        )

        previous = best_by_patch.get(
            patch_key
        )

        candidate = (
            score,
            info,
            group_name,
        )

        if previous is None:
            best_by_patch[
                patch_key
            ] = candidate

        elif score > previous[0]:
            best_by_patch[
                patch_key
            ] = candidate

        elif (
            score == previous[0]
            and info.filename
            < previous[1].filename
        ):
            best_by_patch[
                patch_key
            ] = candidate

    grouped: dict[
        str,
        list[zipfile.ZipInfo],
    ] = defaultdict(list)

    for (
        score,
        info,
        group_name,
    ) in best_by_patch.values():
        if score < 0:
            continue

        grouped[
            group_name
        ].append(
            info
        )

    selected = []

    for group_name in sorted(
        grouped
    ):
        members = sorted(
            grouped[group_name],
            key=lambda item: stable_rank(
                item.filename
            ),
        )

        selected.extend(
            members[
                :maximum_per_group
            ]
        )

    selected = sorted(
        selected,
        key=lambda item: stable_rank(
            item.filename
        ),
    )[:maximum_total]

    return selected, content_rows


def extract_curated_images(
    archive: zipfile.ZipFile,
    selected_members: list[zipfile.ZipInfo],
) -> list[dict[str, Any]]:
    if CURATED_ROOT.exists():
        shutil.rmtree(
            CURATED_ROOT
        )

    CURATED_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []
    failures = []

    for index, info in enumerate(
        selected_members,
        start=1,
    ):
        try:
            raw_bytes = archive.read(
                info
            )

            with Image.open(
                io.BytesIO(raw_bytes)
            ) as source:
                source.load()

                original_mode = str(
                    source.mode
                )

                original_width, original_height = (
                    source.size
                )

                grayscale = source.convert(
                    "L"
                )

            if (
                original_width < 32
                or original_height < 32
            ):
                continue

            group_name = infer_group(
                info.filename
            )

            patch_key = canonical_patch_key(
                info.filename,
                group_name,
            )

            sample_id = hashlib.sha256(
                (
                    "opensarurban|"
                    + info.filename.lower()
                ).encode("utf-8")
            ).hexdigest()[:24]

            destination_directory = (
                CURATED_ROOT
                / group_name
            )

            destination_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

            destination_path = (
                destination_directory
                / f"{sample_id}.png"
            )

            grayscale.save(
                destination_path,
                format="PNG",
                optimize=True,
            )

            rows.append(
                {
                    "sample_id": sample_id,
                    "source_dataset": (
                        "opensarurban"
                    ),
                    "source_group": (
                        group_name
                    ),
                    "binary_label": 0,
                    "binary_label_name": (
                        "unsupported_input"
                    ),
                    "image_path": str(
                        destination_path
                        .resolve()
                        .relative_to(
                            ROOT.resolve()
                        )
                    ),
                    "original_archive_member": (
                        info.filename
                    ),
                    "canonical_patch_key": (
                        patch_key
                    ),
                    "polarization": (
                        detect_polarization(
                            info.filename
                        )
                    ),
                    "original_width": int(
                        original_width
                    ),
                    "original_height": int(
                        original_height
                    ),
                    "original_mode": (
                        original_mode
                    ),
                    "curated_mode": "L",
                    "input_policy": (
                        "reject_before_oil_pipeline"
                    ),
                    "development_role": (
                        "v05_land_sar_hard_negative"
                    ),
                    "difficulty_type": (
                        "non_sea_real_sar"
                    ),
                    "hard_negative_weight": 4.0,
                    "eligible_for_v05_locked_test": (
                        False
                    ),
                }
            )

        except Exception as error:
            failures.append(
                {
                    "archive_member": (
                        info.filename
                    ),
                    "error_type": (
                        type(error).__name__
                    ),
                    "error": str(error),
                }
            )

        if (
            index % 500 == 0
            or index
            == len(selected_members)
        ):
            print(
                f"Curated: {index}/"
                f"{len(selected_members)} "
                f"| başarılı={len(rows)} "
                f"| hatalı={len(failures)}"
            )

    if failures:
        failure_path = (
            OUTPUT_DIR
            / "read_failures.csv"
        )

        with failure_path.open(
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "archive_member",
                    "error_type",
                    "error",
                ],
            )

            writer.writeheader()
            writer.writerows(
                failures
            )

    return rows


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        raise RuntimeError(
            f"Yazılacak satır bulunamadı: {path}"
        )

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def write_report(
    summary: dict[str, Any],
) -> None:
    lines = [
        "# v0.5 OpenSARUrban Envanteri",
        "",
        "## Amaç",
        "",
        "Input-gate modeline gerçek fakat desteklenmeyen "
        "deniz dışı SAR görüntülerini reddetmeyi öğretmek.",
        "",
        "## Veri rolü",
        "",
        "- Kaynak: OpenSARUrban",
        "- Sensör ailesi: Sentinel-1 SAR",
        "- Hedef etiket: `unsupported_input`",
        "- Petrol pipeline kararı: reddet",
        "- v0.5 kilitli test uygunluğu: hayır",
        "",
        "## Curated veri",
        "",
        f"- Seçilen arşiv üyesi: "
        f"{summary['selected_archive_members']}",
        f"- Başarıyla hazırlanan görüntü: "
        f"{summary['curated_images']}",
        f"- Grup sayısı: "
        f"{summary['group_count']}",
        f"- VV görüntüsü: "
        f"{summary['polarization_counts'].get('VV', 0)}",
        f"- VH görüntüsü: "
        f"{summary['polarization_counts'].get('VH', 0)}",
        f"- Polarizasyonu belirlenemeyen: "
        f"{summary['polarization_counts'].get('UNKNOWN', 0)}",
        "",
        "## Arşiv doğrulaması",
        "",
        f"- ZIP boyutu: "
        f"{summary['archive_size_human']}",
        f"- ZIP testi: "
        f"{summary['zip_integrity_valid']}",
        f"- Yerel SHA-256: "
        f"`{summary['archive_sha256']}`",
        "",
        "Resmî kaynak tarafından önceden yayımlanmış "
        "bir checksum bulunmadığı için SHA-256 değeri "
        "yerel indirme kaydı olarak tutulmuştur.",
        "",
        "## Bilimsel kullanım",
        "",
        "Bu görüntüler v0.5 geliştirme hard-negative "
        "verisidir. v0.5 bağımsız test verisi olarak "
        "raporlanmayacaktır.",
        "",
    ]

    REPORT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    if args.maximum_total <= 0:
        raise ValueError(
            "--maximum-total pozitif olmalıdır."
        )

    if args.minimum_total <= 0:
        raise ValueError(
            "--minimum-total pozitif olmalıdır."
        )

    DATASET_ROOT.mkdir(
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
        "OPENSARURBAN DOWNLOAD VE ENVANTER"
    )
    print("=" * 78)

    if args.force_download:
        for path in (
            ARCHIVE_PATH,
            ARCHIVE_PATH.with_suffix(
                ARCHIVE_PATH.suffix
                + ".part"
            ),
        ):
            if path.exists():
                path.unlink()

    if ARCHIVE_PATH.exists():
        print(
            "Arşiv zaten mevcut:",
            ARCHIVE_PATH,
        )

    else:
        download_with_resume(
            DOWNLOAD_URL,
            ARCHIVE_PATH,
            args.maximum_download_gb,
        )

    print(
        "Arşiv boyutu:",
        human_size(
            ARCHIVE_PATH.stat().st_size
        ),
    )

    if not zipfile.is_zipfile(
        ARCHIVE_PATH
    ):
        raise RuntimeError(
            "İndirilen dosya geçerli bir ZIP değil."
        )

    print(
        "ZIP bütünlüğü kontrol ediliyor..."
    )

    with zipfile.ZipFile(
        ARCHIVE_PATH,
        "r",
    ) as archive:
        damaged_member = (
            archive.testzip()
        )

        if damaged_member is not None:
            raise RuntimeError(
                "ZIP içinde bozuk dosya bulundu: "
                f"{damaged_member}"
            )

    print(
        "ZIP bütünlüğü: BAŞARILI"
    )

    print(
        "SHA-256 hesaplanıyor..."
    )

    archive_sha256 = calculate_sha256(
        ARCHIVE_PATH
    )

    print(
        "Yerel SHA-256:",
        archive_sha256,
    )

    with zipfile.ZipFile(
        ARCHIVE_PATH,
        "r",
    ) as archive:
        (
            selected_members,
            content_rows,
        ) = select_archive_members(
            archive,
            maximum_total=(
                args.maximum_total
            ),
            maximum_per_group=(
                args.maximum_per_group
            ),
            preferred_polarization=(
                args.preferred_polarization
            ),
        )

        print(
            "Arşivdeki görüntü dosyası:",
            len(content_rows),
        )

        print(
            "Curated için seçilen:",
            len(selected_members),
        )

        write_csv(
            ARCHIVE_CONTENTS_PATH,
            content_rows,
        )

        if (
            len(selected_members)
            < args.minimum_total
        ):
            raise RuntimeError(
                "Yeterli uygun SAR görüntüsü "
                "seçilemedi. "
                f"Seçilen={len(selected_members)}, "
                f"minimum={args.minimum_total}"
            )

        if (
            args.force_curate
            or not MANIFEST_PATH.exists()
        ):
            curated_rows = (
                extract_curated_images(
                    archive,
                    selected_members,
                )
            )

            write_csv(
                MANIFEST_PATH,
                curated_rows,
            )

        else:
            print(
                "Manifest zaten mevcut. "
                "Yeniden hazırlamak için "
                "--force-curate kullan."
            )

            with MANIFEST_PATH.open(
                "r",
                encoding="utf-8-sig",
                newline="",
            ) as file:
                curated_rows = list(
                    csv.DictReader(file)
                )

    if (
        len(curated_rows)
        < args.minimum_total
    ):
        raise RuntimeError(
            "Başarıyla hazırlanan görüntü "
            "sayısı yetersiz. "
            f"Hazırlanan={len(curated_rows)}"
        )

    group_counts = Counter(
        str(row["source_group"])
        for row in curated_rows
    )

    polarization_counts = Counter(
        str(row["polarization"])
        for row in curated_rows
    )

    group_rows = [
        {
            "source_group": group_name,
            "image_count": int(count),
        }
        for group_name, count
        in sorted(
            group_counts.items()
        )
    ]

    write_csv(
        GROUP_COUNTS_PATH,
        group_rows,
    )

    summary = {
        "dataset": "OpenSARUrban",
        "source_url": DOWNLOAD_URL,
        "archive_path": str(
            ARCHIVE_PATH.resolve()
        ),
        "archive_size_bytes": int(
            ARCHIVE_PATH.stat().st_size
        ),
        "archive_size_human": human_size(
            ARCHIVE_PATH.stat().st_size
        ),
        "archive_sha256": (
            archive_sha256
        ),
        "zip_integrity_valid": True,
        "archive_image_members": int(
            len(content_rows)
        ),
        "selected_archive_members": int(
            len(selected_members)
        ),
        "curated_images": int(
            len(curated_rows)
        ),
        "group_count": int(
            len(group_counts)
        ),
        "group_counts": {
            key: int(value)
            for key, value
            in sorted(
                group_counts.items()
            )
        },
        "polarization_counts": {
            key: int(value)
            for key, value
            in sorted(
                polarization_counts.items()
            )
        },
        "preferred_polarization": (
            args.preferred_polarization
        ),
        "development_only": True,
        "eligible_for_v05_locked_test": False,
        "manifest_path": str(
            MANIFEST_PATH.resolve()
        ),
        "curated_root": str(
            CURATED_ROOT.resolve()
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
        "OPENSARURBAN ENVANTER SONUCU"
    )
    print("=" * 78)

    print(
        "Hazırlanan görüntü:",
        len(curated_rows),
    )

    print(
        "Grup sayısı:",
        len(group_counts),
    )

    print(
        "VV:",
        polarization_counts.get(
            "VV",
            0,
        ),
    )

    print(
        "VH:",
        polarization_counts.get(
            "VH",
            0,
        ),
    )

    print(
        "UNKNOWN:",
        polarization_counts.get(
            "UNKNOWN",
            0,
        ),
    )

    print()
    print(
        "Curated klasör:",
        CURATED_ROOT.resolve(),
    )

    print(
        "Manifest:",
        MANIFEST_PATH.resolve(),
    )

    print(
        "Arşiv içeriği:",
        ARCHIVE_CONTENTS_PATH.resolve(),
    )

    print(
        "Grup dağılımı:",
        GROUP_COUNTS_PATH.resolve(),
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
