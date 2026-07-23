from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_ROOT = Path("data/raw/images")
MASK_ROOT = Path("data/raw/masks")
OUTPUT_PATH = Path("outputs/dataset_summary.json")


def is_real_png(path: Path) -> bool:
    """macOS yardımcı dosyalarını ve gerçek olmayan dosyaları dışarıda bırakır."""
    return (
        path.is_file()
        and path.suffix.lower() == ".png"
        and not path.name.startswith("._")
    )


def find_split(path: Path) -> str:
    """Dosyanın train, val veya test klasöründe olup olmadığını bulur."""
    parts = [part.lower() for part in path.parts]

    if "train" in parts:
        return "train"
    if "val" in parts or "valid" in parts or "validation" in parts:
        return "val"
    if "test" in parts:
        return "test"

    return "unknown"


def create_key(path: Path) -> tuple[str, str]:
    """
    Görüntü ve maskeyi eşleştirmek için anahtar oluşturur.

    Örnek:
    train/palsar_1007.png -> ("train", "palsar_1007")
    """
    return find_split(path), path.stem.lower()


def sensor_name(path: Path) -> str:
    """Dosya adından yaklaşık sensör adını çıkarır."""
    name = path.stem.lower()

    if name.startswith("palsar"):
        return "PALSAR"
    if name.startswith("sentinel"):
        return "Sentinel-1"

    return "Other"


def build_index(files: list[Path]) -> dict[tuple[str, str], list[Path]]:
    """Aynı anahtara sahip dosyaları birlikte tutar."""
    index: dict[tuple[str, str], list[Path]] = defaultdict(list)

    for path in files:
        index[create_key(path)].append(path)

    return dict(index)


def main() -> None:
    image_files = sorted(
        path for path in IMAGE_ROOT.rglob("*.png")
        if is_real_png(path)
    )

    mask_files = sorted(
        path for path in MASK_ROOT.rglob("*.png")
        if is_real_png(path)
    )

    image_index = build_index(image_files)
    mask_index = build_index(mask_files)

    image_keys = set(image_index)
    mask_keys = set(mask_index)

    paired_keys = sorted(image_keys & mask_keys)
    missing_masks = sorted(image_keys - mask_keys)
    missing_images = sorted(mask_keys - image_keys)

    duplicate_images = {
        str(key): [str(path) for path in paths]
        for key, paths in image_index.items()
        if len(paths) > 1
    }

    duplicate_masks = {
        str(key): [str(path) for path in paths]
        for key, paths in mask_index.items()
        if len(paths) > 1
    }

    image_split_counts = Counter(find_split(path) for path in image_files)
    mask_split_counts = Counter(find_split(path) for path in mask_files)
    sensor_counts = Counter(sensor_name(path) for path in image_files)

    image_size_counts: Counter[str] = Counter()
    image_mode_counts: Counter[str] = Counter()
    image_metadata: dict[tuple[str, str], tuple[int, int]] = {}

    print("Görüntüler inceleniyor...")

    for path in image_files:
        with Image.open(path) as image:
            width, height = image.size
            image_size_counts[f"{width}x{height}"] += 1
            image_mode_counts[image.mode] += 1
            image_metadata[create_key(path)] = image.size

    mask_size_counts: Counter[str] = Counter()
    mask_mode_counts: Counter[str] = Counter()
    mask_value_patterns: Counter[str] = Counter()
    mask_metadata: dict[tuple[str, str], tuple[int, int]] = {}

    empty_mask_count = 0
    oil_mask_count = 0
    positive_pixel_count = 0
    total_pixel_count = 0

    print("Maskeler inceleniyor...")

    for path in mask_files:
        with Image.open(path) as mask:
            original_mode = mask.mode
            grayscale_mask = mask.convert("L")
            array = np.asarray(grayscale_mask)

            width, height = grayscale_mask.size
            mask_size_counts[f"{width}x{height}"] += 1
            mask_mode_counts[original_mode] += 1
            mask_metadata[create_key(path)] = grayscale_mask.size

            unique_values = tuple(
                int(value) for value in np.unique(array)
            )
            mask_value_patterns[str(unique_values)] += 1

            positive_pixels = int(np.count_nonzero(array))

            positive_pixel_count += positive_pixels
            total_pixel_count += int(array.size)

            if positive_pixels > 0:
                oil_mask_count += 1
            else:
                empty_mask_count += 1

    size_mismatches = []

    for key in paired_keys:
        image_size = image_metadata.get(key)
        mask_size = mask_metadata.get(key)

        if image_size != mask_size:
            size_mismatches.append(
                {
                    "key": str(key),
                    "image_size": image_size,
                    "mask_size": mask_size,
                }
            )

    positive_pixel_ratio = (
        positive_pixel_count / total_pixel_count
        if total_pixel_count > 0
        else 0.0
    )

    summary = {
        "image_count": len(image_files),
        "mask_count": len(mask_files),
        "paired_count": len(paired_keys),
        "oil_image_count": oil_mask_count,
        "empty_mask_count": empty_mask_count,
        "oil_image_percentage": (
            oil_mask_count / len(mask_files) * 100
            if mask_files
            else 0.0
        ),
        "empty_mask_percentage": (
            empty_mask_count / len(mask_files) * 100
            if mask_files
            else 0.0
        ),
        "positive_pixel_count": positive_pixel_count,
        "total_pixel_count": total_pixel_count,
        "positive_pixel_percentage": positive_pixel_ratio * 100,
        "image_split_counts": dict(image_split_counts),
        "mask_split_counts": dict(mask_split_counts),
        "sensor_counts": dict(sensor_counts),
        "image_size_counts": dict(image_size_counts),
        "mask_size_counts": dict(mask_size_counts),
        "image_mode_counts": dict(image_mode_counts),
        "mask_mode_counts": dict(mask_mode_counts),
        "mask_value_patterns": dict(mask_value_patterns),
        "missing_mask_count": len(missing_masks),
        "missing_image_count": len(missing_images),
        "duplicate_image_count": len(duplicate_images),
        "duplicate_mask_count": len(duplicate_masks),
        "size_mismatch_count": len(size_mismatches),
        "missing_masks": [str(key) for key in missing_masks[:20]],
        "missing_images": [str(key) for key in missing_images[:20]],
        "size_mismatches": size_mismatches[:20],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print()
    print("=" * 60)
    print("VERİ SETİ ÖZETİ")
    print("=" * 60)

    print(f"Gerçek görüntü sayısı : {len(image_files)}")
    print(f"Gerçek maske sayısı   : {len(mask_files)}")
    print(f"Eşleşen çift sayısı   : {len(paired_keys)}")

    print()
    print("Görüntü split dağılımı:", dict(image_split_counts))
    print("Maske split dağılımı  :", dict(mask_split_counts))
    print("Sensör dağılımı       :", dict(sensor_counts))

    print()
    print(f"Petrol bulunan görüntü : {oil_mask_count}")
    print(f"Tamamen petrolsüz      : {empty_mask_count}")
    print(
        f"Petrol bulunan oranı   : "
        f"{summary['oil_image_percentage']:.2f}%"
    )
    print(
        f"Petrolsüz görüntü oranı: "
        f"{summary['empty_mask_percentage']:.2f}%"
    )

    print()
    print(
        f"Petrol piksel oranı    : "
        f"{summary['positive_pixel_percentage']:.4f}%"
    )

    print()
    print("Görüntü boyutları:", dict(image_size_counts))
    print("Maske boyutları  :", dict(mask_size_counts))
    print("Görüntü modları  :", dict(image_mode_counts))
    print("Maske modları    :", dict(mask_mode_counts))
    print("Maske değerleri  :", dict(mask_value_patterns))

    print()
    print(f"Eksik maske sayısı       : {len(missing_masks)}")
    print(f"Eksik görüntü sayısı     : {len(missing_images)}")
    print(f"Tekrarlanan görüntü adı  : {len(duplicate_images)}")
    print(f"Tekrarlanan maske adı    : {len(duplicate_masks)}")
    print(f"Boyutu uyuşmayan çift    : {len(size_mismatches)}")

    print()
    print(f"JSON raporu kaydedildi: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
