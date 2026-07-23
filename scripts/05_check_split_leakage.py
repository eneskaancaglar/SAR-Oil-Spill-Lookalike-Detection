from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


MANIFEST_PATH = Path("data/metadata/dataset_manifest.csv")
OUTPUT_PATH = Path("outputs/split_leakage_report.json")


def hash_array(array: np.ndarray) -> str:
    """Piksel dizisinin SHA-256 özetini üretir."""
    hasher = hashlib.sha256()
    hasher.update(str(array.shape).encode("utf-8"))
    hasher.update(str(array.dtype).encode("utf-8"))
    hasher.update(array.tobytes())
    return hasher.hexdigest()


def load_image_hash(path: Path) -> str:
    """Gri görüntünün piksel hash'ini hesaplar."""
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"), dtype=np.uint8)

    return hash_array(array)


def load_mask_hash(path: Path) -> str:
    """Maskeyi 0/1 yaptıktan sonra piksel hash'ini hesaplar."""
    with Image.open(path) as mask:
        array = np.asarray(mask.convert("L"), dtype=np.uint8)

    binary = (array > 0).astype(np.uint8)
    return hash_array(binary)


def main() -> None:
    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    split_names: dict[str, set[str]] = defaultdict(set)
    image_hashes: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )
    pair_hashes: dict[str, dict[str, list[str]]] = defaultdict(
        lambda: defaultdict(list)
    )

    print(f"{len(rows)} örnek kontrol ediliyor...")

    for index, row in enumerate(rows, start=1):
        split = row["split"].lower()
        sample_name = row["sample_name"]

        image_path = Path(row["image_path"])
        mask_path = Path(row["mask_path"])

        image_hash = load_image_hash(image_path)
        mask_hash = load_mask_hash(mask_path)

        pair_hasher = hashlib.sha256()
        pair_hasher.update(image_hash.encode("utf-8"))
        pair_hasher.update(mask_hash.encode("utf-8"))
        pair_hash = pair_hasher.hexdigest()

        split_names[split].add(sample_name)
        image_hashes[split][image_hash].append(sample_name)
        pair_hashes[split][pair_hash].append(sample_name)

        if index % 1000 == 0:
            print(f"{index}/{len(rows)} tamamlandı.")

    train_names = split_names.get("train", set())
    val_names = split_names.get("val", set())

    train_image_hashes = set(image_hashes.get("train", {}))
    val_image_hashes = set(image_hashes.get("val", {}))

    train_pair_hashes = set(pair_hashes.get("train", {}))
    val_pair_hashes = set(pair_hashes.get("val", {}))

    overlapping_names = sorted(train_names & val_names)
    duplicate_image_hashes = sorted(
        train_image_hashes & val_image_hashes
    )
    duplicate_pair_hashes = sorted(
        train_pair_hashes & val_pair_hashes
    )

    duplicate_image_examples = []

    for digest in duplicate_image_hashes[:20]:
        duplicate_image_examples.append(
            {
                "train": image_hashes["train"][digest],
                "val": image_hashes["val"][digest],
            }
        )

    duplicate_pair_examples = []

    for digest in duplicate_pair_hashes[:20]:
        duplicate_pair_examples.append(
            {
                "train": pair_hashes["train"][digest],
                "val": pair_hashes["val"][digest],
            }
        )

    report = {
        "train_count": sum(
            1 for row in rows if row["split"].lower() == "train"
        ),
        "val_count": sum(
            1 for row in rows if row["split"].lower() == "val"
        ),
        "same_name_across_splits_count": len(overlapping_names),
        "exact_duplicate_image_count": len(duplicate_image_hashes),
        "exact_duplicate_image_mask_pair_count": len(
            duplicate_pair_hashes
        ),
        "same_name_examples": overlapping_names[:20],
        "duplicate_image_examples": duplicate_image_examples,
        "duplicate_pair_examples": duplicate_pair_examples,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print()
    print("=" * 60)
    print("TRAIN–VALIDATION LEAKAGE KONTROLÜ")
    print("=" * 60)
    print(f"Train örnek sayısı: {report['train_count']}")
    print(f"Val örnek sayısı:   {report['val_count']}")
    print()
    print(
        "Her iki split'te aynı isim: "
        f"{report['same_name_across_splits_count']}"
    )
    print(
        "Birebir aynı görüntü:       "
        f"{report['exact_duplicate_image_count']}"
    )
    print(
        "Birebir aynı image-mask:    "
        f"{report['exact_duplicate_image_mask_pair_count']}"
    )
    print()
    print(f"Rapor: {OUTPUT_PATH.resolve()}")

    if duplicate_pair_hashes:
        print()
        print("UYARI: Train ve validation arasında birebir aynı")
        print("görüntü-maske çiftleri bulundu.")
    elif duplicate_image_hashes:
        print()
        print("UYARI: Train ve validation arasında birebir aynı")
        print("görüntüler bulundu.")
    else:
        print()
        print("Temel birebir kopya kontrolünde leakage bulunmadı.")


if __name__ == "__main__":
    main()
