from pathlib import Path
from collections import defaultdict
import csv

import numpy as np
from PIL import Image


IMAGE_ROOT = Path("data/raw/images")
MASK_ROOT = Path("data/raw/masks")
OUTPUT_PATH = Path("data/metadata/dataset_manifest.csv")


def valid_png(path: Path) -> bool:
    return (
        path.is_file()
        and path.suffix.lower() == ".png"
        and not path.name.startswith("._")
    )


def find_split(path: Path) -> str:
    parts = [part.lower() for part in path.parts]

    if "train" in parts:
        return "train"

    if "val" in parts or "valid" in parts:
        return "val"

    return "unknown"


def find_sensor(path: Path) -> str:
    name = path.stem.lower()

    if name.startswith("sentinel"):
        return "Sentinel-1"

    if name.startswith("palsar"):
        return "PALSAR"

    return "Other"


def create_index(root: Path) -> dict[tuple[str, str], Path]:
    grouped: dict[tuple[str, str], list[Path]] = defaultdict(list)

    for path in root.rglob("*.png"):
        if valid_png(path):
            key = (find_split(path), path.stem.lower())
            grouped[key].append(path)

    duplicate_keys = [
        key for key, paths in grouped.items()
        if len(paths) != 1
    ]

    if duplicate_keys:
        raise RuntimeError(
            f"Tekrarlanan dosya anahtarlari bulundu: {duplicate_keys[:10]}"
        )

    return {
        key: paths[0]
        for key, paths in grouped.items()
    }


def main() -> None:
    image_index = create_index(IMAGE_ROOT)
    mask_index = create_index(MASK_ROOT)

    image_keys = set(image_index)
    mask_keys = set(mask_index)

    if image_keys != mask_keys:
        raise RuntimeError(
            "Goruntu ve maske eslesmeleri ayni degil. "
            f"Eksik maske: {len(image_keys - mask_keys)}, "
            f"Eksik goruntu: {len(mask_keys - image_keys)}"
        )

    rows = []

    for split, sample_name in sorted(image_keys):
        image_path = image_index[(split, sample_name)]
        mask_path = mask_index[(split, sample_name)]

        with Image.open(image_path) as image:
            width, height = image.size

        with Image.open(mask_path) as mask:
            mask_array = np.asarray(mask.convert("L"))

        binary_mask = mask_array > 0

        oil_pixels = int(np.count_nonzero(binary_mask))
        total_pixels = int(binary_mask.size)
        oil_ratio = oil_pixels / total_pixels * 100

        rows.append(
            {
                "source_dataset": "refined_sos",
                "sample_name": sample_name,
                "split": split,
                "sensor": find_sensor(image_path),
                "image_path": image_path.as_posix(),
                "mask_path": mask_path.as_posix(),
                "width": width,
                "height": height,
                "has_oil": int(oil_pixels > 0),
                "oil_pixel_ratio": f"{oil_ratio:.6f}",
            }
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "source_dataset",
        "sample_name",
        "split",
        "sensor",
        "image_path",
        "mask_path",
        "width",
        "height",
        "has_oil",
        "oil_pixel_ratio",
    ]

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    oil_count = sum(int(row["has_oil"]) for row in rows)
    no_oil_count = len(rows) - oil_count

    print("Manifest olusturuldu.")
    print(f"Toplam ornek: {len(rows)}")
    print(f"Petrollu:     {oil_count}")
    print(f"Petrolsuz:    {no_oil_count}")
    print(f"Kayit yeri:   {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
