from __future__ import annotations

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


IMAGE_ROOT = Path("data/raw/images")
MASK_ROOT = Path("data/raw/masks")
OUTPUT_PATH = Path("outputs/figures/image_mask_samples.png")

OIL_SAMPLE_COUNT = 4
EMPTY_SAMPLE_COUNT = 2
RANDOM_SEED = 42


def is_real_png(path: Path) -> bool:
    """macOS yardımcı dosyalarını dışarıda bırakır."""
    return (
        path.is_file()
        and path.suffix.lower() == ".png"
        and not path.name.startswith("._")
    )


def find_split(path: Path) -> str:
    """Dosyanın train veya val grubunu bulur."""
    parts = [part.lower() for part in path.parts]

    if "train" in parts:
        return "train"

    if "val" in parts or "valid" in parts or "validation" in parts:
        return "val"

    return "unknown"


def create_key(path: Path) -> tuple[str, str]:
    """
    Görüntü ve maskeleri eşleştirmek için anahtar üretir.

    Örnek:
    train/palsar_1007.png -> ("train", "palsar_1007")
    """
    return find_split(path), path.stem.lower()


def load_grayscale(path: Path) -> np.ndarray:
    """Görüntüyü tek kanallı NumPy dizisi olarak açar."""
    with Image.open(path) as image:
        return np.asarray(image.convert("L"))


def create_overlay(
    image_array: np.ndarray,
    binary_mask: np.ndarray,
) -> np.ndarray:
    """Maskeyi SAR görüntüsünün üzerine yarı saydam kırmızı olarak ekler."""
    image_float = image_array.astype(np.float32)

    minimum = float(image_float.min())
    maximum = float(image_float.max())

    if maximum > minimum:
        normalized = (image_float - minimum) / (maximum - minimum)
    else:
        normalized = np.zeros_like(image_float)

    grayscale = np.stack(
        [normalized, normalized, normalized],
        axis=-1,
    )

    red_layer = np.zeros_like(grayscale)
    red_layer[..., 0] = 1.0

    alpha = 0.45

    overlay = grayscale.copy()
    overlay[binary_mask] = (
        (1 - alpha) * grayscale[binary_mask]
        + alpha * red_layer[binary_mask]
    )

    return overlay

def main() -> None:
    image_files = [
        path
        for path in IMAGE_ROOT.rglob("*.png")
        if is_real_png(path)
    ]

    mask_files = [
        path
        for path in MASK_ROOT.rglob("*.png")
        if is_real_png(path)
    ]

    image_index = {
        create_key(path): path
        for path in image_files
    }

    mask_index = {
        create_key(path): path
        for path in mask_files
    }

    paired_keys = sorted(set(image_index) & set(mask_index))

    oil_keys: list[tuple[str, str]] = []
    empty_keys: list[tuple[str, str]] = []

    print("Maskeler taranıyor...")

    for key in paired_keys:
        mask_array = load_grayscale(mask_index[key])

        if np.any(mask_array > 0):
            oil_keys.append(key)
        else:
            empty_keys.append(key)

    if len(oil_keys) < OIL_SAMPLE_COUNT:
        raise RuntimeError("Yeterli petrol içeren örnek bulunamadı.")

    if len(empty_keys) < EMPTY_SAMPLE_COUNT:
        raise RuntimeError("Yeterli petrolsüz örnek bulunamadı.")

    random_generator = random.Random(RANDOM_SEED)

    selected_keys = (
        random_generator.sample(oil_keys, OIL_SAMPLE_COUNT)
        + random_generator.sample(empty_keys, EMPTY_SAMPLE_COUNT)
    )

    row_count = len(selected_keys)

    figure, axes = plt.subplots(
        row_count,
        3,
        figsize=(12, 3.6 * row_count),
        squeeze=False,
    )

    print()
    print("Seçilen örnekler:")

    for row_index, key in enumerate(selected_keys):
        image_path = image_index[key]
        mask_path = mask_index[key]

        image_array = load_grayscale(image_path)
        mask_array = load_grayscale(mask_path)

        # Şimdilik sıfırdan büyük bütün maske değerlerini pozitif gösteriyoruz.
        binary_mask = mask_array > 0

        overlay = create_overlay(image_array, binary_mask)

        oil_pixel_count = int(np.count_nonzero(binary_mask))
        total_pixel_count = int(binary_mask.size)
        oil_ratio = oil_pixel_count / total_pixel_count * 100

        sample_type = (
            "Petrol içeriyor"
            if oil_pixel_count > 0
            else "Petrolsüz"
        )

        split_name, sample_name = key

        print(
            f"- {split_name}/{sample_name}: "
            f"{sample_type}, petrol piksel oranı %{oil_ratio:.2f}"
        )

        axes[row_index, 0].imshow(image_array, cmap="gray")
        axes[row_index, 0].set_title(
            f"SAR görüntüsü\n{split_name}/{sample_name}"
        )
        axes[row_index, 0].axis("off")

        axes[row_index, 1].imshow(binary_mask, cmap="gray")
        axes[row_index, 1].set_title(
            f"Ground-truth maske\nPetrol: %{oil_ratio:.2f}"
        )
        axes[row_index, 1].axis("off")

        axes[row_index, 2].imshow(overlay)
        axes[row_index, 2].set_title(
            f"Overlay\n{sample_type}"
        )
        axes[row_index, 2].axis("off")

    figure.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=160, bbox_inches="tight")
    plt.close(figure)

    print()
    print(f"Görsel kaydedildi: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
