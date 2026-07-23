from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from oil_spill_dataset import OilSpillDataset


MANIFEST_PATH = PROJECT_ROOT / "data/metadata/dataset_manifest.csv"


def inspect_dataset(
    dataset: OilSpillDataset,
    dataset_name: str,
) -> None:
    """Dataset hakkında temel bilgileri yazdırır."""
    print()
    print("=" * 60)
    print(dataset_name)
    print("=" * 60)

    print(f"Ornek sayisi: {len(dataset)}")

    sample = dataset[0]

    image = sample["image"]
    mask = sample["mask"]
    metadata = sample["metadata"]

    print()
    print("Ilk ornek:")
    print("Goruntu shape:", tuple(image.shape))
    print("Maske shape:  ", tuple(mask.shape))
    print("Goruntu dtype:", image.dtype)
    print("Maske dtype:  ", mask.dtype)
    print("Goruntu min:  ", float(image.min()))
    print("Goruntu max:  ", float(image.max()))
    print(
        "Maske degerleri:",
        torch.unique(mask).tolist(),
    )
    print("Metadata:", metadata)


def main() -> None:
    train_dataset = OilSpillDataset(
        manifest_path=MANIFEST_PATH,
        split="train",
    )

    val_dataset = OilSpillDataset(
        manifest_path=MANIFEST_PATH,
        split="val",
    )

    inspect_dataset(
        dataset=train_dataset,
        dataset_name="TRAIN DATASET",
    )

    inspect_dataset(
        dataset=val_dataset,
        dataset_name="VALIDATION DATASET",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    batch = next(iter(train_loader))

    images = batch["image"]
    masks = batch["mask"]

    print()
    print("=" * 60)
    print("DATALOADER BATCH KONTROLU")
    print("=" * 60)

    print("Batch goruntu shape:", tuple(images.shape))
    print("Batch maske shape:  ", tuple(masks.shape))
    print("Batch dtype:        ", images.dtype)
    print("Maske degerleri:    ", torch.unique(masks).tolist())

    print()
    print("Beklenen batch goruntu şekli:")
    print("[batch, kanal, yukseklik, genislik]")
    print("[4, 1, 256, 256]")


if __name__ == "__main__":
    main()
