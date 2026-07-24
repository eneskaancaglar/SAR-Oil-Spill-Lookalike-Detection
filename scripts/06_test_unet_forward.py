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
from unet import UNet


MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "dataset_manifest.csv"
)


def count_parameters(model: torch.nn.Module) -> int:
    """Modelin öğrenilebilir parametre sayısını hesaplar."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def main() -> None:
    torch.manual_seed(42)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("U-NET ILERI GECIS TESTI")
    print("=" * 60)

    print("Cihaz:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    train_dataset = OilSpillDataset(
        manifest_path=MANIFEST_PATH,
        split="train",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
    )

    batch = next(iter(train_loader))

    images = batch["image"].to(device)
    masks = batch["mask"].to(device)

    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=16,
    ).to(device)

    model.eval()

    with torch.no_grad():
        logits = model(images)
        probabilities = torch.sigmoid(logits)
        predicted_masks = probabilities >= 0.5

    parameter_count = count_parameters(model)

    print()
    print("Model parametre sayisi:", f"{parameter_count:,}")

    print()
    print("Girdi görüntü shape:", tuple(images.shape))
    print("Gerçek maske shape:  ", tuple(masks.shape))
    print("Model logit shape:   ", tuple(logits.shape))
    print("Olasılık shape:      ", tuple(probabilities.shape))
    print("Tahmin maskesi shape:", tuple(predicted_masks.shape))

    print()
    print("Logit minimum:", float(logits.min()))
    print("Logit maksimum:", float(logits.max()))

    print()
    print("Olasılık minimum:", float(probabilities.min()))
    print("Olasılık maksimum:", float(probabilities.max()))

    print()
    print(
        "Gerçek maske değerleri:",
        torch.unique(masks).tolist(),
    )

    print(
        "Tahmin maskesi değerleri:",
        torch.unique(predicted_masks).tolist(),
    )

    print()
    print("Beklenen:")
    print("Girdi  : [2, 1, 256, 256]")
    print("Cikti  : [2, 1, 256, 256]")

    if logits.shape != masks.shape:
        raise RuntimeError(
            "Model çıktısı ile maske boyutu aynı değil."
        )

    print()
    print("BASARILI: U-Net görüntüleri kabul etti ve")
    print("maske ile aynı boyutta tahmin üretti.")

    if device.type == "cuda":
        peak_memory_mb = (
            torch.cuda.max_memory_allocated()
            / 1024**2
        )

        print(
            "GPU maksimum kullanılan bellek:",
            f"{peak_memory_mb:.2f} MB",
        )


if __name__ == "__main__":
    main()
