from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from losses import BCEDiceLoss
from oil_spill_dataset import OilSpillDataset
from unet import UNet


MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "metadata"
    / "dataset_manifest.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"

STEP_COUNT = 200
BATCH_SIZE = 2
LEARNING_RATE = 1e-3
RANDOM_SEED = 42


def set_random_seed(seed: int) -> None:
    """Deneyi mümkün olduğunca tekrar üretilebilir yapar."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def calculate_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, float]:
    """
    İkili tahminlerden Dice, IoU, precision ve recall hesaplar.
    """
    predictions = predictions.float()
    targets = targets.float()

    true_positive = float(
        (predictions * targets).sum().item()
    )

    predicted_positive = float(
        predictions.sum().item()
    )

    actual_positive = float(
        targets.sum().item()
    )

    false_positive = predicted_positive - true_positive
    false_negative = actual_positive - true_positive

    epsilon = 1e-7

    dice = (
        2.0 * true_positive + epsilon
    ) / (
        predicted_positive + actual_positive + epsilon
    )

    iou = (
        true_positive + epsilon
    ) / (
        true_positive
        + false_positive
        + false_negative
        + epsilon
    )

    precision = (
        true_positive + epsilon
    ) / (
        predicted_positive + epsilon
    )

    recall = (
        true_positive + epsilon
    ) / (
        actual_positive + epsilon
    )

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
    }


def get_batch_with_oil(
    data_loader: DataLoader,
) -> dict[str, object]:
    """
    En az bir petrol pikseli bulunan ilk batch'i seçer.
    """
    for batch in data_loader:
        masks = batch["mask"]

        if float(masks.sum()) > 0:
            return batch

    raise RuntimeError(
        "Petrol pikseli bulunan bir batch bulunamadi."
    )


def save_loss_figure(
    history: list[dict[str, float]],
) -> None:
    """Eğitim loss grafiğini kaydeder."""
    steps = [item["step"] for item in history]
    total_losses = [item["total_loss"] for item in history]
    bce_losses = [item["bce_loss"] for item in history]
    dice_losses = [item["dice_loss"] for item in history]

    figure = plt.figure(figsize=(9, 5))
    axis = figure.add_subplot(1, 1, 1)

    axis.plot(steps, total_losses, label="Total Loss")
    axis.plot(steps, bce_losses, label="BCE Loss")
    axis.plot(steps, dice_losses, label="Dice Loss")

    axis.set_title("One-batch overfitting testi")
    axis.set_xlabel("Egitim adimi")
    axis.set_ylabel("Loss")
    axis.grid(True, alpha=0.3)
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        FIGURE_DIR / "one_batch_overfit_loss.png",
        dpi=160,
    )
    plt.close(figure)


def save_prediction_figure(
    images: torch.Tensor,
    masks: torch.Tensor,
    probabilities: torch.Tensor,
) -> None:
    """Girdi, gerçek maske ve tahmin maskesini kaydeder."""
    images_cpu = images.detach().cpu().numpy()
    masks_cpu = masks.detach().cpu().numpy()
    probabilities_cpu = probabilities.detach().cpu().numpy()

    sample_count = min(2, images_cpu.shape[0])

    figure, axes = plt.subplots(
        sample_count,
        3,
        figsize=(11, 3.7 * sample_count),
        squeeze=False,
    )

    for index in range(sample_count):
        image = images_cpu[index, 0]
        ground_truth = masks_cpu[index, 0]
        probability = probabilities_cpu[index, 0]

        axes[index, 0].imshow(image, cmap="gray")
        axes[index, 0].set_title("SAR goruntusu")
        axes[index, 0].axis("off")

        axes[index, 1].imshow(ground_truth, cmap="gray")
        axes[index, 1].set_title("Gercek maske")
        axes[index, 1].axis("off")

        axes[index, 2].imshow(probability, cmap="gray")
        axes[index, 2].set_title("Tahmin olasiligi")
        axes[index, 2].axis("off")

    figure.tight_layout()
    figure.savefig(
        FIGURE_DIR / "one_batch_predictions.png",
        dpi=160,
    )
    plt.close(figure)


def main() -> None:
    set_random_seed(RANDOM_SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 65)
    print("ONE-BATCH OVERFITTING TESTI")
    print("=" * 65)
    print("Cihaz:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        torch.cuda.reset_peak_memory_stats()

    dataset = OilSpillDataset(
        manifest_path=MANIFEST_PATH,
        split="train",
    )

    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    batch = get_batch_with_oil(data_loader)

    images = batch["image"].to(device)
    masks = batch["mask"].to(device)

    print("Goruntu batch shape:", tuple(images.shape))
    print("Maske batch shape:  ", tuple(masks.shape))
    print("Petrol pikseli:     ", int(masks.sum().item()))

    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=16,
    ).to(device)

    loss_function = BCEDiceLoss(
        bce_weight=0.5,
        dice_weight=0.5,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    history: list[dict[str, float]] = []

    model.train()

    for step in range(1, STEP_COUNT + 1):
        # Önceki adımdan kalan gradyanları temizler.
        optimizer.zero_grad(set_to_none=True)

        # Forward pass: Model tahmini üretir.
        logits = model(images)

        # Tahmin ile gerçek maske arasındaki hata.
        total_loss, bce_loss, dice_loss = (
            loss_function.components(logits, masks)
        )

        # Backpropagation: Gradyanları hesaplar.
        total_loss.backward()

        # Model parametrelerini günceller.
        optimizer.step()

        history.append(
            {
                "step": step,
                "total_loss": float(total_loss.item()),
                "bce_loss": float(bce_loss.item()),
                "dice_loss": float(dice_loss.item()),
            }
        )

        if step == 1 or step % 10 == 0:
            print(
                f"Step {step:03d}/{STEP_COUNT} | "
                f"Total: {total_loss.item():.6f} | "
                f"BCE: {bce_loss.item():.6f} | "
                f"Dice: {dice_loss.item():.6f}"
            )

    model.eval()

    with torch.no_grad():
        final_logits = model(images)
        probabilities = torch.sigmoid(final_logits)
        predictions = probabilities >= 0.5

    metrics = calculate_metrics(
        predictions=predictions,
        targets=masks,
    )

    first_loss = history[0]["total_loss"]
    final_loss = history[-1]["total_loss"]

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    save_loss_figure(history)
    save_prediction_figure(
        images=images,
        masks=masks,
        probabilities=probabilities,
    )

    result = {
        "steps": STEP_COUNT,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "first_loss": first_loss,
        "final_loss": final_loss,
        "loss_reduction": first_loss - final_loss,
        "metrics": metrics,
        "history": history,
    }

    result_path = OUTPUT_DIR / "one_batch_overfit_result.json"

    with result_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    print()
    print("=" * 65)
    print("SONUC")
    print("=" * 65)
    print(f"Ilk loss:  {first_loss:.6f}")
    print(f"Son loss:  {final_loss:.6f}")
    print(f"Azalma:    {first_loss - final_loss:.6f}")

    print()
    print(f"Dice:      {metrics['dice']:.4f}")
    print(f"IoU:       {metrics['iou']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")

    print()
    print(
        "Loss grafigi:",
        (FIGURE_DIR / "one_batch_overfit_loss.png").resolve(),
    )

    print(
        "Tahmin gorseli:",
        (FIGURE_DIR / "one_batch_predictions.png").resolve(),
    )

    print("JSON sonucu:", result_path.resolve())

    if device.type == "cuda":
        peak_memory_mb = (
            torch.cuda.max_memory_allocated()
            / 1024**2
        )

        print(
            "GPU maksimum bellek:",
            f"{peak_memory_mb:.2f} MB",
        )

    if final_loss >= first_loss:
        print()
        print("UYARI: Loss azalmadi.")
        print("Model, loss veya veri hatti tekrar kontrol edilmeli.")
    else:
        print()
        print("BASARILI: Model ayni batch uzerinde loss'u azaltti.")


if __name__ == "__main__":
    main()
