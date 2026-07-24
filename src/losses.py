from __future__ import annotations

import torch
from torch import nn


class DiceLoss(nn.Module):
    """
    İkili segmentasyon için Dice Loss.

    Girdi:
        logits:  [B, 1, H, W]
        targets: [B, 1, H, W], değerler 0 veya 1

    Çıktı:
        Tek bir loss değeri
    """

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        if logits.shape != targets.shape:
            raise ValueError(
                "Logit ve hedef boyutlari ayni olmali: "
                f"logits={tuple(logits.shape)}, "
                f"targets={tuple(targets.shape)}"
            )

        probabilities = torch.sigmoid(logits)

        # Her görüntünün kanal, yükseklik ve genişlik
        # boyutları üzerinde toplam alıyoruz.
        dimensions = (1, 2, 3)

        intersection = (
            probabilities * targets
        ).sum(dim=dimensions)

        denominator = (
            probabilities.sum(dim=dimensions)
            + targets.sum(dim=dimensions)
        )

        dice_score = (
            2.0 * intersection + self.smooth
        ) / (
            denominator + self.smooth
        )

        return 1.0 - dice_score.mean()


class BCEDiceLoss(nn.Module):
    """
    BCEWithLogitsLoss ve Dice Loss birleşimi.
    """

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
    ) -> None:
        super().__init__()

        if bce_weight < 0 or dice_weight < 0:
            raise ValueError("Loss agirliklari negatif olamaz.")

        if bce_weight + dice_weight == 0:
            raise ValueError(
                "En az bir loss agirligi sifirdan buyuk olmali."
            )

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()

    def components(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Toplam, BCE ve Dice loss değerlerini ayrı ayrı döndürür.
        """
        bce = self.bce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)

        total = (
            self.bce_weight * bce
            + self.dice_weight * dice
        )

        return total, bce, dice

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        total, _, _ = self.components(logits, targets)
        return total
