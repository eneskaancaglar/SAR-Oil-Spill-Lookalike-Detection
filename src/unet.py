from __future__ import annotations

import torch
from torch import nn


class DoubleConv(nn.Module):
    """
    Arka arkaya iki convolution işlemi uygular.

    Girdi:
        [B, in_channels, H, W]

    Çıktı:
        [B, out_channels, H, W]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DownBlock(nn.Module):
    """
    Görüntü özelliklerini iki kat küçültür ve kanal sayısını artırır.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UpBlock(nn.Module):
    """
    Özellik haritasını büyütür, encoder özelliğiyle birleştirir
    ve convolution uygular.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2,
        )

        # Büyütülen decoder özelliği ve encoder skip özelliği
        # kanal boyutunda birleştirileceği için giriş kanal sayısı
        # 2 * out_channels olur.
        self.conv = DoubleConv(
            in_channels=out_channels * 2,
            out_channels=out_channels,
        )

    def forward(
        self,
        decoder_input: torch.Tensor,
        encoder_skip: torch.Tensor,
    ) -> torch.Tensor:
        decoder_output = self.up(decoder_input)

        if decoder_output.shape[-2:] != encoder_skip.shape[-2:]:
            raise RuntimeError(
                "Decoder ve encoder boyutlari uyusmuyor: "
                f"decoder={tuple(decoder_output.shape)}, "
                f"encoder={tuple(encoder_skip.shape)}"
            )

        combined = torch.cat(
            [encoder_skip, decoder_output],
            dim=1,
        )

        return self.conv(combined)


class UNet(nn.Module):
    """
    Tek kanallı SAR görüntüsünden tek kanallı petrol logit haritası üretir.

    Varsayılan giriş:
        [B, 1, 256, 256]

    Çıkış:
        [B, 1, 256, 256]
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 16,
    ) -> None:
        super().__init__()

        self.input_block = DoubleConv(
            in_channels,
            base_channels,
        )

        self.down1 = DownBlock(
            base_channels,
            base_channels * 2,
        )

        self.down2 = DownBlock(
            base_channels * 2,
            base_channels * 4,
        )

        self.down3 = DownBlock(
            base_channels * 4,
            base_channels * 8,
        )

        self.down4 = DownBlock(
            base_channels * 8,
            base_channels * 16,
        )

        self.up1 = UpBlock(
            base_channels * 16,
            base_channels * 8,
        )

        self.up2 = UpBlock(
            base_channels * 8,
            base_channels * 4,
        )

        self.up3 = UpBlock(
            base_channels * 4,
            base_channels * 2,
        )

        self.up4 = UpBlock(
            base_channels * 2,
            base_channels,
        )

        # 1x1 convolution kanal sayısını 1'e indirir.
        self.output_layer = nn.Conv2d(
            in_channels=base_channels,
            out_channels=out_channels,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoder1 = self.input_block(x)   # [B, 16, 256, 256]
        encoder2 = self.down1(encoder1)  # [B, 32, 128, 128]
        encoder3 = self.down2(encoder2)  # [B, 64, 64, 64]
        encoder4 = self.down3(encoder3)  # [B, 128, 32, 32]
        bottleneck = self.down4(encoder4)  # [B, 256, 16, 16]

        decoder1 = self.up1(bottleneck, encoder4)
        decoder2 = self.up2(decoder1, encoder3)
        decoder3 = self.up3(decoder2, encoder2)
        decoder4 = self.up4(decoder3, encoder1)

        logits = self.output_layer(decoder4)

        return logits
