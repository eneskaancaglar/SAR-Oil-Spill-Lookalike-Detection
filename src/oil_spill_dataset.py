from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class OilSpillDataset(Dataset):
    """
    SAR görüntüsü ve ona ait petrol maskesini yükleyen Dataset sınıfı.

    Her örneğin çıktısı:
        image: [1, H, W] float32 Tensor
        mask:  [1, H, W] float32 Tensor
        metadata: örneğe ait açıklayıcı bilgiler
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.split = split.lower()

        if not self.manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest bulunamadi: {self.manifest_path}"
            )

        self.rows = self._read_manifest()

        if not self.rows:
            raise RuntimeError(
                f"Manifest icinde '{self.split}' split'i bulunamadi."
            )

    def _read_manifest(self) -> list[dict[str, str]]:
        """Manifest dosyasından seçilen split'e ait satırları okur."""
        with self.manifest_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            reader = csv.DictReader(file)

            rows = [
                row
                for row in reader
                if row["split"].lower() == self.split
            ]

        return rows

    def __len__(self) -> int:
        """Dataset içinde kaç örnek olduğunu döndürür."""
        return len(self.rows)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, Any]:
        """
        Belirtilen indeksteki görüntü ve maskeyi yükler.

        Görüntü:
            0-255 uint8 -> 0-1 float32

        Maske:
            0'dan büyük değerler -> 1
            0 değerleri          -> 0
        """
        row = self.rows[index]

        image_path = Path(row["image_path"])
        mask_path = Path(row["mask_path"])

        if not image_path.exists():
            raise FileNotFoundError(
                f"Goruntu bulunamadi: {image_path}"
            )

        if not mask_path.exists():
            raise FileNotFoundError(
                f"Maske bulunamadi: {mask_path}"
            )

        with Image.open(image_path) as image:
            image_array = np.asarray(
                image.convert("L"),
                dtype=np.float32,
            )

        with Image.open(mask_path) as mask:
            mask_array = np.asarray(
                mask.convert("L"),
                dtype=np.uint8,
            )

        if image_array.shape != mask_array.shape:
            raise RuntimeError(
                "Goruntu ve maske boyutlari farkli: "
                f"{image_path.name}={image_array.shape}, "
                f"{mask_path.name}={mask_array.shape}"
            )

        # 0-255 aralığındaki görüntüyü 0-1 aralığına getirir.
        image_array = image_array / 255.0

        # Maskeyi kesin olarak iki sınıfa dönüştürür.
        binary_mask = (mask_array > 0).astype(np.float32)

        # NumPy: [H, W]
        # PyTorch: [C, H, W]
        #
        # Tek kanallı olduğu için başa kanal boyutu ekliyoruz.
        image_tensor = torch.from_numpy(
            image_array
        ).unsqueeze(0)

        mask_tensor = torch.from_numpy(
            binary_mask
        ).unsqueeze(0)

        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "metadata": {
                "sample_name": row["sample_name"],
                "split": row["split"],
                "sensor": row["sensor"],
                "has_oil": int(row["has_oil"]),
                "oil_pixel_ratio": float(
                    row["oil_pixel_ratio"]
                ),
                "image_path": str(image_path),
                "mask_path": str(mask_path),
            },
        }
