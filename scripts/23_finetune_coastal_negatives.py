from __future__ import annotations

import argparse
import json
import random
import runpy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import (
    ConcatDataset,
    DataLoader,
    Dataset,
    WeightedRandomSampler,
)


ROOT = Path(__file__).resolve().parents[1]

SOURCE_SCRIPT = (
    ROOT
    / "scripts"
    / "18_finetune_hard_negatives.py"
)

SOS_MANIFEST = (
    ROOT
    / "data"
    / "metadata"
    / "dataset_manifest.csv"
)

DARTIS_MANIFEST = (
    ROOT
    / "data"
    / "metadata"
    / "dartis_no_oil_scene_split.csv"
)

DARTIS_RAW = (
    ROOT
    / "data"
    / "external"
    / "dartis"
    / "raw"
)

DEFAULT_CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "hard_negative_finetune"
    / "best_balanced.pth"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-name",
        default="coastal_hard_negative_finetune",
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.60,
    )

    parser.add_argument(
        "--nc-weight",
        type=float,
        default=8.0,
    )

    parser.add_argument(
        "--nw-weight",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--sos-weight",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--crop-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class CoastalNegativeDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        crop_size: int,
    ) -> None:
        self.dataframe = dataframe.reset_index(
            drop=True
        )

        self.crop_size = crop_size

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, torch.Tensor]:
        row = self.dataframe.iloc[index]

        image_path = (
            DARTIS_RAW
            / str(row["image_set"])
            / str(row["image_name"])
        )

        with Image.open(image_path) as image:
            image = image.convert("L")

            array = (
                np.asarray(
                    image,
                    dtype=np.float32,
                )
                / 255.0
            )

        height, width = array.shape

        if (
            height < self.crop_size
            or width < self.crop_size
        ):
            resized = Image.fromarray(
                (array * 255).astype(
                    np.uint8
                )
            ).resize(
                (
                    self.crop_size,
                    self.crop_size,
                ),
                Image.Resampling.BILINEAR,
            )

            crop = (
                np.asarray(
                    resized,
                    dtype=np.float32,
                )
                / 255.0
            )

        else:
            top = random.randint(
                0,
                height - self.crop_size,
            )

            left = random.randint(
                0,
                width - self.crop_size,
            )

            crop = array[
                top : top + self.crop_size,
                left : left + self.crop_size,
            ]

        image_tensor = torch.from_numpy(
            crop.copy()
        ).unsqueeze(0)

        return {
            "image": image_tensor,
            "mask": torch.zeros_like(
                image_tensor
            ),
        }


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    metrics: dict,
    base_channels: int,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": (
                model.state_dict()
            ),
            "optimizer_state_dict": (
                optimizer.state_dict()
            ),
            "args": {
                "base_channels": (
                    base_channels
                ),
                "learning_rate": (
                    args.learning_rate
                ),
                "threshold": (
                    args.threshold
                ),
                "nc_weight": (
                    args.nc_weight
                ),
                "nw_weight": (
                    args.nw_weight
                ),
                "source_checkpoint": (
                    str(args.checkpoint)
                ),
            },
            "metrics": metrics,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if not SOURCE_SCRIPT.exists():
        raise FileNotFoundError(
            f"Script 18 bulunamadı: "
            f"{SOURCE_SCRIPT}"
        )

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint bulunamadı: "
            f"{args.checkpoint}"
        )

    source = runpy.run_path(
        str(SOURCE_SCRIPT)
    )

    OilSpillDataset = source[
        "OilSpillDataset"
    ]

    ImageMaskOnlyDataset = source[
        "ImageMaskOnlyDataset"
    ]

    UNet = source["UNet"]
    BCEDiceLoss = source["BCEDiceLoss"]
    evaluate_sos = source["evaluate_sos"]
    evaluate_dartis = source[
        "evaluate_dartis"
    ]

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )

    checkpoint_args = checkpoint.get(
        "args",
        {},
    )

    base_channels = int(
        checkpoint_args.get(
            "base_channels",
            16,
        )
    )

    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=base_channels,
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    loss_function = BCEDiceLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
    )

    sos_train_base = OilSpillDataset(
        manifest_path=SOS_MANIFEST,
        split="train",
    )

    sos_train = ImageMaskOnlyDataset(
        sos_train_base
    )

    sos_val = OilSpillDataset(
        manifest_path=SOS_MANIFEST,
        split="val",
    )

    dartis_dataframe = pd.read_csv(
        DARTIS_MANIFEST,
        encoding="utf-8-sig",
    )

    dartis_train_frame = (
        dartis_dataframe[
            dartis_dataframe["split"]
            == "hard_negative_train"
        ]
        .reset_index(drop=True)
    )

    dartis_val_frame = (
        dartis_dataframe[
            dartis_dataframe["split"]
            == "hard_negative_val"
        ]
        .reset_index(drop=True)
    )

    coastal_train = CoastalNegativeDataset(
        dataframe=dartis_train_frame,
        crop_size=args.crop_size,
    )

    mixed_train = ConcatDataset(
        [
            sos_train,
            coastal_train,
        ]
    )

    sample_weights = [
        args.sos_weight
    ] * len(sos_train)

    for row in dartis_train_frame.itertuples(
        index=False
    ):
        if str(row.image_set) == "nc":
            sample_weights.append(
                args.nc_weight
            )
        else:
            sample_weights.append(
                args.nw_weight
            )

    weighted_sample_count = int(
        len(sos_train)
        + (
            (
                dartis_train_frame[
                    "image_set"
                ]
                == "nc"
            ).sum()
            * args.nc_weight
        )
        + (
            (
                dartis_train_frame[
                    "image_set"
                ]
                == "nw"
            ).sum()
            * args.nw_weight
        )
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)

    sampler = WeightedRandomSampler(
        weights=torch.tensor(
            sample_weights,
            dtype=torch.double,
        ),
        num_samples=weighted_sample_count,
        replacement=True,
        generator=generator,
    )

    train_loader = DataLoader(
        mixed_train,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    sos_val_loader = DataLoader(
        sos_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    checkpoint_dir = (
        ROOT
        / "checkpoints"
        / args.run_name
    )

    output_dir = (
        ROOT
        / "outputs"
        / args.run_name
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    nc_count = int(
        (
            dartis_train_frame[
                "image_set"
            ]
            == "nc"
        ).sum()
    )

    nw_count = int(
        (
            dartis_train_frame[
                "image_set"
            ]
            == "nw"
        ).sum()
    )

    print("=" * 78)
    print(
        "COASTAL HARD-NEGATIVE FINE-TUNING"
    )
    print("=" * 78)
    print("Cihaz:", device)
    print(
        "Başlangıç checkpoint:",
        args.checkpoint,
    )
    print("SOS train:", len(sos_train))
    print("NC train:", nc_count)
    print("NW train:", nw_count)
    print("NC ağırlığı:", args.nc_weight)
    print("NW ağırlığı:", args.nw_weight)
    print(
        "Epoch başına örnek:",
        weighted_sample_count,
    )
    print("SOS validation:", len(sos_val))
    print(
        "DARTIS validation:",
        len(dartis_val_frame),
    )
    print(
        "External test kullanılmıyor."
    )

    history = []
    best_score = float("-inf")

    total_start = time.time()

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        model.train()

        epoch_start = time.time()
        loss_sum = 0.0
        step_count = 0

        for batch in train_loader:
            images = batch["image"].to(
                device,
                non_blocking=True,
            )

            masks = batch["mask"].to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(images)

            loss = loss_function(
                logits,
                masks,
            )

            loss.backward()
            optimizer.step()

            loss_sum += float(
                loss.item()
            )

            step_count += 1

        train_loss = (
            loss_sum / step_count
        )

        sos_metrics = evaluate_sos(
            model=model,
            loader=sos_val_loader,
            device=device,
            threshold=args.threshold,
        )

        dartis_metrics = evaluate_dartis(
            model=model,
            frame=dartis_val_frame,
            device=device,
            threshold=args.threshold,
            max_images=0,
        )

        overall = dartis_metrics[
            "overall"
        ]

        nc_metrics = dartis_metrics["nc"]
        nw_metrics = dartis_metrics["nw"]

        overall_severe = (
            overall[
                "alarm_ge_1_percentage"
            ]
            / 100.0
        )

        nc_severe = (
            nc_metrics[
                "alarm_ge_1_percentage"
            ]
            / 100.0
        )

        recall_shortfall = max(
            0.0,
            0.78
            - sos_metrics["recall"],
        )

        balanced_score = (
            sos_metrics["dice"]
            * (1.0 - overall_severe)
            * (
                1.0
                - 0.5 * nc_severe
            )
            - 2.0 * recall_shortfall
        )

        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "sos_validation": (
                sos_metrics
            ),
            "dartis_validation": (
                dartis_metrics
            ),
            "balanced_score": (
                balanced_score
            ),
        }

        history.append(metrics)

        save_checkpoint(
            checkpoint_dir / "last.pth",
            model,
            optimizer,
            epoch,
            args,
            metrics,
            base_channels,
        )

        is_best = (
            balanced_score > best_score
        )

        if is_best:
            best_score = balanced_score

            save_checkpoint(
                checkpoint_dir
                / "best_balanced.pth",
                model,
                optimizer,
                epoch,
                args,
                metrics,
                base_channels,
            )

        print()
        print(
            f"Epoch {epoch:02d}/"
            f"{args.epochs} | "
            f"{time.time() - epoch_start:.1f} sn"
        )

        print(
            f"Train loss: "
            f"{train_loss:.4f}"
        )

        print(
            "SOS | "
            f"Dice {sos_metrics['dice']:.4f} | "
            f"Recall {sos_metrics['recall']:.4f}"
        )

        print(
            "DARTIS overall | "
            f">=1% "
            f"%{overall['alarm_ge_1_percentage']:.2f} | "
            f"FP pixel "
            f"%{overall['false_positive_pixel_percentage']:.4f}"
        )

        print(
            "DARTIS NC | "
            f">=1% "
            f"%{nc_metrics['alarm_ge_1_percentage']:.2f} | "
            f"FP pixel "
            f"%{nc_metrics['false_positive_pixel_percentage']:.4f}"
        )

        print(
            "DARTIS NW | "
            f">=1% "
            f"%{nw_metrics['alarm_ge_1_percentage']:.2f} | "
            f"FP pixel "
            f"%{nw_metrics['false_positive_pixel_percentage']:.4f}"
        )

        print(
            f"Balanced score: "
            f"{balanced_score:.4f}"
        )

        if is_best:
            print(
                "Yeni en iyi coastal checkpoint."
            )

    with (
        output_dir / "history.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            history,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 78)
    print(
        "COASTAL FINE-TUNING TAMAMLANDI"
    )
    print("=" * 78)
    print(
        f"Toplam süre: "
        f"{time.time() - total_start:.1f} sn"
    )
    print(
        f"En iyi skor: {best_score:.4f}"
    )
    print(
        "Checkpoint:",
        checkpoint_dir.resolve(),
    )


if __name__ == "__main__":
    main()

