from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.ndimage import label
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from oil_spill_dataset import OilSpillDataset
from unet import UNet


CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "hard_negative_finetune"
    / "best_balanced.pth"
)

SOS_MANIFEST = (
    ROOT
    / "data"
    / "metadata"
    / "dataset_manifest.csv"
)

DARTIS_VAL_MANIFEST = (
    ROOT
    / "data"
    / "metadata"
    / "dartis_hard_negative_val_manifest.csv"
)

DARTIS_RAW = (
    ROOT
    / "data"
    / "external"
    / "dartis"
    / "raw"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "postprocessing_validation_sweep"
)

THRESHOLD = 0.60


CONFIGURATIONS = [
    {
        "name": "raw",
        "min_component_ratio": 0.0,
        "border_remove_ratio": None,
    },
    {
        "name": "remove_tiny_005pct",
        "min_component_ratio": 0.0005,
        "border_remove_ratio": None,
    },
    {
        "name": "remove_border_ge_5pct",
        "min_component_ratio": 0.0,
        "border_remove_ratio": 0.05,
    },
    {
        "name": "remove_border_ge_2pct",
        "min_component_ratio": 0.0,
        "border_remove_ratio": 0.02,
    },
    {
        "name": "remove_border_ge_1pct",
        "min_component_ratio": 0.0,
        "border_remove_ratio": 0.01,
    },
    {
        "name": "border_2pct_plus_tiny",
        "min_component_ratio": 0.0005,
        "border_remove_ratio": 0.02,
    },
]


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> torch.nn.Module:
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    checkpoint_args = checkpoint.get(
        "args",
        {},
    )

    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=int(
            checkpoint_args.get(
                "base_channels",
                16,
            )
        ),
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()
    return model


def postprocess(
    binary: np.ndarray,
    min_component_ratio: float,
    border_remove_ratio: float | None,
) -> np.ndarray:
    binary = binary.astype(bool)

    labeled, component_count = label(
        binary,
        structure=np.ones(
            (3, 3),
            dtype=np.uint8,
        ),
    )

    if component_count == 0:
        return binary

    total_pixels = binary.size

    component_sizes = np.bincount(
        labeled.reshape(-1)
    )

    output = np.zeros_like(
        binary,
        dtype=bool,
    )

    for component_id in range(
        1,
        component_count + 1,
    ):
        component_size = int(
            component_sizes[component_id]
        )

        component_ratio = (
            component_size
            / total_pixels
        )

        if (
            component_ratio
            < min_component_ratio
        ):
            continue

        component_mask = (
            labeled == component_id
        )

        touches_border = bool(
            component_mask[0, :].any()
            or component_mask[-1, :].any()
            or component_mask[:, 0].any()
            or component_mask[:, -1].any()
        )

        remove_border_component = (
            border_remove_ratio is not None
            and touches_border
            and component_ratio
            >= border_remove_ratio
        )

        if remove_border_component:
            continue

        output[component_mask] = True

    return output


def create_sos_state() -> dict[str, int]:
    return {
        "tp": 0,
        "fp": 0,
        "fn": 0,
    }


def update_sos_state(
    state: dict[str, int],
    prediction: np.ndarray,
    target: np.ndarray,
) -> None:
    prediction = prediction.astype(bool)
    target = target.astype(bool)

    state["tp"] += int(
        np.logical_and(
            prediction,
            target,
        ).sum()
    )

    state["fp"] += int(
        np.logical_and(
            prediction,
            np.logical_not(target),
        ).sum()
    )

    state["fn"] += int(
        np.logical_and(
            np.logical_not(prediction),
            target,
        ).sum()
    )


def finalize_sos_state(
    state: dict[str, int],
) -> dict[str, float]:
    tp = float(state["tp"])
    fp = float(state["fp"])
    fn = float(state["fn"])

    epsilon = 1e-7

    return {
        "dice": (
            2.0 * tp + epsilon
        ) / (
            2.0 * tp
            + fp
            + fn
            + epsilon
        ),
        "iou": (
            tp + epsilon
        ) / (
            tp
            + fp
            + fn
            + epsilon
        ),
        "precision": (
            tp + epsilon
        ) / (
            tp
            + fp
            + epsilon
        ),
        "recall": (
            tp + epsilon
        ) / (
            tp
            + fn
            + epsilon
        ),
    }


def create_dartis_state() -> dict[str, int]:
    return {
        "image_count": 0,
        "any_alarm": 0,
        "alarm_ge_0_1": 0,
        "alarm_ge_1": 0,
        "positive_pixels": 0,
        "total_pixels": 0,
    }


def update_dartis_state(
    state: dict[str, int],
    prediction: np.ndarray,
) -> None:
    positive_pixels = int(
        prediction.sum()
    )

    total_pixels = int(
        prediction.size
    )

    positive_ratio = (
        positive_pixels
        / total_pixels
    )

    state["image_count"] += 1
    state["positive_pixels"] += (
        positive_pixels
    )
    state["total_pixels"] += total_pixels

    state["any_alarm"] += int(
        positive_pixels > 0
    )

    state["alarm_ge_0_1"] += int(
        positive_ratio >= 0.001
    )

    state["alarm_ge_1"] += int(
        positive_ratio >= 0.01
    )


def finalize_dartis_state(
    state: dict[str, int],
) -> dict[str, float]:
    image_count = state["image_count"]

    return {
        "any_alarm_percentage": (
            state["any_alarm"]
            / image_count
            * 100.0
        ),
        "alarm_ge_0_1_percentage": (
            state["alarm_ge_0_1"]
            / image_count
            * 100.0
        ),
        "alarm_ge_1_percentage": (
            state["alarm_ge_1"]
            / image_count
            * 100.0
        ),
        "false_positive_pixel_percentage": (
            state["positive_pixels"]
            / state["total_pixels"]
            * 100.0
        ),
    }


def apply_configuration(
    binary: np.ndarray,
    configuration: dict,
) -> np.ndarray:
    return postprocess(
        binary=binary,
        min_component_ratio=(
            configuration[
                "min_component_ratio"
            ]
        ),
        border_remove_ratio=(
            configuration[
                "border_remove_ratio"
            ]
        ),
    )


def evaluate_sos(
    model: torch.nn.Module,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    dataset = OilSpillDataset(
        manifest_path=SOS_MANIFEST,
        split="val",
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    states = {
        configuration["name"]:
        create_sos_state()
        for configuration
        in CONFIGURATIONS
    }

    print(
        f"SOS validation: {len(dataset)}"
    )

    with torch.no_grad():
        for index, batch in enumerate(
            loader,
            start=1,
        ):
            image = batch["image"].to(
                device
            )

            target = (
                batch["mask"][0, 0]
                .cpu()
                .numpy()
                >= 0.5
            )

            probability = torch.sigmoid(
                model(image)
            )[0, 0].cpu().numpy()

            raw_binary = (
                probability >= THRESHOLD
            )

            for configuration in (
                CONFIGURATIONS
            ):
                prediction = (
                    apply_configuration(
                        raw_binary,
                        configuration,
                    )
                )

                update_sos_state(
                    states[
                        configuration["name"]
                    ],
                    prediction,
                    target,
                )

            if index % 250 == 0:
                print(
                    f"SOS {index}/{len(dataset)}"
                )

    return {
        configuration["name"]:
        finalize_sos_state(
            states[
                configuration["name"]
            ]
        )
        for configuration
        in CONFIGURATIONS
    }


def evaluate_dartis(
    model: torch.nn.Module,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    dataframe = pd.read_csv(
        DARTIS_VAL_MANIFEST,
        encoding="utf-8-sig",
    )

    states = {
        configuration["name"]:
        create_dartis_state()
        for configuration
        in CONFIGURATIONS
    }

    print(
        f"DARTIS validation: {len(dataframe)}"
    )

    with torch.no_grad():
        for index, row in enumerate(
            dataframe.itertuples(
                index=False
            ),
            start=1,
        ):
            image_path = (
                DARTIS_RAW
                / str(row.image_set)
                / str(row.image_name)
            )

            with Image.open(
                image_path
            ) as image:
                image = image.convert("L")

                image_array = (
                    np.asarray(
                        image,
                        dtype=np.float32,
                    )
                    / 255.0
                )

            tensor = (
                torch.from_numpy(
                    image_array.copy()
                )
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device)
            )

            probability = torch.sigmoid(
                model(tensor)
            )[0, 0].cpu().numpy()

            raw_binary = (
                probability >= THRESHOLD
            )

            for configuration in (
                CONFIGURATIONS
            ):
                prediction = (
                    apply_configuration(
                        raw_binary,
                        configuration,
                    )
                )

                update_dartis_state(
                    states[
                        configuration["name"]
                    ],
                    prediction,
                )

            if index % 100 == 0:
                print(
                    f"DARTIS {index}/{len(dataframe)}"
                )

    return {
        configuration["name"]:
        finalize_dartis_state(
            states[
                configuration["name"]
            ]
        )
        for configuration
        in CONFIGURATIONS
    }


def main() -> None:
    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint bulunamadı: "
            f"{CHECKPOINT}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = load_model(
        CHECKPOINT,
        device,
    )

    print("=" * 80)
    print(
        "POST-PROCESSING VALIDATION SWEEP"
    )
    print("=" * 80)
    print("Cihaz:", device)
    print("Threshold:", THRESHOLD)
    print(
        "External test kullanılmıyor."
    )

    sos_results = evaluate_sos(
        model,
        device,
    )

    dartis_results = evaluate_dartis(
        model,
        device,
    )

    rows = []

    for configuration in CONFIGURATIONS:
        name = configuration["name"]

        sos = sos_results[name]
        dartis = dartis_results[name]

        severe_alarm_fraction = (
            dartis[
                "alarm_ge_1_percentage"
            ]
            / 100.0
        )

        balanced_score = (
            sos["dice"]
            * (
                1.0
                - severe_alarm_fraction
            )
        )

        rows.append(
            {
                "configuration": name,
                "min_component_ratio": (
                    configuration[
                        "min_component_ratio"
                    ]
                ),
                "border_remove_ratio": (
                    configuration[
                        "border_remove_ratio"
                    ]
                ),
                "sos_dice": sos["dice"],
                "sos_iou": sos["iou"],
                "sos_precision": (
                    sos["precision"]
                ),
                "sos_recall": (
                    sos["recall"]
                ),
                "dartis_any_alarm_percentage": (
                    dartis[
                        "any_alarm_percentage"
                    ]
                ),
                "dartis_alarm_ge_0_1_percentage": (
                    dartis[
                        "alarm_ge_0_1_percentage"
                    ]
                ),
                "dartis_alarm_ge_1_percentage": (
                    dartis[
                        "alarm_ge_1_percentage"
                    ]
                ),
                "dartis_fp_pixel_percentage": (
                    dartis[
                        "false_positive_pixel_percentage"
                    ]
                ),
                "balanced_score": (
                    balanced_score
                ),
            }
        )

    result_dataframe = pd.DataFrame(
        rows
    ).sort_values(
        "balanced_score",
        ascending=False,
    )

    result_dataframe.to_csv(
        OUTPUT_DIR
        / "postprocessing_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "threshold": THRESHOLD,
        "best_configuration": (
            result_dataframe.iloc[0]
            .to_dict()
        ),
        "results": (
            result_dataframe
            .to_dict(
                orient="records"
            )
        ),
    }

    with (
        OUTPUT_DIR
        / "postprocessing_results.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 118)
    print("SONUÇLAR")
    print("=" * 118)

    print(
        result_dataframe[
            [
                "configuration",
                "sos_dice",
                "sos_recall",
                "dartis_alarm_ge_1_percentage",
                "dartis_fp_pixel_percentage",
                "balanced_score",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.4f}"
            ),
        )
    )

    print()
    print(
        "En iyi konfigürasyon:",
        result_dataframe.iloc[0][
            "configuration"
        ],
    )

    print(
        "CSV:",
        (
            OUTPUT_DIR
            / "postprocessing_results.csv"
        ).resolve(),
    )


if __name__ == "__main__":
    main()
