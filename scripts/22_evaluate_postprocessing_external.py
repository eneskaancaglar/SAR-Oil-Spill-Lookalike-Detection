from __future__ import annotations

import json
import runpy
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

SOURCE_SCRIPT = (
    ROOT
    / "scripts"
    / "21_postprocessing_validation_sweep.py"
)

EXTERNAL_MANIFEST = (
    ROOT
    / "data"
    / "metadata"
    / "dartis_external_test_manifest.csv"
)

CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "hard_negative_finetune"
    / "best_balanced.pth"
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
    / "postprocessing_external_test"
)

THRESHOLD = 0.60

CONFIGURATIONS = [
    {
        "name": "raw",
        "min_component_ratio": 0.0,
    },
    {
        "name": "remove_tiny_005pct",
        "min_component_ratio": 0.0005,
    },
]


def create_state() -> dict[str, int]:
    return {
        "image_count": 0,
        "any_alarm": 0,
        "alarm_ge_0_01": 0,
        "alarm_ge_0_1": 0,
        "alarm_ge_1": 0,
        "positive_pixels": 0,
        "total_pixels": 0,
    }


def update_state(
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
    state["positive_pixels"] += positive_pixels
    state["total_pixels"] += total_pixels

    state["any_alarm"] += int(
        positive_pixels > 0
    )

    state["alarm_ge_0_01"] += int(
        positive_ratio >= 0.0001
    )

    state["alarm_ge_0_1"] += int(
        positive_ratio >= 0.001
    )

    state["alarm_ge_1"] += int(
        positive_ratio >= 0.01
    )


def finalize_state(
    state: dict[str, int],
) -> dict[str, float | int]:
    image_count = state["image_count"]

    def percentage(value: int) -> float:
        return (
            value / image_count * 100.0
            if image_count > 0
            else 0.0
        )

    return {
        "image_count": image_count,
        "any_alarm_percentage": percentage(
            state["any_alarm"]
        ),
        "alarm_ge_0_01_percentage": percentage(
            state["alarm_ge_0_01"]
        ),
        "alarm_ge_0_1_percentage": percentage(
            state["alarm_ge_0_1"]
        ),
        "alarm_ge_1_percentage": percentage(
            state["alarm_ge_1"]
        ),
        "false_positive_pixel_percentage": (
            state["positive_pixels"]
            / state["total_pixels"]
            * 100.0
            if state["total_pixels"] > 0
            else 0.0
        ),
    }


def main() -> None:
    if not SOURCE_SCRIPT.exists():
        raise FileNotFoundError(
            f"Script 21 bulunamadı: {SOURCE_SCRIPT}"
        )

    if not EXTERNAL_MANIFEST.exists():
        raise FileNotFoundError(
            f"External manifest bulunamadı: "
            f"{EXTERNAL_MANIFEST}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    source = runpy.run_path(
        str(SOURCE_SCRIPT)
    )

    load_model = source["load_model"]
    postprocess = source["postprocess"]

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = load_model(
        CHECKPOINT,
        device,
    )

    dataframe = pd.read_csv(
        EXTERNAL_MANIFEST,
        encoding="utf-8-sig",
    )

    states = {
        configuration["name"]: {
            group: create_state()
            for group in (
                "overall",
                "nc",
                "nw",
            )
        }
        for configuration in CONFIGURATIONS
    }

    per_image_rows = []

    print("=" * 80)
    print("POST-PROCESSING EXTERNAL TEST")
    print("=" * 80)
    print("Cihaz:", device)
    print("Threshold:", THRESHOLD)
    print("Görüntü:", len(dataframe))

    with torch.no_grad():
        for index, row in enumerate(
            dataframe.itertuples(index=False),
            start=1,
        ):
            image_set = str(
                row.image_set
            )

            image_name = str(
                row.image_name
            )

            image_path = (
                DARTIS_RAW
                / image_set
                / image_name
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

            output_row = {
                "image_set": image_set,
                "image_name": image_name,
            }

            for configuration in CONFIGURATIONS:
                name = configuration["name"]

                prediction = postprocess(
                    binary=raw_binary,
                    min_component_ratio=(
                        configuration[
                            "min_component_ratio"
                        ]
                    ),
                    border_remove_ratio=None,
                )

                for group in (
                    "overall",
                    image_set,
                ):
                    update_state(
                        states[name][group],
                        prediction,
                    )

                output_row[
                    f"{name}_positive_ratio"
                ] = float(
                    prediction.mean()
                )

            per_image_rows.append(
                output_row
            )

            if index % 100 == 0:
                print(
                    f"{index}/{len(dataframe)}"
                )

    summary = {
        "threshold": THRESHOLD,
        "configurations": {},
    }

    output_rows = []

    for configuration in CONFIGURATIONS:
        name = configuration["name"]

        summary["configurations"][
            name
        ] = {}

        for group in (
            "overall",
            "nc",
            "nw",
        ):
            metrics = finalize_state(
                states[name][group]
            )

            summary["configurations"][
                name
            ][group] = metrics

            output_rows.append(
                {
                    "configuration": name,
                    "group": group,
                    **metrics,
                }
            )

    result_dataframe = pd.DataFrame(
        output_rows
    )

    result_dataframe.to_csv(
        OUTPUT_DIR
        / "postprocessing_external_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        per_image_rows
    ).to_csv(
        OUTPUT_DIR
        / "postprocessing_external_per_image.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with (
        OUTPUT_DIR
        / "postprocessing_external_summary.json"
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
    print("=" * 110)
    print("EXTERNAL TEST SONUÇLARI")
    print("=" * 110)

    print(
        result_dataframe[
            [
                "configuration",
                "group",
                "image_count",
                "any_alarm_percentage",
                "alarm_ge_0_1_percentage",
                "alarm_ge_1_percentage",
                "false_positive_pixel_percentage",
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
        "CSV:",
        (
            OUTPUT_DIR
            / "postprocessing_external_results.csv"
        ).resolve(),
    )


if __name__ == "__main__":
    main()
