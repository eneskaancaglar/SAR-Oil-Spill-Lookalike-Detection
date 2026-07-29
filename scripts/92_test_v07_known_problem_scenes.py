from __future__ import annotations

import argparse
import json
import math
import runpy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]

SAFE_PIPELINE_SCRIPT = (
    ROOT
    / "scripts"
    / "85_run_safe_water_masked_oil_pipeline.py"
)

V07_CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "verifier_v07"
    / "best.pth"
)

V07_CONFIG = (
    ROOT
    / "checkpoints"
    / "verifier_v07"
    / "calibration_config_v2.json"
)

DEVELOPMENT_MANIFEST = (
    ROOT
    / "data"
    / "metadata"
    / "v06_dartis_water_combined_development_manifest.csv"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "v07_known_scene_regression"
)

DEFAULT_SAMPLE_IDS = [
    "nc:nc-0140-01-000064",
    "nc:nc-0228-02-000068",
    "oc:oc-0108",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Yeni v0.7 LOOK_ALIKE / UNCERTAIN / CONFIRMED_OIL "
            "verifier'ını daha önce sorun çıkaran geliştirme "
            "sahnelerinde test eder. Taze holdout kullanılmaz."
        )
    )

    parser.add_argument(
        "--sample-ids",
        nargs="+",
        default=DEFAULT_SAMPLE_IDS,
    )

    parser.add_argument(
        "--segmentation-threshold",
        type=float,
        default=0.60,
    )

    parser.add_argument(
        "--minimum-component-ratio",
        type=float,
        default=0.0005,
    )

    parser.add_argument(
        "--minimum-component-pixels",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cuda",
            "cpu",
        ],
        default="auto",
    )

    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Gerekli dosya bulunamadı: {path}"
        )


def resolve_path(value: str | Path) -> Path:
    path = Path(
        str(value).replace(
            "\\",
            "/",
        )
    )

    if not path.is_absolute():
        path = ROOT / path

    return path.resolve()


def relative(path: Path) -> str:
    try:
        return str(
            path.resolve().relative_to(
                ROOT.resolve()
            )
        ).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def safe_name(sample_id: str) -> str:
    return (
        sample_id.replace(
            ":",
            "__",
        )
        .replace(
            "/",
            "_",
        )
        .replace(
            "\\",
            "_",
        )
    )


def mask_to_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray(
        (
            np.asarray(mask)
            .astype(np.uint8)
            * 255
        )
    )


def probability_to_image(
    probability: np.ndarray,
) -> Image.Image:
    return Image.fromarray(
        np.clip(
            np.asarray(probability)
            * 255.0,
            0,
            255,
        ).astype(np.uint8)
    )


def load_development_rows(
    sample_ids: list[str],
) -> pd.DataFrame:
    require_file(
        DEVELOPMENT_MANIFEST
    )

    frame = pd.read_csv(
        DEVELOPMENT_MANIFEST,
        encoding="utf-8-sig",
        low_memory=False,
    )

    required = {
        "sample_id",
        "annotation_image_path",
    }

    missing = required - set(
        frame.columns
    )

    if missing:
        raise RuntimeError(
            "Development manifestinde eksik sütunlar: "
            f"{sorted(missing)}"
        )

    selected = frame[
        frame[
            "sample_id"
        ].astype(str).isin(
            sample_ids
        )
    ].copy()

    found = set(
        selected[
            "sample_id"
        ].astype(str)
    )

    missing_ids = [
        sample_id
        for sample_id in sample_ids
        if sample_id not in found
    ]

    if missing_ids:
        raise RuntimeError(
            "Manifestte bulunamayan sample_id: "
            + ", ".join(
                missing_ids
            )
        )

    order = {
        sample_id: index
        for index, sample_id
        in enumerate(
            sample_ids
        )
    }

    selected[
        "requested_order"
    ] = selected[
        "sample_id"
    ].astype(str).map(
        order
    )

    selected = selected.sort_values(
        "requested_order"
    )

    selected[
        "resolved_image_path"
    ] = selected[
        "annotation_image_path"
    ].map(
        resolve_path
    )

    for path in selected[
        "resolved_image_path"
    ]:
        require_file(
            Path(path)
        )

    return selected.reset_index(
        drop=True
    )


def load_models(
    device_name: str,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    for path in (
        SAFE_PIPELINE_SCRIPT,
        V07_CHECKPOINT,
        V07_CONFIG,
    ):
        require_file(path)

    safe_module = runpy.run_path(
        str(
            SAFE_PIPELINE_SCRIPT
        ),
        run_name=(
            "v07_known_scene_safe_pipeline"
        ),
    )

    water_gate = safe_module[
        "load_water_gate"
    ](
        device_name
    )

    oil_pipeline = safe_module[
        "load_oil_pipeline"
    ](
        device_name
    )

    checkpoint = torch.load(
        V07_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    config = json.loads(
        V07_CONFIG.read_text(
            encoding="utf-8-sig"
        )
    )

    if not config.get(
        "gate_passed",
        False,
    ):
        raise RuntimeError(
            "calibration_config_v2 gate PASS değil."
        )

    oil_pipeline[
        "verifier_model"
    ].load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    oil_pipeline[
        "verifier_model"
    ].to(
        oil_pipeline[
            "device"
        ]
    )

    oil_pipeline[
        "verifier_model"
    ].eval()

    return (
        safe_module,
        water_gate,
        {
            **oil_pipeline,
            "v07_config": config,
        },
    )


def verifier_probability(
    crop: Image.Image,
    pipeline: dict[str, Any],
) -> tuple[
    float,
    float,
]:
    tensor = pipeline[
        "verifier_transform"
    ](
        crop.convert("L")
    ).unsqueeze(0).to(
        pipeline[
            "device"
        ]
    )

    with torch.inference_mode():
        with torch.amp.autocast(
            device_type=(
                pipeline[
                    "device"
                ].type
            ),
            enabled=(
                pipeline[
                    "device"
                ].type
                == "cuda"
            ),
        ):
            logits = pipeline[
                "verifier_model"
            ](
                tensor
            )

    if isinstance(
        logits,
        (
            tuple,
            list,
        ),
    ):
        logits = logits[0]

    logit = float(
        logits.reshape(
            -1
        )[0]
        .float()
        .cpu()
        .item()
    )

    temperature = float(
        pipeline[
            "v07_config"
        ][
            "temperature"
        ]
    )

    probability = float(
        1.0
        / (
            1.0
            + math.exp(
                -max(
                    min(
                        logit
                        / temperature,
                        30.0,
                    ),
                    -30.0,
                )
            )
        )
    )

    return (
        logit,
        probability,
    )


def decision_from_probability(
    probability: float,
    config: dict[str, Any],
) -> str:
    lookalike_threshold = float(
        config[
            "lookalike_threshold"
        ]
    )

    confirmed_threshold = float(
        config[
            "confirmed_oil_threshold"
        ]
    )

    if probability <= lookalike_threshold:
        return "LOOK_ALIKE"

    if probability >= confirmed_threshold:
        return "CONFIRMED_OIL"

    return "UNCERTAIN"


def create_final_overlay(
    image: Image.Image,
    final_mask: np.ndarray,
) -> Image.Image:
    base = np.asarray(
        image.convert("RGB"),
        dtype=np.uint8,
    ).copy()

    red = np.zeros_like(
        base
    )

    red[..., 0] = 255

    mask = (
        np.asarray(
            final_mask
        )
        > 0
    )

    blended = base.copy()

    blended[
        mask
    ] = (
        0.45
        * base[
            mask
        ].astype(
            np.float32
        )
        + 0.55
        * red[
            mask
        ].astype(
            np.float32
        )
    ).astype(
        np.uint8
    )

    return Image.fromarray(
        blended
    )


def create_candidate_overlay(
    image: Image.Image,
    candidates: list[dict[str, Any]],
) -> Image.Image:
    overlay = image.convert(
        "RGB"
    ).copy()

    draw = ImageDraw.Draw(
        overlay
    )

    colors = {
        "CONFIRMED_OIL": (
            255,
            0,
            0,
        ),
        "UNCERTAIN": (
            255,
            215,
            0,
        ),
        "LOOK_ALIKE": (
            0,
            170,
            255,
        ),
    }

    for candidate in candidates:
        bbox = tuple(
            int(value)
            for value in candidate[
                "bbox"
            ]
        )

        decision = str(
            candidate[
                "decision"
            ]
        )

        color = colors[
            decision
        ]

        draw.rectangle(
            bbox,
            outline=color,
            width=3,
        )

        label = (
            f"{candidate['component_index']} "
            f"{decision} "
            f"p={candidate['calibrated_probability']:.3f}"
        )

        text_position = (
            bbox[0] + 3,
            max(
                0,
                bbox[1] - 14,
            ),
        )

        draw.text(
            text_position,
            label,
            fill=color,
        )

    return overlay


@torch.inference_mode()
def run_oil_pipeline_v07(
    image: Image.Image,
    safe_water_mask: np.ndarray,
    pipeline: dict[str, Any],
    segmentation_threshold: float,
    minimum_component_ratio: float,
    minimum_component_pixels: int,
) -> dict[str, Any]:
    detector = pipeline[
        "detector"
    ]

    device = pipeline[
        "device"
    ]

    image = image.convert(
        "L"
    )

    image_array = (
        np.asarray(
            image,
            dtype=np.float32,
        )
        / 255.0
    )

    image_width, image_height = (
        image.size
    )

    total_pixels = (
        image_width
        * image_height
    )

    probability = detector[
        "run_segmentation"
    ](
        pipeline[
            "segmentation_model"
        ],
        image_array,
        device,
    )

    water_mask = (
        np.asarray(
            safe_water_mask,
            dtype=np.uint8,
        )
        > 0
    )

    if water_mask.shape != probability.shape:
        resized = Image.fromarray(
            (
                water_mask.astype(
                    np.uint8
                )
                * 255
            )
        ).resize(
            (
                probability.shape[1],
                probability.shape[0],
            ),
            Image.Resampling.NEAREST,
        )

        water_mask = (
            np.asarray(
                resized
            )
            > 0
        )

    safe_water_pixels = int(
        water_mask.sum()
    )

    probability = np.where(
        water_mask,
        probability,
        0.0,
    ).astype(
        np.float32
    )

    raw_mask = (
        probability
        >= segmentation_threshold
    ).astype(
        np.uint8
    )

    components = detector[
        "connected_components"
    ](
        raw_mask
    )

    minimum_area = max(
        int(
            minimum_component_pixels
        ),
        int(
            math.ceil(
                total_pixels
                * minimum_component_ratio
            )
        ),
    )

    eligible_components = [
        component
        for component in components
        if int(
            component[
                "area_pixels"
            ]
        )
        >= minimum_area
    ][:50]

    final_mask = np.zeros(
        raw_mask.shape,
        dtype=np.uint8,
    )

    candidate_results = []

    for component_index, component in enumerate(
        eligible_components,
        start=1,
    ):
        bbox = component[
            "bbox"
        ]

        crop_box = detector[
            "expand_box"
        ](
            bbox,
            image_width,
            image_height,
            0.25,
            64,
        )

        crop = image.crop(
            crop_box
        )

        logit, calibrated_probability = (
            verifier_probability(
                crop,
                pipeline,
            )
        )

        decision = (
            decision_from_probability(
                calibrated_probability,
                pipeline[
                    "v07_config"
                ],
            )
        )

        accepted = (
            decision
            == "CONFIRMED_OIL"
        )

        if accepted:
            for y, x in component[
                "pixels"
            ]:
                final_mask[
                    y,
                    x,
                ] = 1

        candidate_results.append(
            {
                "component_index": (
                    component_index
                ),
                "area_pixels": int(
                    component[
                        "area_pixels"
                    ]
                ),
                "area_percent_total": float(
                    component[
                        "area_pixels"
                    ]
                    / total_pixels
                    * 100.0
                ),
                "bbox": [
                    int(value)
                    for value in bbox
                ],
                "crop_box": [
                    int(value)
                    for value in crop_box
                ],
                "logit": float(
                    logit
                ),
                "calibrated_probability": float(
                    calibrated_probability
                ),
                "lookalike_threshold": float(
                    pipeline[
                        "v07_config"
                    ][
                        "lookalike_threshold"
                    ]
                ),
                "confirmed_oil_threshold": float(
                    pipeline[
                        "v07_config"
                    ][
                        "confirmed_oil_threshold"
                    ]
                ),
                "decision": decision,
                "accepted": bool(
                    accepted
                ),
            }
        )

    final_mask = (
        final_mask
        * water_mask.astype(
            np.uint8
        )
    )

    final_positive_pixels = int(
        final_mask.sum()
    )

    water_denominator = max(
        safe_water_pixels,
        1,
    )

    counts = {
        decision: int(
            sum(
                candidate[
                    "decision"
                ]
                == decision
                for candidate
                in candidate_results
            )
        )
        for decision in (
            "CONFIRMED_OIL",
            "UNCERTAIN",
            "LOOK_ALIKE",
        )
    }

    return {
        "oil_detected": bool(
            final_positive_pixels
            > 0
        ),
        "probability": probability,
        "raw_mask": raw_mask,
        "final_mask": final_mask,
        "overlay": create_final_overlay(
            image,
            final_mask,
        ),
        "candidate_overlay": (
            create_candidate_overlay(
                image,
                candidate_results,
            )
        ),
        "candidate_results": (
            candidate_results
        ),
        "candidate_count": int(
            len(
                candidate_results
            )
        ),
        "decision_counts": counts,
        "confirmed_candidate_count": (
            counts[
                "CONFIRMED_OIL"
            ]
        ),
        "uncertain_candidate_count": (
            counts[
                "UNCERTAIN"
            ]
        ),
        "lookalike_candidate_count": (
            counts[
                "LOOK_ALIKE"
            ]
        ),
        "safe_water_pixels": (
            safe_water_pixels
        ),
        "final_positive_pixels": (
            final_positive_pixels
        ),
        "final_oil_percent_of_water": float(
            final_positive_pixels
            / water_denominator
            * 100.0
        ),
        "minimum_component_pixels": (
            minimum_area
        ),
    }


def save_scene_outputs(
    scene_dir: Path,
    image: Image.Image,
    water_result: dict[str, Any],
    oil_result: dict[str, Any] | None,
    record: dict[str, Any],
) -> None:
    scene_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    image.save(
        scene_dir
        / "input.png"
    )

    mask_to_image(
        water_result[
            "safe_water_mask"
        ]
    ).save(
        scene_dir
        / "safe_water_mask.png"
    )

    mask_to_image(
        water_result[
            "uncertain_mask"
        ]
    ).save(
        scene_dir
        / "water_uncertain_mask.png"
    )

    probability_to_image(
        water_result[
            "probability"
        ]
    ).save(
        scene_dir
        / "water_probability.png"
    )

    if oil_result is not None:
        probability_to_image(
            oil_result[
                "probability"
            ]
        ).save(
            scene_dir
            / "oil_probability_water_masked.png"
        )

        mask_to_image(
            oil_result[
                "raw_mask"
            ]
        ).save(
            scene_dir
            / "oil_raw_candidate_mask.png"
        )

        mask_to_image(
            oil_result[
                "final_mask"
            ]
        ).save(
            scene_dir
            / "oil_final_confirmed_mask.png"
        )

        oil_result[
            "overlay"
        ].save(
            scene_dir
            / "oil_final_overlay.png"
        )

        oil_result[
            "candidate_overlay"
        ].save(
            scene_dir
            / "candidate_decisions_overlay.png"
        )

        pd.DataFrame(
            oil_result[
                "candidate_results"
            ]
        ).to_csv(
            scene_dir
            / "candidate_decisions.csv",
            index=False,
            encoding="utf-8-sig",
        )
    else:
        black = np.zeros(
            (
                image.height,
                image.width,
            ),
            dtype=np.uint8,
        )

        mask_to_image(
            black
        ).save(
            scene_dir
            / "oil_final_confirmed_mask.png"
        )

    (
        scene_dir
        / "result.json"
    ).write_text(
        json.dumps(
            record,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def add_header(
    image: Image.Image,
    text: str,
    width: int,
) -> Image.Image:
    ratio = (
        width
        / image.width
    )

    resized = image.resize(
        (
            width,
            max(
                1,
                int(
                    image.height
                    * ratio
                ),
            ),
        ),
        Image.Resampling.LANCZOS,
    )

    header_height = 44

    canvas = Image.new(
        "RGB",
        (
            width,
            resized.height
            + header_height,
        ),
        "black",
    )

    canvas.paste(
        resized,
        (
            0,
            header_height,
        ),
    )

    draw = ImageDraw.Draw(
        canvas
    )

    draw.text(
        (
            8,
            12,
        ),
        text,
        fill="white",
        font=ImageFont.load_default(),
    )

    return canvas


def build_contact_sheet(
    scene_records: list[dict[str, Any]],
) -> Path:
    rows = []

    for record in scene_records:
        scene_dir = OUTPUT_DIR / safe_name(
            record[
                "sample_id"
            ]
        )

        input_image = Image.open(
            scene_dir
            / "input.png"
        ).convert("RGB")

        water_image = Image.open(
            scene_dir
            / "safe_water_mask.png"
        ).convert("RGB")

        final_mask = Image.open(
            scene_dir
            / "oil_final_confirmed_mask.png"
        ).convert("RGB")

        if (
            scene_dir
            / "candidate_decisions_overlay.png"
        ).exists():
            candidate_overlay = Image.open(
                scene_dir
                / "candidate_decisions_overlay.png"
            ).convert("RGB")
        else:
            candidate_overlay = (
                input_image.copy()
            )

        panel = Image.new(
            "RGB",
            (
                input_image.width
                * 4,
                input_image.height,
            ),
            "black",
        )

        panel.paste(
            input_image,
            (
                0,
                0,
            ),
        )

        panel.paste(
            water_image,
            (
                input_image.width,
                0,
            ),
        )

        panel.paste(
            candidate_overlay,
            (
                input_image.width
                * 2,
                0,
            ),
        )

        panel.paste(
            final_mask,
            (
                input_image.width
                * 3,
                0,
            ),
        )

        title = (
            f"{record['sample_id']} | expected={record['expected']} "
            f"| water={record['water_decision']} "
            f"| result={record['system_result']} "
            f"| confirmed={record['confirmed_candidate_count']} "
            f"uncertain={record['uncertain_candidate_count']} "
            f"look={record['lookalike_candidate_count']}"
        )

        rows.append(
            add_header(
                panel,
                title,
                1600,
            )
        )

    sheet = Image.new(
        "RGB",
        (
            1600,
            sum(
                row.height
                for row in rows
            ),
        ),
        "black",
    )

    y = 0

    for row in rows:
        sheet.paste(
            row,
            (
                0,
                y,
            ),
        )

        y += row.height

    output_path = (
        OUTPUT_DIR
        / "known_scene_regression_contact_sheet.jpg"
    )

    sheet.save(
        output_path,
        quality=92,
    )

    return output_path


def main() -> None:
    args = parse_args()

    rows = load_development_rows(
        args.sample_ids
    )

    (
        safe_module,
        water_gate,
        oil_pipeline,
    ) = load_models(
        args.device
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 78)
    print(
        "v0.7 KNOWN-SCENE REGRESSION TEST"
    )
    print("=" * 78)

    print(
        "Taze holdout kullanılmayacak."
    )

    print(
        "Verifier kararları: LOOK_ALIKE / UNCERTAIN / CONFIRMED_OIL"
    )

    print()

    scene_records = []

    for row in rows.itertuples():
        sample_id = str(
            row.sample_id
        )

        image_path = Path(
            row.resolved_image_path
        )

        image = Image.open(
            image_path
        ).convert("L")

        expected = (
            "NO_OIL"
            if sample_id.startswith(
                "nc:"
            )
            else "OIL"
        )

        with torch.inference_mode():
            water_result = safe_module[
                "run_water_gate"
            ](
                image,
                water_gate,
            )

        oil_result = None

        if water_result[
            "pipeline_allowed"
        ]:
            oil_result = run_oil_pipeline_v07(
                image=image,
                safe_water_mask=(
                    water_result[
                        "safe_water_mask"
                    ]
                ),
                pipeline=oil_pipeline,
                segmentation_threshold=float(
                    args.segmentation_threshold
                ),
                minimum_component_ratio=float(
                    args.minimum_component_ratio
                ),
                minimum_component_pixels=int(
                    args.minimum_component_pixels
                ),
            )

        if not water_result[
            "pipeline_allowed"
        ]:
            system_result = (
                "UNCERTAIN_BLOCK"
            )
        elif oil_result is not None and oil_result[
            "oil_detected"
        ]:
            system_result = (
                "CONFIRMED_OIL"
            )
        elif (
            oil_result is not None
            and oil_result[
                "uncertain_candidate_count"
            ]
            > 0
        ):
            system_result = (
                "UNCERTAIN_NO_FINAL_MASK"
            )
        else:
            system_result = (
                "NO_CONFIRMED_OIL"
            )

        if expected == "NO_OIL":
            regression_pass = (
                system_result
                != "CONFIRMED_OIL"
            )
        else:
            regression_pass = (
                system_result
                == "CONFIRMED_OIL"
            )

        record = {
            "sample_id": sample_id,
            "image_path": relative(
                image_path
            ),
            "expected": expected,
            "water_decision": (
                water_result[
                    "decision"
                ]
            ),
            "safe_water_fraction": float(
                water_result[
                    "predicted_water_fraction"
                ]
            ),
            "water_uncertain_fraction": float(
                water_result[
                    "uncertain_fraction"
                ]
            ),
            "system_result": (
                system_result
            ),
            "oil_model_executed": bool(
                oil_result
                is not None
            ),
            "confirmed_candidate_count": int(
                oil_result[
                    "confirmed_candidate_count"
                ]
                if oil_result is not None
                else 0
            ),
            "uncertain_candidate_count": int(
                oil_result[
                    "uncertain_candidate_count"
                ]
                if oil_result is not None
                else 0
            ),
            "lookalike_candidate_count": int(
                oil_result[
                    "lookalike_candidate_count"
                ]
                if oil_result is not None
                else 0
            ),
            "final_oil_pixels": int(
                oil_result[
                    "final_positive_pixels"
                ]
                if oil_result is not None
                else 0
            ),
            "final_oil_percent_of_water": float(
                oil_result[
                    "final_oil_percent_of_water"
                ]
                if oil_result is not None
                else 0.0
            ),
            "regression_pass": bool(
                regression_pass
            ),
            "candidate_results": (
                oil_result[
                    "candidate_results"
                ]
                if oil_result is not None
                else []
            ),
        }

        scene_records.append(
            record
        )

        save_scene_outputs(
            scene_dir=(
                OUTPUT_DIR
                / safe_name(
                    sample_id
                )
            ),
            image=image,
            water_result=(
                water_result
            ),
            oil_result=(
                oil_result
            ),
            record=record,
        )

        print(
            f"{sample_id} "
            f"| expected={expected} "
            f"| water={water_result['decision']} "
            f"| result={system_result} "
            f"| C={record['confirmed_candidate_count']} "
            f"U={record['uncertain_candidate_count']} "
            f"L={record['lookalike_candidate_count']} "
            f"| pass={regression_pass}"
        )

    summary_frame = pd.DataFrame(
        [
            {
                key: value
                for key, value
                in record.items()
                if key
                != "candidate_results"
            }
            for record in scene_records
        ]
    )

    summary_csv = (
        OUTPUT_DIR
        / "known_scene_regression_results.csv"
    )

    summary_frame.to_csv(
        summary_csv,
        index=False,
        encoding="utf-8-sig",
    )

    contact_sheet = build_contact_sheet(
        scene_records
    )

    negative_records = [
        record
        for record in scene_records
        if record[
            "expected"
        ]
        == "NO_OIL"
    ]

    positive_records = [
        record
        for record in scene_records
        if record[
            "expected"
        ]
        == "OIL"
    ]

    negative_false_alarm_count = int(
        sum(
            record[
                "system_result"
            ]
            == "CONFIRMED_OIL"
            for record
            in negative_records
        )
    )

    positive_confirm_count = int(
        sum(
            record[
                "system_result"
            ]
            == "CONFIRMED_OIL"
            for record
            in positive_records
        )
    )

    all_passed = bool(
        all(
            record[
                "regression_pass"
            ]
            for record
            in scene_records
        )
    )

    summary = {
        "stage": (
            "v07_known_scene_regression"
        ),
        "sample_count": int(
            len(
                scene_records
            )
        ),
        "all_regression_checks_passed": (
            all_passed
        ),
        "negative_scene_count": int(
            len(
                negative_records
            )
        ),
        "negative_confirmed_oil_false_alarm_count": (
            negative_false_alarm_count
        ),
        "positive_scene_count": int(
            len(
                positive_records
            )
        ),
        "positive_confirmed_oil_count": (
            positive_confirm_count
        ),
        "fresh_end_to_end_holdout_used": False,
        "verifier_checkpoint": relative(
            V07_CHECKPOINT
        ),
        "verifier_config": relative(
            V07_CONFIG
        ),
        "results_csv": relative(
            summary_csv
        ),
        "contact_sheet": relative(
            contact_sheet
        ),
        "scientific_note": (
            "Bu sahneler geliştirme sırasında daha önce görülmüş "
            "regresyon örnekleridir; bağımsız başarı kanıtı değildir. "
            "Amaç yalnız yeni verifier entegrasyonunun daha önceki "
            "yanlış pozitifleri bastırıp pozitif kontrolü koruduğunu "
            "doğrulamaktır."
        ),
    }

    summary_path = (
        OUTPUT_DIR
        / "summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print(
        "KNOWN-SCENE REGRESSION SONUCU"
    )
    print("=" * 78)

    print(
        "All checks:",
        (
            "PASS"
            if all_passed
            else "FAIL"
        ),
    )

    print(
        "Negatif confirmed-oil yanlış alarm:",
        negative_false_alarm_count,
        "/",
        len(
            negative_records
        ),
    )

    print(
        "Pozitif confirmed-oil:",
        positive_confirm_count,
        "/",
        len(
            positive_records
        ),
    )

    print(
        "Taze holdout kullanıldı mı: False"
    )

    print(
        "Özet:",
        summary_path.resolve(),
    )

    print(
        "Görsel:",
        contact_sheet.resolve(),
    )


if __name__ == "__main__":
    main()
