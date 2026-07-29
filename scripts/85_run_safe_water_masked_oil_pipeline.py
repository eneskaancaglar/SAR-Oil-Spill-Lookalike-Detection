from __future__ import annotations

import argparse
import json
import math
import runpy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

WATER_GATE_DIR = (
    ROOT
    / "checkpoints"
    / "final_water_gate_v06"
)

WATER_RUNTIME = (
    WATER_GATE_DIR
    / "water_gate_runtime.py"
)

OIL_DETECTOR_SCRIPT = (
    ROOT
    / "scripts"
    / "33_run_end_to_end_oil_detector.py"
)

OIL_MODEL_DIR = (
    ROOT
    / "checkpoints"
    / "final_model_v03"
)

SEGMENTATION_MODEL = (
    OIL_MODEL_DIR
    / "oil_segmentation_unet.pth"
)

VERIFIER_MODEL = (
    OIL_MODEL_DIR
    / "oil_candidate_verifier_resnet18.pth"
)

CALIBRATION_CONFIG = (
    OIL_MODEL_DIR
    / "calibration_config.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "v0.6 seçici kara-su güvenlik kapısını v0.3 "
            "petrol segmentasyonu ve verifier hattına bağlar. "
            "UNCERTAIN_BLOCK durumunda petrol modeli çalıştırılmaz."
        )
    )

    parser.add_argument(
        "--image",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
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


def mask_to_image(
    mask: np.ndarray,
) -> Image.Image:
    return Image.fromarray(
        (
            mask.astype(np.uint8)
            * 255
        )
    )


def probability_to_image(
    probability: np.ndarray,
) -> Image.Image:
    return Image.fromarray(
        np.clip(
            probability * 255.0,
            0,
            255,
        ).astype(
            np.uint8
        )
    )


def load_water_gate(
    requested_device: str,
) -> dict[str, Any]:
    require_file(
        WATER_RUNTIME
    )

    runtime = runpy.run_path(
        str(WATER_RUNTIME),
        run_name=(
            "safe_pipeline_water_runtime"
        ),
    )

    device = runtime[
        "select_device"
    ](
        requested_device
    )

    policy_path = (
        WATER_GATE_DIR
        / "policy.json"
    )

    require_file(
        policy_path
    )

    policy = json.loads(
        policy_path.read_text(
            encoding="utf-8-sig"
        )
    )

    (
        models,
        thresholds,
    ) = runtime[
        "load_models"
    ](
        device
    )

    return {
        "runtime": runtime,
        "device": device,
        "policy": policy,
        "models": models,
        "thresholds": thresholds,
    }


@torch.inference_mode()
def run_water_gate(
    image: Image.Image,
    gate: dict[str, Any],
) -> dict[str, Any]:
    runtime = gate[
        "runtime"
    ]

    device = gate[
        "device"
    ]

    image_array = np.asarray(
        image.convert("L"),
        dtype=np.uint8,
    )

    normalized = runtime[
        "normalize_percentile"
    ](
        image_array
    )

    tensor = torch.from_numpy(
        np.stack(
            [
                normalized,
                normalized,
            ],
            axis=0,
        ).astype(
            np.float32
        )
    ).unsqueeze(0).to(
        device
    )

    aligned_probabilities = []

    for model, threshold in zip(
        gate["models"],
        gate["thresholds"],
    ):
        with torch.amp.autocast(
            device_type=device.type,
            enabled=(
                device.type == "cuda"
            ),
        ):
            logits = model(
                tensor
            )[0, 0]

        probability = torch.sigmoid(
            logits.float()
        ).cpu().numpy()

        aligned = runtime[
            "align_probability"
        ](
            probability,
            threshold,
        )

        aligned_probabilities.append(
            aligned
        )

    stack = np.stack(
        aligned_probabilities,
        axis=0,
    )

    probability = np.median(
        stack,
        axis=0,
    ).astype(
        np.float32
    )

    disagreement = np.std(
        stack,
        axis=0,
    ).astype(
        np.float32
    )

    policy = gate[
        "policy"
    ]

    water_threshold = float(
        policy[
            "water_threshold"
        ]
    )

    land_threshold = float(
        policy[
            "land_threshold"
        ]
    )

    safe_water = (
        probability
        >= water_threshold
    )

    safe_land = (
        probability
        <= land_threshold
    )

    uncertain = (
        (~safe_water)
        & (~safe_land)
    )

    total_pixels = max(
        int(image_array.size),
        1,
    )

    water_fraction = float(
        safe_water.sum()
        / total_pixels
    )

    uncertain_fraction = float(
        uncertain.sum()
        / total_pixels
    )

    accepted = (
        water_fraction
        >= float(
            policy[
                "minimum_predicted_water_fraction"
            ]
        )
        and uncertain_fraction
        <= float(
            policy[
                "maximum_uncertain_fraction"
            ]
        )
    )

    operational_mask = np.zeros(
        safe_water.shape,
        dtype=np.uint8,
    )

    if accepted:
        operational_mask[
            safe_water
        ] = 1

    return {
        "decision": (
            "ACCEPT_WATER"
            if accepted
            else "UNCERTAIN_BLOCK"
        ),
        "pipeline_allowed": bool(
            accepted
        ),
        "safe_water_mask": (
            operational_mask
        ),
        "internal_water_mask": (
            safe_water.astype(
                np.uint8
            )
        ),
        "uncertain_mask": (
            uncertain.astype(
                np.uint8
            )
        ),
        "probability": probability,
        "disagreement": disagreement,
        "predicted_water_fraction": (
            water_fraction
        ),
        "uncertain_fraction": (
            uncertain_fraction
        ),
        "mean_disagreement": float(
            disagreement.mean()
        ),
        "policy": policy,
    }


def load_oil_pipeline(
    requested_device: str,
) -> dict[str, Any]:
    for path in (
        OIL_DETECTOR_SCRIPT,
        SEGMENTATION_MODEL,
        VERIFIER_MODEL,
        CALIBRATION_CONFIG,
    ):
        require_file(path)

    detector = runpy.run_path(
        str(OIL_DETECTOR_SCRIPT),
        run_name=(
            "safe_pipeline_oil_detector"
        ),
    )

    detector[
        "SEGMENTATION_CHECKPOINT"
    ] = SEGMENTATION_MODEL

    detector[
        "VERIFIER_CHECKPOINT"
    ] = VERIFIER_MODEL

    detector[
        "CALIBRATION_CONFIG"
    ] = CALIBRATION_CONFIG

    device = detector[
        "select_device"
    ](
        requested_device
    )

    segmentation_model = detector[
        "build_segmentation_model"
    ](
        device
    )

    (
        verifier_model,
        verifier_config,
    ) = detector[
        "build_verifier_model"
    ](
        device
    )

    verifier_transform = detector[
        "build_verifier_transform"
    ](
        verifier_config
    )

    return {
        "detector": detector,
        "device": device,
        "segmentation_model": (
            segmentation_model
        ),
        "verifier_model": (
            verifier_model
        ),
        "verifier_config": (
            verifier_config
        ),
        "verifier_transform": (
            verifier_transform
        ),
    }


@torch.inference_mode()
def run_oil_pipeline(
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
        water_mask = (
            np.asarray(
                mask_to_image(
                    water_mask.astype(
                        np.uint8
                    )
                ).resize(
                    (
                        probability.shape[1],
                        probability.shape[0],
                    ),
                    Image.Resampling.NEAREST,
                )
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
        if component[
            "area_pixels"
        ]
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

        candidate_crop = image.crop(
            crop_box
        )

        verifier_result = detector[
            "run_verifier"
        ](
            pipeline[
                "verifier_model"
            ],
            candidate_crop,
            pipeline[
                "verifier_transform"
            ],
            pipeline[
                "verifier_config"
            ],
            device,
        )

        accepted = bool(
            verifier_result[
                "accepted"
            ]
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
                "selected_probability": float(
                    verifier_result[
                        "selected_probability"
                    ]
                ),
                "selected_threshold": float(
                    verifier_result[
                        "selected_threshold"
                    ]
                ),
                "verifier_probability": float(
                    verifier_result[
                        "selected_probability"
                    ]
                ),
                "verifier_threshold": float(
                    verifier_result[
                        "selected_threshold"
                    ]
                ),
                "accepted": accepted,
                "decision": (
                    "OIL_CANDIDATE"
                    if accepted
                    else "REJECTED_CANDIDATE"
                ),
            }
        )

    final_mask = (
        final_mask
        * water_mask.astype(
            np.uint8
        )
    )

    raw_positive_pixels = int(
        raw_mask.sum()
    )

    final_positive_pixels = int(
        final_mask.sum()
    )

    water_denominator = max(
        safe_water_pixels,
        1,
    )

    accepted_candidates = [
        item
        for item in candidate_results
        if item["accepted"]
    ]

    oil_detected = (
        final_positive_pixels > 0
    )

    overlay = detector[
        "create_overlay"
    ](
        image,
        final_mask,
    )

    candidate_overlay = detector[
        "create_candidate_overlay"
    ](
        image,
        candidate_results,
    )

    return {
        "oil_detected": bool(
            oil_detected
        ),
        "probability": probability,
        "raw_mask": raw_mask,
        "final_mask": final_mask,
        "overlay": overlay,
        "candidate_overlay": (
            candidate_overlay
        ),
        "candidate_results": (
            candidate_results
        ),
        "raw_positive_pixels": (
            raw_positive_pixels
        ),
        "final_positive_pixels": (
            final_positive_pixels
        ),
        "safe_water_pixels": (
            safe_water_pixels
        ),
        "raw_oil_percent_of_water": float(
            raw_positive_pixels
            / water_denominator
            * 100.0
        ),
        "final_oil_percent_of_water": float(
            final_positive_pixels
            / water_denominator
            * 100.0
        ),
        "candidate_count": int(
            len(candidate_results)
        ),
        "accepted_candidate_count": int(
            len(accepted_candidates)
        ),
        "segmentation_threshold": float(
            segmentation_threshold
        ),
        "minimum_component_pixels": int(
            minimum_area
        ),
    }


def save_outputs(
    output_dir: Path,
    image: Image.Image,
    water_result: dict[str, Any],
    oil_result: dict[str, Any] | None,
) -> Path:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    mask_to_image(
        water_result[
            "safe_water_mask"
        ]
    ).save(
        output_dir
        / "safe_water_mask.png"
    )

    mask_to_image(
        water_result[
            "uncertain_mask"
        ]
    ).save(
        output_dir
        / "water_uncertain_mask.png"
    )

    probability_to_image(
        water_result[
            "probability"
        ]
    ).save(
        output_dir
        / "water_probability.png"
    )

    report: dict[str, Any] = {
        "pipeline_version": (
            "v0.5-input-domain -> "
            "v0.6-water-gate -> "
            "v0.3-oil-detector"
        ),
        "water_gate": {
            "decision": water_result[
                "decision"
            ],
            "pipeline_allowed": (
                water_result[
                    "pipeline_allowed"
                ]
            ),
            "predicted_water_fraction": (
                water_result[
                    "predicted_water_fraction"
                ]
            ),
            "uncertain_fraction": (
                water_result[
                    "uncertain_fraction"
                ]
            ),
            "mean_disagreement": (
                water_result[
                    "mean_disagreement"
                ]
            ),
            "policy": water_result[
                "policy"
            ],
        },
        "oil_analysis": None,
    }

    if oil_result is not None:
        probability_to_image(
            oil_result[
                "probability"
            ]
        ).save(
            output_dir
            / "oil_probability_water_masked.png"
        )

        mask_to_image(
            oil_result[
                "raw_mask"
            ]
        ).save(
            output_dir
            / "oil_raw_candidate_mask.png"
        )

        mask_to_image(
            oil_result[
                "final_mask"
            ]
        ).save(
            output_dir
            / "oil_final_mask.png"
        )

        oil_result[
            "overlay"
        ].save(
            output_dir
            / "oil_overlay.png"
        )

        oil_result[
            "candidate_overlay"
        ].save(
            output_dir
            / "oil_candidate_overlay.png"
        )

        report[
            "oil_analysis"
        ] = {
            "executed": True,
            "oil_detected": (
                oil_result[
                    "oil_detected"
                ]
            ),
            "safe_water_pixels": (
                oil_result[
                    "safe_water_pixels"
                ]
            ),
            "raw_positive_pixels": (
                oil_result[
                    "raw_positive_pixels"
                ]
            ),
            "final_oil_pixels": (
                oil_result[
                    "final_positive_pixels"
                ]
            ),
            "raw_oil_percent_of_water": (
                oil_result[
                    "raw_oil_percent_of_water"
                ]
            ),
            "final_oil_percent_of_water": (
                oil_result[
                    "final_oil_percent_of_water"
                ]
            ),
            "candidate_count": (
                oil_result[
                    "candidate_count"
                ]
            ),
            "accepted_candidate_count": (
                oil_result[
                    "accepted_candidate_count"
                ]
            ),
            "segmentation_threshold": (
                oil_result[
                    "segmentation_threshold"
                ]
            ),
            "minimum_component_pixels": (
                oil_result[
                    "minimum_component_pixels"
                ]
            ),
            "candidate_results": (
                oil_result[
                    "candidate_results"
                ]
            ),
        }
    else:
        black_mask = np.zeros(
            (
                image.height,
                image.width,
            ),
            dtype=np.uint8,
        )

        mask_to_image(
            black_mask
        ).save(
            output_dir
            / "oil_final_mask.png"
        )

        report[
            "oil_analysis"
        ] = {
            "executed": False,
            "oil_detected": False,
            "blocked_reason": (
                "UNCERTAIN_BLOCK from water gate"
            ),
            "final_oil_pixels": 0,
        }

    result_path = (
        output_dir
        / "result.json"
    )

    result_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return result_path


def main() -> None:
    args = parse_args()

    image_path = args.image.resolve()
    output_dir = args.output_dir.resolve()

    if not image_path.exists():
        raise FileNotFoundError(
            f"Görüntü bulunamadı: {image_path}"
        )

    image = Image.open(
        image_path
    ).convert("L")

    print("=" * 78)
    print(
        "SAFE WATER-MASKED OIL PIPELINE"
    )
    print("=" * 78)
    print(
        "Görüntü:",
        image_path,
    )

    water_gate = load_water_gate(
        args.device
    )

    water_result = run_water_gate(
        image,
        water_gate,
    )

    print(
        "Water decision:",
        water_result[
            "decision"
        ],
    )

    print(
        "Safe water fraction:",
        f"{water_result['predicted_water_fraction']:.4f}",
    )

    print(
        "Uncertain fraction:",
        f"{water_result['uncertain_fraction']:.4f}",
    )

    oil_result = None

    if water_result[
        "pipeline_allowed"
    ]:
        oil_pipeline = load_oil_pipeline(
            args.device
        )

        oil_result = run_oil_pipeline(
            image=image,
            safe_water_mask=(
                water_result[
                    "safe_water_mask"
                ]
            ),
            pipeline=oil_pipeline,
            segmentation_threshold=(
                args.segmentation_threshold
            ),
            minimum_component_ratio=(
                args.minimum_component_ratio
            ),
            minimum_component_pixels=(
                args.minimum_component_pixels
            ),
        )

        print(
            "Oil analysis: EXECUTED"
        )

        print(
            "Oil detected:",
            oil_result[
                "oil_detected"
            ],
        )

        print(
            "Final oil pixels:",
            oil_result[
                "final_positive_pixels"
            ],
        )

        print(
            "Oil coverage in safe water:",
            f"{oil_result['final_oil_percent_of_water']:.4f}%",
        )
    else:
        print(
            "Oil analysis: BLOCKED"
        )

        print(
            "Final oil mask: all black"
        )

    result_path = save_outputs(
        output_dir=output_dir,
        image=image,
        water_result=water_result,
        oil_result=oil_result,
    )

    print(
        "Sonuç:",
        result_path,
    )


if __name__ == "__main__":
    main()
