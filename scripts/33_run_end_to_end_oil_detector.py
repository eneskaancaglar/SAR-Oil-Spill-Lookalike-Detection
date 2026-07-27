from __future__ import annotations

import argparse
import json
import math
import sys
from collections import deque
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import transforms
from torchvision.models import resnet18
from torchvision.transforms import InterpolationMode


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.unet import UNet


SEGMENTATION_CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "final_model"
    / "oil_spill_unet_t060.pth"
)

VERIFIER_CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "verifier_v03"
    / "best.pth"
)

CALIBRATION_CONFIG = (
    ROOT
    / "checkpoints"
    / "verifier_v03"
    / "calibration_config.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "U-Net segmentasyon modeli ile ResNet18 verifier "
            "modelini birleştirerek uçtan uca petrol tespiti yapar."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="İşlenecek SAR görüntüsü.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT
            / "outputs"
            / "v03_end_to_end_demo"
        ),
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
        help=(
            "Aday bölgenin görüntü alanına göre minimum oranı. "
            "0.0005 = yüzde 0.05."
        ),
    )

    parser.add_argument(
        "--minimum-component-pixels",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--context",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--minimum-crop-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--maximum-components",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--pixel-size-meters",
        type=float,
        default=None,
        help=(
            "Bir pikselin metre cinsinden kenar uzunluğu. "
            "Bilinmiyorsa verilmemelidir."
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=[
            "auto",
            "cuda",
            "cpu",
        ],
    )

    parser.add_argument(
        "--save-candidate-crops",
        action="store_true",
    )

    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path

    return (
        Path.cwd()
        / path
    ).resolve()


def select_device(
    requested: str,
) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA istendi fakat kullanılamıyor."
            )

        return torch.device("cuda")

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def load_torch_payload(
    path: Path,
    device: torch.device,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint bulunamadı: {path}"
        )

    try:
        payload = torch.load(
            path,
            map_location=device,
            weights_only=False,
        )

    except TypeError:
        payload = torch.load(
            path,
            map_location=device,
        )

    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Checkpoint sözlük formatında değil: {path}"
        )

    return payload


def extract_state_dict(
    payload: dict[str, Any],
) -> dict[str, torch.Tensor]:
    state_dict = None

    for key in (
        "model_state_dict",
        "state_dict",
        "model",
    ):
        candidate = payload.get(key)

        if isinstance(candidate, dict):
            state_dict = candidate
            break

    if state_dict is None:
        if all(
            isinstance(value, torch.Tensor)
            for value in payload.values()
        ):
            state_dict = payload

        else:
            raise RuntimeError(
                "Checkpoint içinde model ağırlıkları bulunamadı."
            )

    cleaned = {}

    for key, value in state_dict.items():
        cleaned_key = str(key)

        changed = True

        while changed:
            changed = False

            for prefix in (
                "module.",
                "_orig_mod.",
                "model.",
            ):
                if cleaned_key.startswith(prefix):
                    cleaned_key = cleaned_key[
                        len(prefix):
                    ]

                    changed = True

        cleaned[cleaned_key] = value

    return cleaned


def build_segmentation_model(
    device: torch.device,
) -> nn.Module:
    payload = load_torch_payload(
        SEGMENTATION_CHECKPOINT,
        device,
    )

    state_dict = extract_state_dict(
        payload
    )

    attempts = [
        (
            "in_channels/out_channels/base_channels",
            lambda: UNet(
                in_channels=1,
                out_channels=1,
                base_channels=16,
            ),
        ),
        (
            "n_channels/n_classes/base_channels",
            lambda: UNet(
                n_channels=1,
                n_classes=1,
                base_channels=16,
            ),
        ),
        (
            "in_channels/out_channels",
            lambda: UNet(
                in_channels=1,
                out_channels=1,
            ),
        ),
        (
            "positional 1,1,16",
            lambda: UNet(1, 1, 16),
        ),
        (
            "positional 1,1",
            lambda: UNet(1, 1),
        ),
        (
            "default",
            lambda: UNet(),
        ),
    ]

    errors = []

    for description, constructor in attempts:
        try:
            model = constructor()

            model.load_state_dict(
                state_dict,
                strict=True,
            )

            model.to(device)
            model.eval()

            print(
                "Segmentasyon modeli:",
                description,
            )

            return model

        except Exception as error:
            errors.append(
                f"{description}: "
                f"{type(error).__name__}: {error}"
            )

    raise RuntimeError(
        "U-Net modeli oluşturulamadı.\n"
        + "\n".join(errors)
    )


def build_verifier_model(
    device: torch.device,
) -> tuple[
    nn.Module,
    dict[str, Any],
]:
    payload = load_torch_payload(
        VERIFIER_CHECKPOINT,
        device,
    )

    model = resnet18(
        weights=None
    )

    model.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=64,
        kernel_size=7,
        stride=2,
        padding=3,
        bias=False,
    )

    model.fc = nn.Linear(
        model.fc.in_features,
        1,
    )

    model.load_state_dict(
        payload["model_state_dict"],
        strict=True,
    )

    model.to(device)
    model.eval()

    if not CALIBRATION_CONFIG.exists():
        raise FileNotFoundError(
            "Calibration config bulunamadı: "
            f"{CALIBRATION_CONFIG}"
        )

    with CALIBRATION_CONFIG.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = json.load(file)

    return model, config


def load_grayscale_image(
    path: Path,
) -> tuple[
    Image.Image,
    np.ndarray,
]:
    if not path.exists():
        raise FileNotFoundError(
            f"Girdi görüntüsü bulunamadı: {path}"
        )

    with Image.open(path) as source:
        image = source.convert("L")

    array = (
        np.asarray(
            image,
            dtype=np.float32,
        )
        / 255.0
    )

    return image, array


def pad_to_multiple(
    tensor: torch.Tensor,
    multiple: int = 16,
) -> tuple[
    torch.Tensor,
    int,
    int,
]:
    height = tensor.shape[-2]
    width = tensor.shape[-1]

    padded_height = int(
        math.ceil(
            height / multiple
        )
        * multiple
    )

    padded_width = int(
        math.ceil(
            width / multiple
        )
        * multiple
    )

    pad_bottom = (
        padded_height
        - height
    )

    pad_right = (
        padded_width
        - width
    )

    padded = F.pad(
        tensor,
        (
            0,
            pad_right,
            0,
            pad_bottom,
        ),
        mode="reflect"
        if height > 1 and width > 1
        else "constant",
    )

    return (
        padded,
        height,
        width,
    )


@torch.inference_mode()
def run_segmentation(
    model: nn.Module,
    image_array: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    tensor = torch.from_numpy(
        image_array.copy()
    ).float()

    tensor = (
        tensor
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )

    padded, original_height, original_width = (
        pad_to_multiple(
            tensor,
            multiple=16,
        )
    )

    amp_enabled = (
        device.type == "cuda"
    )

    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=amp_enabled,
    ):
        output = model(
            padded
        )

        if isinstance(output, dict):
            output = (
                output["out"]
                if "out" in output
                else next(
                    iter(output.values())
                )
            )

        if isinstance(
            output,
            (tuple, list),
        ):
            output = output[0]

        probability = torch.sigmoid(
            output
        )

    probability = probability[
        0,
        0,
        :original_height,
        :original_width,
    ]

    return (
        probability
        .float()
        .cpu()
        .numpy()
    )


def connected_components(
    binary_mask: np.ndarray,
) -> list[dict[str, Any]]:
    height, width = binary_mask.shape

    visited = np.zeros(
        binary_mask.shape,
        dtype=bool,
    )

    components = []

    neighbours = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )

    for start_y in range(height):
        for start_x in range(width):
            if (
                not binary_mask[
                    start_y,
                    start_x,
                ]
                or visited[
                    start_y,
                    start_x,
                ]
            ):
                continue

            queue = deque(
                [
                    (
                        start_y,
                        start_x,
                    )
                ]
            )

            visited[
                start_y,
                start_x,
            ] = True

            pixels = []

            minimum_x = start_x
            maximum_x = start_x
            minimum_y = start_y
            maximum_y = start_y

            while queue:
                y, x = queue.popleft()

                pixels.append(
                    (
                        y,
                        x,
                    )
                )

                minimum_x = min(
                    minimum_x,
                    x,
                )

                maximum_x = max(
                    maximum_x,
                    x,
                )

                minimum_y = min(
                    minimum_y,
                    y,
                )

                maximum_y = max(
                    maximum_y,
                    y,
                )

                for delta_y, delta_x in neighbours:
                    neighbour_y = (
                        y + delta_y
                    )

                    neighbour_x = (
                        x + delta_x
                    )

                    if (
                        neighbour_y < 0
                        or neighbour_y >= height
                        or neighbour_x < 0
                        or neighbour_x >= width
                    ):
                        continue

                    if (
                        binary_mask[
                            neighbour_y,
                            neighbour_x,
                        ]
                        and not visited[
                            neighbour_y,
                            neighbour_x,
                        ]
                    ):
                        visited[
                            neighbour_y,
                            neighbour_x,
                        ] = True

                        queue.append(
                            (
                                neighbour_y,
                                neighbour_x,
                            )
                        )

            components.append(
                {
                    "pixels": pixels,
                    "area_pixels": len(
                        pixels
                    ),
                    "bbox": (
                        minimum_x,
                        minimum_y,
                        maximum_x + 1,
                        maximum_y + 1,
                    ),
                }
            )

    components.sort(
        key=lambda component:
        component["area_pixels"],
        reverse=True,
    )

    return components


def expand_box(
    box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    context_ratio: float,
    minimum_crop_size: int,
) -> tuple[int, int, int, int]:
    left, top, right, bottom = box

    box_width = right - left
    box_height = bottom - top

    centre_x = (
        left + right
    ) / 2.0

    centre_y = (
        top + bottom
    ) / 2.0

    crop_width = max(
        box_width
        * (
            1.0
            + 2.0
            * context_ratio
        ),
        float(
            minimum_crop_size
        ),
    )

    crop_height = max(
        box_height
        * (
            1.0
            + 2.0
            * context_ratio
        ),
        float(
            minimum_crop_size
        ),
    )

    expanded_left = int(
        math.floor(
            centre_x
            - crop_width / 2.0
        )
    )

    expanded_top = int(
        math.floor(
            centre_y
            - crop_height / 2.0
        )
    )

    expanded_right = int(
        math.ceil(
            centre_x
            + crop_width / 2.0
        )
    )

    expanded_bottom = int(
        math.ceil(
            centre_y
            + crop_height / 2.0
        )
    )

    if expanded_left < 0:
        expanded_right -= (
            expanded_left
        )

        expanded_left = 0

    if expanded_top < 0:
        expanded_bottom -= (
            expanded_top
        )

        expanded_top = 0

    if expanded_right > image_width:
        shift = (
            expanded_right
            - image_width
        )

        expanded_left -= shift
        expanded_right = image_width

    if expanded_bottom > image_height:
        shift = (
            expanded_bottom
            - image_height
        )

        expanded_top -= shift
        expanded_bottom = image_height

    expanded_left = max(
        0,
        expanded_left,
    )

    expanded_top = max(
        0,
        expanded_top,
    )

    expanded_right = min(
        image_width,
        max(
            expanded_left + 1,
            expanded_right,
        ),
    )

    expanded_bottom = min(
        image_height,
        max(
            expanded_top + 1,
            expanded_bottom,
        ),
    )

    return (
        expanded_left,
        expanded_top,
        expanded_right,
        expanded_bottom,
    )


def build_verifier_transform(
    config: dict[str, Any],
):
    mean = tuple(
        config.get(
            "normalization_mean",
            [0.449],
        )
    )

    std = tuple(
        config.get(
            "normalization_std",
            [0.226],
        )
    )

    input_size = int(
        config.get(
            "input_size",
            224,
        )
    )

    return transforms.Compose(
        [
            transforms.Resize(
                (
                    input_size,
                    input_size,
                ),
                interpolation=(
                    InterpolationMode.BILINEAR
                ),
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=mean,
                std=std,
            ),
        ]
    )


@torch.inference_mode()
def run_verifier(
    model: nn.Module,
    crop: Image.Image,
    transform,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, float | bool | str]:
    tensor = (
        transform(crop)
        .unsqueeze(0)
        .to(device)
    )

    amp_enabled = (
        device.type == "cuda"
    )

    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=amp_enabled,
    ):
        logit = model(
            tensor
        ).flatten()[0]

    raw_logit = float(
        logit
        .float()
        .cpu()
        .item()
    )

    raw_probability = float(
        torch.sigmoid(
            torch.tensor(
                raw_logit
            )
        ).item()
    )

    temperature = float(
        config.get(
            "temperature",
            1.0,
        )
    )

    probability_mode = str(
        config.get(
            "probability_mode",
            "raw",
        )
    )

    calibrated_probability = float(
        torch.sigmoid(
            torch.tensor(
                raw_logit
                / temperature
            )
        ).item()
    )

    raw_threshold = float(
        config.get(
            "raw_threshold",
            0.15,
        )
    )

    calibrated_threshold = float(
        config.get(
            "calibrated_threshold",
            raw_threshold,
        )
    )

    if probability_mode == "temperature_scaled":
        selected_probability = (
            calibrated_probability
        )

        selected_threshold = (
            calibrated_threshold
        )

    else:
        selected_probability = (
            raw_probability
        )

        selected_threshold = (
            raw_threshold
        )

    accepted = (
        selected_probability
        >= selected_threshold
    )

    return {
        "raw_logit": raw_logit,
        "raw_probability": (
            raw_probability
        ),
        "calibrated_probability": (
            calibrated_probability
        ),
        "selected_probability": (
            selected_probability
        ),
        "selected_threshold": (
            selected_threshold
        ),
        "probability_mode": (
            probability_mode
        ),
        "accepted": bool(
            accepted
        ),
    }


def create_overlay(
    image: Image.Image,
    mask: np.ndarray,
) -> Image.Image:
    base = np.asarray(
        image.convert("RGB"),
        dtype=np.float32,
    ).copy()

    mask_boolean = (
        mask > 0
    )

    base[
        mask_boolean,
        0,
    ] = (
        0.45
        * base[
            mask_boolean,
            0,
        ]
        + 0.55
        * 255.0
    )

    base[
        mask_boolean,
        1,
    ] *= 0.45

    base[
        mask_boolean,
        2,
    ] *= 0.45

    return Image.fromarray(
        np.clip(
            base,
            0,
            255,
        ).astype(
            np.uint8
        )
    )


def create_candidate_overlay(
    image: Image.Image,
    candidates: list[
        dict[str, Any]
    ],
) -> Image.Image:
    overlay = image.convert(
        "RGB"
    ).copy()

    draw = ImageDraw.Draw(
        overlay
    )

    for candidate in candidates:
        left, top, right, bottom = (
            candidate["bbox"]
        )

        accepted = bool(
            candidate["accepted"]
        )

        outline = (
            (255, 0, 0)
            if accepted
            else (255, 255, 0)
        )

        label = (
            f"C{candidate['component_index']} "
            f"P={candidate['selected_probability']:.2f} "
            f"{'KEEP' if accepted else 'REJECT'}"
        )

        draw.rectangle(
            [
                left,
                top,
                right - 1,
                bottom - 1,
            ],
            outline=outline,
            width=3,
        )

        draw.text(
            (
                left + 3,
                top + 3,
            ),
            label,
            fill=(255, 255, 255),
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )

    return overlay


def save_summary_figure(
    image: Image.Image,
    probability: np.ndarray,
    raw_mask: np.ndarray,
    final_mask: np.ndarray,
    destination: Path,
) -> None:
    final_overlay = create_overlay(
        image,
        final_mask,
    )

    figure, axes = plt.subplots(
        1,
        4,
        figsize=(17, 5),
    )

    axes[0].imshow(
        image,
        cmap="gray",
    )

    axes[0].set_title(
        "Original SAR"
    )

    axes[1].imshow(
        probability,
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
    )

    axes[1].set_title(
        "U-Net oil probability"
    )

    axes[2].imshow(
        raw_mask,
        cmap="gray",
        vmin=0,
        vmax=1,
    )

    axes[2].set_title(
        "Raw U-Net mask"
    )

    axes[3].imshow(
        final_overlay
    )

    axes[3].set_title(
        "Verifier-filtered result"
    )

    for axis in axes:
        axis.axis("off")

    figure.tight_layout()

    figure.savefig(
        destination,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def main() -> None:
    args = parse_args()

    input_path = resolve_path(
        args.input
    )

    output_dir = resolve_path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidate_crop_dir = (
        output_dir
        / "candidate_crops"
    )

    if args.save_candidate_crops:
        candidate_crop_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    device = select_device(
        args.device
    )

    print("=" * 78)
    print(
        "ROBUST BINARY OIL DETECTOR v0.3"
    )
    print(
        "END-TO-END INFERENCE"
    )
    print("=" * 78)

    print(
        "Girdi:",
        input_path,
    )

    print(
        "Cihaz:",
        device,
    )

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    segmentation_model = (
        build_segmentation_model(
            device
        )
    )

    (
        verifier_model,
        verifier_config,
    ) = build_verifier_model(
        device
    )

    verifier_transform = (
        build_verifier_transform(
            verifier_config
        )
    )

    image, image_array = (
        load_grayscale_image(
            input_path
        )
    )

    image_width, image_height = (
        image.size
    )

    image_pixel_count = (
        image_width
        * image_height
    )

    print(
        "Görüntü boyutu:",
        f"{image_width}x{image_height}",
    )

    probability = run_segmentation(
        segmentation_model,
        image_array,
        device,
    )

    raw_mask = (
        probability
        >= args.segmentation_threshold
    ).astype(
        np.uint8
    )

    raw_pixel_count = int(
        raw_mask.sum()
    )

    components = connected_components(
        raw_mask
    )

    minimum_area = max(
        int(
            args.minimum_component_pixels
        ),
        int(
            math.ceil(
                image_pixel_count
                * args.minimum_component_ratio
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
    ]

    eligible_components = (
        eligible_components[
            :args.maximum_components
        ]
    )

    print(
        "Ham pozitif piksel:",
        raw_pixel_count,
    )

    print(
        "Bağlantılı bileşen:",
        len(components),
    )

    print(
        "Minimum aday alanı:",
        minimum_area,
        "piksel",
    )

    print(
        "Verifier'a girecek aday:",
        len(
            eligible_components
        ),
    )

    final_mask = np.zeros(
        raw_mask.shape,
        dtype=np.uint8,
    )

    candidate_results = []

    for component_index, component in enumerate(
        eligible_components,
        start=1,
    ):
        bbox = component["bbox"]

        crop_box = expand_box(
            bbox,
            image_width,
            image_height,
            args.context,
            args.minimum_crop_size,
        )

        crop = image.crop(
            crop_box
        )

        verifier_result = run_verifier(
            verifier_model,
            crop,
            verifier_transform,
            verifier_config,
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

        candidate_result = {
            "component_index": (
                component_index
            ),
            "area_pixels": int(
                component[
                    "area_pixels"
                ]
            ),
            "area_percent_of_image": float(
                component[
                    "area_pixels"
                ]
                / image_pixel_count
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
            **verifier_result,
        }

        candidate_results.append(
            candidate_result
        )

        if args.save_candidate_crops:
            crop.save(
                candidate_crop_dir
                / (
                    f"candidate_"
                    f"{component_index:03d}"
                    f"__p_"
                    f"{verifier_result['selected_probability']:.4f}"
                    f"__"
                    f"{'keep' if accepted else 'reject'}"
                    f".png"
                )
            )

        print(
            f"Candidate {component_index:02d} "
            f"| alan={component['area_pixels']:6d} px "
            f"| verifier="
            f"{verifier_result['selected_probability']:.4f} "
            f"| threshold="
            f"{verifier_result['selected_threshold']:.4f} "
            f"| "
            f"{'KEEP' if accepted else 'REJECT'}"
        )

    final_pixel_count = int(
        final_mask.sum()
    )

    raw_coverage_percent = (
        raw_pixel_count
        / image_pixel_count
        * 100.0
    )

    final_coverage_percent = (
        final_pixel_count
        / image_pixel_count
        * 100.0
    )

    kept_candidates = [
        candidate
        for candidate in candidate_results
        if candidate["accepted"]
    ]

    rejected_candidates = [
        candidate
        for candidate in candidate_results
        if not candidate["accepted"]
    ]

    all_candidate_probabilities = [
        float(
            candidate[
                "selected_probability"
            ]
        )
        for candidate in candidate_results
    ]

    kept_probabilities = [
        float(
            candidate[
                "selected_probability"
            ]
        )
        for candidate in kept_candidates
    ]

    maximum_candidate_probability = (
        max(
            all_candidate_probabilities
        )
        if all_candidate_probabilities
        else 0.0
    )

    has_oil = (
        final_pixel_count > 0
    )

    if has_oil:
        decision_confidence = max(
            kept_probabilities
        )

    else:
        decision_confidence = (
            1.0
            - maximum_candidate_probability
        )

    estimated_area_square_meters = None
    estimated_area_square_kilometers = None

    if args.pixel_size_meters is not None:
        if args.pixel_size_meters <= 0:
            raise ValueError(
                "--pixel-size-meters sıfırdan büyük olmalıdır."
            )

        pixel_area = (
            args.pixel_size_meters
            ** 2
        )

        estimated_area_square_meters = (
            final_pixel_count
            * pixel_area
        )

        estimated_area_square_kilometers = (
            estimated_area_square_meters
            / 1_000_000.0
        )

    Image.fromarray(
        (
            probability
            * 255.0
        ).clip(
            0,
            255,
        ).astype(
            np.uint8
        )
    ).save(
        output_dir
        / "segmentation_probability.png"
    )

    Image.fromarray(
        raw_mask
        * 255
    ).save(
        output_dir
        / "raw_segmentation_mask.png"
    )

    Image.fromarray(
        final_mask
        * 255
    ).save(
        output_dir
        / "final_oil_mask.png"
    )

    final_overlay = create_overlay(
        image,
        final_mask,
    )

    final_overlay.save(
        output_dir
        / "final_overlay.png"
    )

    candidate_overlay = (
        create_candidate_overlay(
            image,
            candidate_results,
        )
    )

    candidate_overlay.save(
        output_dir
        / "candidate_decisions.png"
    )

    save_summary_figure(
        image,
        probability,
        raw_mask,
        final_mask,
        output_dir
        / "summary_figure.png",
    )

    candidate_table = pd.DataFrame(
        candidate_results
    )

    candidate_table.to_csv(
        output_dir
        / "candidate_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    result = {
        "input_path": str(
            input_path
        ),
        "image_width": int(
            image_width
        ),
        "image_height": int(
            image_height
        ),
        "segmentation_threshold": float(
            args.segmentation_threshold
        ),
        "minimum_component_pixels": int(
            minimum_area
        ),
        "verifier_probability_mode": (
            verifier_config.get(
                "probability_mode",
                "raw",
            )
        ),
        "verifier_temperature": float(
            verifier_config.get(
                "temperature",
                1.0,
            )
        ),
        "verifier_raw_threshold": float(
            verifier_config.get(
                "raw_threshold",
                0.15,
            )
        ),
        "verifier_calibrated_threshold": float(
            verifier_config.get(
                "calibrated_threshold",
                0.15,
            )
        ),
        "oil_detected": bool(
            has_oil
        ),
        "raw_positive_pixels": int(
            raw_pixel_count
        ),
        "final_oil_pixels": int(
            final_pixel_count
        ),
        "raw_coverage_percent_of_image": float(
            raw_coverage_percent
        ),
        "final_oil_coverage_percent_of_image": float(
            final_coverage_percent
        ),
        "candidate_count": int(
            len(candidate_results)
        ),
        "accepted_candidate_count": int(
            len(kept_candidates)
        ),
        "rejected_candidate_count": int(
            len(rejected_candidates)
        ),
        "maximum_candidate_oil_probability": float(
            maximum_candidate_probability
        ),
        "decision_confidence_estimate": float(
            decision_confidence
        ),
        "pixel_size_meters": (
            float(
                args.pixel_size_meters
            )
            if args.pixel_size_meters
            is not None
            else None
        ),
        "estimated_oil_area_square_meters": (
            float(
                estimated_area_square_meters
            )
            if estimated_area_square_meters
            is not None
            else None
        ),
        "estimated_oil_area_square_kilometers": (
            float(
                estimated_area_square_kilometers
            )
            if estimated_area_square_kilometers
            is not None
            else None
        ),
        "candidate_results": (
            candidate_results
        ),
        "important_note": (
            "Confidence is a candidate-crop verifier score, "
            "not yet a certified whole-image oil probability. "
            "Coverage is calculated over the full image because "
            "a separate land-sea mask is not yet included."
        ),
    }

    with (
        output_dir
        / "result.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("=" * 78)
    print("NİHAİ SONUÇ")
    print("=" * 78)

    print(
        "Petrol durumu:",
        (
            "PETROL TESPİT EDİLDİ"
            if has_oil
            else "PETROL TESPİT EDİLMEDİ"
        ),
    )

    print(
        "Kabul edilen aday:",
        len(kept_candidates),
    )

    print(
        "Reddedilen aday:",
        len(rejected_candidates),
    )

    print(
        "Ham U-Net kaplama oranı:",
        f"%{raw_coverage_percent:.4f}",
    )

    print(
        "Nihai petrol kaplama oranı:",
        f"%{final_coverage_percent:.4f}",
    )

    print(
        "En yüksek aday petrol güveni:",
        f"%{maximum_candidate_probability * 100.0:.2f}",
    )

    print(
        "Karar güven tahmini:",
        f"%{decision_confidence * 100.0:.2f}",
    )

    if (
        estimated_area_square_kilometers
        is not None
    ):
        print(
            "Tahmini petrol alanı:",
            f"{estimated_area_square_kilometers:.6f} km²",
        )

    print()
    print(
        "ÖNEMLİ: Kaplama oranı şu anda bütün "
        "görüntü alanına göre hesaplanmaktadır."
    )

    print(
        "Ayrı kara-deniz maskesi henüz pipeline'a "
        "eklenmemiştir."
    )

    print()
    print(
        "Çıktı klasörü:",
        output_dir,
    )

    print(
        "Özet görsel:",
        output_dir
        / "summary_figure.png",
    )

    print(
        "Nihai maske:",
        output_dir
        / "final_oil_mask.png",
    )

    print(
        "Overlay:",
        output_dir
        / "final_overlay.png",
    )

    print(
        "Sonuç JSON:",
        output_dir
        / "result.json",
    )


if __name__ == "__main__":
    main()
