from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


PACKAGE_DIR = Path(__file__).resolve().parent
MODEL_DEFINITION = PACKAGE_DIR / "model_definition.py"
POLICY_PATH = PACKAGE_DIR / "policy.json"
MANIFEST_PATH = PACKAGE_DIR / "package_manifest.json"


def load_model_module():
    spec = importlib.util.spec_from_file_location(
        "final_water_gate_model_definition",
        MODEL_DEFINITION,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Model tanımı yüklenemedi: {MODEL_DEFINITION}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODEL_MODULE = load_model_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "v0.6 seçici kara-su güvenlik kapısı. "
            "Yalnız yüksek güvenli su maskesi üretir."
        )
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
    )
    return parser.parse_args()


def normalize_percentile(image: np.ndarray) -> np.ndarray:
    array = image.astype(np.float32)
    low = float(np.percentile(array, 2))
    high = float(np.percentile(array, 98))

    if high <= low:
        return (array / 255.0).astype(np.float32)

    return np.clip(
        (array - low) / (high - low),
        0.0,
        1.0,
    ).astype(np.float32)


def align_probability(
    probability: np.ndarray,
    threshold: float,
) -> np.ndarray:
    probability = np.clip(
        probability,
        1e-6,
        1.0 - 1e-6,
    )
    threshold = float(
        np.clip(threshold, 1e-4, 1.0 - 1e-4)
    )

    probability_logit = np.log(
        probability / (1.0 - probability)
    )
    threshold_logit = math.log(
        threshold / (1.0 - threshold)
    )
    shifted = probability_logit - threshold_logit

    return (
        1.0
        / (
            1.0
            + np.exp(
                -np.clip(shifted, -30.0, 30.0)
            )
        )
    ).astype(np.float32)


def select_device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")

    if value == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA istendi ancak kullanılamıyor."
            )
        return torch.device("cuda")

    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def load_models(
    device: torch.device,
) -> tuple[list[torch.nn.Module], list[float]]:
    manifest = json.loads(
        MANIFEST_PATH.read_text(encoding="utf-8-sig")
    )

    models = []
    thresholds = []

    for item in manifest["models"]:
        checkpoint_path = PACKAGE_DIR / item["file"]
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        base_channels = int(
            checkpoint.get(
                "config",
                {},
            ).get(
                "base_channels",
                16,
            )
        )

        model = MODEL_MODULE.SmallUNet(
            in_channels=2,
            base_channels=base_channels,
        ).to(device)

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
        model.eval()

        threshold = float(
            checkpoint.get(
                "validation_metrics",
                {},
            ).get(
                "threshold",
                item["calibration_threshold"],
            )
        )

        models.append(model)
        thresholds.append(threshold)

    if len(models) != 5:
        raise RuntimeError(
            f"Beş model bekleniyordu: {len(models)}"
        )

    return models, thresholds


def save_uint8(array: np.ndarray, path: Path) -> None:
    Image.fromarray(
        array.astype(np.uint8)
    ).save(path)


def main() -> None:
    args = parse_args()

    image_path = args.image.resolve()
    output_dir = args.output_dir.resolve()

    if not image_path.exists():
        raise FileNotFoundError(
            f"Görüntü bulunamadı: {image_path}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    policy = json.loads(
        POLICY_PATH.read_text(encoding="utf-8-sig")
    )

    device = select_device(args.device)
    models, thresholds = load_models(device)

    image = np.array(
        Image.open(image_path).convert("L")
    )
    normalized = normalize_percentile(image)

    tensor = torch.from_numpy(
        np.stack(
            [normalized, normalized],
            axis=0,
        ).astype(np.float32)
    ).unsqueeze(0).to(device)

    aligned_probabilities = []

    with torch.no_grad():
        for model, threshold in zip(
            models,
            thresholds,
        ):
            with torch.amp.autocast(
                device_type=device.type,
                enabled=device.type == "cuda",
            ):
                logits = model(tensor)[0, 0]

            probability = torch.sigmoid(
                logits.float()
            ).cpu().numpy()

            aligned_probabilities.append(
                align_probability(
                    probability,
                    threshold,
                )
            )

    stack = np.stack(
        aligned_probabilities,
        axis=0,
    )

    ensemble_probability = np.median(
        stack,
        axis=0,
    ).astype(np.float32)

    ensemble_std = np.std(
        stack,
        axis=0,
    ).astype(np.float32)

    water_threshold = float(
        policy["water_threshold"]
    )
    land_threshold = float(
        policy["land_threshold"]
    )

    safe_water = (
        ensemble_probability >= water_threshold
    )
    safe_land = (
        ensemble_probability <= land_threshold
    )
    uncertain = (
        (~safe_water) & (~safe_land)
    )

    total_pixels = max(int(image.size), 1)

    predicted_water_fraction = float(
        safe_water.sum() / total_pixels
    )
    uncertain_fraction = float(
        uncertain.sum() / total_pixels
    )

    accepted = (
        predicted_water_fraction
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

    decision = (
        "ACCEPT_WATER"
        if accepted
        else "UNCERTAIN_BLOCK"
    )

    operational_water = np.zeros(
        image.shape,
        dtype=np.uint8,
    )

    if accepted:
        operational_water[safe_water] = 255

    internal_water = np.zeros(
        image.shape,
        dtype=np.uint8,
    )
    internal_water[safe_water] = 255

    uncertain_mask = np.zeros(
        image.shape,
        dtype=np.uint8,
    )
    uncertain_mask[uncertain] = 255

    probability_uint8 = (
        np.clip(
            ensemble_probability,
            0.0,
            1.0,
        )
        * 255.0
    ).astype(np.uint8)

    disagreement_uint8 = (
        np.clip(
            ensemble_std,
            0.0,
            0.5,
        )
        / 0.5
        * 255.0
    ).astype(np.uint8)

    save_uint8(
        operational_water,
        output_dir / "safe_water_mask.png",
    )
    save_uint8(
        internal_water,
        output_dir
        / "internal_high_confidence_water.png",
    )
    save_uint8(
        uncertain_mask,
        output_dir / "uncertain_mask.png",
    )
    save_uint8(
        probability_uint8,
        output_dir / "ensemble_probability.png",
    )
    save_uint8(
        disagreement_uint8,
        output_dir / "ensemble_disagreement.png",
    )

    result = {
        "decision": decision,
        "oil_analysis_allowed": bool(accepted),
        "oil_mask_allowed": bool(accepted),
        "predicted_water_fraction": (
            predicted_water_fraction
        ),
        "uncertain_fraction": uncertain_fraction,
        "mean_ensemble_disagreement": float(
            ensemble_std.mean()
        ),
        "model_count": len(models),
        "aggregation": (
            "median of logit-aligned probabilities"
        ),
        "image": str(image_path),
        "policy": policy,
    }

    result_path = output_dir / "decision.json"
    result_path.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("=" * 72)
    print("FINAL WATER GATE")
    print("=" * 72)
    print("Decision:", decision)
    print("Oil analysis allowed:", accepted)
    print(
        "Predicted water fraction:",
        f"{predicted_water_fraction:.4f}",
    )
    print(
        "Uncertain fraction:",
        f"{uncertain_fraction:.4f}",
    )
    print(
        "Mean disagreement:",
        f"{ensemble_std.mean():.4f}",
    )
    print("Result:", result_path)


if __name__ == "__main__":
    main()
