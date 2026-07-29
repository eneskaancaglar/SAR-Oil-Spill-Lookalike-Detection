from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]

BASE_SCRIPT = (
    ROOT
    / "scripts"
    / "68_evaluate_dartis_water_transfer_gate.py"
)

DEFAULT_DARTIS_MANIFEST = (
    ROOT
    / "data"
    / "metadata"
    / "v06_dartis_water_masks_final.csv"
)

DEFAULT_ACTIVE_MANIFEST = (
    ROOT
    / "data"
    / "metadata"
    / "v06_dartis_water_active_training_manifest.csv"
)

DEFAULT_PILOT_MANIFEST = (
    ROOT
    / "data"
    / "metadata"
    / "v06_manual_land_corrections.csv"
)

DEFAULT_FOLD_DIRECTORY = (
    ROOT
    / "checkpoints"
    / "water_unet_dartis_landaware_cv_v06"
)

DEFAULT_SOURCE_CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "water_unet_flood_v06_robust"
    / "best.pth"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "v06_dartis_water_targeted_hard_batch"
)

ANNOTATION_DIR = (
    ROOT
    / "data"
    / "annotation"
    / "v06_dartis_water_targeted_hard_batch"
)

QUEUE_PATH = (
    ROOT
    / "data"
    / "metadata"
    / "v06_dartis_water_targeted_hard_batch.csv"
)


def load_base_module():
    if not BASE_SCRIPT.exists():
        raise FileNotFoundError(
            f"Gerekli 68 scripti bulunamadı: {BASE_SCRIPT}"
        )

    spec = importlib.util.spec_from_file_location(
        "dartis_water_base_v79",
        BASE_SCRIPT,
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "68 scripti modül olarak yüklenemedi."
        )

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_base_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "DARTIS land-aware CV modellerinin zorlandığı sahne tiplerini "
            "hedefleyerek 8 nc ve 8 oc olmak üzere 16 yeni kara-su "
            "etiketleme adayı seçer. Eski 32 aktif maske ve 7 pilot maske "
            "tekrar seçilmez. Eğitim ve kilitli test kullanımı yapmaz."
        )
    )

    parser.add_argument(
        "--dartis-manifest",
        type=Path,
        default=DEFAULT_DARTIS_MANIFEST,
    )
    parser.add_argument(
        "--active-manifest",
        type=Path,
        default=DEFAULT_ACTIVE_MANIFEST,
    )
    parser.add_argument(
        "--pilot-manifest",
        type=Path,
        default=DEFAULT_PILOT_MANIFEST,
    )
    parser.add_argument(
        "--fold-directory",
        type=Path,
        default=DEFAULT_FOLD_DIRECTORY,
    )
    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        default=DEFAULT_SOURCE_CHECKPOINT,
    )
    parser.add_argument(
        "--per-group",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--inference-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--export-size",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)

    if not path.is_absolute():
        path = ROOT / path

    return path.resolve()


def relative(path: Path) -> str:
    try:
        return str(
            path.resolve().relative_to(ROOT.resolve())
        ).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def normalize_percentile(
    image: np.ndarray,
) -> np.ndarray:
    array = image.astype(np.float32)

    low = float(np.percentile(array, 2))
    high = float(np.percentile(array, 98))

    if high <= low:
        return (
            array / 255.0
        ).astype(np.float32)

    return np.clip(
        (array - low)
        / (high - low),
        0.0,
        1.0,
    ).astype(np.float32)


def coastal_group(sample_id: str) -> str:
    prefix = sample_id.split(
        ":",
        1,
    )[0].lower()

    if prefix in {"nc", "oc"}:
        return prefix

    return ""


def excluded_sample_ids(
    manifest_paths: list[Path],
) -> set[str]:
    excluded: set[str] = set()

    for path in manifest_paths:
        if not path.exists():
            continue

        frame = pd.read_csv(
            path,
            encoding="utf-8-sig",
            low_memory=False,
        )

        if "sample_id" in frame.columns:
            excluded.update(
                frame[
                    "sample_id"
                ].astype(str)
            )

    return excluded


def load_candidates(
    dartis_manifest_path: Path,
    excluded_ids: set[str],
) -> pd.DataFrame:
    frame = pd.read_csv(
        dartis_manifest_path,
        encoding="utf-8-sig",
        low_memory=False,
    )

    required = {
        "sample_id",
        "image_path",
    }

    missing = required - set(frame.columns)

    if missing:
        raise RuntimeError(
            "DARTIS manifestinde eksik sütunlar: "
            f"{sorted(missing)}"
        )

    frame = frame.copy()
    frame["sample_id"] = frame[
        "sample_id"
    ].astype(str)

    frame["coastal_group"] = frame[
        "sample_id"
    ].map(coastal_group)

    frame = frame[
        frame[
            "coastal_group"
        ].isin(["nc", "oc"])
    ].copy()

    frame = frame[
        ~frame[
            "sample_id"
        ].isin(excluded_ids)
    ].copy()

    frame["resolved_image_path"] = frame[
        "image_path"
    ].map(
        lambda value: str(
            resolve_path(value)
        )
    )

    missing_files = frame[
        ~frame[
            "resolved_image_path"
        ].map(
            lambda value: Path(value).exists()
        )
    ]

    if not missing_files.empty:
        examples = ", ".join(
            missing_files[
                "resolved_image_path"
            ].head(5)
        )

        raise FileNotFoundError(
            "Bazı DARTIS görüntüleri bulunamadı: "
            f"{examples}"
        )

    return frame.reset_index(drop=True)


class CandidateDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        image_size: int,
    ) -> None:
        self.records = frame.to_dict(
            orient="records"
        )
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[
        torch.Tensor,
        str,
        torch.Tensor,
        torch.Tensor,
    ]:
        record = self.records[index]

        image = np.array(
            Image.open(
                record[
                    "resolved_image_path"
                ]
            ).convert("L")
        )

        normalized = normalize_percentile(
            image
        )

        resized = cv2.resize(
            normalized,
            (
                self.image_size,
                self.image_size,
            ),
            interpolation=(
                cv2.INTER_AREA
                if max(image.shape)
                > self.image_size
                else cv2.INTER_CUBIC
            ),
        )

        sobel_x = cv2.Sobel(
            resized,
            cv2.CV_32F,
            1,
            0,
            ksize=3,
        )

        sobel_y = cv2.Sobel(
            resized,
            cv2.CV_32F,
            0,
            1,
            ksize=3,
        )

        gradient = cv2.magnitude(
            sobel_x,
            sobel_y,
        )

        edge_density = float(
            np.mean(
                gradient
                > np.percentile(
                    gradient,
                    75,
                )
            )
        )

        descriptor = cv2.resize(
            resized,
            (16, 16),
            interpolation=cv2.INTER_AREA,
        ).reshape(-1)

        descriptor = descriptor.astype(
            np.float32
        )

        descriptor -= float(
            descriptor.mean()
        )

        descriptor /= max(
            float(
                np.linalg.norm(
                    descriptor
                )
            ),
            1e-6,
        )

        two_channel = np.stack(
            [resized, resized],
            axis=0,
        ).astype(np.float32)

        return (
            torch.from_numpy(
                two_channel
            ),
            str(record["sample_id"]),
            torch.tensor(
                edge_density,
                dtype=torch.float32,
            ),
            torch.from_numpy(
                descriptor
            ),
        )


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[
    torch.nn.Module,
    float,
]:
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

    model = BASE.SmallUNet(
        in_channels=2,
        base_channels=base_channels,
    ).to(device)

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    threshold = float(
        checkpoint.get(
            "validation_metrics",
            {},
        ).get(
            "threshold",
            checkpoint.get(
                "selected_threshold",
                0.90,
            ),
        )
    )

    return model, threshold


def load_models(
    fold_directory: Path,
    source_checkpoint_path: Path,
    device: torch.device,
) -> tuple[
    list[torch.nn.Module],
    list[float],
    list[str],
    torch.nn.Module,
    float,
]:
    fold_paths = sorted(
        fold_directory.glob(
            "fold_*_best.pth"
        )
    )

    if len(fold_paths) != 4:
        raise RuntimeError(
            "Dört land-aware fold checkpointi bekleniyordu. "
            f"Bulunan: {len(fold_paths)}"
        )

    models = []
    thresholds = []
    names = []

    for path in fold_paths:
        model, threshold = load_model(
            path,
            device,
        )

        models.append(model)
        thresholds.append(threshold)
        names.append(path.name)

    source_model, source_threshold = (
        load_model(
            source_checkpoint_path,
            device,
        )
    )

    return (
        models,
        thresholds,
        names,
        source_model,
        source_threshold,
    )


def robust_minmax(
    series: pd.Series,
) -> pd.Series:
    values = series.astype(float)

    low = float(
        values.quantile(0.05)
    )
    high = float(
        values.quantile(0.95)
    )

    if high <= low:
        return pd.Series(
            np.zeros(
                len(values),
                dtype=np.float64,
            ),
            index=series.index,
        )

    return (
        (values - low)
        / (high - low)
    ).clip(0.0, 1.0)


def score_candidates(
    frame: pd.DataFrame,
    fold_models: list[
        torch.nn.Module
    ],
    fold_thresholds: list[float],
    source_model: torch.nn.Module,
    source_threshold: float,
    device: torch.device,
    inference_size: int,
    batch_size: int,
    num_workers: int,
) -> pd.DataFrame:
    dataset = CandidateDataset(
        frame,
        inference_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
        drop_last=False,
    )

    threshold_tensor = torch.tensor(
        fold_thresholds,
        dtype=torch.float32,
        device=device,
    ).view(-1, 1, 1, 1)

    rows: list[
        dict[str, Any]
    ] = []

    with torch.no_grad():
        for (
            images,
            sample_ids,
            edge_density,
            descriptors,
        ) in loader:
            images = images.to(
                device,
                non_blocking=True,
            )

            fold_probabilities = []

            for model in fold_models:
                with torch.amp.autocast(
                    device_type=device.type,
                    enabled=(
                        device.type
                        == "cuda"
                    ),
                ):
                    logits = model(images)

                fold_probabilities.append(
                    torch.sigmoid(
                        logits[:, 0].float()
                    )
                )

            stack = torch.stack(
                fold_probabilities,
                dim=0,
            )

            votes = (
                stack
                >= threshold_tensor
            ).float()

            vote_fraction = votes.mean(
                dim=0
            )

            mean_probability = stack.mean(
                dim=0
            )

            standard_deviation = stack.std(
                dim=0,
                unbiased=False,
            )

            consensus_water = (
                vote_fraction >= 0.75
            )

            consensus_land = (
                vote_fraction <= 0.25
            )

            uncertain = (
                (~consensus_water)
                & (~consensus_land)
            )

            with torch.amp.autocast(
                device_type=device.type,
                enabled=(
                    device.type
                    == "cuda"
                ),
            ):
                source_logits = (
                    source_model(images)
                )

            source_probability = torch.sigmoid(
                source_logits[
                    :, 0
                ].float()
            )

            source_water = (
                source_probability
                >= source_threshold
            )

            for batch_index, sample_id in enumerate(
                sample_ids
            ):
                land_fraction = float(
                    consensus_land[
                        batch_index
                    ].float().mean().item()
                )

                water_fraction = float(
                    consensus_water[
                        batch_index
                    ].float().mean().item()
                )

                uncertain_fraction = float(
                    uncertain[
                        batch_index
                    ].float().mean().item()
                )

                source_water_fraction = float(
                    source_water[
                        batch_index
                    ].float().mean().item()
                )

                source_gap = max(
                    0.0,
                    source_water_fraction
                    - water_fraction,
                )

                rows.append(
                    {
                        "sample_id": str(
                            sample_id
                        ),
                        "consensus_land_fraction": (
                            land_fraction
                        ),
                        "consensus_water_fraction": (
                            water_fraction
                        ),
                        "uncertain_fraction": (
                            uncertain_fraction
                        ),
                        "ensemble_std": float(
                            standard_deviation[
                                batch_index
                            ].mean().item()
                        ),
                        "mean_probability": float(
                            mean_probability[
                                batch_index
                            ].mean().item()
                        ),
                        "source_water_fraction": (
                            source_water_fraction
                        ),
                        "source_minus_landaware": (
                            source_gap
                        ),
                        "edge_density": float(
                            edge_density[
                                batch_index
                            ].item()
                        ),
                        "descriptor": "|".join(
                            f"{float(value):.7f}"
                            for value
                            in descriptors[
                                batch_index
                            ].numpy()
                        ),
                    }
                )

    scores = pd.DataFrame(rows)

    merged = frame.merge(
        scores,
        on="sample_id",
        how="inner",
        validate="one_to_one",
    )

    merged[
        "disagreement_score"
    ] = (
        0.65
        * robust_minmax(
            merged[
                "uncertain_fraction"
            ]
        )
        + 0.35
        * robust_minmax(
            merged[
                "ensemble_std"
            ]
        )
    )

    medium_water = (
        1.0
        - np.abs(
            merged[
                "consensus_water_fraction"
            ]
            - 0.5
        )
        * 2.0
    ).clip(0.0, 1.0)

    merged[
        "boundary_score"
    ] = (
        0.60
        * robust_minmax(
            merged[
                "edge_density"
            ]
        )
        + 0.25
        * medium_water
        + 0.15
        * robust_minmax(
            merged[
                "uncertain_fraction"
            ]
        )
    )

    merged[
        "land_heavy_score"
    ] = (
        0.80
        * robust_minmax(
            merged[
                "consensus_land_fraction"
            ]
        )
        + 0.20
        * robust_minmax(
            merged[
                "edge_density"
            ]
        )
    )

    merged[
        "water_miss_risk_score"
    ] = (
        0.75
        * robust_minmax(
            merged[
                "source_minus_landaware"
            ]
        )
        + 0.25
        * robust_minmax(
            merged[
                "uncertain_fraction"
            ]
        )
    )

    return merged


def descriptor_array(
    value: str,
) -> np.ndarray:
    return np.array(
        [
            float(item)
            for item in value.split("|")
            if item
        ],
        dtype=np.float32,
    )


def select_diverse(
    frame: pd.DataFrame,
    score_column: str,
    count: int,
    selected_ids: set[str],
    reason: str,
) -> dict[str, str]:
    selected: dict[str, str] = {}

    candidates = frame[
        ~frame[
            "sample_id"
        ].isin(selected_ids)
    ].sort_values(
        score_column,
        ascending=False,
    ).head(
        min(
            80,
            len(frame),
        )
    ).copy()

    if candidates.empty:
        return selected

    descriptors = {
        str(row.sample_id): descriptor_array(
            row.descriptor
        )
        for row in candidates.itertuples()
    }

    selected_descriptors = [
        descriptor_array(value)
        for value in frame.loc[
            frame[
                "sample_id"
            ].isin(selected_ids),
            "descriptor",
        ].astype(str)
    ]

    while (
        len(selected) < count
        and not candidates.empty
    ):
        best_id = None
        best_value = -1e9

        for row in candidates.itertuples():
            sample_id = str(
                row.sample_id
            )

            if (
                sample_id in selected_ids
                or sample_id in selected
            ):
                continue

            score = float(
                getattr(
                    row,
                    score_column,
                )
            )

            descriptor = descriptors[
                sample_id
            ]

            comparison = (
                selected_descriptors
                + [
                    descriptors[item]
                    for item in selected
                ]
            )

            if comparison:
                minimum_distance = min(
                    float(
                        1.0
                        - np.clip(
                            np.dot(
                                descriptor,
                                other,
                            ),
                            -1.0,
                            1.0,
                        )
                    )
                    for other in comparison
                )
            else:
                minimum_distance = 1.0

            combined = (
                0.75 * score
                + 0.25
                * np.clip(
                    minimum_distance,
                    0.0,
                    2.0,
                )
                / 2.0
            )

            if combined > best_value:
                best_value = combined
                best_id = sample_id

        if best_id is None:
            break

        selected[best_id] = reason

    return selected


def select_group(
    group_frame: pd.DataFrame,
    count: int,
) -> pd.DataFrame:
    if count != 8:
        raise ValueError(
            "Bu aşamada her grup için 8 örnek seçilmelidir."
        )

    selected_reasons: dict[
        str,
        set[str]
    ] = {}

    def add(
        additions: dict[str, str],
    ) -> None:
        for sample_id, reason in additions.items():
            selected_reasons.setdefault(
                sample_id,
                set(),
            ).add(reason)

    add(
        select_diverse(
            group_frame,
            "land_heavy_score",
            3,
            set(selected_reasons),
            "land_heavy",
        )
    )

    add(
        select_diverse(
            group_frame,
            "water_miss_risk_score",
            2,
            set(selected_reasons),
            "water_miss_risk",
        )
    )

    add(
        select_diverse(
            group_frame,
            "disagreement_score",
            2,
            set(selected_reasons),
            "model_disagreement",
        )
    )

    add(
        select_diverse(
            group_frame,
            "boundary_score",
            1,
            set(selected_reasons),
            "boundary_rich",
        )
    )

    if len(selected_reasons) < count:
        fill = select_diverse(
            group_frame,
            "land_heavy_score",
            count - len(
                selected_reasons
            ),
            set(selected_reasons),
            "diversity_fill",
        )
        add(fill)

    selected_ids = list(
        selected_reasons
    )[:count]

    selected = group_frame[
        group_frame[
            "sample_id"
        ].isin(selected_ids)
    ].copy()

    selected[
        "selection_reason"
    ] = selected[
        "sample_id"
    ].map(
        lambda sample_id: "|".join(
            sorted(
                selected_reasons[
                    str(sample_id)
                ]
            )
        )
    )

    return selected


def infer_export_prediction(
    image: np.ndarray,
    fold_models: list[
        torch.nn.Module
    ],
    fold_thresholds: list[float],
    device: torch.device,
    export_size: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    resized = cv2.resize(
        image,
        (
            export_size,
            export_size,
        ),
        interpolation=(
            cv2.INTER_AREA
            if max(image.shape)
            > export_size
            else cv2.INTER_CUBIC
        ),
    )

    normalized = normalize_percentile(
        resized
    )

    tensor = torch.from_numpy(
        np.stack(
            [normalized, normalized],
            axis=0,
        ).astype(np.float32)
    ).unsqueeze(0).to(device)

    probabilities = []

    with torch.no_grad():
        for model in fold_models:
            with torch.amp.autocast(
                device_type=device.type,
                enabled=(
                    device.type
                    == "cuda"
                ),
            ):
                logits = model(tensor)

            probabilities.append(
                torch.sigmoid(
                    logits[
                        0,
                        0,
                    ].float()
                ).cpu().numpy()
            )

    stack = np.stack(
        probabilities,
        axis=0,
    )

    thresholds = np.array(
        fold_thresholds,
        dtype=np.float32,
    ).reshape(-1, 1, 1)

    vote_fraction = (
        stack
        >= thresholds
    ).mean(axis=0)

    prelabel = np.full(
        (
            export_size,
            export_size,
        ),
        128,
        dtype=np.uint8,
    )

    prelabel[
        vote_fraction <= 0.25
    ] = 0

    prelabel[
        vote_fraction >= 0.75
    ] = 255

    return vote_fraction.astype(
        np.float32
    ), prelabel


def add_title(
    image: Image.Image,
    title: str,
    width: int,
) -> Image.Image:
    ratio = width / image.width

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

    header_height = 30

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
        (0, header_height),
    )

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    draw.text(
        (6, 8),
        title,
        fill="white",
        font=font,
    )

    return canvas


def export_package(
    selected: pd.DataFrame,
    fold_models: list[
        torch.nn.Module
    ],
    fold_thresholds: list[float],
    device: torch.device,
    export_size: int,
) -> Path:
    image_directory = (
        ANNOTATION_DIR
        / "images"
    )
    probability_directory = (
        ANNOTATION_DIR
        / "ensemble_vote_fraction"
    )
    prelabel_directory = (
        ANNOTATION_DIR
        / "prelabel_water"
    )
    overlay_directory = (
        ANNOTATION_DIR
        / "review_overlay"
    )
    manual_mask_directory = (
        ANNOTATION_DIR
        / "manual_masks"
    )

    for directory in (
        image_directory,
        probability_directory,
        prelabel_directory,
        overlay_directory,
        manual_mask_directory,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    contact_rows = []

    for row in selected.itertuples():
        sample_id = str(
            row.sample_id
        )

        safe_name = sample_id.replace(
            ":",
            "__",
        )

        original = np.array(
            Image.open(
                row.resolved_image_path
            ).convert("L")
        )

        resized = cv2.resize(
            original,
            (
                export_size,
                export_size,
            ),
            interpolation=(
                cv2.INTER_AREA
                if max(original.shape)
                > export_size
                else cv2.INTER_CUBIC
            ),
        )

        vote_fraction, prelabel = (
            infer_export_prediction(
                original,
                fold_models,
                fold_thresholds,
                device,
                export_size,
            )
        )

        probability_uint8 = (
            np.clip(
                vote_fraction,
                0.0,
                1.0,
            )
            * 255.0
        ).astype(np.uint8)

        heatmap = cv2.applyColorMap(
            probability_uint8,
            cv2.COLORMAP_TURBO,
        )

        heatmap = cv2.cvtColor(
            heatmap,
            cv2.COLOR_BGR2RGB,
        )

        rgb = np.stack(
            [resized, resized, resized],
            axis=2,
        )

        painted = rgb.copy()
        painted[
            prelabel == 255
        ] = (
            0,
            255,
            255,
        )
        painted[
            prelabel == 128
        ] = (
            255,
            0,
            255,
        )

        overlay = np.clip(
            0.60 * rgb
            + 0.40 * painted,
            0,
            255,
        ).astype(np.uint8)

        prelabel_rgb = np.zeros_like(
            rgb
        )
        prelabel_rgb[
            prelabel == 255
        ] = (
            0,
            255,
            255,
        )
        prelabel_rgb[
            prelabel == 128
        ] = (
            255,
            0,
            255,
        )

        image_path = (
            image_directory
            / f"{safe_name}.png"
        )
        probability_path = (
            probability_directory
            / f"{safe_name}.png"
        )
        prelabel_path = (
            prelabel_directory
            / f"{safe_name}.png"
        )
        overlay_path = (
            overlay_directory
            / f"{safe_name}.png"
        )
        manual_path = (
            manual_mask_directory
            / f"{safe_name}.png"
        )

        Image.fromarray(
            resized
        ).save(image_path)

        Image.fromarray(
            heatmap
        ).save(probability_path)

        Image.fromarray(
            prelabel
        ).save(prelabel_path)

        Image.fromarray(
            overlay
        ).save(overlay_path)

        Image.fromarray(
            np.full(
                (
                    export_size,
                    export_size,
                ),
                128,
                dtype=np.uint8,
            )
        ).save(manual_path)

        selected.loc[
            selected[
                "sample_id"
            ].eq(sample_id),
            "annotation_image_path",
        ] = relative(image_path)

        selected.loc[
            selected[
                "sample_id"
            ].eq(sample_id),
            "ensemble_probability_path",
        ] = relative(
            probability_path
        )

        selected.loc[
            selected[
                "sample_id"
            ].eq(sample_id),
            "prelabel_water_path",
        ] = relative(prelabel_path)

        selected.loc[
            selected[
                "sample_id"
            ].eq(sample_id),
            "review_overlay_path",
        ] = relative(overlay_path)

        selected.loc[
            selected[
                "sample_id"
            ].eq(sample_id),
            "manual_mask_path",
        ] = relative(manual_path)

        panel = np.concatenate(
            [
                rgb,
                heatmap,
                prelabel_rgb,
                overlay,
            ],
            axis=1,
        )

        title = (
            f"{sample_id} | "
            f"{row.selection_reason} | "
            f"land={row.consensus_land_fraction:.3f} "
            f"water={row.consensus_water_fraction:.3f} "
            f"uncertain={row.uncertain_fraction:.3f}"
        )

        contact_rows.append(
            add_title(
                Image.fromarray(panel),
                title,
                1400,
            )
        )

    contact_sheet = (
        OUTPUT_DIR
        / "targeted_hard_batch_contact_sheet.jpg"
    )

    sheet = Image.new(
        "RGB",
        (
            1400,
            sum(
                row.height
                for row in contact_rows
            ),
        ),
        "black",
    )

    y = 0

    for row in contact_rows:
        sheet.paste(
            row,
            (0, y),
        )
        y += row.height

    sheet.save(
        contact_sheet,
        quality=90,
    )

    return contact_sheet


def main() -> None:
    args = parse_args()

    if args.per_group != 8:
        raise ValueError(
            "Bu aşamada --per-group 8 kullanılmalıdır."
        )

    dartis_manifest_path = resolve_path(
        args.dartis_manifest
    )
    active_manifest_path = resolve_path(
        args.active_manifest
    )
    pilot_manifest_path = resolve_path(
        args.pilot_manifest
    )
    fold_directory = resolve_path(
        args.fold_directory
    )
    source_checkpoint_path = resolve_path(
        args.source_checkpoint
    )

    for path in (
        dartis_manifest_path,
        active_manifest_path,
        fold_directory,
        source_checkpoint_path,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Gerekli kaynak bulunamadı: {path}"
            )

    excluded_ids = excluded_sample_ids(
        [
            active_manifest_path,
            pilot_manifest_path,
        ]
    )

    candidates = load_candidates(
        dartis_manifest_path,
        excluded_ids,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    (
        fold_models,
        fold_thresholds,
        model_names,
        source_model,
        source_threshold,
    ) = load_models(
        fold_directory,
        source_checkpoint_path,
        device,
    )

    print("=" * 78)
    print(
        "v0.6 DARTIS TARGETED HARD-BATCH SELECTION"
    )
    print("=" * 78)
    print(
        "Dışlanan eski manuel örnek:",
        len(excluded_ids),
    )
    print(
        "Kalan kıyı adayı:",
        len(candidates),
    )
    print(
        "nc aday:",
        int(
            candidates[
                "coastal_group"
            ].eq("nc").sum()
        ),
    )
    print(
        "oc aday:",
        int(
            candidates[
                "coastal_group"
            ].eq("oc").sum()
        ),
    )
    print(
        "Kullanılan land-aware model:",
        len(fold_models),
    )
    print(
        "Cihaz:",
        device,
    )
    print()

    scored = score_candidates(
        candidates,
        fold_models,
        fold_thresholds,
        source_model,
        source_threshold,
        device,
        args.inference_size,
        args.batch_size,
        args.num_workers,
    )

    selected_parts = []

    for group_name in ("nc", "oc"):
        group_frame = scored[
            scored[
                "coastal_group"
            ].eq(group_name)
        ].copy()

        selected_parts.append(
            select_group(
                group_frame,
                args.per_group,
            )
        )

    selected = pd.concat(
        selected_parts,
        ignore_index=True,
    )

    selected = selected.sort_values(
        [
            "coastal_group",
            "selection_reason",
            "land_heavy_score",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    ).reset_index(drop=True)

    selected[
        "annotation_order"
    ] = np.arange(
        1,
        len(selected) + 1,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    ANNOTATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )
    QUEUE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    contact_sheet = export_package(
        selected,
        fold_models,
        fold_thresholds,
        device,
        args.export_size,
    )

    output_columns = [
        "annotation_order",
        "sample_id",
        "coastal_group",
        "selection_reason",
        "consensus_land_fraction",
        "consensus_water_fraction",
        "uncertain_fraction",
        "ensemble_std",
        "source_water_fraction",
        "source_minus_landaware",
        "edge_density",
        "image_path",
        "annotation_image_path",
        "ensemble_probability_path",
        "prelabel_water_path",
        "review_overlay_path",
        "manual_mask_path",
    ]

    selected[
        output_columns
    ].to_csv(
        QUEUE_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    scored_path = (
        OUTPUT_DIR
        / "all_remaining_coastal_scores.csv"
    )

    scored.to_csv(
        scored_path,
        index=False,
        encoding="utf-8-sig",
    )

    reason_counts = (
        selected[
            "selection_reason"
        ]
        .value_counts()
        .to_dict()
    )

    summary = {
        "stage": (
            "v06_dartis_water_targeted_hard_batch"
        ),
        "candidate_count": int(
            len(candidates)
        ),
        "excluded_existing_manual_count": int(
            len(excluded_ids)
        ),
        "selected_count": int(
            len(selected)
        ),
        "selected_nc_count": int(
            selected[
                "coastal_group"
            ].eq("nc").sum()
        ),
        "selected_oc_count": int(
            selected[
                "coastal_group"
            ].eq("oc").sum()
        ),
        "selection_reason_counts": {
            str(key): int(value)
            for key, value
            in reason_counts.items()
        },
        "models_used": model_names,
        "model_thresholds": [
            float(value)
            for value in fold_thresholds
        ],
        "source_checkpoint": relative(
            source_checkpoint_path
        ),
        "source_threshold": float(
            source_threshold
        ),
        "queue_path": relative(
            QUEUE_PATH
        ),
        "annotation_directory": relative(
            ANNOTATION_DIR
        ),
        "contact_sheet": relative(
            contact_sheet
        ),
        "all_scores_path": relative(
            scored_path
        ),
        "training_performed": False,
        "masks_modified": False,
        "locked_test_used": False,
        "scientific_note": (
            "Bu paket eski 32 aktif maske ve 7 pilot maskeyi tekrar "
            "seçmez. Amaç land-heavy, model anlaşmazlığı yüksek, "
            "kıyı sınırı zengin ve olası su-kaçırma tiplerinden yeni "
            "hedef-domain etiketleri toplamaktır."
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

    print("=" * 78)
    print(
        "TARGETED HARD-BATCH HAZIR"
    )
    print("=" * 78)
    print(
        "Seçilen toplam:",
        len(selected),
    )
    print(
        "nc:",
        int(
            selected[
                "coastal_group"
            ].eq("nc").sum()
        ),
    )
    print(
        "oc:",
        int(
            selected[
                "coastal_group"
            ].eq("oc").sum()
        ),
    )
    print(
        "Kuyruk:",
        QUEUE_PATH.resolve(),
    )
    print(
        "Paket:",
        ANNOTATION_DIR.resolve(),
    )
    print(
        "Görsel:",
        contact_sheet.resolve(),
    )
    print(
        "Özet:",
        summary_path.resolve(),
    )
    print()
    print(
        "Eski maskeler değiştirilmedi."
    )
    print(
        "Kilitli test kullanılmadı."
    )


if __name__ == "__main__":
    main()
