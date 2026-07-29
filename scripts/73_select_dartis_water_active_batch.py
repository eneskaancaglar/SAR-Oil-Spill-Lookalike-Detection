from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]

BASE_SCRIPT = (
    ROOT
    / "scripts"
    / "68_evaluate_dartis_water_transfer_gate.py"
)

DEFAULT_SOURCE_CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "water_unet_flood_v06_robust"
    / "best.pth"
)

DEFAULT_FOLD_DIRECTORY = (
    ROOT
    / "checkpoints"
    / "water_unet_dartis_loocv_v06"
)

DEFAULT_DARTIS_MANIFEST = (
    ROOT
    / "data"
    / "metadata"
    / "v06_dartis_water_masks_final.csv"
)

DEFAULT_MANUAL_MANIFEST = (
    ROOT
    / "data"
    / "metadata"
    / "v06_manual_land_corrections.csv"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "v06_dartis_water_active_batch"
)

ANNOTATION_DIR = (
    ROOT
    / "data"
    / "annotation"
    / "v06_dartis_water_active_batch"
)

QUEUE_PATH = (
    ROOT
    / "data"
    / "metadata"
    / "v06_dartis_water_active_batch.csv"
)


def load_base_module():
    if not BASE_SCRIPT.exists():
        raise FileNotFoundError(
            f"Gerekli 68 scripti bulunamadı: {BASE_SCRIPT}"
        )

    spec = importlib.util.spec_from_file_location(
        "dartis_transfer_base_v73",
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
            "Kaynak su modeli ile 7 DARTIS LOOCV fold modelinin "
            "belirsizlik ve anlaşmazlığını kullanarak nc/oc kıyı "
            "sahnelerinden dengeli, yüksek değerli manuel etiketleme "
            "kuyruğu seçer. Eğitim veya toplu maske üretimi yapmaz."
        )
    )

    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        default=DEFAULT_SOURCE_CHECKPOINT,
    )
    parser.add_argument(
        "--fold-directory",
        type=Path,
        default=DEFAULT_FOLD_DIRECTORY,
    )
    parser.add_argument(
        "--dartis-manifest",
        type=Path,
        default=DEFAULT_DARTIS_MANIFEST,
    )
    parser.add_argument(
        "--manual-manifest",
        type=Path,
        default=DEFAULT_MANUAL_MANIFEST,
    )
    parser.add_argument(
        "--per-group",
        type=int,
        default=16,
        help="nc ve oc gruplarının her birinden seçilecek görüntü sayısı.",
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
    parser.add_argument(
        "--maximum-models",
        type=int,
        default=8,
        help="Kaynak model dahil kullanılacak en fazla model sayısı.",
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


def load_candidate_frame(
    dartis_manifest_path: Path,
    manual_manifest_path: Path,
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

    frame["sample_id"] = (
        frame["sample_id"].astype(str)
    )

    frame["coastal_group"] = (
        frame["sample_id"].map(
            coastal_group
        )
    )

    frame = frame[
        frame[
            "coastal_group"
        ].isin(
            ["nc", "oc"]
        )
    ].copy()

    if manual_manifest_path.exists():
        manual = pd.read_csv(
            manual_manifest_path,
            encoding="utf-8-sig",
            low_memory=False,
        )

        if "sample_id" in manual.columns:
            excluded = set(
                manual[
                    "sample_id"
                ].astype(str)
            )

            frame = frame[
                ~frame[
                    "sample_id"
                ].isin(excluded)
            ].copy()

    frame["resolved_image_path"] = (
        frame["image_path"].map(
            lambda value: str(
                resolve_path(value)
            )
        )
    )

    missing_files = frame[
        ~frame[
            "resolved_image_path"
        ].map(
            lambda value: Path(
                value
            ).exists()
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

    return frame.reset_index(
        drop=True
    )


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
                if max(
                    image.shape
                )
                > self.image_size
                else cv2.INTER_CUBIC
            ),
        )

        two_channel = np.stack(
            [
                resized,
                resized,
            ],
            axis=0,
        ).astype(np.float32)

        histogram, _ = np.histogram(
            normalized,
            bins=32,
            range=(0.0, 1.0),
            density=False,
        )

        histogram = histogram.astype(
            np.float32
        )

        histogram /= max(
            float(histogram.sum()),
            1.0,
        )

        return (
            torch.from_numpy(
                two_channel
            ),
            str(record["sample_id"]),
            torch.from_numpy(
                histogram
            ),
        )


def load_models(
    source_checkpoint_path: Path,
    fold_directory: Path,
    maximum_models: int,
    device: torch.device,
) -> tuple[
    list[torch.nn.Module],
    list[str],
]:
    checkpoint_paths = [
        source_checkpoint_path
    ]

    if fold_directory.exists():
        checkpoint_paths.extend(
            sorted(
                fold_directory.glob(
                    "fold_*.pth"
                )
            )
        )

    checkpoint_paths = (
        checkpoint_paths[
            : max(
                maximum_models,
                1,
            )
        ]
    )

    models: list[
        torch.nn.Module
    ] = []

    names: list[str] = []

    for checkpoint_path in (
        checkpoint_paths
    ):
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

        models.append(model)
        names.append(
            checkpoint_path.name
        )

    if len(models) < 2:
        raise RuntimeError(
            "Aktif öğrenme için en az kaynak model ve "
            "bir fold checkpointi gerekli."
        )

    return models, names


def binary_entropy(
    probability: torch.Tensor,
) -> torch.Tensor:
    clipped = probability.clamp(
        1e-6,
        1.0 - 1e-6,
    )

    entropy = -(
        clipped
        * torch.log2(clipped)
        + (
            1.0 - clipped
        )
        * torch.log2(
            1.0 - clipped
        )
    )

    return entropy


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
    ).clip(
        0.0,
        1.0,
    )


def score_candidates(
    frame: pd.DataFrame,
    models: list[
        torch.nn.Module
    ],
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

    rows: list[
        dict[str, Any]
    ] = []

    with torch.no_grad():
        for (
            images,
            sample_ids,
            histograms,
        ) in loader:
            images = images.to(
                device,
                non_blocking=True,
            )

            probabilities = []

            for model in models:
                with torch.amp.autocast(
                    device_type=device.type,
                    enabled=(
                        device.type
                        == "cuda"
                    ),
                ):
                    logits = model(images)

                probabilities.append(
                    torch.sigmoid(
                        logits[:, 0].float()
                    )
                )

            stack = torch.stack(
                probabilities,
                dim=0,
            )

            mean_probability = (
                stack.mean(dim=0)
            )

            standard_deviation = (
                stack.std(
                    dim=0,
                    unbiased=False,
                )
            )

            votes = (
                stack >= 0.5
            ).float()

            vote_fraction = votes.mean(
                dim=0
            )

            vote_disagreement = (
                4.0
                * vote_fraction
                * (
                    1.0
                    - vote_fraction
                )
            )

            entropy = binary_entropy(
                mean_probability
            )

            uncertain_fraction = (
                (
                    mean_probability
                    >= 0.35
                )
                & (
                    mean_probability
                    <= 0.65
                )
            ).float().mean(
                dim=(1, 2)
            )

            water_fraction = (
                mean_probability
                >= 0.5
            ).float().mean(
                dim=(1, 2)
            )

            source_adapted_difference = (
                (
                    stack[0]
                    - stack[1:].mean(
                        dim=0
                    )
                )
                .abs()
                .mean(
                    dim=(1, 2)
                )
            )

            for batch_index, sample_id in enumerate(
                sample_ids
            ):
                rows.append(
                    {
                        "sample_id": (
                            str(sample_id)
                        ),
                        "ensemble_entropy": float(
                            entropy[
                                batch_index
                            ].mean().item()
                        ),
                        "ensemble_std": float(
                            standard_deviation[
                                batch_index
                            ].mean().item()
                        ),
                        "vote_disagreement": float(
                            vote_disagreement[
                                batch_index
                            ].mean().item()
                        ),
                        "uncertain_fraction": float(
                            uncertain_fraction[
                                batch_index
                            ].item()
                        ),
                        "predicted_water_fraction": float(
                            water_fraction[
                                batch_index
                            ].item()
                        ),
                        "source_adapted_difference": float(
                            source_adapted_difference[
                                batch_index
                            ].item()
                        ),
                        "histogram_vector": "|".join(
                            f"{float(value):.8f}"
                            for value
                            in histograms[
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

    metric_columns = (
        "ensemble_entropy",
        "ensemble_std",
        "vote_disagreement",
        "uncertain_fraction",
        "source_adapted_difference",
    )

    for column in metric_columns:
        merged[
            f"{column}_norm"
        ] = robust_minmax(
            merged[column]
        )

    merged["selection_score"] = (
        0.25
        * merged[
            "ensemble_entropy_norm"
        ]
        + 0.25
        * merged[
            "ensemble_std_norm"
        ]
        + 0.20
        * merged[
            "vote_disagreement_norm"
        ]
        + 0.15
        * merged[
            "uncertain_fraction_norm"
        ]
        + 0.15
        * merged[
            "source_adapted_difference_norm"
        ]
    )

    return merged


def histogram_array(
    value: str,
) -> np.ndarray:
    array = np.array(
        [
            float(item)
            for item in value.split("|")
            if item
        ],
        dtype=np.float32,
    )

    norm = float(
        np.linalg.norm(array)
    )

    if norm > 0.0:
        array /= norm

    return array


def diversity_fill(
    candidates: pd.DataFrame,
    selected_ids: set[str],
    count: int,
) -> list[str]:
    if count <= 0:
        return []

    pool = candidates[
        ~candidates[
            "sample_id"
        ].isin(
            selected_ids
        )
    ].copy()

    if pool.empty:
        return []

    pool = pool.sort_values(
        "selection_score",
        ascending=False,
    ).head(
        min(
            120,
            len(pool),
        )
    )

    vectors = {
        row.sample_id: histogram_array(
            row.histogram_vector
        )
        for row in pool.itertuples()
    }

    chosen: list[str] = []

    initial_id = str(
        pool.iloc[0][
            "sample_id"
        ]
    )

    chosen.append(initial_id)

    while (
        len(chosen) < count
        and len(chosen) < len(pool)
    ):
        best_id = None
        best_value = -1.0

        for sample_id in pool[
            "sample_id"
        ].astype(str):
            if sample_id in chosen:
                continue

            vector = vectors[sample_id]

            distances = []

            for chosen_id in chosen:
                distances.append(
                    float(
                        1.0
                        - np.dot(
                            vector,
                            vectors[
                                chosen_id
                            ],
                        )
                    )
                )

            minimum_distance = min(
                distances
            )

            row_score = float(
                pool.loc[
                    pool[
                        "sample_id"
                    ].eq(
                        sample_id
                    ),
                    "selection_score",
                ].iloc[0]
            )

            combined = (
                0.70
                * minimum_distance
                + 0.30
                * row_score
            )

            if combined > best_value:
                best_value = combined
                best_id = sample_id

        if best_id is None:
            break

        chosen.append(best_id)

    return chosen[:count]


def select_group(
    group_frame: pd.DataFrame,
    count: int,
) -> pd.DataFrame:
    if len(group_frame) <= count:
        selected = group_frame.copy()
        selected[
            "selection_reason"
        ] = "all_available"
        return selected

    selected_reasons: dict[
        str,
        set[str]
    ] = {}

    def add_rows(
        frame: pd.DataFrame,
        reason: str,
        maximum: int,
    ) -> None:
        for sample_id in frame[
            "sample_id"
        ].astype(str).head(
            maximum
        ):
            selected_reasons.setdefault(
                sample_id,
                set(),
            ).add(reason)

    uncertainty_count = min(
        5,
        count,
    )
    disagreement_count = min(
        3,
        max(
            count
            - uncertainty_count,
            0,
        ),
    )
    low_water_count = min(
        2,
        max(
            count
            - uncertainty_count
            - disagreement_count,
            0,
        ),
    )
    high_water_count = min(
        2,
        max(
            count
            - uncertainty_count
            - disagreement_count
            - low_water_count,
            0,
        ),
    )

    add_rows(
        group_frame.sort_values(
            "selection_score",
            ascending=False,
        ),
        "high_uncertainty",
        uncertainty_count,
    )

    add_rows(
        group_frame.sort_values(
            "source_adapted_difference",
            ascending=False,
        ),
        "source_adapt_disagreement",
        disagreement_count,
    )

    add_rows(
        group_frame.sort_values(
            "predicted_water_fraction",
            ascending=True,
        ),
        "low_predicted_water",
        low_water_count,
    )

    add_rows(
        group_frame.sort_values(
            "predicted_water_fraction",
            ascending=False,
        ),
        "high_predicted_water",
        high_water_count,
    )

    selected_ids = set(
        selected_reasons
    )

    remaining = (
        count
        - len(selected_ids)
    )

    diverse_ids = diversity_fill(
        group_frame,
        selected_ids,
        remaining,
    )

    for sample_id in diverse_ids:
        selected_reasons.setdefault(
            sample_id,
            set(),
        ).add(
            "histogram_diversity"
        )

    if len(selected_reasons) < count:
        for sample_id in group_frame.sort_values(
            "selection_score",
            ascending=False,
        )[
            "sample_id"
        ].astype(str):
            if sample_id not in selected_reasons:
                selected_reasons[
                    sample_id
                ] = {
                    "score_fill"
                }

            if len(
                selected_reasons
            ) >= count:
                break

    selected_ids = list(
        selected_reasons
    )[:count]

    selected = group_frame[
        group_frame[
            "sample_id"
        ].isin(
            selected_ids
        )
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


def export_selected_package(
    selected: pd.DataFrame,
    models: list[
        torch.nn.Module
    ],
    device: torch.device,
    export_size: int,
) -> Path:
    image_directory = (
        ANNOTATION_DIR
        / "images"
    )
    probability_directory = (
        ANNOTATION_DIR
        / "ensemble_probability"
    )
    prelabel_directory = (
        ANNOTATION_DIR
        / "prelabel_water"
    )
    overlay_directory = (
        ANNOTATION_DIR
        / "review_overlay"
    )
    blank_mask_directory = (
        ANNOTATION_DIR
        / "manual_masks"
    )

    for directory in (
        image_directory,
        probability_directory,
        prelabel_directory,
        overlay_directory,
        blank_mask_directory,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    preview_rows: list[
        Image.Image
    ] = []

    for order, row in enumerate(
        selected.itertuples(),
        start=1,
    ):
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
                if max(
                    original.shape
                )
                > export_size
                else cv2.INTER_CUBIC
            ),
        )

        normalized = normalize_percentile(
            resized
        )

        tensor = torch.from_numpy(
            np.stack(
                [
                    normalized,
                    normalized,
                ],
                axis=0,
            ).astype(np.float32)
        ).unsqueeze(0).to(device)

        probabilities = []

        with torch.no_grad():
            for model in models:
                logits = model(tensor)

                probabilities.append(
                    torch.sigmoid(
                        logits[0, 0].float()
                    ).cpu().numpy()
                )

        mean_probability = np.mean(
            probabilities,
            axis=0,
        )

        prelabel = (
            mean_probability
            >= 0.5
        )

        probability_uint8 = (
            np.clip(
                mean_probability,
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
            [
                resized,
                resized,
                resized,
            ],
            axis=2,
        )

        overlay = rgb.copy()

        overlay[
            prelabel
        ] = (
            255,
            255,
            0,
        )

        blended = cv2.addWeighted(
            rgb,
            0.60,
            overlay,
            0.40,
            0.0,
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
        blank_path = (
            blank_mask_directory
            / f"{safe_name}.png"
        )

        Image.fromarray(
            resized
        ).save(image_path)

        Image.fromarray(
            heatmap
        ).save(probability_path)

        Image.fromarray(
            prelabel.astype(
                np.uint8
            )
            * 255
        ).save(prelabel_path)

        Image.fromarray(
            blended
        ).save(overlay_path)

        Image.fromarray(
            np.zeros(
                (
                    export_size,
                    export_size,
                ),
                dtype=np.uint8,
            )
        ).save(blank_path)

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
        ] = relative(blank_path)

        prelabel_rgb = np.stack(
            [
                prelabel.astype(
                    np.uint8
                )
                * 255,
            ]
            * 3,
            axis=2,
        )

        row_image = np.concatenate(
            [
                rgb,
                heatmap,
                prelabel_rgb,
                blended,
            ],
            axis=1,
        )

        preview_rows.append(
            Image.fromarray(
                row_image
            )
        )

    contact_sheet_path = (
        OUTPUT_DIR
        / "active_batch_contact_sheet.jpg"
    )

    if preview_rows:
        thumbnail_width = 1400
        resized_rows: list[
            Image.Image
        ] = []

        for row_image in preview_rows:
            ratio = (
                thumbnail_width
                / row_image.width
            )

            resized_rows.append(
                row_image.resize(
                    (
                        thumbnail_width,
                        max(
                            1,
                            int(
                                row_image.height
                                * ratio
                            ),
                        ),
                    ),
                    Image.Resampling.LANCZOS,
                )
            )

        sheet = Image.new(
            "RGB",
            (
                thumbnail_width,
                sum(
                    image.height
                    for image
                    in resized_rows
                ),
            ),
            "black",
        )

        y = 0

        for row_image in resized_rows:
            sheet.paste(
                row_image,
                (0, y),
            )
            y += row_image.height

        contact_sheet_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        sheet.save(
            contact_sheet_path,
            quality=90,
        )

    return contact_sheet_path


def main() -> None:
    args = parse_args()

    source_checkpoint_path = (
        resolve_path(
            args.source_checkpoint
        )
    )
    fold_directory = resolve_path(
        args.fold_directory
    )
    dartis_manifest_path = (
        resolve_path(
            args.dartis_manifest
        )
    )
    manual_manifest_path = (
        resolve_path(
            args.manual_manifest
        )
    )

    for path in (
        source_checkpoint_path,
        dartis_manifest_path,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Gerekli dosya bulunamadı: {path}"
            )

    if not fold_directory.exists():
        raise FileNotFoundError(
            "LOOCV fold checkpoint klasörü bulunamadı: "
            f"{fold_directory}"
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

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    models, model_names = load_models(
        source_checkpoint_path=(
            source_checkpoint_path
        ),
        fold_directory=fold_directory,
        maximum_models=(
            args.maximum_models
        ),
        device=device,
    )

    candidates = load_candidate_frame(
        dartis_manifest_path=(
            dartis_manifest_path
        ),
        manual_manifest_path=(
            manual_manifest_path
        ),
    )

    print("=" * 78)
    print(
        "v0.6 DARTIS WATER ACTIVE BATCH SELECTION"
    )
    print("=" * 78)
    print("Cihaz:", device)
    print(
        "Kullanılan model:",
        len(models),
    )
    print(
        "Aday kıyı görüntüsü:",
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
    print()

    scored = score_candidates(
        frame=candidates,
        models=models,
        device=device,
        inference_size=(
            args.inference_size
        ),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    selected_parts = []

    for group_name in (
        "nc",
        "oc",
    ):
        group_frame = scored[
            scored[
                "coastal_group"
            ].eq(group_name)
        ].copy()

        if group_frame.empty:
            raise RuntimeError(
                f"{group_name} kıyı grubu boş."
            )

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
            "selection_score",
        ],
        ascending=[
            True,
            False,
        ],
    ).reset_index(
        drop=True
    )

    selected[
        "annotation_order"
    ] = np.arange(
        1,
        len(selected) + 1,
    )

    contact_sheet_path = (
        export_selected_package(
            selected=selected,
            models=models,
            device=device,
            export_size=args.export_size,
        )
    )

    output_columns = [
        "annotation_order",
        "sample_id",
        "coastal_group",
        "selection_reason",
        "selection_score",
        "ensemble_entropy",
        "ensemble_std",
        "vote_disagreement",
        "uncertain_fraction",
        "predicted_water_fraction",
        "source_adapted_difference",
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
        / "all_coastal_scores.csv"
    )

    scored.to_csv(
        scored_path,
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "stage": (
            "v06_dartis_water_active_batch_selection"
        ),
        "candidate_count": int(
            len(candidates)
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
        "excluded_existing_manual_count": (
            int(
                len(
                    pd.read_csv(
                        manual_manifest_path,
                        encoding="utf-8-sig",
                        low_memory=False,
                    )
                )
            )
            if manual_manifest_path.exists()
            else 0
        ),
        "models_used": (
            model_names
        ),
        "selection_strategy": [
            "ensemble uncertainty",
            "source-adapted disagreement",
            "predicted water-fraction extremes",
            "intensity-histogram diversity",
            "balanced nc/oc sampling",
        ],
        "queue_path": relative(
            QUEUE_PATH
        ),
        "all_scores_path": relative(
            scored_path
        ),
        "annotation_directory": relative(
            ANNOTATION_DIR
        ),
        "contact_sheet": relative(
            contact_sheet_path
        ),
        "training_performed": False,
        "new_ground_truth_created": False,
        "all_dartis_masks_generated": False,
        "official_sen1floods11_test_used": False,
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
        "AKTİF ETİKETLEME PAKETİ HAZIR"
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
        contact_sheet_path.resolve(),
    )
    print(
        "Özet:",
        summary_path.resolve(),
    )
    print()
    print(
        "Eğitim yapılmadı."
    )
    print(
        "Yeni ground-truth maske üretilmedi."
    )


if __name__ == "__main__":
    main()
