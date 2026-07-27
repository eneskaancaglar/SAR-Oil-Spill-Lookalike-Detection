from __future__ import annotations

import argparse
import inspect
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.unet import UNet


SCENE_MANIFEST = (
    ROOT
    / "data"
    / "metadata"
    / "dartis_no_oil_scene_split.csv"
)

RAW_ROOT = (
    ROOT
    / "data"
    / "external"
    / "dartis"
    / "raw"
)

CHECKPOINT_PATH = (
    ROOT
    / "checkpoints"
    / "final_model"
    / "oil_spill_unet_t060.pth"
)

CROP_ROOT = (
    ROOT
    / "data"
    / "external"
    / "dartis"
    / "processed"
    / "verifier_negative"
)

OUTPUT_MANIFEST = (
    ROOT
    / "data"
    / "metadata"
    / "dartis_negative_verifier_crops.csv"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "v03_dartis_negative_crops"
)

QC_DIR = OUTPUT_DIR / "qc_overlays"

SUMMARY_PATH = (
    OUTPUT_DIR
    / "negative_crop_summary.json"
)

REPORT_PATH = (
    ROOT
    / "reports"
    / "v03_dartis_negative_crops.md"
)


ALLOWED_SOURCE_SPLITS = {
    "hard_negative_train": "train",
    "hard_negative_val": "validation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Petrolsüz DARTIS görüntülerinden model tabanlı "
            "zor negatif ve kolay negatif verifier crop'ları üretir."
        )
    )

    parser.add_argument(
        "--crop-size",
        type=int,
        default=224,
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=112,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.60,
    )

    parser.add_argument(
        "--minimum-positive-ratio",
        type=float,
        default=0.001,
        help=(
            "Bir pencerenin güçlü hard-negative sayılması için "
            "modelin petrol dediği minimum piksel oranı."
        ),
    )

    parser.add_argument(
        "--hard-per-image",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--easy-per-image",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--maximum-iou",
        type=float,
        default=0.40,
    )

    parser.add_argument(
        "--qc-images",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="0 bütün uygun görüntüleri işler.",
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
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if text.lower() in {
        "",
        "nan",
        "none",
        "<na>",
    }:
        return ""

    return text


def find_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
    required: bool = True,
) -> str | None:
    column_map = {
        str(column).strip().lower(): str(column)
        for column in dataframe.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower()

        if key in column_map:
            return column_map[key]

    if required:
        raise RuntimeError(
            f"Sütun bulunamadı: {candidates}\n"
            f"Mevcut sütunlar: {list(dataframe.columns)}"
        )

    return None


def select_device(
    requested_device: str,
) -> torch.device:
    if requested_device == "cpu":
        return torch.device("cpu")

    if requested_device == "cuda":
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


def read_checkpoint(
    path: Path,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    if not path.exists():
        raise FileNotFoundError(
            f"Final checkpoint bulunamadı: {path}"
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
            "Checkpoint beklenen sözlük formatında değil."
        )

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

    cleaned_state = {}

    for key, value in state_dict.items():
        cleaned_key = str(key)

        for prefix in (
            "module.",
            "_orig_mod.",
            "model.",
        ):
            if cleaned_key.startswith(prefix):
                cleaned_key = cleaned_key[
                    len(prefix):
                ]

        cleaned_state[cleaned_key] = value

    return cleaned_state


def build_model(
    state_dict: dict[str, torch.Tensor],
    device: torch.device,
) -> torch.nn.Module:
    constructor_attempts = [
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

    for description, constructor in constructor_attempts:
        try:
            model = constructor()

            model.load_state_dict(
                state_dict,
                strict=True,
            )

            model.to(device)
            model.eval()

            print(
                "UNet constructor:",
                description,
            )

            return model

        except Exception as error:
            errors.append(
                f"{description}: "
                f"{type(error).__name__}: {error}"
            )

    signature = inspect.signature(
        UNet
    )

    raise RuntimeError(
        "UNet modeli oluşturulamadı.\n"
        f"UNet signature: {signature}\n"
        + "\n".join(errors)
    )


def resolve_image_path(
    row: dict[str, Any],
    group_name: str,
    image_name: str,
    path_column: str | None,
) -> Path | None:
    if path_column is not None:
        path_text = clean_text(
            row.get(path_column, "")
        )

        if path_text:
            path = Path(path_text)

            if not path.is_absolute():
                path = ROOT / path

            if path.exists():
                return path

    direct_path = (
        RAW_ROOT
        / group_name
        / image_name
    )

    if direct_path.exists():
        return direct_path

    directory = (
        RAW_ROOT
        / group_name
    )

    if not directory.exists():
        return None

    stem = Path(image_name).stem

    for suffix in (
        ".jpg",
        ".jpeg",
        ".png",
        ".tif",
        ".tiff",
    ):
        candidate = (
            directory
            / f"{stem}{suffix}"
        )

        if candidate.exists():
            return candidate

    return None


def sliding_positions(
    length: int,
    crop_size: int,
    stride: int,
) -> list[int]:
    if length <= crop_size:
        return [0]

    positions = list(
        range(
            0,
            length - crop_size + 1,
            stride,
        )
    )

    final_position = (
        length - crop_size
    )

    if positions[-1] != final_position:
        positions.append(
            final_position
        )

    return positions


def box_iou(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> float:
    first_left, first_top, first_right, first_bottom = (
        first
    )

    second_left, second_top, second_right, second_bottom = (
        second
    )

    intersection_left = max(
        first_left,
        second_left,
    )

    intersection_top = max(
        first_top,
        second_top,
    )

    intersection_right = min(
        first_right,
        second_right,
    )

    intersection_bottom = min(
        first_bottom,
        second_bottom,
    )

    intersection_width = max(
        0,
        intersection_right
        - intersection_left,
    )

    intersection_height = max(
        0,
        intersection_bottom
        - intersection_top,
    )

    intersection_area = (
        intersection_width
        * intersection_height
    )

    first_area = (
        first_right - first_left
    ) * (
        first_bottom - first_top
    )

    second_area = (
        second_right - second_left
    ) * (
        second_bottom - second_top
    )

    union_area = (
        first_area
        + second_area
        - intersection_area
    )

    if union_area <= 0:
        return 0.0

    return (
        intersection_area
        / union_area
    )


def run_model(
    model: torch.nn.Module,
    image_array: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    tensor = torch.from_numpy(
        image_array
    ).float()

    tensor = tensor.unsqueeze(0).unsqueeze(0)
    tensor = tensor.to(device)

    autocast_enabled = (
        device.type == "cuda"
    )

    with torch.inference_mode():
        with torch.autocast(
            device_type=device.type,
            enabled=autocast_enabled,
            dtype=torch.float16,
        ):
            output = model(tensor)

            if isinstance(output, dict):
                if "out" in output:
                    output = output["out"]
                else:
                    output = next(
                        iter(output.values())
                    )

            if isinstance(
                output,
                (tuple, list),
            ):
                output = output[0]

            probability = torch.sigmoid(
                output
            )

    probability_array = (
        probability[
            0,
            0,
        ]
        .float()
        .cpu()
        .numpy()
    )

    return probability_array


def build_candidates(
    probability: np.ndarray,
    crop_size: int,
    stride: int,
    threshold: float,
) -> list[dict[str, Any]]:
    height, width = probability.shape

    x_positions = sliding_positions(
        width,
        crop_size,
        stride,
    )

    y_positions = sliding_positions(
        height,
        crop_size,
        stride,
    )

    candidates = []

    for top in y_positions:
        for left in x_positions:
            right = min(
                width,
                left + crop_size,
            )

            bottom = min(
                height,
                top + crop_size,
            )

            tile = probability[
                top:bottom,
                left:right,
            ]

            positive_ratio = float(
                (
                    tile >= threshold
                ).mean()
            )

            mean_probability = float(
                tile.mean()
            )

            maximum_probability = float(
                tile.max()
            )

            percentile_95 = float(
                np.percentile(
                    tile,
                    95,
                )
            )

            score = (
                positive_ratio * 4.0
                + percentile_95
                + mean_probability * 0.25
            )

            candidates.append(
                {
                    "box": (
                        left,
                        top,
                        right,
                        bottom,
                    ),
                    "positive_ratio": (
                        positive_ratio
                    ),
                    "mean_probability": (
                        mean_probability
                    ),
                    "maximum_probability": (
                        maximum_probability
                    ),
                    "percentile_95": (
                        percentile_95
                    ),
                    "score": float(
                        score
                    ),
                }
            )

    return candidates


def select_non_overlapping(
    candidates: list[dict[str, Any]],
    count: int,
    maximum_iou: float,
    blocked_boxes: list[
        tuple[int, int, int, int]
    ] | None = None,
) -> list[dict[str, Any]]:
    selected = []

    blocked = list(
        blocked_boxes or []
    )

    for candidate in candidates:
        candidate_box = candidate[
            "box"
        ]

        overlaps = any(
            box_iou(
                candidate_box,
                existing_box,
            )
            > maximum_iou
            for existing_box in (
                blocked
                + [
                    selected_candidate[
                        "box"
                    ]
                    for selected_candidate
                    in selected
                ]
            )
        )

        if overlaps:
            continue

        selected.append(
            candidate
        )

        if len(selected) >= count:
            break

    return selected


def save_crop(
    image: Image.Image,
    box: tuple[int, int, int, int],
    destination: Path,
    crop_size: int,
    overwrite: bool,
) -> None:
    if (
        destination.exists()
        and not overwrite
    ):
        return

    crop = image.crop(
        box
    )

    if crop.size != (
        crop_size,
        crop_size,
    ):
        crop = crop.resize(
            (
                crop_size,
                crop_size,
            ),
            resample=Image.Resampling.BILINEAR,
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    crop.save(
        destination,
        format="PNG",
        optimize=True,
    )


def save_qc_overlay(
    image: Image.Image,
    hard_candidates: list[
        dict[str, Any]
    ],
    easy_candidates: list[
        dict[str, Any]
    ],
    destination: Path,
) -> None:
    overlay = image.convert(
        "RGB"
    ).copy()

    draw = ImageDraw.Draw(
        overlay
    )

    for index, candidate in enumerate(
        hard_candidates,
        start=1,
    ):
        left, top, right, bottom = (
            candidate["box"]
        )

        draw.rectangle(
            [
                left,
                top,
                right - 1,
                bottom - 1,
            ],
            outline=(255, 0, 0),
            width=3,
        )

        draw.text(
            (
                left + 4,
                top + 4,
            ),
            f"H{index}",
            fill=(255, 255, 255),
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )

    for index, candidate in enumerate(
        easy_candidates,
        start=1,
    ):
        left, top, right, bottom = (
            candidate["box"]
        )

        draw.rectangle(
            [
                left,
                top,
                right - 1,
                bottom - 1,
            ],
            outline=(255, 255, 0),
            width=3,
        )

        draw.text(
            (
                left + 4,
                top + 4,
            ),
            f"E{index}",
            fill=(255, 255, 255),
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    overlay.save(
        destination,
        quality=95,
    )


def write_report(
    summary: dict[str, Any],
) -> None:
    lines = [
        "# DARTIS Negatif Verifier Crop Hazırlığı",
        "",
        "## Amaç",
        "",
        "Petrolsüz DARTIS `nc` ve `nw` görüntülerinde "
        "mevcut petrol segmentasyon modelinin en fazla petrol "
        "olasılığı verdiği bölgeleri zor negatif örnek olarak toplamak.",
        "",
        "Bütün örneklerin verifier etiketi:",
        "",
        "- `0`: Petrol değil",
        "",
        "## Bilimsel koruma",
        "",
        "`external_test` görüntüleri eğitim verisi üretiminde "
        "kullanılmamıştır.",
        "",
        "## Sonuç",
        "",
        f"- İşlenen görüntü: {summary['processed_images']}",
        f"- Başarılı görüntü: {summary['successful_images']}",
        f"- Hatalı görüntü: {summary['failed_images']}",
        f"- Toplam negatif crop: {summary['saved_crops']}",
        f"- Zor negatif crop: {summary['hard_crops']}",
        f"- Kolay negatif crop: {summary['easy_crops']}",
        f"- Model threshold: {summary['threshold']}",
        "",
        "## Split sonuçları",
        "",
        "| Kaynak split | Çıktı split | Görüntü | Crop |",
        "|---|---|---:|---:|",
    ]

    for source_split, result in (
        summary[
            "source_splits"
        ].items()
    ):
        lines.append(
            f"| {source_split} "
            f"| {result['output_split']} "
            f"| {result['images']} "
            f"| {result['crops']} |"
        )

    lines.extend(
        [
            "",
            "## Grup sonuçları",
            "",
            "| Grup | Görüntü | Hard | Easy | Toplam |",
            "|---|---:|---:|---:|---:|",
        ]
    )

    for group_name in (
        "nc",
        "nw",
    ):
        result = summary[
            "groups"
        ].get(
            group_name,
            {},
        )

        lines.append(
            f"| {group_name} "
            f"| {result.get('images', 0)} "
            f"| {result.get('hard', 0)} "
            f"| {result.get('easy', 0)} "
            f"| {result.get('total', 0)} |"
        )

    lines.extend(
        [
            "",
            "## Sonraki aşama",
            "",
            "Pozitif ve negatif crop manifestleri sahne bazında "
            "birleştirilecek ve verifier modeli için dengeli "
            "train, validation ve calibration ayrımı oluşturulacaktır.",
            "",
        ]
    )

    REPORT_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    if not SCENE_MANIFEST.exists():
        raise FileNotFoundError(
            f"DARTIS scene manifest bulunamadı: "
            f"{SCENE_MANIFEST}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    QC_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_MANIFEST.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = pd.read_csv(
        SCENE_MANIFEST,
        encoding="utf-8-sig",
        low_memory=False,
    )

    split_column = find_column(
        dataframe,
        [
            "split",
            "scene_split",
        ],
    )

    group_column = find_column(
        dataframe,
        [
            "image_set",
            "group",
            "source_group",
        ],
    )

    image_column = find_column(
        dataframe,
        [
            "image_name",
            "sample_name",
            "filename",
        ],
    )

    scene_column = find_column(
        dataframe,
        [
            "scene_id",
            "ID_3",
            "id_3",
        ],
        required=False,
    )

    path_column = find_column(
        dataframe,
        [
            "local_image_path",
            "image_path",
        ],
        required=False,
    )

    dataframe[
        "normalized_split"
    ] = (
        dataframe[split_column]
        .astype(str)
        .str.strip()
    )

    dataframe[
        "normalized_group"
    ] = (
        dataframe[group_column]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    dataframe = dataframe[
        dataframe[
            "normalized_split"
        ].isin(
            ALLOWED_SOURCE_SPLITS
        )
        & dataframe[
            "normalized_group"
        ].isin(
            [
                "nc",
                "nw",
            ]
        )
    ].copy()

    dataframe = dataframe.drop_duplicates(
        subset=[
            split_column,
            group_column,
            image_column,
        ]
    ).reset_index(drop=True)

    if args.limit > 0:
        dataframe = (
            dataframe
            .head(args.limit)
            .copy()
        )

    device = select_device(
        args.device
    )

    state_dict = read_checkpoint(
        CHECKPOINT_PATH,
        device,
    )

    model = build_model(
        state_dict,
        device,
    )

    print("=" * 78)
    print(
        "DARTIS ZOR NEGATİF VERIFIER CROP HAZIRLIĞI"
    )
    print("=" * 78)
    print("Cihaz:", device)
    print("İşlenecek görüntü:", len(dataframe))
    print("Threshold:", args.threshold)
    print("Crop boyutu:", args.crop_size)
    print("Stride:", args.stride)
    print(
        "External test kullanımı: YOK"
    )
    print()

    output_rows = []
    failure_rows = []

    group_images = Counter()
    group_hard = Counter()
    group_easy = Counter()

    source_split_images = Counter()
    source_split_crops = Counter()

    qc_saved = 0

    for position, row in enumerate(
        dataframe.to_dict(
            orient="records"
        ),
        start=1,
    ):
        source_split = clean_text(
            row.get(
                split_column,
                "",
            )
        )

        output_split = (
            ALLOWED_SOURCE_SPLITS[
                source_split
            ]
        )

        group_name = clean_text(
            row.get(
                group_column,
                "",
            )
        ).lower()

        image_name = clean_text(
            row.get(
                image_column,
                "",
            )
        )

        scene_id = (
            clean_text(
                row.get(
                    scene_column,
                    "",
                )
            )
            if scene_column
            else ""
        )

        image_path = resolve_image_path(
            row,
            group_name,
            image_name,
            path_column,
        )

        if image_path is None:
            failure_rows.append(
                {
                    "image_set": group_name,
                    "image_name": image_name,
                    "reason": "image_missing",
                }
            )

            print(
                f"[{position:04d}/{len(dataframe):04d}] "
                f"HATA görüntü yok: "
                f"{group_name}/{image_name}"
            )

            continue

        try:
            with Image.open(
                image_path
            ) as source_image:
                image = source_image.convert(
                    "L"
                )

            image_array = (
                np.asarray(
                    image,
                    dtype=np.float32,
                )
                / 255.0
            )

            probability = run_model(
                model,
                image_array,
                device,
            )

            if probability.shape != (
                image.height,
                image.width,
            ):
                raise RuntimeError(
                    "Model çıktı boyutu görüntüyle uyuşmuyor: "
                    f"{probability.shape} != "
                    f"{(image.height, image.width)}"
                )

            candidates = build_candidates(
                probability,
                args.crop_size,
                args.stride,
                args.threshold,
            )

            strong_hard_candidates = [
                candidate
                for candidate in candidates
                if candidate[
                    "positive_ratio"
                ]
                >= args.minimum_positive_ratio
            ]

            strong_hard_candidates.sort(
                key=lambda candidate: (
                    candidate[
                        "positive_ratio"
                    ],
                    candidate[
                        "percentile_95"
                    ],
                    candidate[
                        "mean_probability"
                    ],
                ),
                reverse=True,
            )

            remaining_candidates = sorted(
                candidates,
                key=lambda candidate: (
                    candidate[
                        "score"
                    ],
                    candidate[
                        "maximum_probability"
                    ],
                ),
                reverse=True,
            )

            hard_pool = list(
                strong_hard_candidates
            )

            existing_boxes = {
                candidate["box"]
                for candidate in hard_pool
            }

            hard_pool.extend(
                candidate
                for candidate
                in remaining_candidates
                if candidate["box"]
                not in existing_boxes
            )

            selected_hard = (
                select_non_overlapping(
                    hard_pool,
                    args.hard_per_image,
                    args.maximum_iou,
                )
            )

            selected_hard_boxes = [
                candidate["box"]
                for candidate
                in selected_hard
            ]

            easy_pool = sorted(
                candidates,
                key=lambda candidate: (
                    candidate[
                        "mean_probability"
                    ],
                    candidate[
                        "percentile_95"
                    ],
                ),
            )

            selected_easy = (
                select_non_overlapping(
                    easy_pool,
                    args.easy_per_image,
                    args.maximum_iou,
                    blocked_boxes=(
                        selected_hard_boxes
                    ),
                )
            )

            selected_groups = [
                (
                    "hard",
                    selected_hard,
                ),
                (
                    "easy",
                    selected_easy,
                ),
            ]

            image_saved_crops = 0

            for negative_type, selected in (
                selected_groups
            ):
                for crop_index, candidate in enumerate(
                    selected,
                    start=1,
                ):
                    (
                        left,
                        top,
                        right,
                        bottom,
                    ) = candidate["box"]

                    crop_name = (
                        f"{Path(image_name).stem}"
                        f"__{negative_type}"
                        f"_{crop_index:02d}"
                        f"__x{left}_y{top}.png"
                    )

                    crop_path = (
                        CROP_ROOT
                        / output_split
                        / group_name
                        / negative_type
                        / crop_name
                    )

                    save_crop(
                        image,
                        candidate["box"],
                        crop_path,
                        args.crop_size,
                        args.overwrite,
                    )

                    output_rows.append(
                        {
                            "source_dataset": (
                                "DARTIS"
                            ),
                            "binary_label": 0,
                            "binary_label_name": (
                                "non_oil"
                            ),
                            "negative_type": (
                                negative_type
                            ),
                            "image_set": (
                                group_name
                            ),
                            "scene_id": (
                                scene_id
                            ),
                            "source_split": (
                                source_split
                            ),
                            "split": (
                                output_split
                            ),
                            "source_image_name": (
                                image_name
                            ),
                            "source_image_path": str(
                                image_path
                            ),
                            "crop_index": (
                                crop_index
                            ),
                            "crop_left": left,
                            "crop_top": top,
                            "crop_right": right,
                            "crop_bottom": bottom,
                            "crop_size": (
                                args.crop_size
                            ),
                            "model_threshold": (
                                args.threshold
                            ),
                            "predicted_positive_ratio": (
                                candidate[
                                    "positive_ratio"
                                ]
                            ),
                            "mean_oil_probability": (
                                candidate[
                                    "mean_probability"
                                ]
                            ),
                            "maximum_oil_probability": (
                                candidate[
                                    "maximum_probability"
                                ]
                            ),
                            "percentile_95_probability": (
                                candidate[
                                    "percentile_95"
                                ]
                            ),
                            "selection_score": (
                                candidate[
                                    "score"
                                ]
                            ),
                            "crop_name": (
                                crop_name
                            ),
                            "crop_path": str(
                                crop_path
                            ),
                        }
                    )

                    image_saved_crops += 1

                    if negative_type == "hard":
                        group_hard[
                            group_name
                        ] += 1
                    else:
                        group_easy[
                            group_name
                        ] += 1

            group_images[
                group_name
            ] += 1

            source_split_images[
                source_split
            ] += 1

            source_split_crops[
                source_split
            ] += image_saved_crops

            if (
                qc_saved
                < args.qc_images
            ):
                qc_destination = (
                    QC_DIR
                    / output_split
                    / group_name
                    / (
                        f"{Path(image_name).stem}"
                        f"__selected.jpg"
                    )
                )

                save_qc_overlay(
                    image,
                    selected_hard,
                    selected_easy,
                    qc_destination,
                )

                qc_saved += 1

            maximum_ratio = max(
                (
                    candidate[
                        "positive_ratio"
                    ]
                    for candidate
                    in selected_hard
                ),
                default=0.0,
            )

            print(
                f"[{position:04d}/{len(dataframe):04d}] "
                f"OK {source_split} "
                f"{group_name}/{image_name} "
                f"| hard={len(selected_hard)} "
                f"| easy={len(selected_easy)} "
                f"| max_fp={maximum_ratio:.4f}"
            )

        except Exception as error:
            failure_rows.append(
                {
                    "image_set": group_name,
                    "image_name": image_name,
                    "reason": (
                        f"{type(error).__name__}: "
                        f"{error}"
                    ),
                }
            )

            print(
                f"[{position:04d}/{len(dataframe):04d}] "
                f"HATA {group_name}/{image_name}: "
                f"{type(error).__name__}: {error}"
            )

    output_dataframe = pd.DataFrame(
        output_rows
    )

    output_dataframe.to_csv(
        OUTPUT_MANIFEST,
        index=False,
        encoding="utf-8-sig",
    )

    hard_crops = int(
        (
            output_dataframe[
                "negative_type"
            ]
            == "hard"
        ).sum()
    ) if not output_dataframe.empty else 0

    easy_crops = int(
        (
            output_dataframe[
                "negative_type"
            ]
            == "easy"
        ).sum()
    ) if not output_dataframe.empty else 0

    summary = {
        "processed_images": int(
            len(dataframe)
        ),
        "successful_images": int(
            sum(
                group_images.values()
            )
        ),
        "failed_images": int(
            len(failure_rows)
        ),
        "saved_crops": int(
            len(output_dataframe)
        ),
        "hard_crops": hard_crops,
        "easy_crops": easy_crops,
        "threshold": float(
            args.threshold
        ),
        "crop_size": int(
            args.crop_size
        ),
        "stride": int(
            args.stride
        ),
        "external_test_used": False,
        "groups": {
            group_name: {
                "images": int(
                    group_images[
                        group_name
                    ]
                ),
                "hard": int(
                    group_hard[
                        group_name
                    ]
                ),
                "easy": int(
                    group_easy[
                        group_name
                    ]
                ),
                "total": int(
                    group_hard[
                        group_name
                    ]
                    + group_easy[
                        group_name
                    ]
                ),
            }
            for group_name in (
                "nc",
                "nw",
            )
        },
        "source_splits": {
            source_split: {
                "output_split": (
                    ALLOWED_SOURCE_SPLITS[
                        source_split
                    ]
                ),
                "images": int(
                    source_split_images[
                        source_split
                    ]
                ),
                "crops": int(
                    source_split_crops[
                        source_split
                    ]
                ),
            }
            for source_split in (
                "hard_negative_train",
                "hard_negative_val",
            )
        },
        "failure_examples": (
            failure_rows[:30]
        ),
    }

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
            ensure_ascii=False,
        )

    write_report(
        summary
    )

    print()
    print("=" * 78)
    print("NEGATİF CROP SONUCU")
    print("=" * 78)

    print(
        "İşlenen görüntü:",
        summary[
            "processed_images"
        ],
    )

    print(
        "Başarılı görüntü:",
        summary[
            "successful_images"
        ],
    )

    print(
        "Toplam negatif crop:",
        summary[
            "saved_crops"
        ],
    )

    print(
        "Zor negatif:",
        summary[
            "hard_crops"
        ],
    )

    print(
        "Kolay negatif:",
        summary[
            "easy_crops"
        ],
    )

    print()
    print(
        "Manifest:",
        OUTPUT_MANIFEST.resolve(),
    )

    print(
        "QC:",
        QC_DIR.resolve(),
    )

    print(
        "Özet:",
        SUMMARY_PATH.resolve(),
    )

    print(
        "Rapor:",
        REPORT_PATH.resolve(),
    )


if __name__ == "__main__":
    main()
