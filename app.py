from __future__ import annotations

import hashlib
import io
import json
import math
import runpy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18
from torchvision.transforms import InterpolationMode


ROOT = Path(__file__).resolve().parent

DETECTOR_SCRIPT = (
    ROOT
    / "scripts"
    / "33_run_end_to_end_oil_detector.py"
)

FINAL_MODEL_DIR = (
    ROOT
    / "checkpoints"
    / "final_model_v03"
)

SEGMENTATION_MODEL = (
    FINAL_MODEL_DIR
    / "oil_segmentation_unet.pth"
)

VERIFIER_MODEL = (
    FINAL_MODEL_DIR
    / "oil_candidate_verifier_resnet18.pth"
)

CALIBRATION_CONFIG = (
    FINAL_MODEL_DIR
    / "calibration_config.json"
)

MODEL_CARD = (
    FINAL_MODEL_DIR
    / "model_card.json"
)


INPUT_GATE_DIR = (
    ROOT
    / "checkpoints"
    / "input_gate_v05"
)

INPUT_GATE_CHECKPOINT = (
    INPUT_GATE_DIR
    / "best_input_gate_v05.pth"
)

INPUT_GATE_CONFIG = (
    INPUT_GATE_DIR
    / "input_gate_config_v05.json"
)

INPUT_GATE_PROTOTYPES = (
    INPUT_GATE_DIR
    / "supported_prototypes_v05.npz"
)

INPUT_GATE_LOCKED_SUMMARY = (
    ROOT
    / "outputs"
    / "v05_input_gate_negative_locked_test"
    / "locked_negative_summary.json"
)


st.set_page_config(
    page_title="SAR Petrol Tespit Sistemi",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.25rem;
        font-weight: 750;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #666;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    .result-positive {
        padding: 1rem;
        border-radius: 0.7rem;
        background: rgba(220, 53, 69, 0.12);
        border: 1px solid rgba(220, 53, 69, 0.45);
        font-size: 1.25rem;
        font-weight: 700;
    }

    .result-negative {
        padding: 1rem;
        border-radius: 0.7rem;
        background: rgba(25, 135, 84, 0.12);
        border: 1px solid rgba(25, 135, 84, 0.45);
        font-size: 1.25rem;
        font-weight: 700;
    }

    .gate-accept {
        padding: 1rem;
        border-radius: 0.7rem;
        background: rgba(25, 135, 84, 0.12);
        border: 1px solid rgba(25, 135, 84, 0.45);
        font-size: 1.1rem;
        font-weight: 700;
    }

    .gate-reject {
        padding: 1rem;
        border-radius: 0.7rem;
        background: rgba(220, 53, 69, 0.12);
        border: 1px solid rgba(220, 53, 69, 0.45);
        font-size: 1.1rem;
        font-weight: 700;
    }

    .gate-uncertain {
        padding: 1rem;
        border-radius: 0.7rem;
        background: rgba(255, 193, 7, 0.12);
        border: 1px solid rgba(255, 193, 7, 0.45);
        font-size: 1.1rem;
        font-weight: 700;
    }

    .scientific-note {
        padding: 0.9rem;
        border-radius: 0.6rem;
        background: rgba(255, 193, 7, 0.12);
        border: 1px solid rgba(255, 193, 7, 0.45);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Gerekli dosya bulunamadı: {path}"
        )


def load_json_if_exists(
    path: Path,
) -> dict[str, Any]:
    if not path.exists():
        return {}

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def load_model_card() -> dict[str, Any]:
    return load_json_if_exists(
        MODEL_CARD
    )


@st.cache_resource(
    show_spinner=False
)
def load_pipeline():
    require_file(
        DETECTOR_SCRIPT
    )

    require_file(
        SEGMENTATION_MODEL
    )

    require_file(
        VERIFIER_MODEL
    )

    require_file(
        CALIBRATION_CONFIG
    )

    detector = runpy.run_path(
        str(DETECTOR_SCRIPT),
        run_name=(
            "streamlit_detector_module"
        ),
    )

    # Script 33'ün model yollarını final v0.3
    # paketindeki dosyalara yönlendiriyoruz.
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
        "auto"
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


def select_gate_device() -> torch.device:
    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def forward_gate_features(
    model: nn.Module,
    images: torch.Tensor,
) -> torch.Tensor:
    x = model.conv1(images)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = model.avgpool(x)

    return torch.flatten(
        x,
        1,
    )


@st.cache_resource(
    show_spinner=False
)
def load_input_gate() -> dict[str, Any]:
    require_file(
        INPUT_GATE_CHECKPOINT
    )
    require_file(
        INPUT_GATE_CONFIG
    )
    require_file(
        INPUT_GATE_PROTOTYPES
    )

    device = select_gate_device()

    try:
        checkpoint = torch.load(
            INPUT_GATE_CHECKPOINT,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            INPUT_GATE_CHECKPOINT,
            map_location=device,
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

    model.fc = nn.Sequential(
        nn.Dropout(p=0.30),
        nn.Linear(
            model.fc.in_features,
            1,
        ),
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    model.to(device)
    model.eval()

    config = load_json_if_exists(
        INPUT_GATE_CONFIG
    )

    if not config:
        raise RuntimeError(
            "Input-gate config dosyası boş veya okunamadı."
        )

    prototype_data = np.load(
        INPUT_GATE_PROTOTYPES,
        allow_pickle=False,
    )

    prototype_names = prototype_data[
        "group_names"
    ].astype(str)

    prototypes = prototype_data[
        "centroids"
    ].astype(
        np.float32
    )

    input_size = int(
        config.get(
            "input_size",
            checkpoint.get(
                "input_size",
                224,
            ),
        )
    )

    mean = tuple(
        config.get(
            "normalization_mean",
            checkpoint.get(
                "normalization_mean",
                [0.5],
            ),
        )
    )

    std = tuple(
        config.get(
            "normalization_std",
            checkpoint.get(
                "normalization_std",
                [0.25],
            ),
        )
    )

    gate_transform = transforms.Compose(
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

    return {
        "device": device,
        "model": model,
        "config": config,
        "prototype_names": (
            prototype_names
        ),
        "prototypes": prototypes,
        "transform": gate_transform,
    }


@torch.inference_mode()
def run_input_gate(
    image: Image.Image,
    gate: dict[str, Any],
) -> dict[str, Any]:
    grayscale = image.convert("L")

    image_tensor = gate[
        "transform"
    ](
        grayscale
    ).unsqueeze(0).to(
        gate["device"]
    )

    features = forward_gate_features(
        gate["model"],
        image_tensor,
    )

    logits = gate[
        "model"
    ].fc(
        features
    ).flatten()

    normalized_features = F.normalize(
        features.float(),
        p=2,
        dim=1,
    )

    config = gate[
        "config"
    ]

    temperature = float(
        config[
            "temperature"
        ]
    )

    if temperature <= 0:
        raise RuntimeError(
            "Input-gate temperature değeri pozitif olmalıdır."
        )

    probability = float(
        torch.sigmoid(
            logits.float()
            / temperature
        )[0]
        .cpu()
        .item()
    )

    feature_array = (
        normalized_features
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )

    similarities = (
        feature_array
        @ gate[
            "prototypes"
        ].T
    )[0]

    nearest_index = int(
        np.argmax(
            similarities
        )
    )

    similarity = float(
        similarities[
            nearest_index
        ]
    )

    accept_probability = float(
        config[
            "accept_probability_threshold"
        ]
    )
    accept_similarity = float(
        config[
            "accept_similarity_threshold"
        ]
    )
    reject_probability = float(
        config[
            "reject_probability_threshold"
        ]
    )
    reject_similarity = float(
        config[
            "reject_similarity_threshold"
        ]
    )

    accepted = (
        probability
        >= accept_probability
        and similarity
        >= accept_similarity
    )

    rejected = (
        probability
        <= reject_probability
        or similarity
        <= reject_similarity
    )

    if accepted:
        decision = "ACCEPT"
    elif rejected:
        decision = "REJECT"
    else:
        decision = "UNCERTAIN"

    return {
        "decision": decision,
        "pipeline_allowed": (
            decision == "ACCEPT"
        ),
        "supported_probability": (
            probability
        ),
        "prototype_similarity": (
            similarity
        ),
        "nearest_prototype": str(
            gate[
                "prototype_names"
            ][nearest_index]
        ),
        "temperature": temperature,
        "accept_probability_threshold": (
            accept_probability
        ),
        "accept_similarity_threshold": (
            accept_similarity
        ),
        "reject_probability_threshold": (
            reject_probability
        ),
        "reject_similarity_threshold": (
            reject_similarity
        ),
        "important_note": (
            "Input-gate skoru sertifikalı gerçek-dünya "
            "olasılığı değildir. Yalnız görüntünün eğitimde "
            "tanımlanan DARTIS benzeri deniz/kıyı SAR veri "
            "alanına uygunluğunu ölçer."
        ),
    }


def render_input_gate_result(
    gate_result: dict[str, Any],
) -> None:
    decision = gate_result[
        "decision"
    ]

    if decision == "ACCEPT":
        st.markdown(
            '<div class="gate-accept">'
            '✅ Girdi kabul edildi: DARTIS benzeri '
            'deniz/kıyı SAR görüntüsü olarak doğrulandı.'
            '</div>',
            unsafe_allow_html=True,
        )
    elif decision == "REJECT":
        st.markdown(
            '<div class="gate-reject">'
            '⛔ Girdi reddedildi: Bu görüntü desteklenen '
            'deniz/kıyı SAR veri alanına uygun değil. '
            'Petrol analizi çalıştırılmadı.'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="gate-uncertain">'
            '⚠️ Girdi belirsiz: Görüntü güvenli biçimde '
            'doğrulanamadı. Petrol analizi çalıştırılmadı.'
            '</div>',
            unsafe_allow_html=True,
        )

    columns = st.columns(3)

    columns[0].metric(
        "Desteklenen giriş skoru",
        f"%{gate_result['supported_probability'] * 100.0:.3f}",
    )

    columns[1].metric(
        "DARTIS benzerliği",
        f"{gate_result['prototype_similarity']:.4f}",
    )

    columns[2].metric(
        "En yakın prototip",
        gate_result[
            "nearest_prototype"
        ],
    )

    st.caption(
        "Bu değerler gerçek petrol olasılığı değildir. "
        "Yalnız giriş uygunluğu için kullanılır."
    )


def pil_to_png_bytes(
    image: Image.Image,
) -> bytes:
    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG",
    )

    return buffer.getvalue()


def mask_to_image(
    mask: np.ndarray,
) -> Image.Image:
    return Image.fromarray(
        (
            mask.astype(
                np.uint8
            )
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


def analyse_image(
    image: Image.Image,
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
                "area_percent": float(
                    component[
                        "area_pixels"
                    ]
                    / total_pixels
                    * 100.0
                ),
                "bbox": tuple(
                    int(value)
                    for value in bbox
                ),
                "crop_box": tuple(
                    int(value)
                    for value
                    in crop_box
                ),
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
                "accepted": accepted,
                "decision": (
                    "Petrol"
                    if accepted
                    else "Petrol değil"
                ),
            }
        )

    raw_positive_pixels = int(
        raw_mask.sum()
    )

    final_positive_pixels = int(
        final_mask.sum()
    )

    raw_coverage_percent = (
        raw_positive_pixels
        / total_pixels
        * 100.0
    )

    final_coverage_percent = (
        final_positive_pixels
        / total_pixels
        * 100.0
    )

    accepted_candidates = [
        candidate
        for candidate
        in candidate_results
        if candidate["accepted"]
    ]

    rejected_candidates = [
        candidate
        for candidate
        in candidate_results
        if not candidate["accepted"]
    ]

    maximum_candidate_score = max(
        (
            candidate[
                "selected_probability"
            ]
            for candidate
            in candidate_results
        ),
        default=0.0,
    )

    maximum_accepted_score = max(
        (
            candidate[
                "selected_probability"
            ]
            for candidate
            in accepted_candidates
        ),
        default=0.0,
    )

    oil_detected = (
        final_positive_pixels > 0
    )

    if oil_detected:
        decision_score = (
            maximum_accepted_score
        )

    elif candidate_results:
        decision_score = (
            1.0
            - maximum_candidate_score
        )

    else:
        decision_score = None

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

    result_json = {
        "oil_detected": bool(
            oil_detected
        ),
        "image_width": int(
            image_width
        ),
        "image_height": int(
            image_height
        ),
        "segmentation_threshold": float(
            segmentation_threshold
        ),
        "minimum_component_pixels": int(
            minimum_area
        ),
        "raw_positive_pixels": (
            raw_positive_pixels
        ),
        "final_oil_pixels": (
            final_positive_pixels
        ),
        "raw_coverage_percent": float(
            raw_coverage_percent
        ),
        "final_oil_coverage_percent": float(
            final_coverage_percent
        ),
        "candidate_count": int(
            len(candidate_results)
        ),
        "accepted_candidate_count": int(
            len(accepted_candidates)
        ),
        "rejected_candidate_count": int(
            len(rejected_candidates)
        ),
        "maximum_candidate_score": float(
            maximum_candidate_score
        ),
        "decision_score": (
            float(decision_score)
            if decision_score
            is not None
            else None
        ),
        "candidate_results": (
            candidate_results
        ),
        "important_note": (
            "Kaplama oranı bütün görüntü alanına göre "
            "hesaplanmıştır. Ayrı kara-deniz maskesi yoktur. "
            "Gösterilen güven değeri sertifikalı bütün-görüntü "
            "petrol olasılığı değil, verifier karar skorudur."
        ),
    }

    return {
        "image": image,
        "probability": probability,
        "probability_image": (
            probability_to_image(
                probability
            )
        ),
        "raw_mask": raw_mask,
        "raw_mask_image": (
            mask_to_image(
                raw_mask
            )
        ),
        "final_mask": final_mask,
        "final_mask_image": (
            mask_to_image(
                final_mask
            )
        ),
        "overlay": overlay,
        "candidate_overlay": (
            candidate_overlay
        ),
        "candidate_results": (
            candidate_results
        ),
        "result_json": result_json,
        "oil_detected": (
            oil_detected
        ),
        "raw_coverage_percent": (
            raw_coverage_percent
        ),
        "final_coverage_percent": (
            final_coverage_percent
        ),
        "accepted_candidate_count": (
            len(
                accepted_candidates
            )
        ),
        "candidate_count": (
            len(
                candidate_results
            )
        ),
        "decision_score": (
            decision_score
        ),
    }


st.markdown(
    '<div class="main-title">'
    '🛰️ SAR Petrol Sızıntısı Tespit Sistemi'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'v0.5 güvenli giriş kapısı + v0.3 petrol '
    'segmentasyonu ve aday doğrulaması'
    '</div>',
    unsafe_allow_html=True,
)


model_card = load_model_card()
gate_locked_summary = load_json_if_exists(
    INPUT_GATE_LOCKED_SUMMARY
)

with st.sidebar:
    st.header(
        "Model ayarları"
    )

    segmentation_threshold = (
        st.slider(
            "U-Net segmentasyon eşiği",
            min_value=0.10,
            max_value=0.95,
            value=0.60,
            step=0.05,
            disabled=True,
            help=(
                "Kilitli testte kullanılan "
                "dondurulmuş eşiktir."
            ),
        )
    )

    minimum_component_ratio = (
        st.number_input(
            "Minimum aday alan oranı",
            min_value=0.0001,
            max_value=0.0100,
            value=0.0005,
            step=0.0001,
            format="%.4f",
            disabled=True,
        )
    )

    minimum_component_pixels = (
        st.number_input(
            "Minimum aday pikseli",
            min_value=1,
            max_value=10000,
            value=16,
            step=1,
            disabled=True,
        )
    )

    st.divider()

    st.subheader(
        "v0.5 giriş kapısı"
    )

    if gate_locked_summary:
        st.write(
            "Yeni negatif kilitli test:",
            int(
                gate_locked_summary.get(
                    "total_negative_samples",
                    0,
                )
            ),
        )
        st.write(
            "Yanlış kabul:",
            int(
                gate_locked_summary.get(
                    "false_accept_count",
                    0,
                )
            ),
        )
        st.write(
            "Güvenli engelleme:",
            f"%{float(gate_locked_summary.get('safe_block_rate', 0)) * 100:.2f}",
        )

    st.caption(
        "Input-gate yalnız DARTIS benzeri deniz/kıyı "
        "SAR giriş alanı için geliştirilmiştir."
    )

    st.divider()

    st.subheader(
        "v0.3 petrol modeli"
    )

    if model_card:
        metrics = model_card.get(
            "locked_test_metrics",
            {},
        )

        st.write(
            "Pozitif görüntü recall:",
            f"%{float(metrics.get('positive_image_recall', 0)) * 100:.2f}",
        )

        st.write(
            "Kutu recall:",
            f"%{float(metrics.get('positive_box_recall', 0)) * 100:.2f}",
        )

        st.write(
            "Ciddi yanlış alarm:",
            f"%{float(metrics.get('negative_severe_alarm_percent', 0)):.2f}",
        )

        st.write(
            "FP piksel:",
            f"%{float(metrics.get('negative_fp_pixel_percent', 0)):.4f}",
        )

    st.warning(
        "Bu uygulama araştırma prototipidir. "
        "Uzman doğrulaması olmadan operasyonel "
        "karar için kullanılmamalıdır."
    )


uploaded_file = st.file_uploader(
    "Görüntüyü yükleyin",
    type=[
        "png",
        "jpg",
        "jpeg",
        "tif",
        "tiff",
    ],
    help=(
        "Sistem yalnız DARTIS benzeri deniz veya kıyı "
        "SAR görüntülerini petrol analizine kabul eder. "
        "Optik, hava, şehir ve deniz dışı SAR görüntüleri "
        "reddedilir ya da belirsiz olarak durdurulur."
    ),
)


if uploaded_file is None:
    st.info(
        "Başlamak için bir görüntü yükleyin. "
        "Petrol analizi öncesinde giriş uygunluğu kontrol edilir."
    )

    st.stop()


uploaded_bytes = (
    uploaded_file.getvalue()
)

upload_hash = hashlib.sha256(
    uploaded_bytes
).hexdigest()

if st.session_state.get(
    "active_upload_hash"
) != upload_hash:
    for state_key in (
        "input_gate_result",
        "oil_result",
        "oil_filename",
        "analysed_upload_hash",
    ):
        st.session_state.pop(
            state_key,
            None,
        )

    st.session_state[
        "active_upload_hash"
    ] = upload_hash

try:
    uploaded_image = (
        Image.open(
            io.BytesIO(
                uploaded_bytes
            )
        )
        .convert("L")
        .copy()
    )

except Exception as error:
    st.error(
        "Görüntü okunamadı: "
        f"{type(error).__name__}: {error}"
    )

    st.stop()


image_width, image_height = (
    uploaded_image.size
)

if (
    image_width > 2048
    or image_height > 2048
):
    st.error(
        "Sunum modu en fazla 2048×2048 piksel "
        "görüntü kabul etmektedir."
    )

    st.stop()


preview_column, info_column = (
    st.columns(
        [2, 1]
    )
)

with preview_column:
    st.image(
        uploaded_image,
        caption=(
            "Yüklenen görüntü — model girdisi olarak "
            "gri seviyede gösteriliyor"
        ),
        use_container_width=True,
    )

with info_column:
    st.write(
        "**Dosya:**",
        uploaded_file.name,
    )

    st.write(
        "**Boyut:**",
        f"{image_width} × {image_height}",
    )

    st.write(
        "**İşlem sırası:**",
        "Giriş kontrolü → kabul edilirse petrol analizi",
    )

    analyse_button = st.button(
        "Girdiyi doğrula ve analizi başlat",
        type="primary",
        use_container_width=True,
    )


if analyse_button:
    for state_key in (
        "input_gate_result",
        "oil_result",
        "oil_filename",
        "analysed_upload_hash",
    ):
        st.session_state.pop(
            state_key,
            None,
        )

    try:
        with st.spinner(
            "Görüntünün desteklenen deniz/kıyı SAR "
            "girişi olup olmadığı kontrol ediliyor..."
        ):
            input_gate = load_input_gate()

            gate_result = run_input_gate(
                uploaded_image,
                input_gate,
            )

        st.session_state[
            "input_gate_result"
        ] = gate_result

        st.session_state[
            "analysed_upload_hash"
        ] = upload_hash

        if gate_result[
            "pipeline_allowed"
        ]:
            with st.spinner(
                "Girdi kabul edildi. Petrol modelleri "
                "yükleniyor ve görüntü analiz ediliyor..."
            ):
                pipeline = load_pipeline()

                result = analyse_image(
                    uploaded_image,
                    pipeline,
                    segmentation_threshold,
                    minimum_component_ratio,
                    int(
                        minimum_component_pixels
                    ),
                )

            result[
                "result_json"
            ][
                "input_gate"
            ] = gate_result

            st.session_state[
                "oil_result"
            ] = result

            st.session_state[
                "oil_filename"
            ] = Path(
                uploaded_file.name
            ).stem

    except Exception as error:
        st.exception(error)


if st.session_state.get(
    "analysed_upload_hash"
) != upload_hash:
    st.stop()


gate_result = st.session_state.get(
    "input_gate_result"
)

if gate_result is None:
    st.stop()


st.divider()
st.subheader(
    "1. Giriş doğrulama sonucu"
)

render_input_gate_result(
    gate_result
)

if not gate_result[
    "pipeline_allowed"
]:
    st.stop()


if "oil_result" not in st.session_state:
    st.error(
        "Girdi kabul edildi ancak petrol analizi sonucu "
        "oluşturulamadı. Terminaldeki hata mesajını kontrol edin."
    )
    st.stop()


st.divider()
st.subheader(
    "2. Petrol analiz sonucu"
)

result = st.session_state[
    "oil_result"
]

filename_stem = st.session_state[
    "oil_filename"
]


if result[
    "oil_detected"
]:
    st.markdown(
        '<div class="result-positive">'
        '🔴 Sonuç: PETROL ADAYI TESPİT EDİLDİ'
        '</div>',
        unsafe_allow_html=True,
    )

else:
    st.markdown(
        '<div class="result-negative">'
        '🟢 Sonuç: PETROL TESPİT EDİLMEDİ'
        '</div>',
        unsafe_allow_html=True,
    )


metric_columns = st.columns(4)

metric_columns[0].metric(
    "Nihai kaplama",
    (
        f"%{result['final_coverage_percent']:.4f}"
    ),
)

metric_columns[1].metric(
    "Ham U-Net kaplama",
    (
        f"%{result['raw_coverage_percent']:.4f}"
    ),
)

metric_columns[2].metric(
    "Kabul edilen aday",
    (
        f"{result['accepted_candidate_count']}"
        f"/{result['candidate_count']}"
    ),
)

decision_score = result[
    "decision_score"
]

metric_columns[3].metric(
    "Verifier karar skoru",
    (
        f"%{decision_score * 100.0:.2f}"
        if decision_score
        is not None
        else "Hesaplanamadı"
    ),
)


st.markdown(
    """
    <div class="scientific-note">
    <b>Bilimsel not:</b>
    Giriş kapısı yalnız desteklenen veri alanını kontrol eder.
    Kaplama oranı bütün görüntü piksellerine göre
    hesaplanmaktadır. Ayrı bir kara-deniz maskesi henüz
    bulunmamaktadır. Verifier skoru sertifikalı bir
    bütün-görüntü petrol olasılığı değildir.
    </div>
    """,
    unsafe_allow_html=True,
)


tabs = st.tabs(
    [
        "Nihai sonuç",
        "Model aşamaları",
        "Aday kararları",
        "Dosyaları indir",
    ]
)


with tabs[0]:
    first_column, second_column = (
        st.columns(2)
    )

    with first_column:
        st.image(
            result[
                "final_mask_image"
            ],
            caption="Nihai petrol maskesi",
            use_container_width=True,
        )

    with second_column:
        st.image(
            result["overlay"],
            caption=(
                "SAR görüntüsü üzerinde "
                "nihai petrol maskesi"
            ),
            use_container_width=True,
        )


with tabs[1]:
    stage_columns = st.columns(
        3
    )

    with stage_columns[0]:
        st.image(
            result[
                "probability_image"
            ],
            caption=(
                "U-Net petrol olasılık haritası"
            ),
            use_container_width=True,
        )

    with stage_columns[1]:
        st.image(
            result[
                "raw_mask_image"
            ],
            caption="Ham U-Net maskesi",
            use_container_width=True,
        )

    with stage_columns[2]:
        st.image(
            result[
                "candidate_overlay"
            ],
            caption=(
                "Kırmızı: kabul — "
                "Sarı: reddedildi"
            ),
            use_container_width=True,
        )


with tabs[2]:
    candidate_dataframe = (
        pd.DataFrame(
            result[
                "candidate_results"
            ]
        )
    )

    if candidate_dataframe.empty:
        st.info(
            "Minimum alan koşulunu sağlayan "
            "bir petrol adayı bulunamadı."
        )

    else:
        display_columns = [
            "component_index",
            "decision",
            "area_pixels",
            "area_percent",
            "selected_probability",
            "selected_threshold",
        ]

        st.dataframe(
            candidate_dataframe[
                display_columns
            ],
            use_container_width=True,
            hide_index=True,
        )


with tabs[3]:
    result_json_bytes = (
        json.dumps(
            result[
                "result_json"
            ],
            ensure_ascii=False,
            indent=2,
        ).encode(
            "utf-8"
        )
    )

    download_columns = st.columns(
        3
    )

    with download_columns[0]:
        st.download_button(
            "Nihai maskeyi indir",
            data=pil_to_png_bytes(
                result[
                    "final_mask_image"
                ]
            ),
            file_name=(
                f"{filename_stem}"
                f"_oil_mask.png"
            ),
            mime="image/png",
            use_container_width=True,
        )

    with download_columns[1]:
        st.download_button(
            "Overlay'i indir",
            data=pil_to_png_bytes(
                result[
                    "overlay"
                ]
            ),
            file_name=(
                f"{filename_stem}"
                f"_oil_overlay.png"
            ),
            mime="image/png",
            use_container_width=True,
        )

    with download_columns[2]:
        st.download_button(
            "JSON raporunu indir",
            data=result_json_bytes,
            file_name=(
                f"{filename_stem}"
                f"_oil_result.json"
            ),
            mime="application/json",
            use_container_width=True,
        )
