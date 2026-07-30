from __future__ import annotations

import hashlib
import io
import json
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

SAFE_PIPELINE_SCRIPT = (
    ROOT
    / "scripts"
    / "85_run_safe_water_masked_oil_pipeline.py"
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

WATER_GATE_LOCKED_SUMMARY = (
    ROOT
    / "checkpoints"
    / "final_water_gate_v06"
    / "validation_reports"
    / "locked_test_summary.json"
)


V08_PIPELINE_SCRIPT = (
    ROOT
    / "scripts"
    / "92_test_v07_known_problem_scenes.py"
)

V08_VERIFIER_CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "verifier_v08"
    / "best.pth"
)

V08_OPERATIONAL_CONFIG = (
    ROOT
    / "checkpoints"
    / "verifier_v08"
    / "operational_gate_config.json"
)


st.set_page_config(
    page_title="SAR Petrol Taraması — Final Hassas Mod",
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

    .accept-box {
        padding: 1rem;
        border-radius: 0.7rem;
        background: rgba(25, 135, 84, 0.12);
        border: 1px solid rgba(25, 135, 84, 0.45);
        font-size: 1.1rem;
        font-weight: 700;
    }

    .reject-box {
        padding: 1rem;
        border-radius: 0.7rem;
        background: rgba(220, 53, 69, 0.12);
        border: 1px solid rgba(220, 53, 69, 0.45);
        font-size: 1.1rem;
        font-weight: 700;
    }

    .uncertain-box {
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


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    return json.loads(
        path.read_text(
            encoding="utf-8-sig"
        )
    )


def pil_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(
        buffer,
        format="PNG",
    )
    return buffer.getvalue()


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



def build_sensitive_screening_mask(
    image: Image.Image,
    probability: np.ndarray,
    analysis_mask: np.ndarray,
    absolute_mask: np.ndarray,
) -> tuple[np.ndarray, str, float]:
    probability_array = np.asarray(
        probability,
        dtype=np.float32,
    )

    valid_mask = (
        np.asarray(
            analysis_mask,
            dtype=np.uint8,
        )
        > 0
    )

    absolute_array = (
        np.asarray(
            absolute_mask,
            dtype=np.uint8,
        )
        > 0
    )

    if absolute_array.any():
        return (
            absolute_array.astype(
                np.uint8
            ),
            "ABSOLUTE_UNET_THRESHOLD",
            float(
                st.session_state.get(
                    "final_scan_threshold",
                    0.20,
                )
            ),
        )

    if (
        not valid_mask.any()
        or probability_array.size == 0
    ):
        return (
            np.zeros_like(
                probability_array,
                dtype=np.uint8,
            ),
            "NO_VALID_AREA",
            0.0,
        )

    gray = (
        np.asarray(
            image.convert("L"),
            dtype=np.float32,
        )
        / 255.0
    )

    if gray.shape != probability_array.shape:
        gray = np.asarray(
            image.convert("L").resize(
                (
                    probability_array.shape[1],
                    probability_array.shape[0],
                ),
                Image.Resampling.BILINEAR,
            ),
            dtype=np.float32,
        ) / 255.0

    probability_tensor = torch.from_numpy(
        probability_array
    ).unsqueeze(0).unsqueeze(0)

    smoothed_probability = (
        F.avg_pool2d(
            probability_tensor,
            kernel_size=9,
            stride=1,
            padding=4,
        )
        .squeeze()
        .numpy()
        .astype(np.float32)
    )

    gray_tensor = torch.from_numpy(
        gray
    ).unsqueeze(0).unsqueeze(0)

    local_background = (
        F.avg_pool2d(
            gray_tensor,
            kernel_size=41,
            stride=1,
            padding=20,
        )
        .squeeze()
        .numpy()
        .astype(np.float32)
    )

    local_darkness = np.clip(
        local_background - gray,
        0.0,
        None,
    )

    def robust_normalize(
        values: np.ndarray,
    ) -> np.ndarray:
        valid_values = values[
            valid_mask
        ]

        scale = float(
            np.quantile(
                valid_values,
                0.99,
            )
        )

        if scale <= 1e-8:
            scale = float(
                valid_values.max(
                    initial=0.0
                )
            )

        if scale <= 1e-8:
            return np.zeros_like(
                values,
                dtype=np.float32,
            )

        return np.clip(
            values / scale,
            0.0,
            1.0,
        ).astype(
            np.float32
        )

    probability_score = robust_normalize(
        smoothed_probability
    )

    darkness_score = robust_normalize(
        local_darkness
    )

    combined_score = (
        0.75
        * probability_score
        + 0.25
        * darkness_score
    )

    combined_tensor = torch.from_numpy(
        combined_score
    ).unsqueeze(0).unsqueeze(0)

    combined_score = (
        F.avg_pool2d(
            combined_tensor,
            kernel_size=7,
            stride=1,
            padding=3,
        )
        .squeeze()
        .numpy()
        .astype(np.float32)
    )

    valid_scores = combined_score[
        valid_mask
    ]

    if (
        valid_scores.size == 0
        or float(
            valid_scores.max(
                initial=0.0
            )
        )
        <= 1e-8
    ):
        return (
            np.zeros_like(
                probability_array,
                dtype=np.uint8,
            ),
            "NO_MODEL_RESPONSE",
            0.0,
        )

    relative_cutoff = float(
        np.quantile(
            valid_scores,
            0.995,
        )
    )

    screening_mask = (
        valid_mask
        & (
            combined_score
            >= relative_cutoff
        )
    ).astype(
        np.uint8
    )

    return (
        screening_mask,
        "RELATIVE_TOP_0_5_PERCENT",
        relative_cutoff,
    )

def select_device() -> torch.device:
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

    device = select_device()

    checkpoint = torch.load(
        INPUT_GATE_CHECKPOINT,
        map_location=device,
        weights_only=False,
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

    config = load_json(
        INPUT_GATE_CONFIG
    )

    if not config:
        raise RuntimeError(
            "Input-gate config boş veya okunamadı."
        )

    prototype_data = np.load(
        INPUT_GATE_PROTOTYPES,
        allow_pickle=False,
    )

    prototypes = prototype_data[
        "centroids"
    ].astype(np.float32)

    prototype_names = prototype_data[
        "group_names"
    ].astype(str)

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

    transform = transforms.Compose(
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
        "prototypes": prototypes,
        "prototype_names": (
            prototype_names
        ),
        "transform": transform,
    }


@torch.inference_mode()
def run_input_gate(
    image: Image.Image,
    gate: dict[str, Any],
) -> dict[str, Any]:
    tensor = gate[
        "transform"
    ](
        image.convert("L")
    ).unsqueeze(0).to(
        gate["device"]
    )

    features = forward_gate_features(
        gate["model"],
        tensor,
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
        config["temperature"]
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
        .astype(np.float32)
    )

    similarities = (
        feature_array
        @ gate["prototypes"].T
    )[0]

    nearest_index = int(
        np.argmax(similarities)
    )

    similarity = float(
        similarities[
            nearest_index
        ]
    )

    accepted = (
        probability
        >= float(
            config[
                "accept_probability_threshold"
            ]
        )
        and similarity
        >= float(
            config[
                "accept_similarity_threshold"
            ]
        )
    )

    rejected = (
        probability
        <= float(
            config[
                "reject_probability_threshold"
            ]
        )
        or similarity
        <= float(
            config[
                "reject_similarity_threshold"
            ]
        )
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
    }


@st.cache_resource(
    show_spinner=False
)
def load_safe_module() -> dict[str, Any]:
    require_file(
        SAFE_PIPELINE_SCRIPT
    )

    return runpy.run_path(
        str(SAFE_PIPELINE_SCRIPT),
        run_name=(
            "streamlit_safe_pipeline"
        ),
    )


@st.cache_resource(
    show_spinner=False
)
def load_water_gate_cached() -> dict[str, Any]:
    module = load_safe_module()
    return module[
        "load_water_gate"
    ](
        "auto"
    )


@st.cache_resource(
    show_spinner=False
)
def load_v08_module() -> dict[str, Any]:
    require_file(V08_PIPELINE_SCRIPT)
    return runpy.run_path(
        str(V08_PIPELINE_SCRIPT),
        run_name="streamlit_v08_verifier_pipeline",
    )


@st.cache_resource(
    show_spinner=False
)
def load_oil_pipeline_cached() -> dict[str, Any]:
    require_file(V08_VERIFIER_CHECKPOINT)
    require_file(V08_OPERATIONAL_CONFIG)

    safe_module = load_safe_module()
    pipeline = safe_module["load_oil_pipeline"]("auto")

    checkpoint = torch.load(
        V08_VERIFIER_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )
    config = load_json(V08_OPERATIONAL_CONFIG)

    if not config.get("operational_gate_passed", False):
        raise RuntimeError("v0.8 operational gate PASS değil.")

    pipeline["verifier_model"].load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )
    pipeline["verifier_model"].to(pipeline["device"])
    pipeline["verifier_model"].eval()

    return {
        **pipeline,
        "v07_config": config,
    }


def render_input_gate(
    result: dict[str, Any],
) -> None:
    decision = result[
        "decision"
    ]

    if decision == "ACCEPT":
        st.markdown(
            '<div class="accept-box">'
            '✅ Girdi desteklenen deniz/kıyı SAR '
            'alanına kabul edildi.'
            '</div>',
            unsafe_allow_html=True,
        )
    elif decision == "REJECT":
        st.markdown(
            '<div class="reject-box">'
            '⛔ Girdi desteklenen SAR alanına uygun değil. '
            'Sonraki modeller çalıştırılmadı.'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="uncertain-box">'
            '⚠️ Girdi güvenli biçimde doğrulanamadı. '
            'Sonraki modeller çalıştırılmadı.'
            '</div>',
            unsafe_allow_html=True,
        )

    columns = st.columns(3)

    columns[0].metric(
        "Karar",
        decision,
    )

    columns[1].metric(
        "Desteklenen giriş skoru",
        (
            f"%{result['supported_probability'] * 100.0:.3f}"
        ),
    )

    columns[2].metric(
        "DARTIS benzerliği",
        (
            f"{result['prototype_similarity']:.4f}"
        ),
    )


def render_water_gate(
    result: dict[str, Any],
) -> None:
    if result["pipeline_allowed"]:
        st.markdown(
            '<div class="accept-box">'
            '✅ Kara-su modeli güvenli su alanını kabul etti. '
            'Hassas tarama modunda kesin kara dışındaki güvenli ve '
            'belirsiz alanlar taranıyor.'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="uncertain-box">'
            '⚠️ Kara-su ayrımı belirsiz. Petrol analizi '
            'durdurulmadı; kesin kara dışındaki alanlar '
            'hassas tarama modunda taranıyor.'
            '</div>',
            unsafe_allow_html=True,
        )

    analysis_mask = result.get(
        "analysis_water_mask",
        np.maximum(
            np.asarray(
                result["internal_water_mask"],
                dtype=np.uint8,
            ),
            np.asarray(
                result["uncertain_mask"],
                dtype=np.uint8,
            ),
        ),
    )

    columns = st.columns(4)
    columns[0].metric(
        "Su kararı",
        result["decision"],
    )
    columns[1].metric(
        "Güvenli su oranı",
        f"%{result['predicted_water_fraction'] * 100.0:.2f}",
    )
    columns[2].metric(
        "Belirsiz alan",
        f"%{result['uncertain_fraction'] * 100.0:.2f}",
    )
    columns[3].metric(
        "Taranan alan",
        f"%{float(np.asarray(analysis_mask).mean()) * 100.0:.2f}",
    )

    views = st.columns(4)

    with views[0]:
        st.image(
            mask_to_image(
                result["internal_water_mask"]
            ),
            caption="Yüksek güvenli su",
            use_container_width=True,
        )

    with views[1]:
        st.image(
            mask_to_image(
                result["uncertain_mask"]
            ),
            caption="Belirsiz fakat taramaya açık alan",
            use_container_width=True,
        )

    with views[2]:
        st.image(
            mask_to_image(
                analysis_mask
            ),
            caption="Hassas tarama analiz maskesi",
            use_container_width=True,
        )

    with views[3]:
        st.image(
            probability_to_image(
                result["probability"]
            ),
            caption="Kara-su ensemble olasılık haritası",
            use_container_width=True,
        )


def render_oil_result(
    result: dict[str, Any],
) -> None:
    screening_mask = np.asarray(
        result.get(
            "screening_mask",
            result["raw_mask"],
        ),
        dtype=np.uint8,
    )

    screening_pixels = int(
        screening_mask.sum()
    )

    screening_mode = str(
        result.get(
            "screening_mode",
            "ABSOLUTE_UNET_THRESHOLD",
        )
    )
    if result["oil_detected"]:
        st.markdown(
            '<div class="reject-box">'
            '🔴 PETROL ADAYI — UZMAN İNCELEMESİ GEREKLİ'
            '</div>',
            unsafe_allow_html=True,
        )
    elif screening_pixels > 0:
        st.markdown(
            '<div class="uncertain-box">'
            '🟡 ŞÜPHELİ BÖLGE BULUNDU — HASSAS TARAMA MASKESİNDE '
            'GÖSTERİLİYOR; KESİN PETROL KARARI DEĞİLDİR'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="accept-box">'
            '🟢 YÜKSEK GÜVENLİ PETROL ADAYI BULUNMADI<br><small>Bu sonuç görüntüde petrol bulunmadığını kanıtlamaz.</small>'
            '</div>',
            unsafe_allow_html=True,
        )

    columns = st.columns(4)
    columns[0].metric(
        "Hassas tarama pikseli",
        screening_pixels,
    )
    columns[1].metric(
        "Onaylı aday pikseli",
        result["final_positive_pixels"],
    )
    columns[2].metric(
        "Onaylı / toplam aday",
        f"{result['confirmed_candidate_count']}/{result['candidate_count']}",
    )
    columns[3].metric(
        "Belirsiz / look-alike",
        f"{result['uncertain_candidate_count']}/{result['lookalike_candidate_count']}",
    )

    tabs = st.tabs(
        [
            "Nihai araştırma sonucu",
            "Hassas tarama maskesi",
            "Aday kararları",
            "Dosyaları indir",
        ]
    )

    with tabs[0]:
        first, second = st.columns(2)
        with first:
            st.image(
                mask_to_image(result["final_mask"]),
                caption="Yalnız CONFIRMED_OIL adaylarından oluşan araştırma maskesi",
                use_container_width=True,
            )
        with second:
            st.image(
                result["overlay"],
                caption="SAR üzerinde uzman incelemesi gereken adaylar",
                use_container_width=True,
            )

    with tabs[1]:
        stage_columns = st.columns(3)
        with stage_columns[0]:
            st.image(
                probability_to_image(result["probability"]),
                caption="Kesin kara dışındaki alanda U-Net olasılık haritası",
                use_container_width=True,
            )
        with stage_columns[1]:
            st.image(
                mask_to_image(screening_mask),
                caption=(
                    "Final hassas şüpheli-bölge maskesi — "
                    f"mod: {screening_mode}"
                ),
                use_container_width=True,
            )
        with stage_columns[2]:
            st.image(
                result["candidate_overlay"],
                caption="CONFIRMED_OIL / UNCERTAIN / LOOK_ALIKE kararları",
                use_container_width=True,
            )

    with tabs[2]:
        frame = pd.DataFrame(result["candidate_results"])
        if frame.empty:
            st.info("Minimum alan koşulunu sağlayan aday bulunmadı.")
        else:
            preferred_columns = [
                "component_index",
                "decision",
                "area_pixels",
                "area_percent_total",
                "calibrated_probability",
                "lookalike_threshold",
                "confirmed_oil_threshold",
            ]
            available_columns = [
                column
                for column in preferred_columns
                if column in frame.columns
            ]
            st.dataframe(
                frame[available_columns],
                use_container_width=True,
                hide_index=True,
            )

    with tabs[3]:
        output_json = {
            "decision_policy": {
                "CONFIRMED_OIL": "research candidate; expert review required",
                "UNCERTAIN": "excluded from final candidate mask",
                "LOOK_ALIKE": "excluded from final candidate mask",
            },
            "water_gate": (
                st.session_state["water_result"]
                | {
                    "safe_water_mask": "binary mask omitted",
                    "internal_water_mask": "binary mask omitted",
                    "uncertain_mask": "binary mask omitted",
                    "analysis_water_mask": "binary mask omitted",
                    "probability": "array omitted",
                    "disagreement": "array omitted",
                }
            ),
            "oil_analysis": {
                key: value
                for key, value in result.items()
                if key not in {
                    "probability",
                    "raw_mask",
                    "final_mask",
                    "overlay",
                    "candidate_overlay",
                }
            },
        }

        download_columns = st.columns(3)
        with download_columns[0]:
            st.download_button(
                "Aday maskesini indir",
                data=pil_to_png_bytes(mask_to_image(result["final_mask"])),
                file_name=(
                    f"{st.session_state['filename_stem']}"
                    f"_confirmed_candidate_mask.png"
                ),
                mime="image/png",
                use_container_width=True,
            )
        with download_columns[1]:
            st.download_button(
                "Aday overlay'ini indir",
                data=pil_to_png_bytes(result["overlay"]),
                file_name=(
                    f"{st.session_state['filename_stem']}"
                    f"_candidate_overlay.png"
                ),
                mime="image/png",
                use_container_width=True,
            )
        with download_columns[2]:
            st.download_button(
                "JSON raporunu indir",
                data=json.dumps(
                    output_json,
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                ).encode("utf-8"),
                file_name=(
                    f"{st.session_state['filename_stem']}"
                    f"_result.json"
                ),
                mime="application/json",
                use_container_width=True,
            )

st.markdown(
    '<div class="main-title">'
    '🛰️ SAR Petrol Taraması — Final Hassas Mod'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Girdi doğrulama → kesin karayı dışlama → '
    'hassas petrol taraması → aday doğrulama'
    '</div>',
    unsafe_allow_html=True,
)


input_summary = load_json(
    INPUT_GATE_LOCKED_SUMMARY
)

water_summary = load_json(
    WATER_GATE_LOCKED_SUMMARY
)


with st.sidebar:
    st.header(
        "Final hassas tarama ayarları"
    )

    segmentation_threshold = st.slider(
        "Hassas petrol tarama eşiği",
        min_value=0.10,
        max_value=0.60,
        value=0.20,
        step=0.05,
        help=(
            "0.20 varsayılan hassas taramadır. Eşik düştükçe "
            "kaçırma azalabilir fakat yanlış alarm artabilir."
        ),
    )

    st.session_state[
        "final_scan_threshold"
    ] = float(
        segmentation_threshold
    )

    minimum_component_ratio = (
        st.number_input(
            "Minimum aday oranı",
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
        "Girdi doğrulama modeli"
    )

    if input_summary:
        st.write(
            "Negatif kilitli test:",
            input_summary.get(
                "total_negative_samples",
                0,
            ),
        )

        st.write(
            "Yanlış kabul:",
            input_summary.get(
                "false_accept_count",
                0,
            ),
        )

    st.divider()
    st.subheader(
        "Kara-su güvenlik modeli"
    )

    if water_summary:
        metrics = water_summary.get(
            "accepted_scene_metrics",
            {},
        )

        st.write(
            "Kilitli test kabul:",
            (
                f"{water_summary.get('accepted_count', 0)}"
                f"/{water_summary.get('locked_test_count', 0)}"
            ),
        )

        st.write(
            "Su precision:",
            (
                f"%{float(metrics.get('water_precision', 0)) * 100.0:.3f}"
            ),
        )

        st.write(
            "Kara sızıntısı:",
            (
                f"%{float(metrics.get('land_leakage', 0)) * 100.0:.4f}"
            ),
        )

    st.warning(
        "Araştırma prototipidir. Uzman doğrulaması "
        "olmadan operasyonel karar için kullanılmamalıdır."
    )


uploaded_file = st.file_uploader(
    "SAR görüntüsünü yükleyin",
    type=[
        "png",
        "jpg",
        "jpeg",
        "tif",
        "tiff",
    ],
)


if uploaded_file is None:
    st.info(
        "Başlamak için bir SAR görüntüsü yükleyin."
    )
    st.stop()


uploaded_bytes = uploaded_file.getvalue()
upload_hash = hashlib.sha256(
    uploaded_bytes
).hexdigest()


if st.session_state.get(
    "active_upload_hash"
) != upload_hash:
    for state_key in (
        "input_result",
        "water_result",
        "oil_result",
        "filename_stem",
        "analysed_hash",
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
        f"Görüntü okunamadı: {error}"
    )
    st.stop()


preview, info = st.columns(
    [2, 1]
)

with preview:
    st.image(
        uploaded_image,
        caption="Yüklenen gri-seviye SAR görüntüsü",
        use_container_width=True,
    )

with info:
    st.write(
        "**Dosya:**",
        uploaded_file.name,
    )

    st.write(
        "**Boyut:**",
        (
            f"{uploaded_image.width} × "
            f"{uploaded_image.height}"
        ),
    )

    st.write(
        "**Akış:**",
        (
            "Girdi doğrulama → kesin karayı dışla → seçilen eşikle petrol tara → aday doğrulama"
        ),
    )

    analyse_button = st.button(
        "Hassas taramayı başlat",
        type="primary",
        use_container_width=True,
    )


if analyse_button:
    for state_key in (
        "input_result",
        "water_result",
        "oil_result",
        "filename_stem",
        "analysed_hash",
    ):
        st.session_state.pop(
            state_key,
            None,
        )

    try:
        with st.spinner(
            "Girdi alanı doğrulanıyor..."
        ):
            input_gate = load_input_gate()
            input_result = run_input_gate(
                uploaded_image,
                input_gate,
            )

        st.session_state[
            "input_result"
        ] = input_result

        st.session_state[
            "analysed_hash"
        ] = upload_hash

        if input_result[
            "pipeline_allowed"
        ]:
            module = load_safe_module()

            with st.spinner(
                "Kara-su güvenlik kapısı çalıştırılıyor..."
            ):
                water_gate = (
                    load_water_gate_cached()
                )

                water_result = module[
                    "run_water_gate"
                ](
                    uploaded_image,
                    water_gate,
                )

            st.session_state[
                "water_result"
            ] = water_result

            analysis_water_mask = np.maximum(
                np.asarray(
                    water_result[
                        "internal_water_mask"
                    ],
                    dtype=np.uint8,
                ),
                np.asarray(
                    water_result[
                        "uncertain_mask"
                    ],
                    dtype=np.uint8,
                ),
            ).astype(
                np.uint8
            )

            water_result[
                "analysis_water_mask"
            ] = analysis_water_mask

            with st.spinner(
                "Hassas tarama modu: kesin kara dışındaki alanlar "
                f"{segmentation_threshold:.2f} eşikle taranıyor..."
            ):
                oil_pipeline = (
                    load_oil_pipeline_cached()
                )
                verifier_module = load_v08_module()
                oil_result = verifier_module[
                    "run_oil_pipeline_v07"
                ](
                    image=uploaded_image,
                    safe_water_mask=(
                        analysis_water_mask
                    ),
                    pipeline=oil_pipeline,
                    segmentation_threshold=float(segmentation_threshold),
                    minimum_component_ratio=float(
                        minimum_component_ratio
                    ),
                    minimum_component_pixels=int(
                        minimum_component_pixels
                    ),
                )


                screening_mask, screening_mode, screening_cutoff = (
                    build_sensitive_screening_mask(
                        image=uploaded_image,
                        probability=oil_result[
                            "probability"
                        ],
                        analysis_mask=(
                            analysis_water_mask
                        ),
                        absolute_mask=oil_result[
                            "raw_mask"
                        ],
                    )
                )

                oil_result[
                    "screening_mask"
                ] = screening_mask

                oil_result[
                    "screening_mode"
                ] = screening_mode

                oil_result[
                    "screening_cutoff"
                ] = float(
                    screening_cutoff
                )

                oil_result[
                    "scan_threshold"
                ] = float(
                    segmentation_threshold
                )
                oil_result[
                    "water_gate_original_decision"
                ] = water_result[
                    "decision"
                ]

            st.session_state[
                "oil_result"
            ] = oil_result

            st.session_state[
                "filename_stem"
            ] = Path(
                uploaded_file.name
            ).stem

    except Exception as error:
        st.exception(error)


if st.session_state.get(
    "analysed_hash"
) != upload_hash:
    st.stop()


input_result = st.session_state.get(
    "input_result"
)

if input_result is None:
    st.stop()


st.divider()
st.subheader(
    "1. Giriş doğrulama"
)

render_input_gate(
    input_result
)


if not input_result[
    "pipeline_allowed"
]:
    st.stop()


water_result = st.session_state.get(
    "water_result"
)

if water_result is None:
    st.error(
        "Giriş kabul edildi fakat kara-su sonucu "
        "oluşturulamadı."
    )
    st.stop()


st.divider()
st.subheader(
    "2. Kara-su güvenliği"
)

render_water_gate(
    water_result
)


if not water_result[
    "pipeline_allowed"
]:
    st.markdown(
        """
        <div class="scientific-note">
        <b>Hassas tarama modu:</b> Su kapısı belirsiz olsa da
        petrol analizi çalıştırıldı. Kesin kara dışındaki güvenli
        ve belirsiz alanlar tarandı. Bu yaklaşım kaçırmayı azaltmayı
        hedefler fakat yanlış alarmı artırabilir.
        </div>
        """,
        unsafe_allow_html=True,
    )


oil_result = st.session_state.get(
    "oil_result"
)
oil_result = st.session_state.get(
    "oil_result"
)

if oil_result is None:
    st.error(
        "Su kapısı geçti fakat petrol sonucu "
        "oluşturulamadı."
    )
    st.stop()


st.divider()
st.subheader(
    "3. Final hassas petrol taraması ve aday doğrulama"
)

render_oil_result(
    oil_result
)


st.markdown(
    """
    <div class="scientific-note">
    <b>Bilimsel not:</b>
    Final hassas tarama kesin kara dışındaki güvenli ve
    belirsiz alanları seçilen eşikle inceler. Mutlak U-Net
    maskesi boş kalırsa göreli model yanıtı ve yerel koyuluk
    yalnız şüpheli-bölge taraması için kullanılır. Ham tarama maskesi
    şüpheli bölgeleri görünür tutar ve yanlış pozitif içerebilir.
    Yalnız CONFIRMED_OIL kararı alan bölgeler onaylı maskeye
    eklenir. Her iki çıktı da uzman incelemesi gerektirir.
    </div>
    """,
    unsafe_allow_html=True,
)



