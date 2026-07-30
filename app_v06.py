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


st.set_page_config(
    page_title="SAR Petrol Tespit Sistemi v0.6",
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
def load_oil_pipeline_cached() -> dict[str, Any]:
    module = load_safe_module()
    return module[
        "load_oil_pipeline"
    ](
        "auto"
    )


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
    if result[
        "pipeline_allowed"
    ]:
        st.markdown(
            '<div class="accept-box">'
            '✅ Kara-su güvenlik kapısı geçti. '
            'Petrol analizi yalnız güvenli su '
            'piksellerinde çalıştırıldı.'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="uncertain-box">'
            '⚠️ Kara-su ayrımı yeterince güvenilir değil. '
            'Petrol analizi engellendi ve petrol maskesi '
            'üretilmedi.'
            '</div>',
            unsafe_allow_html=True,
        )

    columns = st.columns(3)

    columns[0].metric(
        "Su kararı",
        result["decision"],
    )

    columns[1].metric(
        "Güvenli su oranı",
        (
            f"%{result['predicted_water_fraction'] * 100.0:.2f}"
        ),
    )

    columns[2].metric(
        "Belirsiz alan",
        (
            f"%{result['uncertain_fraction'] * 100.0:.2f}"
        ),
    )

    views = st.columns(3)

    with views[0]:
        st.image(
            mask_to_image(
                result[
                    "safe_water_mask"
                ]
            ),
            caption=(
                "Operasyonel güvenli su maskesi"
            ),
            use_container_width=True,
        )

    with views[1]:
        st.image(
            mask_to_image(
                result[
                    "uncertain_mask"
                ]
            ),
            caption=(
                "Belirsiz ve petrol analizine kapalı alan"
            ),
            use_container_width=True,
        )

    with views[2]:
        st.image(
            probability_to_image(
                result["probability"]
            ),
            caption=(
                "Kara-su ensemble olasılık haritası"
            ),
            use_container_width=True,
        )


def render_oil_result(
    result: dict[str, Any],
) -> None:
    if result[
        "oil_detected"
    ]:
        st.markdown(
            '<div class="reject-box">'
            '🔴 PETROL ADAYI TESPİT EDİLDİ'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="accept-box">'
            '🟢 PETROL TESPİT EDİLMEDİ'
            '</div>',
            unsafe_allow_html=True,
        )

    columns = st.columns(4)

    columns[0].metric(
        "Su içindeki petrol kaplaması",
        (
            f"%{result['final_oil_percent_of_water']:.4f}"
        ),
    )

    columns[1].metric(
        "Nihai petrol pikseli",
        result[
            "final_positive_pixels"
        ],
    )

    columns[2].metric(
        "Kabul edilen aday",
        (
            f"{result['accepted_candidate_count']}"
            f"/{result['candidate_count']}"
        ),
    )

    columns[3].metric(
        "Güvenli su pikseli",
        result[
            "safe_water_pixels"
        ],
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
        first, second = st.columns(2)

        with first:
            st.image(
                mask_to_image(
                    result[
                        "final_mask"
                    ]
                ),
                caption=(
                    "Nihai petrol maskesi"
                ),
                use_container_width=True,
            )

        with second:
            st.image(
                result["overlay"],
                caption=(
                    "SAR üzerinde nihai petrol maskesi"
                ),
                use_container_width=True,
            )

    with tabs[1]:
        stage_columns = st.columns(3)

        with stage_columns[0]:
            st.image(
                probability_to_image(
                    result[
                        "probability"
                    ]
                ),
                caption=(
                    "Su dışı sıfırlanmış petrol olasılığı"
                ),
                use_container_width=True,
            )

        with stage_columns[1]:
            st.image(
                mask_to_image(
                    result[
                        "raw_mask"
                    ]
                ),
                caption=(
                    "Ham petrol aday maskesi"
                ),
                use_container_width=True,
            )

        with stage_columns[2]:
            st.image(
                result[
                    "candidate_overlay"
                ],
                caption=(
                    "Aday bileşen ve verifier kararları"
                ),
                use_container_width=True,
            )

    with tabs[2]:
        frame = pd.DataFrame(
            result[
                "candidate_results"
            ]
        )

        if frame.empty:
            st.info(
                "Minimum alan koşulunu sağlayan "
                "petrol adayı bulunmadı."
            )
        else:
            columns_to_show = [
                "component_index",
                "decision",
                "area_pixels",
                "area_percent_total",
                "selected_probability",
                "selected_threshold",
            ]

            st.dataframe(
                frame[
                    columns_to_show
                ],
                use_container_width=True,
                hide_index=True,
            )

    with tabs[3]:
        output_json = {
            "water_gate": (
                st.session_state[
                    "water_result"
                ]
                | {
                    "safe_water_mask": (
                        "binary mask omitted"
                    ),
                    "internal_water_mask": (
                        "binary mask omitted"
                    ),
                    "uncertain_mask": (
                        "binary mask omitted"
                    ),
                    "probability": (
                        "array omitted"
                    ),
                    "disagreement": (
                        "array omitted"
                    ),
                }
            ),
            "oil_analysis": {
                key: value
                for key, value
                in result.items()
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
                "Nihai maskeyi indir",
                data=pil_to_png_bytes(
                    mask_to_image(
                        result[
                            "final_mask"
                        ]
                    )
                ),
                file_name=(
                    f"{st.session_state['filename_stem']}"
                    f"_oil_mask.png"
                ),
                mime="image/png",
                use_container_width=True,
            )

        with download_columns[1]:
            st.download_button(
                "Overlay'i indir",
                data=pil_to_png_bytes(
                    result["overlay"]
                ),
                file_name=(
                    f"{st.session_state['filename_stem']}"
                    f"_oil_overlay.png"
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
    '🛰️ SAR Petrol Sızıntısı Tespit Sistemi'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'v0.5 giriş doğrulama → v0.6 seçici kara-su '
    'güvenliği → v0.3 suyla sınırlandırılmış petrol analizi'
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
        "Dondurulmuş model ayarları"
    )

    segmentation_threshold = st.slider(
        "Petrol U-Net eşiği",
        min_value=0.10,
        max_value=0.95,
        value=0.60,
        step=0.05,
        disabled=True,
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
        "v0.5 giriş kapısı"
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
        "v0.6 kara-su kapısı"
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
            "Giriş kapısı → kara-su kapısı → "
            "yalnız güvenli suda petrol analizi"
        ),
    )

    analyse_button = st.button(
        "Güvenli analizi başlat",
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

            if water_result[
                "pipeline_allowed"
            ]:
                with st.spinner(
                    "Petrol modeli yalnız güvenli "
                    "su piksellerinde çalıştırılıyor..."
                ):
                    oil_pipeline = (
                        load_oil_pipeline_cached()
                    )

                    oil_result = module[
                        "run_oil_pipeline"
                    ](
                        image=uploaded_image,
                        safe_water_mask=(
                            water_result[
                                "safe_water_mask"
                            ]
                        ),
                        pipeline=oil_pipeline,
                        segmentation_threshold=float(
                            segmentation_threshold
                        ),
                        minimum_component_ratio=float(
                            minimum_component_ratio
                        ),
                        minimum_component_pixels=int(
                            minimum_component_pixels
                        ),
                    )

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
        Petrol modeli bilinçli olarak çalıştırılmadı.
        Bu sahnede operasyonel petrol maskesi tamamen
        siyah kabul edilir.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


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
    "3. Güvenli suyla sınırlandırılmış petrol analizi"
)

render_oil_result(
    oil_result
)


st.markdown(
    """
    <div class="scientific-note">
    <b>Bilimsel not:</b>
    Petrol adayları yalnız seçici kara-su kapısının
    kabul ettiği yüksek güvenli su piksellerinde
    üretilebilir. Kara ve belirsiz alanlar nihai petrol
    maskesine giremez. Bu sistem geniş ölçekli operasyonel
    geçerlilik iddiası olmayan bir araştırma prototipidir.
    </div>
    """,
    unsafe_allow_html=True,
)
