from __future__ import annotations

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
from PIL import Image


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


def load_model_card() -> dict[str, Any]:
    if not MODEL_CARD.exists():
        return {}

    with MODEL_CARD.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


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
    'Robust Binary Oil Detector v0.3 — '
    'Petrol / petrol değil segmentasyonu'
    '</div>',
    unsafe_allow_html=True,
)


model_card = load_model_card()

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
        "Final kilitli test"
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
    "SAR görüntüsünü yükleyin",
    type=[
        "png",
        "jpg",
        "jpeg",
        "tif",
        "tiff",
    ],
    help=(
        "Gri seviye veya RGB görüntü yüklenebilir. "
        "Görüntü otomatik olarak gri seviyeye çevrilir."
    ),
)


if uploaded_file is None:
    st.info(
        "Analizi başlatmak için bir SAR "
        "görüntüsü yükleyin."
    )

    st.stop()


uploaded_bytes = (
    uploaded_file.getvalue()
)

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
        caption="Yüklenen SAR görüntüsü",
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
        "**Mod:**",
        "Gri seviye",
    )

    analyse_button = st.button(
        "Petrol analizini başlat",
        type="primary",
        use_container_width=True,
    )


if analyse_button:
    try:
        with st.spinner(
            "Modeller yükleniyor ve görüntü analiz ediliyor..."
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


if "oil_result" not in st.session_state:
    st.stop()


result = st.session_state[
    "oil_result"
]

filename_stem = st.session_state[
    "oil_filename"
]


st.divider()

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
