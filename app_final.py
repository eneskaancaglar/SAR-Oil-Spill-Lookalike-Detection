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


PAGE_ICON_PATH = ROOT / "SAR_Oil_Spill_Scanner.ico"

st.set_page_config(
    page_title="SAR Oil Spill Scanner",
    page_icon=(
        str(PAGE_ICON_PATH)
        if PAGE_ICON_PATH.exists()
        else "🛰️"
    ),
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


TRANSLATIONS: dict[str, dict[str, str]] = {
    "en": {
        "language_label": "Language / Dil",
        "app_title": "🛰️ SAR Oil Spill Scanner — Final Sensitive Mode",
        "app_subtitle": (
            "Input validation → definite-land exclusion → "
            "sensitive oil screening → candidate verification"
        ),
        "sidebar_header": "Final sensitive screening settings",
        "scan_threshold": "Sensitive oil-screening threshold",
        "scan_threshold_help": (
            "0.20 is the default sensitive-screening threshold. "
            "Lower values may reduce missed detections but can increase false alarms."
        ),
        "minimum_candidate_ratio": "Minimum candidate ratio",
        "minimum_candidate_pixels": "Minimum candidate pixels",
        "input_gate_model": "Input-validation model",
        "locked_negative_test": "Locked negative test:",
        "false_accepts": "False accepts:",
        "water_gate_model": "Land-water safety model",
        "locked_test_acceptance": "Locked-test acceptance:",
        "water_precision": "Water precision:",
        "land_leakage": "Land leakage:",
        "research_warning": (
            "This is a research prototype. It must not be used for operational "
            "decisions without expert validation."
        ),
        "upload_label": "Upload a SAR image",
        "upload_prompt": "Upload a SAR image to begin.",
        "image_read_error": "The image could not be read: {error}",
        "uploaded_image_caption": "Uploaded grayscale SAR image",
        "file_label": "**File:**",
        "dimensions_label": "**Dimensions:**",
        "workflow_label": "**Workflow:**",
        "workflow_text": (
            "Validate input → exclude definite land → screen for oil at the "
            "selected threshold → verify candidates"
        ),
        "start_scan": "Start sensitive screening",
        "spinner_input": "Validating the input domain...",
        "spinner_water": "Running the land-water safety gate...",
        "spinner_oil": (
            "Sensitive screening mode: scanning non-definite-land areas "
            "at threshold {threshold:.2f}..."
        ),
        "section_input": "1. Input validation",
        "section_water": "2. Land-water safety",
        "section_oil": "3. Final sensitive oil screening and candidate verification",
        "water_result_missing": (
            "The input was accepted, but the land-water result could not be generated."
        ),
        "oil_result_missing": (
            "The water stage completed, but the oil-analysis result could not be generated."
        ),
        "sensitive_mode_note": (
            "<div class=\"scientific-note\">"
            "<b>Sensitive screening mode:</b> Oil analysis was performed even though "
            "the water gate was uncertain. Safe and uncertain areas outside definite "
            "land were screened. This approach aims to reduce missed detections but "
            "may increase false alarms."
            "</div>"
        ),
        "scientific_note": (
            "<div class=\"scientific-note\">"
            "<b>Scientific note:</b> Final sensitive screening examines safe and "
            "uncertain areas outside definite land using the selected threshold. "
            "When the absolute U-Net mask is empty, the relative model response and "
            "local darkness are used only for suspicious-region screening. The raw "
            "screening mask keeps suspicious regions visible and may contain false "
            "positives. Only regions classified as CONFIRMED_OIL are added to the "
            "confirmed mask. Both outputs require expert review."
            "</div>"
        ),
        "required_file_missing": "Required file not found: {path}",
        "input_gate_config_error": "The input-gate configuration is empty or unreadable.",
        "operational_gate_error": "The v0.8 operational gate did not pass.",
        "input_accept": (
            "<div class=\"accept-box\">"
            "✅ The input was accepted as a supported marine/coastal SAR scene."
            "</div>"
        ),
        "input_reject": (
            "<div class=\"reject-box\">"
            "⛔ The input is outside the supported SAR domain. "
            "The downstream models were not run."
            "</div>"
        ),
        "input_uncertain": (
            "<div class=\"uncertain-box\">"
            "⚠️ The input could not be validated safely. "
            "The downstream models were not run."
            "</div>"
        ),
        "decision": "Decision",
        "supported_input_score": "Supported-input score",
        "dartis_similarity": "DARTIS similarity",
        "decision_ACCEPT": "ACCEPTED",
        "decision_REJECT": "REJECTED",
        "decision_UNCERTAIN": "UNCERTAIN",
        "water_accept": (
            "<div class=\"accept-box\">"
            "✅ The land-water model accepted a safe water area. In sensitive "
            "screening mode, safe and uncertain regions outside definite land "
            "are being screened."
            "</div>"
        ),
        "water_uncertain": (
            "<div class=\"uncertain-box\">"
            "⚠️ The land-water separation is uncertain. Oil analysis was not "
            "stopped; areas outside definite land are being screened in "
            "sensitive mode."
            "</div>"
        ),
        "water_decision": "Water decision",
        "safe_water_fraction": "Safe-water fraction",
        "uncertain_area": "Uncertain area",
        "screened_area": "Screened area",
        "safe_water_caption": "High-confidence water",
        "uncertain_area_caption": "Uncertain area included in screening",
        "analysis_mask_caption": "Sensitive-screening analysis mask",
        "water_probability_caption": "Land-water ensemble probability map",
        "oil_candidate": (
            "<div class=\"reject-box\">"
            "🔴 OIL CANDIDATE — EXPERT REVIEW REQUIRED"
            "</div>"
        ),
        "suspicious_region": (
            "<div class=\"uncertain-box\">"
            "🟡 SUSPICIOUS REGION FOUND — DISPLAYED IN THE SENSITIVE-SCREENING "
            "MASK; THIS IS NOT A CONFIRMED OIL DECISION"
            "</div>"
        ),
        "no_high_confidence_candidate": (
            "<div class=\"accept-box\">"
            "🟢 NO HIGH-CONFIDENCE OIL CANDIDATE FOUND"
            "<br><small>This result does not prove that the image contains no oil.</small>"
            "</div>"
        ),
        "screening_pixels": "Sensitive-screening pixels",
        "confirmed_pixels": "Confirmed-candidate pixels",
        "confirmed_total": "Confirmed / total candidates",
        "uncertain_lookalike": "Uncertain / look-alike",
        "tab_final_result": "Final research result",
        "tab_screening_mask": "Sensitive-screening mask",
        "tab_candidate_decisions": "Candidate decisions",
        "tab_downloads": "Download files",
        "confirmed_mask_caption": (
            "Research mask containing only CONFIRMED_OIL candidates"
        ),
        "expert_overlay_caption": "Candidates requiring expert review on the SAR image",
        "unet_probability_caption": (
            "U-Net probability map outside definite land"
        ),
        "screening_mask_caption": (
            "Final sensitive suspicious-region mask — mode: {mode}"
        ),
        "candidate_overlay_caption": (
            "CONFIRMED_OIL / UNCERTAIN / LOOK_ALIKE decisions"
        ),
        "no_candidates": "No candidate met the minimum-area requirement.",
        "download_mask": "Download candidate mask",
        "download_overlay": "Download candidate overlay",
        "download_json": "Download JSON report",
        "policy_confirmed": "research candidate; expert review required",
        "policy_uncertain": "excluded from the final candidate mask",
        "policy_lookalike": "excluded from the final candidate mask",
        "mode_ABSOLUTE_UNET_THRESHOLD": "absolute U-Net threshold",
        "mode_RELATIVE_TOP_0_5_PERCENT": "relative top 0.5%",
        "mode_NO_VALID_AREA": "no valid analysis area",
        "mode_NO_MODEL_RESPONSE": "no model response",
        "candidate_CONFIRMED_OIL": "Confirmed oil",
        "candidate_UNCERTAIN": "Uncertain",
        "candidate_LOOK_ALIKE": "Look-alike",
        "column_component_index": "Component",
        "column_decision": "Decision",
        "column_area_pixels": "Area (pixels)",
        "column_area_percent_total": "Area (%)",
        "column_calibrated_probability": "Calibrated probability",
        "column_lookalike_threshold": "Look-alike threshold",
        "column_confirmed_oil_threshold": "Confirmed-oil threshold",
    },
    "tr": {
        "language_label": "Dil / Language",
        "app_title": "🛰️ SAR Petrol Sızıntısı Tarayıcısı — Final Hassas Mod",
        "app_subtitle": (
            "Girdi doğrulama → kesin karayı dışlama → "
            "hassas petrol taraması → aday doğrulama"
        ),
        "sidebar_header": "Final hassas tarama ayarları",
        "scan_threshold": "Hassas petrol tarama eşiği",
        "scan_threshold_help": (
            "0.20 varsayılan hassas tarama eşiğidir. Eşik düştükçe "
            "kaçırma azalabilir ancak yanlış alarm artabilir."
        ),
        "minimum_candidate_ratio": "Minimum aday oranı",
        "minimum_candidate_pixels": "Minimum aday pikseli",
        "input_gate_model": "Girdi doğrulama modeli",
        "locked_negative_test": "Negatif kilitli test:",
        "false_accepts": "Yanlış kabul:",
        "water_gate_model": "Kara-su güvenlik modeli",
        "locked_test_acceptance": "Kilitli test kabulü:",
        "water_precision": "Su hassasiyeti:",
        "land_leakage": "Kara sızıntısı:",
        "research_warning": (
            "Bu bir araştırma prototipidir. Uzman doğrulaması olmadan "
            "operasyonel kararlar için kullanılmamalıdır."
        ),
        "upload_label": "SAR görüntüsü yükleyin",
        "upload_prompt": "Başlamak için bir SAR görüntüsü yükleyin.",
        "image_read_error": "Görüntü okunamadı: {error}",
        "uploaded_image_caption": "Yüklenen gri seviye SAR görüntüsü",
        "file_label": "**Dosya:**",
        "dimensions_label": "**Boyut:**",
        "workflow_label": "**İşlem akışı:**",
        "workflow_text": (
            "Girdiyi doğrula → kesin karayı dışla → seçilen eşikle petrol tara "
            "→ adayları doğrula"
        ),
        "start_scan": "Hassas taramayı başlat",
        "spinner_input": "Girdi alanı doğrulanıyor...",
        "spinner_water": "Kara-su güvenlik kapısı çalıştırılıyor...",
        "spinner_oil": (
            "Hassas tarama modu: kesin kara dışındaki alanlar "
            "{threshold:.2f} eşiğiyle taranıyor..."
        ),
        "section_input": "1. Girdi doğrulama",
        "section_water": "2. Kara-su güvenliği",
        "section_oil": "3. Final hassas petrol taraması ve aday doğrulama",
        "water_result_missing": (
            "Girdi kabul edildi ancak kara-su sonucu oluşturulamadı."
        ),
        "oil_result_missing": (
            "Su aşaması tamamlandı ancak petrol analizi sonucu oluşturulamadı."
        ),
        "sensitive_mode_note": (
            "<div class=\"scientific-note\">"
            "<b>Hassas tarama modu:</b> Su kapısı belirsiz olsa da petrol "
            "analizi çalıştırıldı. Kesin kara dışındaki güvenli ve belirsiz "
            "alanlar tarandı. Bu yaklaşım kaçırmayı azaltmayı hedefler ancak "
            "yanlış alarmı artırabilir."
            "</div>"
        ),
        "scientific_note": (
            "<div class=\"scientific-note\">"
            "<b>Bilimsel not:</b> Final hassas tarama, kesin kara dışındaki "
            "güvenli ve belirsiz alanları seçilen eşikle inceler. Mutlak U-Net "
            "maskesi boş kalırsa göreli model yanıtı ve yerel koyuluk yalnızca "
            "şüpheli bölge taraması için kullanılır. Ham tarama maskesi şüpheli "
            "bölgeleri görünür tutar ve yanlış pozitif içerebilir. Yalnızca "
            "CONFIRMED_OIL olarak sınıflandırılan bölgeler onaylı maskeye "
            "eklenir. Her iki çıktı da uzman incelemesi gerektirir."
            "</div>"
        ),
        "required_file_missing": "Gerekli dosya bulunamadı: {path}",
        "input_gate_config_error": "Girdi kapısı yapılandırması boş veya okunamadı.",
        "operational_gate_error": "v0.8 operasyonel kapısı başarılı değil.",
        "input_accept": (
            "<div class=\"accept-box\">"
            "✅ Girdi desteklenen deniz/kıyı SAR alanına kabul edildi."
            "</div>"
        ),
        "input_reject": (
            "<div class=\"reject-box\">"
            "⛔ Girdi desteklenen SAR alanına uygun değil. "
            "Sonraki modeller çalıştırılmadı."
            "</div>"
        ),
        "input_uncertain": (
            "<div class=\"uncertain-box\">"
            "⚠️ Girdi güvenli biçimde doğrulanamadı. "
            "Sonraki modeller çalıştırılmadı."
            "</div>"
        ),
        "decision": "Karar",
        "supported_input_score": "Desteklenen girdi skoru",
        "dartis_similarity": "DARTIS benzerliği",
        "decision_ACCEPT": "KABUL EDİLDİ",
        "decision_REJECT": "REDDEDİLDİ",
        "decision_UNCERTAIN": "BELİRSİZ",
        "water_accept": (
            "<div class=\"accept-box\">"
            "✅ Kara-su modeli güvenli su alanını kabul etti. Hassas tarama "
            "modunda kesin kara dışındaki güvenli ve belirsiz alanlar taranıyor."
            "</div>"
        ),
        "water_uncertain": (
            "<div class=\"uncertain-box\">"
            "⚠️ Kara-su ayrımı belirsiz. Petrol analizi durdurulmadı; "
            "kesin kara dışındaki alanlar hassas modda taranıyor."
            "</div>"
        ),
        "water_decision": "Su kararı",
        "safe_water_fraction": "Güvenli su oranı",
        "uncertain_area": "Belirsiz alan",
        "screened_area": "Taranan alan",
        "safe_water_caption": "Yüksek güvenli su",
        "uncertain_area_caption": "Taramaya dahil edilen belirsiz alan",
        "analysis_mask_caption": "Hassas tarama analiz maskesi",
        "water_probability_caption": "Kara-su ensemble olasılık haritası",
        "oil_candidate": (
            "<div class=\"reject-box\">"
            "🔴 PETROL ADAYI — UZMAN İNCELEMESİ GEREKLİ"
            "</div>"
        ),
        "suspicious_region": (
            "<div class=\"uncertain-box\">"
            "🟡 ŞÜPHELİ BÖLGE BULUNDU — HASSAS TARAMA MASKESİNDE "
            "GÖSTERİLİYOR; BU KESİN PETROL KARARI DEĞİLDİR"
            "</div>"
        ),
        "no_high_confidence_candidate": (
            "<div class=\"accept-box\">"
            "🟢 YÜKSEK GÜVENLİ PETROL ADAYI BULUNMADI"
            "<br><small>Bu sonuç görüntüde petrol bulunmadığını kanıtlamaz.</small>"
            "</div>"
        ),
        "screening_pixels": "Hassas tarama pikselleri",
        "confirmed_pixels": "Onaylı aday pikselleri",
        "confirmed_total": "Onaylı / toplam aday",
        "uncertain_lookalike": "Belirsiz / benzer yapı",
        "tab_final_result": "Nihai araştırma sonucu",
        "tab_screening_mask": "Hassas tarama maskesi",
        "tab_candidate_decisions": "Aday kararları",
        "tab_downloads": "Dosyaları indir",
        "confirmed_mask_caption": (
            "Yalnızca CONFIRMED_OIL adaylarından oluşan araştırma maskesi"
        ),
        "expert_overlay_caption": "SAR görüntüsü üzerinde uzman incelemesi gereken adaylar",
        "unet_probability_caption": (
            "Kesin kara dışındaki alanda U-Net olasılık haritası"
        ),
        "screening_mask_caption": (
            "Final hassas şüpheli bölge maskesi — mod: {mode}"
        ),
        "candidate_overlay_caption": (
            "CONFIRMED_OIL / UNCERTAIN / LOOK_ALIKE kararları"
        ),
        "no_candidates": "Minimum alan koşulunu sağlayan aday bulunamadı.",
        "download_mask": "Aday maskesini indir",
        "download_overlay": "Aday katmanını indir",
        "download_json": "JSON raporunu indir",
        "policy_confirmed": "araştırma adayı; uzman incelemesi gerekli",
        "policy_uncertain": "nihai aday maskesinden çıkarıldı",
        "policy_lookalike": "nihai aday maskesinden çıkarıldı",
        "mode_ABSOLUTE_UNET_THRESHOLD": "mutlak U-Net eşiği",
        "mode_RELATIVE_TOP_0_5_PERCENT": "göreli en yüksek %0,5",
        "mode_NO_VALID_AREA": "geçerli analiz alanı yok",
        "mode_NO_MODEL_RESPONSE": "model yanıtı yok",
        "candidate_CONFIRMED_OIL": "Onaylı petrol",
        "candidate_UNCERTAIN": "Belirsiz",
        "candidate_LOOK_ALIKE": "Benzer yapı",
        "column_component_index": "Bileşen",
        "column_decision": "Karar",
        "column_area_pixels": "Alan (piksel)",
        "column_area_percent_total": "Alan (%)",
        "column_calibrated_probability": "Kalibre edilmiş olasılık",
        "column_lookalike_threshold": "Benzer yapı eşiği",
        "column_confirmed_oil_threshold": "Onaylı petrol eşiği",
    },
}


if "app_language" not in st.session_state:
    st.session_state["app_language"] = "English"


def current_language_code() -> str:
    return (
        "tr"
        if st.session_state.get("app_language") == "Türkçe"
        else "en"
    )


def t(key: str, **values: Any) -> str:
    template = TRANSLATIONS[current_language_code()][key]
    return template.format(**values)


def localize_decision(value: Any) -> str:
    key = f"decision_{str(value)}"
    return TRANSLATIONS[current_language_code()].get(
        key,
        str(value),
    )


def localize_candidate_decision(value: Any) -> str:
    key = f"candidate_{str(value)}"
    return TRANSLATIONS[current_language_code()].get(
        key,
        str(value),
    )


def localize_screening_mode(value: Any) -> str:
    key = f"mode_{str(value)}"
    return TRANSLATIONS[current_language_code()].get(
        key,
        str(value),
    )


def localize_candidate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    localized = frame.copy()

    if "decision" in localized.columns:
        localized["decision"] = localized["decision"].map(
            localize_candidate_decision
        )

    rename_map = {
        "component_index": t("column_component_index"),
        "decision": t("column_decision"),
        "area_pixels": t("column_area_pixels"),
        "area_percent_total": t("column_area_percent_total"),
        "calibrated_probability": t("column_calibrated_probability"),
        "lookalike_threshold": t("column_lookalike_threshold"),
        "confirmed_oil_threshold": t("column_confirmed_oil_threshold"),
    }

    return localized.rename(columns=rename_map)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            t("required_file_missing", path=path)
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


def prepare_image_for_models(
    image: Image.Image,
    multiple: int = 32,
    maximum_side: int = 1024,
) -> Image.Image:
    """
    Resize large images while preserving aspect ratio, then pad the
    right and bottom edges so both dimensions are multiples of 32.
    """
    grayscale = image.convert("L")

    width, height = grayscale.size
    longest_side = max(
        width,
        height,
    )

    if longest_side > maximum_side:
        scale = (
            maximum_side
            / float(longest_side)
        )

        resized_width = max(
            32,
            int(round(width * scale)),
        )

        resized_height = max(
            32,
            int(round(height * scale)),
        )

        grayscale = grayscale.resize(
            (
                resized_width,
                resized_height,
            ),
            Image.Resampling.BILINEAR,
        )

    image_array = np.asarray(
        grayscale,
        dtype=np.uint8,
    )

    height, width = image_array.shape

    target_height = (
        (height + multiple - 1)
        // multiple
        * multiple
    )

    target_width = (
        (width + multiple - 1)
        // multiple
        * multiple
    )

    pad_bottom = (
        target_height - height
    )

    pad_right = (
        target_width - width
    )

    if (
        pad_bottom > 0
        or pad_right > 0
    ):
        image_array = np.pad(
            image_array,
            (
                (0, pad_bottom),
                (0, pad_right),
            ),
            mode="edge",
        )

    return Image.fromarray(
        image_array,
        mode="L",
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
            t("input_gate_config_error")
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
        raise RuntimeError(t("operational_gate_error"))

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
    decision = result["decision"]

    if decision == "ACCEPT":
        message = t("input_accept")
    elif decision == "REJECT":
        message = t("input_reject")
    else:
        message = t("input_uncertain")

    st.markdown(
        message,
        unsafe_allow_html=True,
    )

    columns = st.columns(3)

    columns[0].metric(
        t("decision"),
        localize_decision(decision),
    )

    columns[1].metric(
        t("supported_input_score"),
        f"{result['supported_probability'] * 100.0:.3f}%",
    )

    columns[2].metric(
        t("dartis_similarity"),
        f"{result['prototype_similarity']:.4f}",
    )


def render_water_gate(
    result: dict[str, Any],
) -> None:
    st.markdown(
        (
            t("water_accept")
            if result["pipeline_allowed"]
            else t("water_uncertain")
        ),
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
        t("water_decision"),
        localize_decision(result["decision"]),
    )
    columns[1].metric(
        t("safe_water_fraction"),
        f"{result['predicted_water_fraction'] * 100.0:.2f}%",
    )
    columns[2].metric(
        t("uncertain_area"),
        f"{result['uncertain_fraction'] * 100.0:.2f}%",
    )
    columns[3].metric(
        t("screened_area"),
        f"{float(np.asarray(analysis_mask).mean()) * 100.0:.2f}%",
    )

    views = st.columns(4)

    with views[0]:
        st.image(
            mask_to_image(
                result["internal_water_mask"]
            ),
            caption=t("safe_water_caption"),
            use_container_width=True,
        )

    with views[1]:
        st.image(
            mask_to_image(
                result["uncertain_mask"]
            ),
            caption=t("uncertain_area_caption"),
            use_container_width=True,
        )

    with views[2]:
        st.image(
            mask_to_image(
                analysis_mask
            ),
            caption=t("analysis_mask_caption"),
            use_container_width=True,
        )

    with views[3]:
        st.image(
            probability_to_image(
                result["probability"]
            ),
            caption=t("water_probability_caption"),
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
        status_message = t("oil_candidate")
    elif screening_pixels > 0:
        status_message = t("suspicious_region")
    else:
        status_message = t("no_high_confidence_candidate")

    st.markdown(
        status_message,
        unsafe_allow_html=True,
    )

    columns = st.columns(4)
    columns[0].metric(
        t("screening_pixels"),
        screening_pixels,
    )
    columns[1].metric(
        t("confirmed_pixels"),
        result["final_positive_pixels"],
    )
    columns[2].metric(
        t("confirmed_total"),
        (
            f"{result['confirmed_candidate_count']}"
            f"/{result['candidate_count']}"
        ),
    )
    columns[3].metric(
        t("uncertain_lookalike"),
        (
            f"{result['uncertain_candidate_count']}"
            f"/{result['lookalike_candidate_count']}"
        ),
    )

    tabs = st.tabs(
        [
            t("tab_final_result"),
            t("tab_screening_mask"),
            t("tab_candidate_decisions"),
            t("tab_downloads"),
        ]
    )

    with tabs[0]:
        first, second = st.columns(2)
        with first:
            st.image(
                mask_to_image(result["final_mask"]),
                caption=t("confirmed_mask_caption"),
                use_container_width=True,
            )
        with second:
            st.image(
                result["overlay"],
                caption=t("expert_overlay_caption"),
                use_container_width=True,
            )

    with tabs[1]:
        stage_columns = st.columns(3)
        with stage_columns[0]:
            st.image(
                probability_to_image(result["probability"]),
                caption=t("unet_probability_caption"),
                use_container_width=True,
            )
        with stage_columns[1]:
            st.image(
                mask_to_image(screening_mask),
                caption=t(
                    "screening_mask_caption",
                    mode=localize_screening_mode(
                        screening_mode
                    ),
                ),
                use_container_width=True,
            )
        with stage_columns[2]:
            st.image(
                result["candidate_overlay"],
                caption=t("candidate_overlay_caption"),
                use_container_width=True,
            )

    with tabs[2]:
        frame = pd.DataFrame(
            result["candidate_results"]
        )
        if frame.empty:
            st.info(t("no_candidates"))
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
            display_frame = localize_candidate_frame(
                frame[available_columns]
            )
            st.dataframe(
                display_frame,
                use_container_width=True,
                hide_index=True,
            )

    with tabs[3]:
        output_json = {
            "interface_language": current_language_code(),
            "decision_policy": {
                "CONFIRMED_OIL": t("policy_confirmed"),
                "UNCERTAIN": t("policy_uncertain"),
                "LOOK_ALIKE": t("policy_lookalike"),
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
                t("download_mask"),
                data=pil_to_png_bytes(
                    mask_to_image(
                        result["final_mask"]
                    )
                ),
                file_name=(
                    f"{st.session_state['filename_stem']}"
                    f"_confirmed_candidate_mask.png"
                ),
                mime="image/png",
                use_container_width=True,
            )
        with download_columns[1]:
            st.download_button(
                t("download_overlay"),
                data=pil_to_png_bytes(
                    result["overlay"]
                ),
                file_name=(
                    f"{st.session_state['filename_stem']}"
                    f"_candidate_overlay.png"
                ),
                mime="image/png",
                use_container_width=True,
            )
        with download_columns[2]:
            st.download_button(
                t("download_json"),
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
    (
        '<div class="main-title">'
        f'{t("app_title")}'
        '</div>'
    ),
    unsafe_allow_html=True,
)

st.markdown(
    (
        '<div class="subtitle">'
        f'{t("app_subtitle")}'
        '</div>'
    ),
    unsafe_allow_html=True,
)


input_summary = load_json(
    INPUT_GATE_LOCKED_SUMMARY
)

water_summary = load_json(
    WATER_GATE_LOCKED_SUMMARY
)


with st.sidebar:
    st.selectbox(
        t("language_label"),
        options=[
            "English",
            "Türkçe",
        ],
        key="app_language",
    )

    st.header(
        t("sidebar_header")
    )

    segmentation_threshold = st.slider(
        t("scan_threshold"),
        min_value=0.10,
        max_value=0.60,
        value=0.20,
        step=0.05,
        help=t("scan_threshold_help"),
    )

    st.session_state[
        "final_scan_threshold"
    ] = float(
        segmentation_threshold
    )

    minimum_component_ratio = (
        st.number_input(
            t("minimum_candidate_ratio"),
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
            t("minimum_candidate_pixels"),
            min_value=1,
            max_value=10000,
            value=16,
            step=1,
            disabled=True,
        )
    )

    st.divider()
    st.subheader(
        t("input_gate_model")
    )

    if input_summary:
        st.write(
            t("locked_negative_test"),
            input_summary.get(
                "total_negative_samples",
                0,
            ),
        )

        st.write(
            t("false_accepts"),
            input_summary.get(
                "false_accept_count",
                0,
            ),
        )

    st.divider()
    st.subheader(
        t("water_gate_model")
    )

    if water_summary:
        metrics = water_summary.get(
            "accepted_scene_metrics",
            {},
        )

        st.write(
            t("locked_test_acceptance"),
            (
                f"{water_summary.get('accepted_count', 0)}"
                f"/{water_summary.get('locked_test_count', 0)}"
            ),
        )

        st.write(
            t("water_precision"),
            (
                f"{float(metrics.get('water_precision', 0)) * 100.0:.3f}%"
            ),
        )

        st.write(
            t("land_leakage"),
            (
                f"{float(metrics.get('land_leakage', 0)) * 100.0:.4f}%"
            ),
        )

    st.warning(
        t("research_warning")
    )


uploaded_file = st.file_uploader(
    t("upload_label"),
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
        t("upload_prompt")
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
        t(
            "image_read_error",
            error=error,
        )
    )
    st.stop()


model_image = prepare_image_for_models(
    uploaded_image
)


preview, info = st.columns(
    [2, 1]
)

with preview:
    st.image(
        uploaded_image,
        caption=t(
            "uploaded_image_caption"
        ),
        use_container_width=True,
    )

with info:
    st.write(
        t("file_label"),
        uploaded_file.name,
    )

    st.write(
        t("dimensions_label"),
        (
            f"{uploaded_image.width} × "
            f"{uploaded_image.height}"
        ),
    )

    st.write(
        t("workflow_label"),
        t("workflow_text"),
    )

    analyse_button = st.button(
        t("start_scan"),
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
            t("spinner_input")
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
                t("spinner_water")
            ):
                water_gate = (
                    load_water_gate_cached()
                )

                water_result = module[
                    "run_water_gate"
                ](
                    model_image,
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
                t(
                    "spinner_oil",
                    threshold=(
                        segmentation_threshold
                    ),
                )
            ):
                oil_pipeline = (
                    load_oil_pipeline_cached()
                )
                verifier_module = (
                    load_v08_module()
                )
                oil_result = verifier_module[
                    "run_oil_pipeline_v07"
                ](
                    image=model_image,
                    safe_water_mask=(
                        analysis_water_mask
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

                (
                    screening_mask,
                    screening_mode,
                    screening_cutoff,
                ) = build_sensitive_screening_mask(
                    image=model_image,
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
    t("section_input")
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
        t("water_result_missing")
    )
    st.stop()


st.divider()
st.subheader(
    t("section_water")
)

render_water_gate(
    water_result
)


if not water_result[
    "pipeline_allowed"
]:
    st.markdown(
        t("sensitive_mode_note"),
        unsafe_allow_html=True,
    )


oil_result = st.session_state.get(
    "oil_result"
)

if oil_result is None:
    st.error(
        t("oil_result_missing")
    )
    st.stop()


st.divider()
st.subheader(
    t("section_oil")
)

render_oil_result(
    oil_result
)


st.markdown(
    t("scientific_note"),
    unsafe_allow_html=True,
)
