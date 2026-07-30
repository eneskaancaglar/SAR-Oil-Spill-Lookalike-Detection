from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "app_v11.py"
TARGET = ROOT / "app_final.py"


def replace_exact_once(
    text: str,
    old: str,
    new: str,
    label: str,
) -> str:
    count = text.count(old)

    if count != 1:
        raise RuntimeError(
            f"{label}: beklenen eşleşme 1, bulunan {count}."
        )

    return text.replace(
        old,
        new,
        1,
    )


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(
            f"Kaynak uygulama bulunamadı: {SOURCE}"
        )

    text = SOURCE.read_text(
        encoding="utf-8-sig"
    )

    text = replace_exact_once(
        text,
        'page_title="SAR Petrol Tarama Sistemi v1.1"',
        'page_title="SAR Petrol Taraması — Final Hassas Mod"',
        'sayfa başlığı',
    )

    text = replace_exact_once(
        text,
        "'🛰️ SAR Petrol Tarama Sistemi — Recall First'",
        "'🛰️ SAR Petrol Taraması — Final Hassas Mod'",
        'ana başlık',
    )

    text = replace_exact_once(
        text,
        "'v0.5 giriş doğrulama → kesin karayı dışla → '\n    '0.35 eşikle petrol tara → v0.8 aday doğrulama'",
        "'Girdi doğrulama → kesin karayı dışlama → '\n    'hassas petrol taraması → aday doğrulama'",
        'alt başlık',
    )

    text = replace_exact_once(
        text,
        '"Recall-first tarama ayarları"',
        '"Final hassas tarama ayarları"',
        'sidebar başlığı',
    )

    text = replace_exact_once(
        text,
        '"v0.5 giriş kapısı"',
        '"Girdi doğrulama modeli"',
        'girdi modeli etiketi',
    )

    text = replace_exact_once(
        text,
        '"v0.6 kara-su kapısı"',
        '"Kara-su güvenlik modeli"',
        'kara-su modeli etiketi',
    )

    text = replace_exact_once(
        text,
        '"3. v1.1 recall-first petrol taraması ve v0.8 doğrulama"',
        '"3. Final hassas petrol taraması ve aday doğrulama"',
        'sonuç başlığı',
    )

    text = replace_exact_once(
        text,
        'caption="0.35 recall-first şüpheli petrol maskesi"',
        'caption="Final hassas şüpheli-bölge maskesi"',
        'hassas maske başlığı',
    )

    text = replace_exact_once(
        text,
        '    segmentation_threshold = st.slider(\n        "Petrol U-Net tarama eşiği",\n        min_value=0.10,\n        max_value=0.95,\n        value=0.35,\n        step=0.05,\n        disabled=True,\n        help=(\n            "Kaçırmayı azaltmak için kullanılan recall-first eşik. "\n            "Yanlış alarmı artırabilir."\n        ),\n    )\n',
        '    segmentation_threshold = st.slider(\n        "Hassas petrol tarama eşiği",\n        min_value=0.10,\n        max_value=0.60,\n        value=0.20,\n        step=0.05,\n        help=(\n            "0.20 varsayılan hassas taramadır. Eşik düştükçe "\n            "kaçırma azalabilir fakat yanlış alarm artabilir."\n        ),\n    )\n\n    st.session_state[\n        "final_scan_threshold"\n    ] = float(\n        segmentation_threshold\n    )\n',
        "etkileşimli hassas eşik",
    )

    text = text.replace(
        "segmentation_threshold=0.35,",
        "segmentation_threshold=float(segmentation_threshold),",
        1,
    )

    text = text.replace(
        '"0.35 eşikle taranıyor..."',
        'f"{segmentation_threshold:.2f} eşikle taranıyor..."',
        1,
    )

    helper_marker = "\ndef select_device() -> torch.device:\n"
    helper_position = text.find(
        helper_marker
    )

    if helper_position < 0:
        raise RuntimeError(
            "Hassas maske yardımcı fonksiyonu için ekleme noktası bulunamadı."
        )

    text = (
        text[:helper_position]
        + "\n"
        + '\ndef build_sensitive_screening_mask(\n    image: Image.Image,\n    probability: np.ndarray,\n    analysis_mask: np.ndarray,\n    absolute_mask: np.ndarray,\n) -> tuple[np.ndarray, str, float]:\n    probability_array = np.asarray(\n        probability,\n        dtype=np.float32,\n    )\n\n    valid_mask = (\n        np.asarray(\n            analysis_mask,\n            dtype=np.uint8,\n        )\n        > 0\n    )\n\n    absolute_array = (\n        np.asarray(\n            absolute_mask,\n            dtype=np.uint8,\n        )\n        > 0\n    )\n\n    if absolute_array.any():\n        return (\n            absolute_array.astype(\n                np.uint8\n            ),\n            "ABSOLUTE_UNET_THRESHOLD",\n            float(\n                st.session_state.get(\n                    "final_scan_threshold",\n                    0.20,\n                )\n            ),\n        )\n\n    if (\n        not valid_mask.any()\n        or probability_array.size == 0\n    ):\n        return (\n            np.zeros_like(\n                probability_array,\n                dtype=np.uint8,\n            ),\n            "NO_VALID_AREA",\n            0.0,\n        )\n\n    gray = (\n        np.asarray(\n            image.convert("L"),\n            dtype=np.float32,\n        )\n        / 255.0\n    )\n\n    if gray.shape != probability_array.shape:\n        gray = np.asarray(\n            image.convert("L").resize(\n                (\n                    probability_array.shape[1],\n                    probability_array.shape[0],\n                ),\n                Image.Resampling.BILINEAR,\n            ),\n            dtype=np.float32,\n        ) / 255.0\n\n    probability_tensor = torch.from_numpy(\n        probability_array\n    ).unsqueeze(0).unsqueeze(0)\n\n    smoothed_probability = (\n        F.avg_pool2d(\n            probability_tensor,\n            kernel_size=9,\n            stride=1,\n            padding=4,\n        )\n        .squeeze()\n        .numpy()\n        .astype(np.float32)\n    )\n\n    gray_tensor = torch.from_numpy(\n        gray\n    ).unsqueeze(0).unsqueeze(0)\n\n    local_background = (\n        F.avg_pool2d(\n            gray_tensor,\n            kernel_size=41,\n            stride=1,\n            padding=20,\n        )\n        .squeeze()\n        .numpy()\n        .astype(np.float32)\n    )\n\n    local_darkness = np.clip(\n        local_background - gray,\n        0.0,\n        None,\n    )\n\n    def robust_normalize(\n        values: np.ndarray,\n    ) -> np.ndarray:\n        valid_values = values[\n            valid_mask\n        ]\n\n        scale = float(\n            np.quantile(\n                valid_values,\n                0.99,\n            )\n        )\n\n        if scale <= 1e-8:\n            scale = float(\n                valid_values.max(\n                    initial=0.0\n                )\n            )\n\n        if scale <= 1e-8:\n            return np.zeros_like(\n                values,\n                dtype=np.float32,\n            )\n\n        return np.clip(\n            values / scale,\n            0.0,\n            1.0,\n        ).astype(\n            np.float32\n        )\n\n    probability_score = robust_normalize(\n        smoothed_probability\n    )\n\n    darkness_score = robust_normalize(\n        local_darkness\n    )\n\n    combined_score = (\n        0.75\n        * probability_score\n        + 0.25\n        * darkness_score\n    )\n\n    combined_tensor = torch.from_numpy(\n        combined_score\n    ).unsqueeze(0).unsqueeze(0)\n\n    combined_score = (\n        F.avg_pool2d(\n            combined_tensor,\n            kernel_size=7,\n            stride=1,\n            padding=3,\n        )\n        .squeeze()\n        .numpy()\n        .astype(np.float32)\n    )\n\n    valid_scores = combined_score[\n        valid_mask\n    ]\n\n    if (\n        valid_scores.size == 0\n        or float(\n            valid_scores.max(\n                initial=0.0\n            )\n        )\n        <= 1e-8\n    ):\n        return (\n            np.zeros_like(\n                probability_array,\n                dtype=np.uint8,\n            ),\n            "NO_MODEL_RESPONSE",\n            0.0,\n        )\n\n    relative_cutoff = float(\n        np.quantile(\n            valid_scores,\n            0.995,\n        )\n    )\n\n    screening_mask = (\n        valid_mask\n        & (\n            combined_score\n            >= relative_cutoff\n        )\n    ).astype(\n        np.uint8\n    )\n\n    return (\n        screening_mask,\n        "RELATIVE_TOP_0_5_PERCENT",\n        relative_cutoff,\n    )\n'
        + text[helper_position:]
    )

    text = replace_exact_once(
        text,
        '                oil_result[\n                    "scan_threshold"\n                ] = 0.35\n',
        '\n                screening_mask, screening_mode, screening_cutoff = (\n                    build_sensitive_screening_mask(\n                        image=uploaded_image,\n                        probability=oil_result[\n                            "probability"\n                        ],\n                        analysis_mask=(\n                            analysis_water_mask\n                        ),\n                        absolute_mask=oil_result[\n                            "raw_mask"\n                        ],\n                    )\n                )\n\n                oil_result[\n                    "screening_mask"\n                ] = screening_mask\n\n                oil_result[\n                    "screening_mode"\n                ] = screening_mode\n\n                oil_result[\n                    "screening_cutoff"\n                ] = float(\n                    screening_cutoff\n                )\n\n                oil_result[\n                    "scan_threshold"\n                ] = float(\n                    segmentation_threshold\n                )\n',
        "hassas tarama üretimi",
    )

    text = replace_exact_once(
        text,
        'def render_oil_result(\n    result: dict[str, Any],\n) -> None:\n',
        'def render_oil_result(\n    result: dict[str, Any],\n) -> None:\n    screening_mask = np.asarray(\n        result.get(\n            "screening_mask",\n            result["raw_mask"],\n        ),\n        dtype=np.uint8,\n    )\n\n    screening_pixels = int(\n        screening_mask.sum()\n    )\n\n    screening_mode = str(\n        result.get(\n            "screening_mode",\n            "ABSOLUTE_UNET_THRESHOLD",\n        )\n    )\n',
        "sonuç fonksiyonu başlangıcı",
    )

    text = replace_exact_once(
        text,
        '    elif result["candidate_count"] > 0:\n',
        '    elif screening_pixels > 0:\n',
        "şüpheli bölge kararı",
    )

    text = text.replace(
        "'🟡 ŞÜPHELİ PETROL BÖLGESİ BULUNDU — '\n            'HAM TARAMA MASKESİNDE GÖSTERİLİYOR'",
        "'🟡 ŞÜPHELİ BÖLGE BULUNDU — HASSAS TARAMA MASKESİNDE '\n            'GÖSTERİLİYOR; KESİN PETROL KARARI DEĞİLDİR'",
        1,
    )

    text = replace_exact_once(
        text,
        '    columns[0].metric(\n        "Tarama adayı pikseli",\n        int(\n            np.asarray(\n                result["raw_mask"],\n                dtype=np.uint8,\n            ).sum()\n        ),\n    )\n',
        '    columns[0].metric(\n        "Hassas tarama pikseli",\n        screening_pixels,\n    )\n',
        "hassas tarama metriği",
    )

    text = text.replace(
        'mask_to_image(result["raw_mask"])',
        "mask_to_image(screening_mask)",
        1,
    )

    text = text.replace(
        'caption="Final hassas şüpheli-bölge maskesi",',
        'caption=(\n                    "Final hassas şüpheli-bölge maskesi — "\n                    f"mod: {screening_mode}"\n                ),',
        1,
    )

    text = text.replace(
        'Recall-first tarama kesin kara dışındaki güvenli ve\n    belirsiz alanları 0.35 eşikle inceler.',
        'Final hassas tarama kesin kara dışındaki güvenli ve\n    belirsiz alanları seçilen eşikle inceler. Mutlak U-Net\n    maskesi boş kalırsa göreli model yanıtı ve yerel koyuluk\n    yalnız şüpheli-bölge taraması için kullanılır.',
        1,
    )

    required = [
        "Final Hassas Mod",
        "value=0.20",
        "segmentation_threshold=float(segmentation_threshold)",
        "build_sensitive_screening_mask",
        "RELATIVE_TOP_0_5_PERCENT",
        "screening_mask",
        "Hassas tarama pikseli",
    ]

    missing = [
        token
        for token in required
        if token not in text
    ]

    if missing:
        raise RuntimeError(
            "Final uygulamada eksik ifadeler: "
            + ", ".join(missing)
        )

    compile(
        text,
        str(TARGET),
        "exec",
    )

    TARGET.write_text(
        text,
        encoding="utf-8",
    )

    print("=" * 78)
    print("FINAL HASSAS PETROL TARAMA UYGULAMASI")
    print("=" * 78)
    print("Durum: PASS")
    print("Yeni uygulama:", TARGET)
    print("Varsayılan hassas eşik: 0.20")
    print("Eşik kullanıcı tarafından değiştirilebilir: True")
    print("Su kapısı belirsizken analiz devam eder: True")
    print("Mutlak maske boşsa göreli şüpheli-bölge taraması: True")
    print("Kesin petrol kararı: yalnız verifier onaylarsa")


if __name__ == "__main__":
    main()
