from __future__ import annotations

import argparse
import hashlib
import json
import runpy
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]

REGRESSION_SCRIPT = (
    ROOT
    / "scripts"
    / "92_test_v07_known_problem_scenes.py"
)

HOLDOUT_MANIFEST = (
    ROOT
    / "data"
    / "metadata"
    / "v07_fresh_end_to_end_holdout_scenes.csv"
)

VERIFIER_CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "verifier_v07"
    / "best.pth"
)

VERIFIER_CONFIG = (
    ROOT
    / "checkpoints"
    / "verifier_v07"
    / "calibration_config_v2.json"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "v07_fresh_negative_holdout"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "summary.json"
)

PER_SCENE_PATH = (
    OUTPUT_DIR
    / "per_scene_results.csv"
)

CONTACT_SHEET_PATH = (
    OUTPUT_DIR
    / "fresh_negative_holdout_contact_sheet.jpg"
)

CONSUMED_MARKER = (
    ROOT
    / "data"
    / "metadata"
    / "v07_fresh_negative_holdout_consumed.json"
)

FALSE_ALARM_DIR = (
    OUTPUT_DIR
    / "confirmed_false_alarms"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Daha önce açılmamış nc/nw DARTIS sahnelerinde v0.7 "
            "water-gated verifier için tek seferlik negatif yanlış "
            "alarm testi yapar. Bütün sahnelerin gerçek etiketi NO_OIL'dir."
        )
    )

    parser.add_argument(
        "--consume-fresh-holdout",
        action="store_true",
        help=(
            "Taze holdout görüntülerini tek seferlik değerlendirmeye "
            "açmayı açıkça onaylar."
        ),
    )

    parser.add_argument(
        "--minimum-water-accepted-scenes",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--maximum-confirmed-oil-false-alarm-scenes",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--maximum-total-final-oil-pixels",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--segmentation-threshold",
        type=float,
        default=0.60,
    )

    parser.add_argument(
        "--minimum-component-ratio",
        type=float,
        default=0.0005,
    )

    parser.add_argument(
        "--minimum-component-pixels",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--device",
        choices=[
            "auto",
            "cuda",
            "cpu",
        ],
        default="auto",
    )

    return parser.parse_args()


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Gerekli dosya bulunamadı: {path}"
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def resolve_path(value: str | Path) -> Path:
    path = Path(
        str(value).strip().replace(
            "\\",
            "/",
        )
    )

    if not path.is_absolute():
        path = ROOT / path

    return path.resolve()


def relative(path: Path) -> str:
    try:
        return str(
            path.resolve().relative_to(
                ROOT.resolve()
            )
        ).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def parse_false(value: Any) -> bool:
    text = str(value).strip().lower()

    return text in {
        "",
        "0",
        "false",
        "no",
        "none",
        "nan",
    }


def load_holdout() -> pd.DataFrame:
    require_file(
        HOLDOUT_MANIFEST
    )

    frame = pd.read_csv(
        HOLDOUT_MANIFEST,
        encoding="utf-8-sig",
        low_memory=False,
    )

    required = {
        "scene_id",
        "group",
        "expected_scene_label",
        "split_role",
        "image_path",
        "image_opened",
        "model_inference_run",
    }

    missing = required - set(
        frame.columns
    )

    if missing:
        raise RuntimeError(
            "Fresh holdout manifestinde eksik sütunlar: "
            f"{sorted(missing)}"
        )

    if frame.empty:
        raise RuntimeError(
            "Fresh holdout manifesti boş."
        )

    invalid_groups = set(
        frame[
            "group"
        ].astype(str).str.lower()
    ) - {
        "nc",
        "nw",
    }

    if invalid_groups:
        raise RuntimeError(
            "Bu aşama yalnız nc/nw negatif holdout içindir. "
            f"Geçersiz gruplar: {sorted(invalid_groups)}"
        )

    invalid_labels = frame[
        ~frame[
            "expected_scene_label"
        ].astype(str).str.upper().eq(
            "NO_OIL"
        )
    ]

    if not invalid_labels.empty:
        raise RuntimeError(
            "Fresh negatif holdout içinde NO_OIL dışı etiket bulundu."
        )

    already_opened = frame[
        ~frame[
            "image_opened"
        ].map(
            parse_false
        )
        |
        ~frame[
            "model_inference_run"
        ].map(
            parse_false
        )
    ]

    if not already_opened.empty:
        raise RuntimeError(
            "Manifestte daha önce açılmış veya inference yapılmış "
            "holdout kaydı var. Tek seferlik test güvenilir değil."
        )

    frame = frame.copy()

    frame[
        "resolved_image_path"
    ] = frame[
        "image_path"
    ].map(
        resolve_path
    )

    missing_images = frame[
        ~frame[
            "resolved_image_path"
        ].map(
            lambda value: Path(
                value
            ).exists()
        )
    ]

    if not missing_images.empty:
        raise FileNotFoundError(
            "Eksik holdout görüntüleri: "
            + ", ".join(
                missing_images[
                    "scene_id"
                ].astype(str).head(5)
            )
        )

    duplicate_scene_count = int(
        frame[
            "scene_id"
        ].astype(str).duplicated().sum()
    )

    if duplicate_scene_count:
        raise RuntimeError(
            f"Holdout manifestinde {duplicate_scene_count} tekrar sahne var."
        )

    return frame.reset_index(
        drop=True
    )


def create_row_panel(
    image: Image.Image,
    water_mask: np.ndarray,
    candidate_overlay: Image.Image,
    final_mask: np.ndarray,
    title: str,
    regression_module: dict[str, Any],
) -> Image.Image:
    panel_width = 1200
    column_width = (
        panel_width // 4
    )

    images = [
        image.convert("RGB"),
        regression_module[
            "mask_to_image"
        ](
            water_mask
        ).convert("RGB"),
        candidate_overlay.convert(
            "RGB"
        ),
        regression_module[
            "mask_to_image"
        ](
            final_mask
        ).convert("RGB"),
    ]

    resized = []

    for item in images:
        copy = item.copy()

        copy.thumbnail(
            (
                column_width,
                300,
            ),
            Image.Resampling.LANCZOS,
        )

        canvas = Image.new(
            "RGB",
            (
                column_width,
                300,
            ),
            "black",
        )

        canvas.paste(
            copy,
            (
                (
                    column_width
                    - copy.width
                )
                // 2,
                (
                    300
                    - copy.height
                )
                // 2,
            ),
        )

        resized.append(
            canvas
        )

    header_height = 42

    row = Image.new(
        "RGB",
        (
            panel_width,
            300
            + header_height,
        ),
        "black",
    )

    for index, item in enumerate(
        resized
    ):
        row.paste(
            item,
            (
                index
                * column_width,
                header_height,
            ),
        )

    draw = ImageDraw.Draw(
        row
    )

    draw.text(
        (
            8,
            13,
        ),
        title[:190],
        fill="white",
        font=ImageFont.load_default(),
    )

    return row


def main() -> None:
    args = parse_args()

    if not args.consume_fresh_holdout:
        raise RuntimeError(
            "Bu test holdout'u tüketir. Çalıştırmak için "
            "--consume-fresh-holdout parametresi zorunludur."
        )

    if CONSUMED_MARKER.exists():
        consumed = json.loads(
            CONSUMED_MARKER.read_text(
                encoding="utf-8-sig"
            )
        )

        raise RuntimeError(
            "Fresh negatif holdout daha önce tüketilmiş. "
            f"Tarih: {consumed.get('consumed_at_utc', 'bilinmiyor')} | "
            f"Sonuç: {consumed.get('gate_passed', 'bilinmiyor')}"
        )

    for path in (
        REGRESSION_SCRIPT,
        HOLDOUT_MANIFEST,
        VERIFIER_CHECKPOINT,
        VERIFIER_CONFIG,
    ):
        require_file(path)

    holdout = load_holdout()

    if OUTPUT_DIR.exists():
        shutil.rmtree(
            OUTPUT_DIR
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    regression = runpy.run_path(
        str(
            REGRESSION_SCRIPT
        ),
        run_name=(
            "v07_fresh_negative_holdout_module"
        ),
    )

    (
        safe_module,
        water_gate,
        oil_pipeline,
    ) = regression[
        "load_models"
    ](
        args.device
    )

    print("=" * 78)
    print(
        "v0.7 FRESH NEGATIVE HOLDOUT"
    )
    print("=" * 78)

    print(
        "Fresh scene:",
        len(holdout),
    )

    print(
        "Groups:",
        holdout.groupby(
            "group"
        ).size().to_dict(),
    )

    print(
        "Expected label: NO_OIL for every scene"
    )

    print(
        "Bu test yalnız bir kez tüketilecek."
    )

    print()

    rows: list[
        dict[str, Any]
    ] = []

    contact_rows: list[
        Image.Image
    ] = []

    for index, row in enumerate(
        holdout.itertuples(),
        start=1,
    ):
        scene_id = str(
            row.scene_id
        )

        image_path = Path(
            row.resolved_image_path
        )

        image = Image.open(
            image_path
        ).convert("L")

        with torch.inference_mode():
            water_result = safe_module[
                "run_water_gate"
            ](
                image,
                water_gate,
            )

        oil_result = None

        if water_result[
            "pipeline_allowed"
        ]:
            oil_result = regression[
                "run_oil_pipeline_v07"
            ](
                image=image,
                safe_water_mask=(
                    water_result[
                        "safe_water_mask"
                    ]
                ),
                pipeline=oil_pipeline,
                segmentation_threshold=float(
                    args.segmentation_threshold
                ),
                minimum_component_ratio=float(
                    args.minimum_component_ratio
                ),
                minimum_component_pixels=int(
                    args.minimum_component_pixels
                ),
            )

        if not water_result[
            "pipeline_allowed"
        ]:
            system_result = (
                "UNCERTAIN_BLOCK"
            )
        elif (
            oil_result is not None
            and oil_result[
                "oil_detected"
            ]
        ):
            system_result = (
                "CONFIRMED_OIL_FALSE_ALARM"
            )
        elif (
            oil_result is not None
            and oil_result[
                "uncertain_candidate_count"
            ]
            > 0
        ):
            system_result = (
                "UNCERTAIN_NO_FINAL_MASK"
            )
        else:
            system_result = (
                "NO_CONFIRMED_OIL"
            )

        confirmed_count = int(
            oil_result[
                "confirmed_candidate_count"
            ]
            if oil_result is not None
            else 0
        )

        uncertain_count = int(
            oil_result[
                "uncertain_candidate_count"
            ]
            if oil_result is not None
            else 0
        )

        lookalike_count = int(
            oil_result[
                "lookalike_candidate_count"
            ]
            if oil_result is not None
            else 0
        )

        final_pixels = int(
            oil_result[
                "final_positive_pixels"
            ]
            if oil_result is not None
            else 0
        )

        final_mask = (
            oil_result[
                "final_mask"
            ]
            if oil_result is not None
            else np.zeros(
                (
                    image.height,
                    image.width,
                ),
                dtype=np.uint8,
            )
        )

        candidate_overlay = (
            oil_result[
                "candidate_overlay"
            ]
            if oil_result is not None
            else image.convert(
                "RGB"
            )
        )

        record = {
            "scene_id": scene_id,
            "group": str(
                row.group
            ),
            "expected": (
                "NO_OIL"
            ),
            "image_path": relative(
                image_path
            ),
            "water_decision": (
                water_result[
                    "decision"
                ]
            ),
            "safe_water_fraction": float(
                water_result[
                    "predicted_water_fraction"
                ]
            ),
            "water_uncertain_fraction": float(
                water_result[
                    "uncertain_fraction"
                ]
            ),
            "oil_model_executed": bool(
                oil_result
                is not None
            ),
            "system_result": (
                system_result
            ),
            "confirmed_candidate_count": (
                confirmed_count
            ),
            "uncertain_candidate_count": (
                uncertain_count
            ),
            "lookalike_candidate_count": (
                lookalike_count
            ),
            "final_oil_pixels": (
                final_pixels
            ),
            "false_alarm": bool(
                confirmed_count > 0
                or final_pixels > 0
            ),
        }

        rows.append(
            record
        )

        title = (
            f"{scene_id} | water={record['water_decision']} "
            f"| result={system_result} | "
            f"C={confirmed_count} U={uncertain_count} "
            f"L={lookalike_count} pixels={final_pixels}"
        )

        contact_rows.append(
            create_row_panel(
                image=image,
                water_mask=(
                    water_result[
                        "safe_water_mask"
                    ]
                ),
                candidate_overlay=(
                    candidate_overlay
                ),
                final_mask=(
                    final_mask
                ),
                title=title,
                regression_module=(
                    regression
                ),
            )
        )

        if record[
            "false_alarm"
        ]:
            FALSE_ALARM_DIR.mkdir(
                parents=True,
                exist_ok=True,
            )

            safe_scene_name = (
                scene_id.replace(
                    ":",
                    "__",
                )
                .replace(
                    "/",
                    "_",
                )
            )

            scene_dir = (
                FALSE_ALARM_DIR
                / safe_scene_name
            )

            scene_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            image.save(
                scene_dir
                / "input.png"
            )

            regression[
                "mask_to_image"
            ](
                water_result[
                    "safe_water_mask"
                ]
            ).save(
                scene_dir
                / "safe_water_mask.png"
            )

            candidate_overlay.save(
                scene_dir
                / "candidate_decisions_overlay.png"
            )

            regression[
                "mask_to_image"
            ](
                final_mask
            ).save(
                scene_dir
                / "final_confirmed_oil_mask.png"
            )

            if oil_result is not None:
                pd.DataFrame(
                    oil_result[
                        "candidate_results"
                    ]
                ).to_csv(
                    scene_dir
                    / "candidate_decisions.csv",
                    index=False,
                    encoding="utf-8-sig",
                )

        print(
            f"[{index:02d}/{len(holdout):02d}] "
            f"{scene_id} "
            f"| water={record['water_decision']} "
            f"| result={system_result} "
            f"| C={confirmed_count} "
            f"U={uncertain_count} "
            f"L={lookalike_count} "
            f"| pixels={final_pixels}"
        )

    result_frame = pd.DataFrame(
        rows
    )

    result_frame.to_csv(
        PER_SCENE_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    contact_sheet = Image.new(
        "RGB",
        (
            1200,
            sum(
                row.height
                for row in contact_rows
            ),
        ),
        "black",
    )

    y = 0

    for row in contact_rows:
        contact_sheet.paste(
            row,
            (
                0,
                y,
            ),
        )

        y += row.height

    contact_sheet.save(
        CONTACT_SHEET_PATH,
        quality=90,
    )

    water_accepted_count = int(
        result_frame[
            "oil_model_executed"
        ].eq(
            True
        ).sum()
    )

    water_blocked_count = int(
        len(
            result_frame
        )
        - water_accepted_count
    )

    false_alarm_scene_count = int(
        result_frame[
            "false_alarm"
        ].eq(
            True
        ).sum()
    )

    total_final_oil_pixels = int(
        result_frame[
            "final_oil_pixels"
        ].sum()
    )

    total_confirmed_candidates = int(
        result_frame[
            "confirmed_candidate_count"
        ].sum()
    )

    total_uncertain_candidates = int(
        result_frame[
            "uncertain_candidate_count"
        ].sum()
    )

    total_lookalike_candidates = int(
        result_frame[
            "lookalike_candidate_count"
        ].sum()
    )

    checks = {
        "minimum_water_accepted_scenes": (
            water_accepted_count
            >= args.minimum_water_accepted_scenes
        ),
        "maximum_confirmed_oil_false_alarm_scenes": (
            false_alarm_scene_count
            <= args.maximum_confirmed_oil_false_alarm_scenes
        ),
        "maximum_total_final_oil_pixels": (
            total_final_oil_pixels
            <= args.maximum_total_final_oil_pixels
        ),
    }

    failed_checks = [
        name
        for name, passed
        in checks.items()
        if not passed
    ]

    gate_passed = (
        len(
            failed_checks
        )
        == 0
    )

    summary = {
        "stage": (
            "v07_fresh_negative_holdout"
        ),
        "holdout_count": int(
            len(
                result_frame
            )
        ),
        "group_counts": {
            str(key): int(value)
            for key, value in result_frame.groupby(
                "group"
            ).size().to_dict().items()
        },
        "expected_label": (
            "NO_OIL"
        ),
        "water_accepted_scene_count": (
            water_accepted_count
        ),
        "water_blocked_scene_count": (
            water_blocked_count
        ),
        "water_accepted_rate": float(
            water_accepted_count
            / max(
                len(
                    result_frame
                ),
                1,
            )
        ),
        "confirmed_oil_false_alarm_scene_count": (
            false_alarm_scene_count
        ),
        "confirmed_oil_false_alarm_rate_overall": float(
            false_alarm_scene_count
            / max(
                len(
                    result_frame
                ),
                1,
            )
        ),
        "confirmed_oil_false_alarm_rate_among_water_accepted": float(
            false_alarm_scene_count
            / max(
                water_accepted_count,
                1,
            )
        ),
        "total_confirmed_candidates": (
            total_confirmed_candidates
        ),
        "total_uncertain_candidates": (
            total_uncertain_candidates
        ),
        "total_lookalike_candidates": (
            total_lookalike_candidates
        ),
        "total_final_oil_pixels": (
            total_final_oil_pixels
        ),
        "gate_passed": bool(
            gate_passed
        ),
        "failed_checks": (
            failed_checks
        ),
        "criteria": {
            "minimum_water_accepted_scenes": int(
                args.minimum_water_accepted_scenes
            ),
            "maximum_confirmed_oil_false_alarm_scenes": int(
                args.maximum_confirmed_oil_false_alarm_scenes
            ),
            "maximum_total_final_oil_pixels": int(
                args.maximum_total_final_oil_pixels
            ),
        },
        "fresh_holdout_consumed": True,
        "verifier_checkpoint": relative(
            VERIFIER_CHECKPOINT
        ),
        "verifier_config": relative(
            VERIFIER_CONFIG
        ),
        "holdout_manifest": relative(
            HOLDOUT_MANIFEST
        ),
        "per_scene_results": relative(
            PER_SCENE_PATH
        ),
        "contact_sheet": relative(
            CONTACT_SHEET_PATH
        ),
        "input_gate_retested": False,
        "scientific_note": (
            "Bu tek seferlik test yalnız daha önce açılmamış nc/nw "
            "DARTIS negatif sahnelerinde water-gated petrol segmenteri "
            "ve v0.7 verifier'ın kesin CONFIRMED_OIL yanlış alarmını "
            "ölçer. v0.5 input gate bu aşamada yeniden test edilmemiştir; "
            "onun bağımsız negatif kilitli testi ayrıdır. Pozitif taze "
            "holdout bulunmadığı için bu sonuç petrol recall kanıtı değildir."
        ),
    }

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    marker = {
        "stage": (
            "v07_fresh_negative_holdout_consumed"
        ),
        "consumed_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "gate_passed": bool(
            gate_passed
        ),
        "holdout_count": int(
            len(
                result_frame
            )
        ),
        "confirmed_oil_false_alarm_scene_count": (
            false_alarm_scene_count
        ),
        "total_final_oil_pixels": (
            total_final_oil_pixels
        ),
        "holdout_manifest_sha256": (
            sha256_file(
                HOLDOUT_MANIFEST
            )
        ),
        "verifier_checkpoint_sha256": (
            sha256_file(
                VERIFIER_CHECKPOINT
            )
        ),
        "verifier_config_sha256": (
            sha256_file(
                VERIFIER_CONFIG
            )
        ),
        "summary": relative(
            SUMMARY_PATH
        ),
    }

    CONSUMED_MARKER.write_text(
        json.dumps(
            marker,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print(
        "FRESH NEGATIVE HOLDOUT NİHAİ SONUCU"
    )
    print("=" * 78)

    print(
        "Gate:",
        (
            "PASS"
            if gate_passed
            else "FAIL"
        ),
    )

    print(
        "Water accepted:",
        f"{water_accepted_count}/{len(result_frame)}",
    )

    print(
        "Confirmed-oil false alarm scenes:",
        f"{false_alarm_scene_count}/{len(result_frame)}",
    )

    print(
        "False alarm among water accepted:",
        (
            f"{false_alarm_scene_count / max(water_accepted_count, 1):.4f}"
        ),
    )

    print(
        "Confirmed candidates:",
        total_confirmed_candidates,
    )

    print(
        "Uncertain candidates:",
        total_uncertain_candidates,
    )

    print(
        "Look-alike candidates:",
        total_lookalike_candidates,
    )

    print(
        "Total final oil pixels:",
        total_final_oil_pixels,
    )

    print(
        "Failed:",
        failed_checks,
    )

    print(
        "Fresh holdout consumed:",
        True,
    )

    print(
        "Özet:",
        SUMMARY_PATH.resolve(),
    )

    print(
        "Görsel:",
        CONTACT_SHEET_PATH.resolve(),
    )


if __name__ == "__main__":
    main()
