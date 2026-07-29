from __future__ import annotations

import argparse
import hashlib
import json
import math
import runpy
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

SAFE_PIPELINE_SCRIPT = (
    ROOT
    / "scripts"
    / "85_run_safe_water_masked_oil_pipeline.py"
)

HOLDOUT_MANIFEST = (
    ROOT
    / "data"
    / "metadata"
    / "v08_fresh_negative_holdout_scenes.csv"
)

VERIFIER_CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "verifier_v08"
    / "best.pth"
)

OPERATIONAL_CONFIG = (
    ROOT
    / "checkpoints"
    / "verifier_v08"
    / "operational_gate_config.json"
)

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "v08_fresh_negative_holdout"
)

PER_SCENE_PATH = (
    OUTPUT_DIR
    / "per_scene_results.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "summary.json"
)

CONTACT_SHEET_PATH = (
    OUTPUT_DIR
    / "fresh_negative_holdout_contact_sheet.jpg"
)

PANEL_DIR = (
    OUTPUT_DIR
    / "panels"
)

FALSE_ALARM_DIR = (
    OUTPUT_DIR
    / "confirmed_false_alarms"
)

CONSUMED_MARKER = (
    ROOT
    / "data"
    / "metadata"
    / "v08_fresh_negative_holdout_consumed.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "v0.8 verifier ve seçici operasyonel kapıyı, daha önce "
            "açılmamış nc/nw DARTIS sahnelerinde tek seferlik negatif "
            "yanlış-alarm testine sokar."
        )
    )

    parser.add_argument(
        "--consume-fresh-holdout",
        action="store_true",
    )

    parser.add_argument(
        "--resume-interrupted",
        action="store_true",
        help=(
            "Yalnız teknik olarak yarıda kalmış, marker durumu RUNNING "
            "olan aynı dondurulmuş testi kalan sahnelerden sürdürür."
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


def safe_name(value: str) -> str:
    return (
        value.replace(
            ":",
            "__",
        )
        .replace(
            "/",
            "_",
        )
        .replace(
            "\\",
            "_",
        )
    )


def false_like(value: Any) -> bool:
    return str(value).strip().lower() in {
        "",
        "0",
        "false",
        "no",
        "none",
        "nan",
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(
        path.read_text(
            encoding="utf-8-sig"
        )
    )


def write_json(
    path: Path,
    value: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


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
            "v0.8 holdout manifestinde eksik sütunlar: "
            f"{sorted(missing)}"
        )

    if frame.empty:
        raise RuntimeError(
            "v0.8 holdout manifesti boş."
        )

    groups = set(
        frame[
            "group"
        ].astype(str).str.lower()
    )

    if not groups.issubset(
        {
            "nc",
            "nw",
        }
    ):
        raise RuntimeError(
            "Bu test yalnız nc/nw negatif sahneleri içermelidir. "
            f"Bulunan gruplar: {sorted(groups)}"
        )

    if not frame[
        "expected_scene_label"
    ].astype(str).str.upper().eq(
        "NO_OIL"
    ).all():
        raise RuntimeError(
            "Negatif holdout içinde NO_OIL dışı etiket bulundu."
        )

    if not frame[
        "image_opened"
    ].map(
        false_like
    ).all():
        raise RuntimeError(
            "Holdout manifestinde daha önce açılmış görüntü kaydı var."
        )

    if not frame[
        "model_inference_run"
    ].map(
        false_like
    ).all():
        raise RuntimeError(
            "Holdout manifestinde daha önce inference yapılmış kayıt var."
        )

    if frame[
        "scene_id"
    ].astype(str).duplicated().any():
        raise RuntimeError(
            "Holdout manifestinde tekrar sahne var."
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
            "Eksik holdout görüntüsü: "
            + ", ".join(
                missing_images[
                    "scene_id"
                ].astype(str).head(5)
            )
        )

    return frame.reset_index(
        drop=True
    )


def prepare_run(
    args: argparse.Namespace,
    holdout: pd.DataFrame,
) -> set[str]:
    manifest_hash = sha256_file(
        HOLDOUT_MANIFEST
    )

    checkpoint_hash = sha256_file(
        VERIFIER_CHECKPOINT
    )

    config_hash = sha256_file(
        OPERATIONAL_CONFIG
    )

    if CONSUMED_MARKER.exists():
        marker = load_json(
            CONSUMED_MARKER
        )

        status = str(
            marker.get(
                "status",
                "",
            )
        ).upper()

        same_frozen_run = (
            marker.get(
                "holdout_manifest_sha256"
            )
            == manifest_hash
            and marker.get(
                "verifier_checkpoint_sha256"
            )
            == checkpoint_hash
            and marker.get(
                "operational_config_sha256"
            )
            == config_hash
        )

        if status == "COMPLETED":
            raise RuntimeError(
                "v0.8 fresh negatif holdout daha önce tamamlanarak "
                "tüketilmiş. Aynı test tekrar çalıştırılamaz."
            )

        if status == "RUNNING":
            if not args.resume_interrupted:
                raise RuntimeError(
                    "v0.8 holdout testi RUNNING durumunda yarıda kalmış. "
                    "Aynı dondurulmuş testi sürdürmek için "
                    "--resume-interrupted ekleyin."
                )

            if not same_frozen_run:
                raise RuntimeError(
                    "Yarıda kalan testte manifest/model/config hash'i "
                    "değişmiş. Bilimsel olarak devam edilemez."
                )

            if PER_SCENE_PATH.exists():
                completed = pd.read_csv(
                    PER_SCENE_PATH,
                    encoding="utf-8-sig",
                    low_memory=False,
                )

                return set(
                    completed[
                        "scene_id"
                    ].astype(str)
                )

            return set()

        raise RuntimeError(
            "Holdout marker durumu tanınmıyor: "
            f"{status!r}"
        )

    if args.resume_interrupted:
        raise RuntimeError(
            "--resume-interrupted yalnız mevcut RUNNING marker ile "
            "kullanılabilir."
        )

    if not args.consume_fresh_holdout:
        raise RuntimeError(
            "Bu tek seferlik test için --consume-fresh-holdout zorunludur."
        )

    if OUTPUT_DIR.exists():
        raise RuntimeError(
            "Çıktı klasörü zaten var fakat tüketim marker'ı yok. "
            "Klasörü elle incelemeden test başlatılmamalıdır: "
            f"{OUTPUT_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    PANEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    marker = {
        "stage": (
            "v08_fresh_negative_holdout"
        ),
        "status": "RUNNING",
        "started_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "holdout_count": int(
            len(
                holdout
            )
        ),
        "holdout_manifest_sha256": (
            manifest_hash
        ),
        "verifier_checkpoint_sha256": (
            checkpoint_hash
        ),
        "operational_config_sha256": (
            config_hash
        ),
        "processed_scene_count": 0,
    }

    write_json(
        CONSUMED_MARKER,
        marker,
    )

    return set()


def load_v08_runtime(
    device_name: str,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    for path in (
        REGRESSION_SCRIPT,
        SAFE_PIPELINE_SCRIPT,
        VERIFIER_CHECKPOINT,
        OPERATIONAL_CONFIG,
    ):
        require_file(path)

    regression = runpy.run_path(
        str(
            REGRESSION_SCRIPT
        ),
        run_name=(
            "v08_fresh_holdout_regression_module"
        ),
    )

    safe_module = runpy.run_path(
        str(
            SAFE_PIPELINE_SCRIPT
        ),
        run_name=(
            "v08_fresh_holdout_safe_pipeline"
        ),
    )

    config = load_json(
        OPERATIONAL_CONFIG
    )

    if not config.get(
        "operational_gate_passed",
        False,
    ):
        raise RuntimeError(
            "v0.8 operational_gate_config PASS değil."
        )

    water_gate = safe_module[
        "load_water_gate"
    ](
        device_name
    )

    oil_pipeline = safe_module[
        "load_oil_pipeline"
    ](
        device_name
    )

    checkpoint = torch.load(
        VERIFIER_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    oil_pipeline[
        "verifier_model"
    ].load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )

    oil_pipeline[
        "verifier_model"
    ].to(
        oil_pipeline[
            "device"
        ]
    )

    oil_pipeline[
        "verifier_model"
    ].eval()

    oil_pipeline[
        "v07_config"
    ] = config

    return (
        regression,
        safe_module,
        water_gate,
        oil_pipeline,
    )


def create_panel(
    image: Image.Image,
    water_mask: np.ndarray,
    candidate_overlay: Image.Image,
    final_mask: np.ndarray,
    title: str,
    mask_to_image: Any,
) -> Image.Image:
    column_width = 300
    image_height = 300
    header_height = 44

    items = [
        image.convert("RGB"),
        mask_to_image(
            water_mask
        ).convert("RGB"),
        candidate_overlay.convert(
            "RGB"
        ),
        mask_to_image(
            final_mask
        ).convert("RGB"),
    ]

    panel = Image.new(
        "RGB",
        (
            column_width * 4,
            image_height
            + header_height,
        ),
        "black",
    )

    for index, item in enumerate(
        items
    ):
        copy = item.copy()

        copy.thumbnail(
            (
                column_width,
                image_height,
            ),
            Image.Resampling.LANCZOS,
        )

        x = (
            index
            * column_width
            + (
                column_width
                - copy.width
            )
            // 2
        )

        y = (
            header_height
            + (
                image_height
                - copy.height
            )
            // 2
        )

        panel.paste(
            copy,
            (
                x,
                y,
            ),
        )

    draw = ImageDraw.Draw(
        panel
    )

    draw.text(
        (
            8,
            14,
        ),
        title[:185],
        fill="white",
        font=ImageFont.load_default(),
    )

    return panel


def save_results(
    rows: list[dict[str, Any]],
) -> None:
    pd.DataFrame(
        rows
    ).to_csv(
        PER_SCENE_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    marker = load_json(
        CONSUMED_MARKER
    )

    marker[
        "processed_scene_count"
    ] = int(
        len(
            rows
        )
    )

    marker[
        "last_update_utc"
    ] = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    write_json(
        CONSUMED_MARKER,
        marker,
    )


def build_contact_sheet(
    holdout: pd.DataFrame,
) -> None:
    panels = []

    for scene_id in holdout[
        "scene_id"
    ].astype(str):
        panel_path = (
            PANEL_DIR
            / f"{safe_name(scene_id)}.jpg"
        )

        require_file(
            panel_path
        )

        panels.append(
            Image.open(
                panel_path
            ).convert("RGB")
        )

    sheet = Image.new(
        "RGB",
        (
            1200,
            sum(
                panel.height
                for panel in panels
            ),
        ),
        "black",
    )

    y = 0

    for panel in panels:
        sheet.paste(
            panel,
            (
                0,
                y,
            ),
        )

        y += panel.height

    sheet.save(
        CONTACT_SHEET_PATH,
        quality=90,
    )


def main() -> None:
    args = parse_args()

    for path in (
        HOLDOUT_MANIFEST,
        VERIFIER_CHECKPOINT,
        OPERATIONAL_CONFIG,
        REGRESSION_SCRIPT,
        SAFE_PIPELINE_SCRIPT,
    ):
        require_file(path)

    holdout = load_holdout()

    completed_scene_ids = prepare_run(
        args,
        holdout,
    )

    existing_rows: list[
        dict[str, Any]
    ] = []

    if PER_SCENE_PATH.exists():
        existing_rows = pd.read_csv(
            PER_SCENE_PATH,
            encoding="utf-8-sig",
            low_memory=False,
        ).to_dict(
            orient="records"
        )

    (
        regression,
        safe_module,
        water_gate,
        oil_pipeline,
    ) = load_v08_runtime(
        args.device
    )

    print("=" * 78)
    print(
        "v0.8 FRESH NEGATIVE HOLDOUT"
    )
    print("=" * 78)

    print(
        "Fresh scene:",
        len(
            holdout
        ),
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
        "Dondurulmuş v0.8 operational gate kullanılacak."
    )

    print(
        "Daha önce tamamlanan:",
        len(
            completed_scene_ids
        ),
    )

    print()

    rows = list(
        existing_rows
    )

    for index, row in enumerate(
        holdout.itertuples(),
        start=1,
    ):
        scene_id = str(
            row.scene_id
        )

        if scene_id in completed_scene_ids:
            print(
                f"[{index:02d}/{len(holdout):02d}] "
                f"{scene_id} | RESUME: daha önce tamamlandı"
            )

            continue

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
            f"| result={system_result} | C={confirmed_count} "
            f"U={uncertain_count} L={lookalike_count} "
            f"pixels={final_pixels}"
        )

        panel = create_panel(
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
            mask_to_image=(
                regression[
                    "mask_to_image"
                ]
            ),
        )

        panel.save(
            PANEL_DIR
            / f"{safe_name(scene_id)}.jpg",
            quality=90,
        )

        if record[
            "false_alarm"
        ]:
            scene_dir = (
                FALSE_ALARM_DIR
                / safe_name(
                    scene_id
                )
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

        save_results(
            rows
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

    if len(
        result_frame
    ) != len(
        holdout
    ):
        raise RuntimeError(
            "Bütün holdout sahneleri tamamlanmadı. "
            f"Sonuç: {len(result_frame)}/{len(holdout)}"
        )

    result_frame = (
        result_frame.drop_duplicates(
            subset=[
                "scene_id",
            ],
            keep="last",
        )
        .set_index(
            "scene_id"
        )
        .loc[
            holdout[
                "scene_id"
            ].astype(str)
        ]
        .reset_index()
    )

    result_frame.to_csv(
        PER_SCENE_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    build_contact_sheet(
        holdout
    )

    water_accepted_count = int(
        result_frame[
            "oil_model_executed"
        ].eq(
            True
        ).sum()
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
            "v08_fresh_negative_holdout"
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
        "water_blocked_scene_count": int(
            len(
                result_frame
            )
            - water_accepted_count
        ),
        "water_accepted_rate": float(
            water_accepted_count
            / len(
                result_frame
            )
        ),
        "confirmed_oil_false_alarm_scene_count": (
            false_alarm_scene_count
        ),
        "confirmed_oil_false_alarm_rate_overall": float(
            false_alarm_scene_count
            / len(
                result_frame
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
        "operational_config": relative(
            OPERATIONAL_CONFIG
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
            "Bu tek seferlik test yalnız tamamen yeni nc/nw DARTIS "
            "negatif sahnelerinde v0.6 water gate, petrol segmenteri ve "
            "v0.8 seçici verifier'ın CONFIRMED_OIL yanlış alarmını ölçer. "
            "v0.5 input gate bu aşamada yeniden test edilmemiştir. Pozitif "
            "taze holdout bulunmadığı için bu sonuç petrol recall kanıtı değildir."
        ),
    }

    write_json(
        SUMMARY_PATH,
        summary,
    )

    marker = load_json(
        CONSUMED_MARKER
    )

    marker.update(
        {
            "status": "COMPLETED",
            "completed_at_utc": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
            "processed_scene_count": int(
                len(
                    result_frame
                )
            ),
            "gate_passed": bool(
                gate_passed
            ),
            "confirmed_oil_false_alarm_scene_count": (
                false_alarm_scene_count
            ),
            "total_final_oil_pixels": (
                total_final_oil_pixels
            ),
            "summary": relative(
                SUMMARY_PATH
            ),
        }
    )

    write_json(
        CONSUMED_MARKER,
        marker,
    )

    print()
    print("=" * 78)
    print(
        "v0.8 FRESH NEGATIVE HOLDOUT NİHAİ SONUCU"
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
