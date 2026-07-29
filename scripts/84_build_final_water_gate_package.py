from __future__ import annotations

import argparse
import hashlib
import json
import py_compile
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]

SOURCE_MODEL_DEFINITION = (
    ROOT
    / "scripts"
    / "68_evaluate_dartis_water_transfer_gate.py"
)

SOURCE_RUNTIME = (
    ROOT
    / "scripts"
    / "84_water_gate_runtime.py"
)

SOURCE_CHECKPOINT_DIR = (
    ROOT
    / "checkpoints"
    / "water_unet_dartis_combined_landaware_cv_v06"
)

SOURCE_POLICY_SUMMARY = (
    ROOT
    / "outputs"
    / "v06_dartis_selective_water_gate"
    / "summary.json"
)

SOURCE_LOCKED_SUMMARY = (
    ROOT
    / "outputs"
    / "v06_dartis_selective_water_locked_test"
    / "summary.json"
)

SOURCE_LOCKED_PER_SCENE = (
    ROOT
    / "outputs"
    / "v06_dartis_selective_water_locked_test"
    / "locked_test_per_scene.csv"
)

SOURCE_LOCKED_CONTACT_SHEET = (
    ROOT
    / "outputs"
    / "v06_dartis_selective_water_locked_test"
    / "locked_test_contact_sheet.jpg"
)

SOURCE_CONSUMED_MARKER = (
    ROOT
    / "data"
    / "metadata"
    / "v06_dartis_locked_test_consumed.json"
)

PACKAGE_DIR = (
    ROOT
    / "checkpoints"
    / "final_water_gate_v06"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Kilitli testi geçen v0.6 seçici kara-su "
            "güvenlik modülünü dondurur."
        )
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()


def checkpoint_metadata(
    path: Path,
) -> dict[str, Any]:
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    threshold = float(
        checkpoint.get(
            "validation_metrics",
            {},
        ).get(
            "threshold",
            0.5,
        )
    )

    return {
        "calibration_threshold": threshold,
        "best_epoch": int(
            checkpoint.get("best_epoch", 0)
        ),
    }


def main() -> None:
    args = parse_args()

    model_sources = [
        SOURCE_CHECKPOINT_DIR
        / f"fold_{index:02d}_best.pth"
        for index in range(1, 6)
    ]

    required = [
        SOURCE_MODEL_DEFINITION,
        SOURCE_RUNTIME,
        SOURCE_POLICY_SUMMARY,
        SOURCE_LOCKED_SUMMARY,
        SOURCE_LOCKED_PER_SCENE,
        SOURCE_LOCKED_CONTACT_SHEET,
        SOURCE_CONSUMED_MARKER,
        *model_sources,
    ]

    missing = [
        path
        for path in required
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Eksik kaynaklar:\n"
            + "\n".join(
                str(path)
                for path in missing
            )
        )

    policy_summary = json.loads(
        SOURCE_POLICY_SUMMARY.read_text(
            encoding="utf-8-sig"
        )
    )

    locked_summary = json.loads(
        SOURCE_LOCKED_SUMMARY.read_text(
            encoding="utf-8-sig"
        )
    )

    consumed = json.loads(
        SOURCE_CONSUMED_MARKER.read_text(
            encoding="utf-8-sig"
        )
    )

    if not policy_summary.get(
        "policy_passed",
        False,
    ):
        raise RuntimeError(
            "Development policy PASS değil."
        )

    if not locked_summary.get(
        "final_gate_passed",
        False,
    ):
        raise RuntimeError(
            "Kilitli test final gate PASS değil."
        )

    if not locked_summary.get(
        "locked_test_consumed",
        False,
    ):
        raise RuntimeError(
            "Kilitli test tüketilmemiş görünüyor."
        )

    if not consumed.get(
        "final_gate_passed",
        False,
    ):
        raise RuntimeError(
            "Consumed marker PASS değil."
        )

    if PACKAGE_DIR.exists():
        if not args.force:
            raise FileExistsError(
                "Paket zaten var. Yeniden oluşturmak "
                "için --force kullanın."
            )

        shutil.rmtree(PACKAGE_DIR)

    PACKAGE_DIR.mkdir(
        parents=True,
        exist_ok=False,
    )

    shutil.copy2(
        SOURCE_MODEL_DEFINITION,
        PACKAGE_DIR
        / "model_definition.py",
    )

    shutil.copy2(
        SOURCE_RUNTIME,
        PACKAGE_DIR
        / "water_gate_runtime.py",
    )

    policy = dict(
        policy_summary["selected_policy"]
    )

    (
        PACKAGE_DIR
        / "policy.json"
    ).write_text(
        json.dumps(
            policy,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    model_entries = []

    for index, source_path in enumerate(
        model_sources,
        start=1,
    ):
        destination = (
            PACKAGE_DIR
            / f"fold_{index:02d}_best.pth"
        )

        shutil.copy2(
            source_path,
            destination,
        )

        metadata = checkpoint_metadata(
            destination
        )

        model_entries.append(
            {
                "fold": index,
                "file": destination.name,
                "sha256": sha256(destination),
                **metadata,
            }
        )

    reports_dir = (
        PACKAGE_DIR
        / "validation_reports"
    )
    reports_dir.mkdir()

    shutil.copy2(
        SOURCE_POLICY_SUMMARY,
        reports_dir
        / "development_policy_summary.json",
    )
    shutil.copy2(
        SOURCE_LOCKED_SUMMARY,
        reports_dir
        / "locked_test_summary.json",
    )
    shutil.copy2(
        SOURCE_LOCKED_PER_SCENE,
        reports_dir
        / "locked_test_per_scene.csv",
    )
    shutil.copy2(
        SOURCE_LOCKED_CONTACT_SHEET,
        reports_dir
        / "locked_test_contact_sheet.jpg",
    )
    shutil.copy2(
        SOURCE_CONSUMED_MARKER,
        reports_dir
        / "locked_test_consumed.json",
    )

    package_manifest = {
        "package": "final_water_gate_v06",
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "model_count": 5,
        "models": model_entries,
        "policy": policy,
        "runtime": "water_gate_runtime.py",
        "model_definition": "model_definition.py",
        "decision_contract": {
            "ACCEPT_WATER": (
                "safe_water_mask.png petrol analizine "
                "aktarılabilir."
            ),
            "UNCERTAIN_BLOCK": (
                "Petrol analizi durdurulur ve operasyonel "
                "su maskesi siyah kalır."
            ),
        },
        "locked_test_result": {
            "final_gate_passed": True,
            "accepted_count": (
                locked_summary["accepted_count"]
            ),
            "locked_test_count": (
                locked_summary["locked_test_count"]
            ),
            "accepted_scene_metrics": (
                locked_summary[
                    "accepted_scene_metrics"
                ]
            ),
        },
    }

    manifest_path = (
        PACKAGE_DIR
        / "package_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            package_manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    py_compile.compile(
        str(
            PACKAGE_DIR
            / "water_gate_runtime.py"
        ),
        doraise=True,
    )

    print("=" * 78)
    print("FINAL WATER GATE PAKETİ HAZIR")
    print("=" * 78)
    print("Paket:", PACKAGE_DIR.resolve())
    print("Model:", len(model_entries))
    print(
        "Runtime:",
        (
            PACKAGE_DIR
            / "water_gate_runtime.py"
        ).resolve(),
    )
    print("Manifest:", manifest_path.resolve())
    print(
        "Locked test:",
        f"{locked_summary['accepted_count']}/"
        f"{locked_summary['locked_test_count']}",
    )
    print("Final gate: PASS")


if __name__ == "__main__":
    main()
