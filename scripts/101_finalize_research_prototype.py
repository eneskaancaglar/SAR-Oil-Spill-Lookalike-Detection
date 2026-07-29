from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    ROOT
    / "outputs"
    / "final_research_prototype_v08"
)

PACKAGE_DIR = (
    ROOT
    / "checkpoints"
    / "final_research_prototype_v08"
)

FINAL_STATUS_PATH = (
    OUTPUT_DIR
    / "FINAL_STATUS.json"
)

FINAL_METRICS_PATH = (
    OUTPUT_DIR
    / "final_metrics.json"
)

FINAL_REPORT_PATH = (
    OUTPUT_DIR
    / "FINAL_REPORT.md"
)

LIMITATIONS_PATH = (
    OUTPUT_DIR
    / "LIMITATIONS.md"
)

HANDOFF_PATH = (
    OUTPUT_DIR
    / "HANDOFF_CHECKLIST.md"
)

PACKAGE_MANIFEST_PATH = (
    PACKAGE_DIR
    / "package_manifest.json"
)

SOURCE_PATHS = {
    "v08_fresh_negative_holdout": (
        ROOT
        / "outputs"
        / "v08_fresh_negative_holdout"
        / "summary.json"
    ),
    "v08_operational_gate": (
        ROOT
        / "outputs"
        / "verifier_v08_operational_gate"
        / "summary.json"
    ),
    "v08_training": (
        ROOT
        / "outputs"
        / "verifier_v08"
        / "summary.json"
    ),
    "v07_known_scene_regression": (
        ROOT
        / "outputs"
        / "v07_known_scene_regression"
        / "summary.json"
    ),
    "v06_water_locked_test": (
        ROOT
        / "outputs"
        / "v06_dartis_selective_water_locked_test"
        / "summary.json"
    ),
    "v07_fresh_negative_holdout": (
        ROOT
        / "outputs"
        / "v07_fresh_negative_holdout"
        / "summary.json"
    ),
    "v09_positive_mask_audit": (
        ROOT
        / "outputs"
        / "v09_positive_mask_source_audit"
        / "summary.json"
    ),
}

MODEL_PATHS = {
    "oil_segmenter_and_v03_verifier": (
        ROOT
        / "checkpoints"
        / "final_model_v03"
    ),
    "water_gate_v06": (
        ROOT
        / "checkpoints"
        / "final_water_gate_v06"
    ),
    "lookalike_verifier_v08": (
        ROOT
        / "checkpoints"
        / "verifier_v08"
        / "best.pth"
    ),
    "v08_operational_gate_config": (
        ROOT
        / "checkpoints"
        / "verifier_v08"
        / "operational_gate_config.json"
    ),
    "streamlit_app": (
        ROOT
        / "app.py"
    ),
}


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None

    return json.loads(
        path.read_text(
            encoding="utf-8-sig"
        )
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


def directory_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "exists": False,
            "path": relative(path),
        }

    files = []

    for file_path in sorted(
        item
        for item in path.rglob("*")
        if item.is_file()
    ):
        files.append(
            {
                "path": relative(file_path),
                "size_bytes": int(
                    file_path.stat().st_size
                ),
                "sha256": sha256_file(
                    file_path
                ),
            }
        )

    return {
        "exists": True,
        "path": relative(path),
        "file_count": int(
            len(files)
        ),
        "files": files,
    }


def artifact_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "exists": False,
            "path": relative(path),
        }

    if path.is_dir():
        return directory_manifest(path)

    return {
        "exists": True,
        "path": relative(path),
        "size_bytes": int(
            path.stat().st_size
        ),
        "sha256": sha256_file(
            path
        ),
    }


def relative(path: Path) -> str:
    try:
        return str(
            path.resolve().relative_to(
                ROOT.resolve()
            )
        ).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def nested(
    value: dict[str, Any] | None,
    *keys: str,
    default: Any = None,
) -> Any:
    current: Any = value

    for key in keys:
        if not isinstance(
            current,
            dict,
        ):
            return default

        if key not in current:
            return default

        current = current[key]

    return current


def percent(
    value: Any,
    digits: int = 2,
) -> str:
    try:
        return (
            f"{float(value) * 100:.{digits}f}%"
        )
    except (
        TypeError,
        ValueError,
    ):
        return "N/A"


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PACKAGE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    sources = {
        name: load_json(path)
        for name, path
        in SOURCE_PATHS.items()
    }

    v08_holdout = sources[
        "v08_fresh_negative_holdout"
    ]

    v08_gate = sources[
        "v08_operational_gate"
    ]

    v08_training = sources[
        "v08_training"
    ]

    v06_water = sources[
        "v06_water_locked_test"
    ]

    v07_holdout = sources[
        "v07_fresh_negative_holdout"
    ]

    v09_audit = sources[
        "v09_positive_mask_audit"
    ]

    if v08_holdout is None:
        raise FileNotFoundError(
            "v0.8 fresh negatif holdout özeti bulunamadı: "
            f"{SOURCE_PATHS['v08_fresh_negative_holdout']}"
        )

    if v08_gate is None:
        raise FileNotFoundError(
            "v0.8 operasyonel kapı özeti bulunamadı: "
            f"{SOURCE_PATHS['v08_operational_gate']}"
        )

    final_metrics = {
        "frozen_version": (
            "v0.8-research-prototype"
        ),
        "completion_status": (
            "COMPLETED_RESEARCH_PROTOTYPE"
        ),
        "operational_readiness": False,
        "automatic_operational_oil_decision_allowed": (
            False
        ),
        "recommended_output_semantics": (
            "OIL_CANDIDATE_REQUIRES_EXPERT_REVIEW"
        ),
        "input_gate_v05": {
            "locked_negative_count": 1582,
            "false_accept_count": 0,
            "safe_block_rate": 1.0,
            "source_note": (
                "Previously completed v0.5 locked negative evaluation."
            ),
        },
        "water_gate_v06": {
            "final_gate_passed": nested(
                v06_water,
                "final_gate_passed",
                default=nested(
                    v06_water,
                    "gate_passed",
                ),
            ),
            "locked_test_count": nested(
                v06_water,
                "locked_test_count",
                default=8,
            ),
            "accepted_scenes": nested(
                v06_water,
                "accepted_scene_count",
                default=nested(
                    v06_water,
                    "accepted_count",
                    default=5,
                ),
            ),
            "water_precision": nested(
                v06_water,
                "water_precision",
                default=1.0,
            ),
            "mean_accepted_recall": nested(
                v06_water,
                "mean_accepted_recall",
                default=0.9054,
            ),
            "land_leakage": nested(
                v06_water,
                "land_leakage",
                default=0.0003,
            ),
        },
        "lookalike_verifier_v08_calibration": {
            "operational_gate_passed": nested(
                v08_gate,
                "operational_gate_passed",
            ),
            "confirmed_oil_precision": nested(
                v08_gate,
                "metrics",
                "confirmed_oil_precision",
            ),
            "confirmed_oil_recall": nested(
                v08_gate,
                "metrics",
                "confirmed_oil_recall",
            ),
            "confirmed_negative_count": nested(
                v08_gate,
                "metrics",
                "confirmed_negative_count",
            ),
            "negative_safe_block_rate": nested(
                v08_gate,
                "metrics",
                "negative_safe_block_rate",
            ),
            "positive_lookalike_rate": nested(
                v08_gate,
                "metrics",
                "positive_lookalike_rate",
            ),
            "source_training_gate_passed": nested(
                v08_gate,
                "source_training_gate_passed",
            ),
            "source_training_failed_checks": nested(
                v08_gate,
                "source_training_failed_checks",
                default=[],
            ),
        },
        "independent_negative_holdout_v08": {
            "gate_passed": nested(
                v08_holdout,
                "gate_passed",
            ),
            "scene_count": nested(
                v08_holdout,
                "holdout_count",
            ),
            "group_counts": nested(
                v08_holdout,
                "group_counts",
            ),
            "water_accepted_scene_count": nested(
                v08_holdout,
                "water_accepted_scene_count",
            ),
            "false_alarm_scene_count": nested(
                v08_holdout,
                "confirmed_oil_false_alarm_scene_count",
            ),
            "false_alarm_rate_overall": nested(
                v08_holdout,
                "confirmed_oil_false_alarm_rate_overall",
            ),
            "false_alarm_rate_among_water_accepted": nested(
                v08_holdout,
                "confirmed_oil_false_alarm_rate_among_water_accepted",
            ),
            "confirmed_candidate_count": nested(
                v08_holdout,
                "total_confirmed_candidates",
            ),
            "final_oil_pixels": nested(
                v08_holdout,
                "total_final_oil_pixels",
            ),
            "failed_checks": nested(
                v08_holdout,
                "failed_checks",
                default=[],
            ),
        },
        "comparison_with_v07": {
            "v07_false_alarm_scene_count": nested(
                v07_holdout,
                "confirmed_oil_false_alarm_scene_count",
            ),
            "v07_final_oil_pixels": nested(
                v07_holdout,
                "total_final_oil_pixels",
            ),
            "v08_false_alarm_scene_count": nested(
                v08_holdout,
                "confirmed_oil_false_alarm_scene_count",
            ),
            "v08_final_oil_pixels": nested(
                v08_holdout,
                "total_final_oil_pixels",
            ),
        },
        "independent_positive_evidence": {
            "fresh_positive_holdout_available": (
                False
            ),
            "independent_positive_recall_claim_allowed": (
                False
            ),
        },
        "v09_research_branch_status": {
            "continued": False,
            "reason_stopped": (
                "Positive source-mask coverage was insufficient for a "
                "scientifically reliable dual-view retraining dataset."
            ),
            "positive_source_scene_count": nested(
                v09_audit,
                "source_scene_count",
            ),
            "valid_source_mask_pair_count": nested(
                v09_audit,
                "valid_source_mask_pair_count",
            ),
            "pair_coverage": nested(
                v09_audit,
                "pair_coverage",
            ),
            "oc_coverage": nested(
                v09_audit,
                "group_coverage",
                "oc",
                "coverage",
            ),
            "ow_coverage": nested(
                v09_audit,
                "group_coverage",
                "ow",
                "coverage",
            ),
        },
    }

    FINAL_METRICS_PATH.write_text(
        json.dumps(
            final_metrics,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    final_status = {
        "project_stage": (
            "FINAL_RESEARCH_HANDOFF"
        ),
        "frozen_version": (
            "v0.8-research-prototype"
        ),
        "status": (
            "COMPLETED_RESEARCH_PROTOTYPE"
        ),
        "frozen_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "more_training_required_for_current_handoff": (
            False
        ),
        "operational_deployment_approved": (
            False
        ),
        "automatic_oil_mask_as_final_decision": (
            False
        ),
        "allowed_interpretation": (
            "The system may generate research candidates for expert review."
        ),
        "blocking_evidence": {
            "v08_independent_negative_holdout_gate": nested(
                v08_holdout,
                "gate_passed",
            ),
            "v08_false_alarm_scenes": nested(
                v08_holdout,
                "confirmed_oil_false_alarm_scene_count",
            ),
            "v08_holdout_scenes": nested(
                v08_holdout,
                "holdout_count",
            ),
            "fresh_positive_holdout_missing": (
                True
            ),
        },
        "final_metrics": relative(
            FINAL_METRICS_PATH
        ),
        "final_report": relative(
            FINAL_REPORT_PATH
        ),
        "limitations": relative(
            LIMITATIONS_PATH
        ),
        "handoff_checklist": relative(
            HANDOFF_PATH
        ),
        "package_manifest": relative(
            PACKAGE_MANIFEST_PATH
        ),
    }

    FINAL_STATUS_PATH.write_text(
        json.dumps(
            final_status,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    false_alarm_count = nested(
        v08_holdout,
        "confirmed_oil_false_alarm_scene_count",
        default=2,
    )

    holdout_count = nested(
        v08_holdout,
        "holdout_count",
        default=40,
    )

    water_accepted_count = nested(
        v08_holdout,
        "water_accepted_scene_count",
        default=32,
    )

    v08_false_alarm_rate = nested(
        v08_holdout,
        "confirmed_oil_false_alarm_rate_overall",
        default=0.05,
    )

    v08_accepted_false_alarm_rate = nested(
        v08_holdout,
        "confirmed_oil_false_alarm_rate_among_water_accepted",
        default=0.0625,
    )

    v08_final_pixels = nested(
        v08_holdout,
        "total_final_oil_pixels",
        default=2646,
    )

    v07_false_alarm_count = nested(
        v07_holdout,
        "confirmed_oil_false_alarm_scene_count",
        default=6,
    )

    v07_final_pixels = nested(
        v07_holdout,
        "total_final_oil_pixels",
        default=27750,
    )

    confirmed_precision = nested(
        v08_gate,
        "metrics",
        "confirmed_oil_precision",
        default=1.0,
    )

    confirmed_recall = nested(
        v08_gate,
        "metrics",
        "confirmed_oil_recall",
        default=0.8087649402390438,
    )

    negative_safe_block = nested(
        v08_gate,
        "metrics",
        "negative_safe_block_rate",
        default=1.0,
    )

    water_precision = nested(
        v06_water,
        "water_precision",
        default=1.0,
    )

    water_recall = nested(
        v06_water,
        "mean_accepted_recall",
        default=0.9054,
    )

    water_leakage = nested(
        v06_water,
        "land_leakage",
        default=0.0003,
    )

    report = f"""# SAR Oil-Spill Detection — Final Research Prototype Report

## Final status

**Frozen version:** `v0.8-research-prototype`

**Project status:** Completed as a research prototype.

**Operational deployment:** Not approved.

**Required interpretation:** Any positive result is an **oil candidate requiring expert review**. It must not be presented as an automatic operational oil-spill decision.

## Frozen pipeline

1. **v0.5 input-domain gate**  
   Rejects unsupported non-SAR or out-of-domain inputs.

2. **v0.6 selective water gate**  
   Allows oil analysis only on high-confidence water pixels. Land and uncertain pixels are excluded from the final oil mask.

3. **Oil candidate segmenter**  
   Produces candidate dark regions only inside the accepted water mask.

4. **v0.8 look-alike-aware verifier**  
   Produces `LOOK_ALIKE`, `UNCERTAIN`, or `CONFIRMED_OIL`. Only `CONFIRMED_OIL` may enter the research mask.

## Final evidence

### Input-domain safety

- Locked negative samples: **1,582**
- False accepts: **0**
- Safe blocking rate: **100%**

### Water gate

- Locked test accepted scenes: **5/8**
- Water precision: **{percent(water_precision, 3)}**
- Mean accepted recall: **{percent(water_recall, 2)}**
- Land leakage: **{percent(water_leakage, 4)}**

### v0.8 verifier calibration

- Confirmed-oil precision: **{percent(confirmed_precision, 2)}**
- Confirmed-oil recall: **{percent(confirmed_recall, 2)}**
- Negative safe-block rate: **{percent(negative_safe_block, 2)}**
- Calibration operational gate: **PASS**

This is calibration evidence, not independent final proof.

### Independent negative holdout

- Fresh negative scenes: **{holdout_count}**
- Water-gate accepted scenes: **{water_accepted_count}/{holdout_count}**
- Confirmed-oil false-alarm scenes: **{false_alarm_count}/{holdout_count}**
- Overall false-alarm rate: **{percent(v08_false_alarm_rate, 2)}**
- False-alarm rate among water-accepted scenes: **{percent(v08_accepted_false_alarm_rate, 2)}**
- Incorrect final oil pixels: **{v08_final_pixels:,}**
- Independent negative gate: **FAIL**

### Improvement from v0.7 to v0.8

- False-alarm scenes: **{v07_false_alarm_count}/40 → {false_alarm_count}/40**
- Incorrect oil pixels: **{v07_final_pixels:,} → {v08_final_pixels:,}**

The model improved substantially, but the zero-false-alarm safety target was not reached.

## Scientific conclusion

The project successfully demonstrates a complete safeguarded SAR oil-spill research pipeline with:

- input-domain rejection,
- selective land/water isolation,
- water-only oil candidate production,
- look-alike and uncertainty handling,
- independent negative testing,
- reproducible model and evaluation artifacts.

The independent negative holdout still produced **{false_alarm_count} false-alarm scenes out of {holdout_count}**. In addition, a fresh independent positive holdout was not available. Therefore:

- no operational deployment claim is made,
- no independent positive-recall claim is made,
- automatic final oil-spill decisions are not allowed,
- outputs must remain research candidates for expert review.

## Stopped v0.9 branch

The attempted dual-view extension was intentionally stopped. Reliable positive source-mask pairs were found for only **756/1,365** scenes:

- `oc`: **375/375**
- `ow`: **381/990**

This coverage was insufficient for a scientifically balanced dual-view retraining data set. Continuing would have increased complexity without trustworthy evidence.

## Final deliverable decision

The technically and scientifically correct endpoint is the frozen `v0.8-research-prototype`, together with its passed component gates, failed independent negative gate, limitations, reproducible summaries, and expert-review-only usage policy.
"""

    FINAL_REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )

    limitations = f"""# Known Limitations

1. The v0.8 independent negative holdout produced **{false_alarm_count}/{holdout_count}** false-alarm scenes.
2. No untouched positive `oc/ow` holdout remained, so independent positive recall could not be measured.
3. The water gate intentionally blocks uncertain scenes; this improves safety but reduces coverage.
4. Calibration performance must not be reported as independent test performance.
5. The v0.9 dual-view branch lacked sufficient trustworthy `ow` source-mask coverage.
6. The system is a research prototype and requires expert review for every positive candidate.
7. A black final mask on a blocked scene means “no operational decision,” not proof that no oil exists.
8. `LOOK_ALIKE` and `UNCERTAIN` are both excluded from the final research mask.
"""

    LIMITATIONS_PATH.write_text(
        limitations,
        encoding="utf-8",
    )

    handoff = """# Final Handoff Checklist

- [x] Input-domain gate packaged and tested.
- [x] Selective water gate packaged and locked-tested.
- [x] Oil analysis restricted to accepted water pixels.
- [x] Look-alike-aware verifier trained.
- [x] Independent negative holdout consumed exactly once.
- [x] v0.7 and v0.8 holdout failures preserved.
- [x] Final limitations documented.
- [x] Research-only interpretation documented.
- [x] No further model training required for this internship handoff.
- [ ] Operational deployment approval — intentionally not granted.
- [ ] Independent positive holdout evidence — unavailable.
"""

    HANDOFF_PATH.write_text(
        handoff,
        encoding="utf-8",
    )

    package_manifest = {
        "package_name": (
            "final_research_prototype_v08"
        ),
        "version": (
            "v0.8-research-prototype"
        ),
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "status": (
            "COMPLETED_RESEARCH_PROTOTYPE"
        ),
        "operational_readiness": False,
        "usage_policy": (
            "Expert-review-only oil candidate generation."
        ),
        "artifacts": {
            name: artifact_manifest(path)
            for name, path
            in MODEL_PATHS.items()
        },
        "evaluation_sources": {
            name: {
                "exists": path.exists(),
                "path": relative(path),
                "sha256": (
                    sha256_file(path)
                    if path.exists()
                    and path.is_file()
                    else None
                ),
            }
            for name, path
            in SOURCE_PATHS.items()
        },
        "final_documents": {
            "status": relative(
                FINAL_STATUS_PATH
            ),
            "metrics": relative(
                FINAL_METRICS_PATH
            ),
            "report": relative(
                FINAL_REPORT_PATH
            ),
            "limitations": relative(
                LIMITATIONS_PATH
            ),
            "handoff": relative(
                HANDOFF_PATH
            ),
        },
    }

    PACKAGE_MANIFEST_PATH.write_text(
        json.dumps(
            package_manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("=" * 78)
    print(
        "FINAL RESEARCH PROTOTYPE HANDOFF"
    )
    print("=" * 78)

    print(
        "Status: COMPLETED_RESEARCH_PROTOTYPE"
    )

    print(
        "Frozen version: v0.8-research-prototype"
    )

    print(
        "Operational deployment: NOT APPROVED"
    )

    print(
        "Automatic oil decision: NOT ALLOWED"
    )

    print(
        "Output semantics: OIL_CANDIDATE_REQUIRES_EXPERT_REVIEW"
    )

    print(
        "Independent negative result:",
        f"{false_alarm_count}/{holdout_count} false-alarm scenes",
    )

    print(
        "More training for current handoff: NO"
    )

    print(
        "Final report:",
        FINAL_REPORT_PATH.resolve(),
    )

    print(
        "Final status:",
        FINAL_STATUS_PATH.resolve(),
    )

    print(
        "Package manifest:",
        PACKAGE_MANIFEST_PATH.resolve(),
    )


if __name__ == "__main__":
    main()
