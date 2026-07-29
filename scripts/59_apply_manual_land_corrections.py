from __future__ import annotations

from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SOURCE_MANIFEST = (
    ROOT / "data" / "metadata"
    / "v06_dartis_water_masks_homography.csv"
)

CORRECTIONS_MANIFEST = (
    ROOT / "data" / "metadata"
    / "v06_manual_land_corrections.csv"
)

OUTPUT_MANIFEST = (
    ROOT / "data" / "metadata"
    / "v06_dartis_water_masks_final.csv"
)

OUTPUT_SUMMARY = (
    ROOT / "outputs"
    / "v06_manual_land_corrections"
    / "apply_summary.json"
)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Gerekli dosya bulunamadı: {path}"
        )


def main() -> None:
    require_file(SOURCE_MANIFEST)
    require_file(CORRECTIONS_MANIFEST)

    source = pd.read_csv(
        SOURCE_MANIFEST,
        encoding="utf-8-sig",
        low_memory=False,
    )

    corrections = pd.read_csv(
        CORRECTIONS_MANIFEST,
        encoding="utf-8-sig",
        low_memory=False,
    )

    if source["sample_id"].duplicated().any():
        raise RuntimeError(
            "Kaynak manifestte tekrarlı sample_id var."
        )

    if corrections["sample_id"].duplicated().any():
        raise RuntimeError(
            "Düzeltme manifestinde tekrarlı sample_id var."
        )

    missing_ids = sorted(
        set(corrections["sample_id"])
        - set(source["sample_id"])
    )

    if missing_ids:
        raise RuntimeError(
            "Kaynak manifestte bulunmayan düzeltmeler:\n"
            + "\n".join(missing_ids)
        )

    result = source.copy()

    result["final_land_mask_path"] = (
        result["land_mask_path"]
    )
    result["final_safe_water_mask_path"] = (
        result["safe_water_mask_path"]
    )
    result["final_mask_source"] = (
        result.get(
            "mask_source",
            pd.Series(
                "GEOSPATIAL",
                index=result.index,
            ),
        )
    )
    result["manual_verified"] = False

    # Açık-su gruplarında kara yoktur; bütün alan su etiketi
    # DARTIS grup semantiğinden gelir.
    result["eligible_for_water_model"] = (
        result["surface_context"].eq("water")
    )

    corrections_index = corrections.set_index(
        "sample_id"
    )

    for index, row in result.iterrows():
        sample_id = row["sample_id"]

        if sample_id not in corrections_index.index:
            continue

        correction = corrections_index.loc[
            sample_id
        ]

        land_path = (
            ROOT
            / str(
                correction["land_mask_path"]
            )
        )

        water_path = (
            ROOT
            / str(
                correction[
                    "safe_water_mask_path"
                ]
            )
        )

        require_file(land_path)
        require_file(water_path)

        result.at[
            index,
            "final_land_mask_path",
        ] = str(
            correction["land_mask_path"]
        )

        result.at[
            index,
            "final_safe_water_mask_path",
        ] = str(
            correction[
                "safe_water_mask_path"
            ]
        )

        result.at[
            index,
            "final_mask_source",
        ] = "MANUAL_CORRECTION"

        result.at[
            index,
            "manual_verified",
        ] = True

        result.at[
            index,
            "eligible_for_water_model",
        ] = True

    OUTPUT_MANIFEST.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_SUMMARY.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_MANIFEST,
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "total_records": int(len(result)),
        "open_water_eligible": int(
            (
                result["surface_context"].eq("water")
                & result[
                    "eligible_for_water_model"
                ].astype(bool)
            ).sum()
        ),
        "manual_coastal_corrections": int(
            result[
                "manual_verified"
            ].astype(bool).sum()
        ),
        "other_coastal_excluded": int(
            (
                result["surface_context"].eq("coast")
                & ~result[
                    "manual_verified"
                ].astype(bool)
            ).sum()
        ),
        "output_manifest": str(
            OUTPUT_MANIFEST.resolve()
        ),
        "important_note": (
            "Elle doğrulanmamış diğer kıyı maskeleri "
            "water-model eğitiminden dışlandı."
        ),
    }

    OUTPUT_SUMMARY.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 78)
    print("v0.6 MANUEL KARA MASKESİ DÜZELTMELERİ UYGULANDI")
    print("=" * 78)
    print(
        "Toplam kayıt:",
        summary["total_records"],
    )
    print(
        "Açık su eligible:",
        summary["open_water_eligible"],
    )
    print(
        "Manuel kıyı düzeltmesi:",
        summary[
            "manual_coastal_corrections"
        ],
    )
    print(
        "Diğer kıyılılar eğitim dışı:",
        summary[
            "other_coastal_excluded"
        ],
    )
    print(
        "Final manifest:",
        OUTPUT_MANIFEST.resolve(),
    )


if __name__ == "__main__":
    main()
