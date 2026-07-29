from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
INPUT_MANIFEST = ROOT / "data" / "metadata" / "v06_dartis_water_masks_homography.csv"
OUTPUT_ROOT = ROOT / "data" / "derived" / "v06_water_masks_refined"
LAND_ROOT = OUTPUT_ROOT / "land"
SAFE_WATER_ROOT = OUTPUT_ROOT / "safe_water"
OUTPUT_MANIFEST = ROOT / "data" / "metadata" / "v06_dartis_water_masks_refined.csv"
OUTPUT_DIR = ROOT / "outputs" / "v06_dartis_water_masks_refined"
REVIEW_DIR = OUTPUT_DIR / "review"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
STATUS_PATH = OUTPUT_DIR / "status_counts.csv"
FAILURES_PATH = OUTPUT_DIR / "failures.csv"
REPORT_PATH = ROOT / "reports" / "v06_dartis_water_masks_refined.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uncertain-band", type=int, default=48)
    parser.add_argument("--coast-buffer-pixels", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--minimum-component-pixels", type=int, default=64)
    parser.add_argument("--review-count", type=int, default=160)
    parser.add_argument(
        "--sample-id",
        action="append",
        default=[],
        help=(
            "Yalnız verilen sample_id üzerinde çalışır. "
            "Birden fazla kez kullanılabilir."
        ),
    )
    return parser.parse_args()


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def resolve_path(value: Any) -> Path:
    path = Path(str(value).strip())
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8)


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > 127


def save_mask(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(path)


def ellipse_kernel(radius: int) -> np.ndarray:
    radius = max(int(radius), 1)
    size = radius * 2 + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def erode(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool, copy=True)
    return cv2.erode(mask.astype(np.uint8), ellipse_kernel(radius), iterations=1) > 0


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool, copy=True)
    return cv2.dilate(mask.astype(np.uint8), ellipse_kernel(radius), iterations=1) > 0


def boundary(mask: np.ndarray) -> np.ndarray:
    return dilate(mask, 1) & ~erode(mask, 1)


def normalize_uint8(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros(array.shape, dtype=np.uint8)
    low = float(np.quantile(finite, 0.01))
    high = float(np.quantile(finite, 0.99))
    if high <= low + 1e-8:
        return np.zeros(array.shape, dtype=np.uint8)
    return (np.clip((array - low) / (high - low), 0.0, 1.0) * 255.0).astype(np.uint8)


def build_feature_image(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    local_mean = cv2.GaussianBlur(equalized, (0, 0), sigmaX=3.0, sigmaY=3.0)
    image_float = equalized.astype(np.float32)
    mean = cv2.GaussianBlur(image_float, (0, 0), sigmaX=4.0, sigmaY=4.0)
    mean_square = cv2.GaussianBlur(image_float * image_float, (0, 0), sigmaX=4.0, sigmaY=4.0)
    local_std = np.sqrt(np.maximum(mean_square - mean * mean, 0.0))
    std_uint8 = normalize_uint8(local_std)
    return cv2.merge([equalized, local_mean, std_uint8]), std_uint8


def border_connected_water_seed(gray: np.ndarray, texture: np.ndarray) -> np.ndarray:
    height, width = gray.shape
    border = np.zeros((height, width), dtype=bool)
    border_width = max(8, min(height, width) // 32)
    border[:border_width, :] = True
    border[-border_width:, :] = True
    border[:, :border_width] = True
    border[:, -border_width:] = True
    candidate = (
        (texture <= float(np.quantile(texture, 0.45)))
        & (gray <= float(np.quantile(gray, 0.75)))
    ).astype(np.uint8)
    count, labels, _, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    seed = np.zeros_like(candidate, dtype=bool)
    for label in range(1, count):
        component = labels == label
        if (component & border).any():
            seed |= component
    return erode(seed, 2)


def remove_tiny_components(
    land: np.ndarray,
    prior_land: np.ndarray,
    minimum_pixels: int,
) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        land.astype(np.uint8),
        connectivity=8,
    )
    output = np.zeros_like(land, dtype=bool)
    for label in range(1, count):
        component = labels == label
        area = int(stats[label, cv2.CC_STAT_AREA])
        overlap = int((component & prior_land).sum())
        if area >= minimum_pixels or overlap >= max(16, minimum_pixels // 4):
            output |= component
    return output


def boundary_distance_statistics(
    first_mask: np.ndarray,
    second_mask: np.ndarray,
) -> tuple[float, float]:
    first_boundary = boundary(first_mask)
    second_boundary = boundary(second_mask)
    if not first_boundary.any() or not second_boundary.any():
        return float("inf"), float("inf")
    target = np.where(second_boundary, 0, 255).astype(np.uint8)
    distance = cv2.distanceTransform(target, cv2.DIST_L2, 3)
    values = distance[first_boundary]
    reverse_target = np.where(first_boundary, 0, 255).astype(np.uint8)
    reverse_distance = cv2.distanceTransform(reverse_target, cv2.DIST_L2, 3)
    reverse_values = reverse_distance[second_boundary]
    combined = np.concatenate([values, reverse_values])
    return float(np.median(combined)), float(np.quantile(combined, 0.95))


def refine_coastal_mask(
    gray: np.ndarray,
    prior_land: np.ndarray,
    uncertain_band: int,
    iterations: int,
    minimum_component_pixels: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    feature_image, texture = build_feature_image(gray)
    half_band = max(8, uncertain_band // 2)
    definite_land = erode(prior_land, half_band)
    definite_water = erode(~prior_land, half_band)
    if int(definite_land.sum()) < 100:
        definite_land = erode(prior_land, 4)
    if int(definite_water.sum()) < 100:
        definite_water = erode(~prior_land, 4)
    fallback_water_used = False
    if int(definite_water.sum()) < 100:
        definite_water |= border_connected_water_seed(gray, texture)
        fallback_water_used = True
    if int(definite_land.sum()) < 100:
        raise RuntimeError("Yeterli güvenli kara tohumu üretilemedi.")
    if int(definite_water.sum()) < 100:
        raise RuntimeError("Yeterli güvenli su tohumu üretilemedi.")

    gc_mask = np.where(prior_land, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
    gc_mask[definite_land] = cv2.GC_FGD
    gc_mask[definite_water] = cv2.GC_BGD
    bg_model = np.zeros((1, 65), dtype=np.float64)
    fg_model = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(
        feature_image,
        gc_mask,
        None,
        bg_model,
        fg_model,
        int(iterations),
        cv2.GC_INIT_WITH_MASK,
    )
    refined_land = np.isin(gc_mask, [cv2.GC_FGD, cv2.GC_PR_FGD])
    refined_land[definite_land] = True
    refined_land[definite_water] = False
    refined_land = cv2.morphologyEx(
        refined_land.astype(np.uint8),
        cv2.MORPH_CLOSE,
        ellipse_kernel(2),
        iterations=1,
    ) > 0
    refined_land = cv2.morphologyEx(
        refined_land.astype(np.uint8),
        cv2.MORPH_OPEN,
        ellipse_kernel(1),
        iterations=1,
    ) > 0
    refined_land = remove_tiny_components(
        refined_land,
        prior_land,
        minimum_component_pixels,
    )
    refined_land[definite_land] = True
    refined_land[definite_water] = False

    prior_ratio = float(prior_land.mean())
    refined_ratio = float(refined_land.mean())
    median_shift, p95_shift = boundary_distance_statistics(prior_land, refined_land)
    return refined_land, {
        "prior_land_ratio": prior_ratio,
        "refined_land_ratio": refined_ratio,
        "land_ratio_change": float(refined_ratio - prior_ratio),
        "absolute_land_ratio_change": float(abs(refined_ratio - prior_ratio)),
        "boundary_median_change_pixels": median_shift,
        "boundary_p95_change_pixels": p95_shift,
        "definite_land_pixels": int(definite_land.sum()),
        "definite_water_pixels": int(definite_water.sum()),
        "fallback_water_seed_used": bool(fallback_water_used),
    }


def determine_status(
    metrics: dict[str, Any],
    safe_water_ratio: float,
) -> tuple[str, str]:
    if (
        not np.isfinite(metrics["boundary_p95_change_pixels"])
        or safe_water_ratio <= 0.005
        or safe_water_ratio >= 0.995
    ):
        return "REVIEW_REQUIRED", "Tek sınıfa yakın maske veya geçersiz sınır."
    reasons = []
    if metrics["absolute_land_ratio_change"] > 0.35:
        reasons.append("Kara alan oranı çok fazla değişti")
    if metrics["boundary_p95_change_pixels"] > 60:
        reasons.append("Kıyı sınırı çok büyük yerel değişim gösterdi")
    if metrics["fallback_water_seed_used"]:
        reasons.append("Su tohumu görüntüden türetildi")
    if reasons:
        return "REVIEW_REQUIRED", " | ".join(reasons)
    return "AUTO_ACCEPT_CANDIDATE", "GrabCut yerel düzeltmesi tamamlandı."


def create_review_image(
    gray: np.ndarray,
    prior_land: np.ndarray,
    refined_land: np.ndarray,
    safe_water: np.ndarray,
    destination: Path,
) -> None:
    original = np.stack([gray, gray, gray], axis=-1).astype(np.float32)

    prior_overlay = original.copy()
    prior_overlay[prior_land] = (
        prior_overlay[prior_land] * 0.35
        + np.array([255, 40, 40], dtype=np.float32) * 0.65
    )
    prior_overlay[boundary(prior_land)] = np.array(
        [255, 215, 20],
        dtype=np.float32,
    )

    refined_overlay = original.copy()
    refined_overlay[refined_land] = (
        refined_overlay[refined_land] * 0.35
        + np.array([40, 170, 255], dtype=np.float32) * 0.65
    )
    refined_overlay[boundary(refined_land)] = np.array(
        [20, 255, 120],
        dtype=np.float32,
    )

    water_only = original.copy()
    water_only[~safe_water] = 0

    combined = np.concatenate(
        [original, prior_overlay, refined_overlay, water_only],
        axis=1,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(
        np.clip(combined, 0, 255).astype(np.uint8)
    ).save(destination)


def main() -> None:
    args = parse_args()

    if not INPUT_MANIFEST.exists():
        raise FileNotFoundError(f"Manifest bulunamadı: {INPUT_MANIFEST}")

    OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if REVIEW_DIR.exists():
        shutil.rmtree(REVIEW_DIR)

    dataframe = pd.read_csv(
        INPUT_MANIFEST,
        encoding="utf-8-sig",
        low_memory=False,
    )

    required = {
        "sample_id",
        "subset",
        "surface_context",
        "image_path",
        "land_mask_path",
        "generation_status",
    }
    missing = required - set(dataframe.columns)
    if missing:
        raise RuntimeError(f"Manifestte eksik sütunlar: {sorted(missing)}")

    if args.sample_id:
        wanted = {
            str(value).strip()
            for value in args.sample_id
        }
        dataframe = dataframe[
            dataframe["sample_id"]
            .astype(str)
            .isin(wanted)
        ].copy()

        missing_ids = sorted(
            wanted
            - set(
                dataframe["sample_id"]
                .astype(str)
            )
        )

        if missing_ids:
            raise RuntimeError(
                "Manifestte bulunmayan sample_id: "
                + ", ".join(missing_ids)
            )

    print("=" * 78)
    print("ROBUST BINARY OIL DETECTOR v0.6")
    print("SAR-GUIDED KIYI MASKESİ YEREL DÜZELTME")
    print("=" * 78)
    print("Toplam kayıt:", len(dataframe))
    print("Belirsiz kıyı bandı:", args.uncertain_band, "piksel")
    print("GrabCut iterasyonu:", args.iterations)

    result_rows = []
    failures = []

    for index, row in enumerate(dataframe.to_dict(orient="records"), start=1):
        sample_id = str(row["sample_id"])
        subset = str(row["subset"])
        image_path = resolve_path(row["image_path"])
        prior_land_path = resolve_path(row["land_mask_path"])
        output_land_path = LAND_ROOT / subset / f"{image_path.stem}.png"
        output_water_path = SAFE_WATER_ROOT / subset / f"{image_path.stem}.png"

        try:
            if not image_path.exists():
                raise FileNotFoundError(str(image_path))
            gray = load_gray(image_path)

            if str(row["surface_context"]) == "water":
                refined_land = np.zeros(gray.shape, dtype=bool)
                metrics = {
                    "prior_land_ratio": 0.0,
                    "refined_land_ratio": 0.0,
                    "land_ratio_change": 0.0,
                    "absolute_land_ratio_change": 0.0,
                    "boundary_median_change_pixels": 0.0,
                    "boundary_p95_change_pixels": 0.0,
                    "definite_land_pixels": 0,
                    "definite_water_pixels": int(gray.size),
                    "fallback_water_seed_used": False,
                }
                status = "OPEN_WATER"
                reason = "DARTIS açık-su grubu; kara maskesi yok."
            else:
                if not prior_land_path.exists():
                    raise FileNotFoundError(str(prior_land_path))
                prior_land = load_mask(prior_land_path)
                if prior_land.shape != gray.shape:
                    raise RuntimeError("Görüntü ve öncül maske boyutları eşleşmiyor.")
                refined_land, metrics = refine_coastal_mask(
                    gray,
                    prior_land,
                    args.uncertain_band,
                    args.iterations,
                    args.minimum_component_pixels,
                )
                safe_preview = ~dilate(refined_land, args.coast_buffer_pixels)
                status, reason = determine_status(
                    metrics,
                    float(safe_preview.mean()),
                )

            safe_water = ~dilate(refined_land, args.coast_buffer_pixels)
            save_mask(refined_land, output_land_path)
            save_mask(safe_water, output_water_path)

            result_rows.append(
                {
                    **row,
                    "refined_land_mask_path": relative(output_land_path),
                    "refined_safe_water_mask_path": relative(output_water_path),
                    **metrics,
                    "safe_water_ratio_refined": float(safe_water.mean()),
                    "refinement_status": status,
                    "refinement_reason": reason,
                    "eligible_for_water_model": bool(
                        status in {"OPEN_WATER", "AUTO_ACCEPT_CANDIDATE"}
                    ),
                    "locked_test_used": False,
                }
            )
        except Exception as error:
            failures.append(
                {
                    "sample_id": sample_id,
                    "image_path": str(image_path),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            result_rows.append(
                {
                    **row,
                    "refinement_status": "FAILED",
                    "refinement_reason": f"{type(error).__name__}: {error}",
                    "eligible_for_water_model": False,
                    "locked_test_used": False,
                }
            )

        if index % 100 == 0 or index == len(dataframe):
            print(f"{index}/{len(dataframe)} | hata={len(failures)}")

    result = pd.DataFrame(result_rows)
    result.to_csv(
        OUTPUT_MANIFEST,
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        failures,
        columns=["sample_id", "image_path", "error_type", "error"],
    ).to_csv(
        FAILURES_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    coastal = result[
        result["surface_context"].eq("coast")
        & ~result["refinement_status"].eq("FAILED")
    ].copy()
    coastal = coastal.sort_values(
        [
            "refinement_status",
            "boundary_p95_change_pixels",
            "sample_id",
        ],
        ascending=[True, False, True],
    )

    for review_index, review_row in enumerate(
        coastal.head(args.review_count).to_dict(orient="records"),
        start=1,
    ):
        gray = load_gray(resolve_path(review_row["image_path"]))
        prior_land = load_mask(resolve_path(review_row["land_mask_path"]))
        refined_land = load_mask(
            resolve_path(review_row["refined_land_mask_path"])
        )
        safe_water = load_mask(
            resolve_path(review_row["refined_safe_water_mask_path"])
        )
        destination = (
            REVIEW_DIR
            / (
                f"{review_index:03d}"
                f"__{review_row['refinement_status']}"
                f"__p95_{float(review_row['boundary_p95_change_pixels']):.1f}"
                f"__{review_row['sample_id'].replace(':', '_')}.png"
            )
        )
        create_review_image(
            gray,
            prior_land,
            refined_land,
            safe_water,
            destination,
        )

    status_counts = (
        result.groupby("refinement_status", dropna=False)
        .size()
        .reset_index(name="sample_count")
        .sort_values("refinement_status")
    )
    status_counts.to_csv(
        STATUS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    summary = {
        "stage": "v06_dartis_water_masks_refined",
        "total_records": int(len(result)),
        "failed_records": int(len(failures)),
        "status_counts": status_counts.to_dict(orient="records"),
        "method": (
            "GrabCut with geospatial prior and SAR intensity-texture features"
        ),
        "geospatial_mask_is_ground_truth": False,
        "refined_mask_is_ground_truth": False,
        "manual_review_required": True,
        "training_performed": False,
        "locked_test_used": False,
    }
    with SUMMARY_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    REPORT_PATH.write_text(
        f"""# v0.6 SAR-Guided Kara–Su Maskesi Düzeltme

Global sınır kaydırma yaklaşımı terk edilmiştir.

Yeni yöntem, coğrafi kara maskesini yalnız başlangıç öncülü olarak
kullanır. SAR yoğunluk ve yerel doku özellikleriyle GrabCut çalıştırarak
kıyı sınırını görüntü içinde yerel olarak günceller.

- Toplam kayıt: {len(result)}
- Hatalı kayıt: {len(failures)}
- Eğitim yapıldı: Hayır
- Kilitli test kullanıldı: Hayır
""",
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("v0.6 YEREL KIYI DÜZELTME SONUCU")
    print("=" * 78)
    print(status_counts.to_string(index=False))
    print("Hata:", len(failures))
    print("Manifest:", OUTPUT_MANIFEST.resolve())
    print("İnceleme:", REVIEW_DIR.resolve())
    print()
    print("Bu aşamada model eğitilmedi.")


if __name__ == "__main__":
    main()
