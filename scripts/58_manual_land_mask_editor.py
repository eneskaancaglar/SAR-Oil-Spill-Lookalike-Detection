from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SOURCE_MANIFEST = (
    ROOT
    / "data"
    / "metadata"
    / "v06_dartis_water_masks_homography.csv"
)

DEFAULT_QUEUE_PATH = (
    ROOT
    / "data"
    / "metadata"
    / "v06_manual_land_annotation_queue.csv"
)

ANNOTATION_ROOT = (
    ROOT
    / "data"
    / "annotations"
    / "v06_land_manual"
)

LAND_ROOT = ANNOTATION_ROOT / "land"
SAFE_WATER_ROOT = ANNOTATION_ROOT / "safe_water"

PROBLEM_SAMPLE_IDS = [
    "nc:nc-0313-04-000034",
    "nc:nc-0284-04-000005",
    "nc:nc-0287-04-000008",
    "nc:nc-0043-00-000043",
    "nc:nc-0095-01-000019",
    "nc:nc-0031-00-000031",
    "nc:nc-0155-01-000079",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "DARTIS kıyı görüntülerinde kara maskelerini "
            "doğrudan SAR görüntüsü üzerinde elle düzeltir."
        )
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=DEFAULT_SOURCE_MANIFEST,
    )
    parser.add_argument(
        "--queue",
        type=Path,
        default=DEFAULT_QUEUE_PATH,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=150,
        help="İlk anotasyon kuyruğuna alınacak kıyı görüntüsü.",
    )
    parser.add_argument(
        "--coast-buffer-pixels",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--brush-size",
        type=int,
        default=24,
    )
    parser.add_argument(
        "--reset-queue",
        action="store_true",
    )
    parser.add_argument(
        "--blank-start",
        action="store_true",
        help="Coğrafi maskeden değil tamamen boş maskeden başla.",
    )
    return parser.parse_args()


def relative(path: Path) -> str:
    try:
        return str(
            path.resolve().relative_to(ROOT.resolve())
        ).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def resolve_path(value: Any) -> Path:
    path = Path(str(value).strip())
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(
            image.convert("L"),
            dtype=np.uint8,
        )


def load_binary_mask(
    path: Path,
    shape: tuple[int, int],
) -> np.ndarray:
    if not path.exists():
        return np.zeros(
            shape,
            dtype=bool,
        )

    with Image.open(path) as image:
        mask = np.asarray(
            image.convert("L"),
            dtype=np.uint8,
        )

    if mask.shape != shape:
        mask = cv2.resize(
            mask,
            (shape[1], shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )

    return mask > 127


def save_binary_mask(
    mask: np.ndarray,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    Image.fromarray(
        mask.astype(np.uint8) * 255,
        mode="L",
    ).save(path)


def ellipse_kernel(radius: int) -> np.ndarray:
    radius = max(int(radius), 1)
    size = radius * 2 + 1
    return cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (size, size),
    )


def dilate(
    mask: np.ndarray,
    radius: int,
) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool, copy=True)

    return (
        cv2.dilate(
            mask.astype(np.uint8),
            ellipse_kernel(radius),
            iterations=1,
        )
        > 0
    )


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones(
        (3, 3),
        dtype=np.uint8,
    )
    return (
        cv2.morphologyEx(
            mask.astype(np.uint8),
            cv2.MORPH_GRADIENT,
            kernel,
        )
        > 0
    )


def build_queue(
    source_manifest: Path,
    queue_path: Path,
    limit: int,
) -> pd.DataFrame:
    if not source_manifest.exists():
        raise FileNotFoundError(
            f"Kaynak manifest bulunamadı: {source_manifest}"
        )

    dataframe = pd.read_csv(
        source_manifest,
        encoding="utf-8-sig",
        low_memory=False,
    )

    required = {
        "sample_id",
        "subset",
        "surface_context",
        "image_path",
        "land_mask_path",
    }

    missing = required - set(dataframe.columns)

    if missing:
        raise RuntimeError(
            f"Manifestte eksik sütunlar: {sorted(missing)}"
        )

    coastal = dataframe[
        dataframe["surface_context"].eq("coast")
    ].copy()

    if coastal.empty:
        raise RuntimeError(
            "Kıyılı görüntü bulunamadı."
        )

    coastal["land_ratio_for_selection"] = pd.to_numeric(
        coastal.get(
            "land_ratio",
            pd.Series(
                np.nan,
                index=coastal.index,
            ),
        ),
        errors="coerce",
    ).fillna(0.5)

    coastal["_priority"] = coastal[
        "sample_id"
    ].isin(
        PROBLEM_SAMPLE_IDS
    ).astype(int)

    coastal["_ratio_bin"] = pd.cut(
        coastal[
            "land_ratio_for_selection"
        ],
        bins=np.linspace(0.0, 1.0, 11),
        labels=False,
        include_lowest=True,
    ).fillna(5).astype(int)

    fixed = coastal[
        coastal["_priority"].eq(1)
    ].sort_values("sample_id")

    remaining = coastal[
        coastal["_priority"].eq(0)
    ].copy()

    selected_frames = [fixed]

    remaining_slots = max(
        int(limit) - len(fixed),
        0,
    )

    if remaining_slots > 0:
        groups = []
        for (
            subset,
            ratio_bin,
        ), group in remaining.groupby(
            [
                "subset",
                "_ratio_bin",
            ],
            dropna=False,
        ):
            group = group.sort_values(
                "sample_id"
            ).copy()
            group["_stable"] = group[
                "sample_id"
            ].map(
                lambda value: hash(
                    (
                        "v06-manual-queue",
                        str(value),
                    )
                )
            )
            group = group.sort_values(
                "_stable"
            )
            groups.append(group)

        chosen_rows = []
        group_positions = [0] * len(groups)

        while (
            len(chosen_rows) < remaining_slots
            and groups
        ):
            progressed = False

            for group_index, group in enumerate(groups):
                position = group_positions[
                    group_index
                ]

                if (
                    position < len(group)
                    and len(chosen_rows)
                    < remaining_slots
                ):
                    chosen_rows.append(
                        group.iloc[position]
                    )
                    group_positions[
                        group_index
                    ] += 1
                    progressed = True

            if not progressed:
                break

        if chosen_rows:
            selected_frames.append(
                pd.DataFrame(chosen_rows)
            )

    selected = pd.concat(
        selected_frames,
        ignore_index=True,
    ).head(limit)

    selected = selected.drop_duplicates(
        subset=["sample_id"],
        keep="first",
    ).reset_index(drop=True)

    selected["queue_index"] = np.arange(
        1,
        len(selected) + 1,
    )
    selected["annotation_status"] = "PENDING"
    selected["manual_land_mask_path"] = ""
    selected["manual_safe_water_mask_path"] = ""
    selected["annotation_note"] = ""
    selected["updated_at"] = ""

    selected = selected.drop(
        columns=[
            "_priority",
            "_ratio_bin",
            "_stable",
        ],
        errors="ignore",
    )

    queue_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    selected.to_csv(
        queue_path,
        index=False,
        encoding="utf-8-sig",
    )

    return selected


class LandMaskEditor:
    WINDOW_NAME = "v0.6 Manual Land Mask Editor"

    def __init__(
        self,
        queue: pd.DataFrame,
        queue_path: Path,
        coast_buffer_pixels: int,
        initial_brush_size: int,
        blank_start: bool,
    ) -> None:
        self.queue = queue.reset_index(
            drop=True
        )
        self.queue_path = queue_path
        self.coast_buffer_pixels = max(
            0,
            int(coast_buffer_pixels),
        )
        self.brush_size = max(
            2,
            int(initial_brush_size),
        )
        self.blank_start = bool(
            blank_start
        )

        self.current_index = self._first_pending_index()
        self.image: np.ndarray | None = None
        self.mask: np.ndarray | None = None
        self.initial_mask: np.ndarray | None = None

        self.mode = "BRUSH"
        self.paint_value = True
        self.show_overlay = True
        self.dragging = False
        self.drag_value = True
        self.polygon_points: list[
            tuple[int, int]
        ] = []

        self.banner_height = 126
        self.display_scale = 1.0

        self._load_current()

    def _first_pending_index(self) -> int:
        pending = self.queue.index[
            self.queue[
                "annotation_status"
            ].eq("PENDING")
        ].tolist()

        if pending:
            return int(pending[0])

        return 0

    def _current_row(self) -> pd.Series:
        return self.queue.iloc[
            self.current_index
        ]

    def _manual_land_path(
        self,
        row: pd.Series,
    ) -> Path:
        image_path = resolve_path(
            row["image_path"]
        )
        return (
            LAND_ROOT
            / str(row["subset"])
            / (
                image_path.stem
                + ".png"
            )
        )

    def _manual_water_path(
        self,
        row: pd.Series,
    ) -> Path:
        image_path = resolve_path(
            row["image_path"]
        )
        return (
            SAFE_WATER_ROOT
            / str(row["subset"])
            / (
                image_path.stem
                + ".png"
            )
        )

    def _load_current(self) -> None:
        row = self._current_row()
        image_path = resolve_path(
            row["image_path"]
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Görüntü bulunamadı: {image_path}"
            )

        self.image = load_gray(
            image_path
        )

        manual_path = self._manual_land_path(
            row
        )

        if manual_path.exists():
            starting_mask = load_binary_mask(
                manual_path,
                self.image.shape,
            )
        elif self.blank_start:
            starting_mask = np.zeros(
                self.image.shape,
                dtype=bool,
            )
        else:
            prior_path = resolve_path(
                row["land_mask_path"]
            )
            starting_mask = load_binary_mask(
                prior_path,
                self.image.shape,
            )

        self.mask = starting_mask.copy()
        self.initial_mask = starting_mask.copy()
        self.polygon_points = []

    def _save_queue(self) -> None:
        self.queue.to_csv(
            self.queue_path,
            index=False,
            encoding="utf-8-sig",
        )

    def save_current(self) -> None:
        assert self.mask is not None

        row = self._current_row()

        land_path = self._manual_land_path(
            row
        )
        water_path = self._manual_water_path(
            row
        )

        safe_water = ~dilate(
            self.mask,
            self.coast_buffer_pixels,
        )

        save_binary_mask(
            self.mask,
            land_path,
        )
        save_binary_mask(
            safe_water,
            water_path,
        )

        self.queue.at[
            self.current_index,
            "annotation_status",
        ] = "DONE"
        self.queue.at[
            self.current_index,
            "manual_land_mask_path",
        ] = relative(land_path)
        self.queue.at[
            self.current_index,
            "manual_safe_water_mask_path",
        ] = relative(water_path)
        self.queue.at[
            self.current_index,
            "updated_at",
        ] = datetime.now().isoformat(
            timespec="seconds"
        )

        self._save_queue()

    def skip_current(self) -> None:
        self.queue.at[
            self.current_index,
            "annotation_status",
        ] = "SKIPPED"
        self.queue.at[
            self.current_index,
            "updated_at",
        ] = datetime.now().isoformat(
            timespec="seconds"
        )
        self._save_queue()

    def next_image(self) -> None:
        self.current_index = min(
            self.current_index + 1,
            len(self.queue) - 1,
        )
        self._load_current()

    def previous_image(self) -> None:
        self.current_index = max(
            self.current_index - 1,
            0,
        )
        self._load_current()

    def _image_coordinates(
        self,
        x: int,
        y: int,
    ) -> tuple[int, int] | None:
        assert self.image is not None

        adjusted_y = (
            y
            - self.banner_height
        )

        if adjusted_y < 0:
            return None

        image_x = int(
            round(
                x
                / self.display_scale
            )
        )
        image_y = int(
            round(
                adjusted_y
                / self.display_scale
            )
        )

        height, width = self.image.shape

        if not (
            0 <= image_x < width
            and 0 <= image_y < height
        ):
            return None

        return image_x, image_y

    def _paint(
        self,
        x: int,
        y: int,
        value: bool,
    ) -> None:
        assert self.mask is not None

        radius = max(
            1,
            self.brush_size // 2,
        )

        cv2.circle(
            self.mask,
            (int(x), int(y)),
            radius,
            1 if value else 0,
            thickness=-1,
        )

    def mouse_callback(
        self,
        event: int,
        x: int,
        y: int,
        flags: int,
        userdata: Any,
    ) -> None:
        point = self._image_coordinates(
            x,
            y,
        )

        if point is None:
            return

        image_x, image_y = point

        if self.mode == "POLYGON":
            if event == cv2.EVENT_LBUTTONDOWN:
                self.polygon_points.append(
                    (
                        image_x,
                        image_y,
                    )
                )
            elif event == cv2.EVENT_RBUTTONDOWN:
                if self.polygon_points:
                    self.polygon_points.pop()
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.drag_value = self.paint_value
            self._paint(
                image_x,
                image_y,
                self.drag_value,
            )

        elif event == cv2.EVENT_RBUTTONDOWN:
            self.dragging = True
            self.drag_value = (
                not self.paint_value
            )
            self._paint(
                image_x,
                image_y,
                self.drag_value,
            )

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.dragging:
                self._paint(
                    image_x,
                    image_y,
                    self.drag_value,
                )

        elif event in {
            cv2.EVENT_LBUTTONUP,
            cv2.EVENT_RBUTTONUP,
        }:
            self.dragging = False

        elif event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:
                self.brush_size = min(
                    self.brush_size + 4,
                    200,
                )
            else:
                self.brush_size = max(
                    self.brush_size - 4,
                    2,
                )

    def commit_polygon(self) -> None:
        assert self.mask is not None

        if len(
            self.polygon_points
        ) < 3:
            return

        polygon = np.asarray(
            self.polygon_points,
            dtype=np.int32,
        ).reshape(
            (-1, 1, 2)
        )

        cv2.fillPoly(
            self.mask,
            [polygon],
            1 if self.paint_value else 0,
        )

        self.polygon_points = []

    def reset_current(self) -> None:
        assert self.initial_mask is not None
        self.mask = self.initial_mask.copy()
        self.polygon_points = []

    def clear_current(self) -> None:
        assert self.mask is not None
        self.mask[:] = False
        self.polygon_points = []

    def _render(self) -> np.ndarray:
        assert self.image is not None
        assert self.mask is not None

        grayscale_rgb = np.stack(
            [
                self.image,
                self.image,
                self.image,
            ],
            axis=-1,
        ).astype(np.float32)

        display = grayscale_rgb.copy()

        if self.show_overlay:
            display[
                self.mask
            ] = (
                display[
                    self.mask
                ] * 0.35
                + np.array(
                    [255, 35, 35],
                    dtype=np.float32,
                ) * 0.65
            )

            boundary = mask_boundary(
                self.mask
            )
            display[
                boundary
            ] = np.array(
                [255, 220, 20],
                dtype=np.float32,
            )

        for point_index, (
            point_x,
            point_y,
        ) in enumerate(
            self.polygon_points
        ):
            cv2.circle(
                display,
                (
                    point_x,
                    point_y,
                ),
                4,
                (
                    20,
                    255,
                    100,
                ),
                thickness=-1,
            )

            if point_index > 0:
                previous = self.polygon_points[
                    point_index - 1
                ]
                cv2.line(
                    display,
                    previous,
                    (
                        point_x,
                        point_y,
                    ),
                    (
                        20,
                        255,
                        100,
                    ),
                    thickness=2,
                )

        height, width = self.image.shape

        self.display_scale = min(
            1.0,
            1200.0 / width,
            780.0 / height,
        )

        display = cv2.resize(
            np.clip(
                display,
                0,
                255,
            ).astype(np.uint8),
            (
                int(
                    round(
                        width
                        * self.display_scale
                    )
                ),
                int(
                    round(
                        height
                        * self.display_scale
                    )
                ),
            ),
            interpolation=cv2.INTER_AREA,
        )

        banner = np.zeros(
            (
                self.banner_height,
                display.shape[1],
                3,
            ),
            dtype=np.uint8,
        )

        row = self._current_row()
        status = str(
            row["annotation_status"]
        )

        lines = [
            (
                f"{self.current_index + 1}/{len(self.queue)}  "
                f"{row['sample_id']}  STATUS={status}"
            ),
            (
                f"MODE={self.mode}  LABEL="
                f"{'LAND' if self.paint_value else 'WATER'}  "
                f"BRUSH={self.brush_size}px"
            ),
            (
                "L=land W=water B=brush G=polygon Enter=fill "
                "RightClick=opposite/undo"
            ),
            (
                "S=save+next N=next P=previous R=reset "
                "C=clear M=overlay [ ]=brush Q=quit"
            ),
        ]

        for index, text in enumerate(lines):
            cv2.putText(
                banner,
                text,
                (
                    12,
                    26 + index * 28,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (
                    240,
                    240,
                    240,
                ),
                thickness=1,
                lineType=cv2.LINE_AA,
            )

        return np.vstack(
            [
                banner,
                display,
            ]
        )

    def run(self) -> None:
        cv2.namedWindow(
            self.WINDOW_NAME,
            cv2.WINDOW_NORMAL,
        )
        cv2.setMouseCallback(
            self.WINDOW_NAME,
            self.mouse_callback,
        )

        while True:
            cv2.imshow(
                self.WINDOW_NAME,
                self._render(),
            )

            key = cv2.waitKey(20) & 0xFF

            if key == 255:
                continue

            if key in {
                ord("q"),
                27,
            }:
                break

            if key == ord("l"):
                self.paint_value = True

            elif key == ord("w"):
                self.paint_value = False

            elif key == ord("b"):
                self.mode = "BRUSH"
                self.polygon_points = []

            elif key == ord("g"):
                self.mode = "POLYGON"
                self.polygon_points = []

            elif key in {
                13,
                10,
            }:
                if self.mode == "POLYGON":
                    self.commit_polygon()

            elif key == 8:
                if self.polygon_points:
                    self.polygon_points.pop()

            elif key == ord("s"):
                self.save_current()
                self.next_image()

            elif key == ord("n"):
                self.next_image()

            elif key == ord("p"):
                self.previous_image()

            elif key == ord("x"):
                self.skip_current()
                self.next_image()

            elif key == ord("r"):
                self.reset_current()

            elif key == ord("c"):
                self.clear_current()

            elif key == ord("m"):
                self.show_overlay = (
                    not self.show_overlay
                )

            elif key == ord("["):
                self.brush_size = max(
                    self.brush_size - 4,
                    2,
                )

            elif key == ord("]"):
                self.brush_size = min(
                    self.brush_size + 4,
                    200,
                )

        cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()

    queue_path = args.queue
    if not queue_path.is_absolute():
        queue_path = ROOT / queue_path

    if (
        args.reset_queue
        or not queue_path.exists()
    ):
        queue = build_queue(
            args.source_manifest,
            queue_path,
            args.limit,
        )
    else:
        queue = pd.read_csv(
            queue_path,
            encoding="utf-8-sig",
            low_memory=False,
        )

    if queue.empty:
        raise RuntimeError(
            "Anotasyon kuyruğu boş."
        )

    print("=" * 78)
    print(
        "v0.6 MANUEL KARA MASKESİ DÜZENLEYİCİ"
    )
    print("=" * 78)
    print(
        "Kuyruk:",
        queue_path.resolve(),
    )
    print(
        "Görüntü:",
        len(queue),
    )
    print(
        "Kaydetme:",
        ANNOTATION_ROOT.resolve(),
    )
    print()
    print(
        "Sol fare mevcut etiketi boyar; "
        "sağ fare ters etiketi boyar."
    )
    print(
        "S = kaydet ve sonraki görüntü."
    )

    editor = LandMaskEditor(
        queue=queue,
        queue_path=queue_path,
        coast_buffer_pixels=(
            args.coast_buffer_pixels
        ),
        initial_brush_size=(
            args.brush_size
        ),
        blank_start=args.blank_start,
    )
    editor.run()


if __name__ == "__main__":
    main()
