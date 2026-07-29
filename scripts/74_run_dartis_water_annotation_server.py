from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import mimetypes
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_QUEUE = (
    ROOT
    / "data"
    / "metadata"
    / "v06_dartis_water_active_batch.csv"
)

DEFAULT_STATUS = (
    ROOT
    / "data"
    / "metadata"
    / "v06_dartis_water_active_status.json"
)


HTML = r"""<!doctype html>
<html lang="tr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DARTIS Kara-Su Etiketleme</title>
<style>
:root {
  font-family: Arial, Helvetica, sans-serif;
  color-scheme: dark;
}
body {
  margin: 0;
  background: #111;
  color: #eee;
}
header {
  padding: 10px 14px;
  background: #1d1d1d;
  border-bottom: 1px solid #444;
  display: flex;
  gap: 16px;
  align-items: center;
  flex-wrap: wrap;
}
main {
  padding: 12px;
}
.toolbar {
  display: flex;
  gap: 7px;
  flex-wrap: wrap;
  align-items: center;
  margin-bottom: 10px;
}
button, select, input {
  font-size: 14px;
}
button {
  border: 1px solid #555;
  background: #292929;
  color: #eee;
  border-radius: 6px;
  padding: 7px 10px;
  cursor: pointer;
}
button:hover {
  background: #383838;
}
button.active {
  outline: 3px solid #eee;
}
button.water {
  background: #006d76;
}
button.land {
  background: #762020;
}
button.ignore {
  background: #69306d;
}
button.complete {
  background: #176b35;
}
button.danger {
  background: #782424;
}
.canvases {
  display: grid;
  grid-template-columns: minmax(300px, 1fr) minmax(300px, 1fr);
  gap: 12px;
  align-items: start;
}
.panel {
  background: #191919;
  border: 1px solid #444;
  border-radius: 8px;
  padding: 8px;
}
.panel h3 {
  margin: 0 0 7px;
  font-size: 15px;
}
canvas {
  width: 100%;
  height: auto;
  image-rendering: pixelated;
  background: #000;
  cursor: crosshair;
  touch-action: none;
}
.legend {
  display: flex;
  gap: 14px;
  flex-wrap: wrap;
  margin-top: 8px;
  font-size: 13px;
}
.swatch {
  display: inline-block;
  width: 14px;
  height: 14px;
  vertical-align: -2px;
  margin-right: 4px;
  border: 1px solid #ddd;
}
.status {
  margin-top: 10px;
  background: #1c1c1c;
  border: 1px solid #444;
  border-radius: 7px;
  padding: 8px;
  min-height: 20px;
  white-space: pre-wrap;
}
.progress {
  font-weight: bold;
}
.small {
  font-size: 12px;
  opacity: 0.82;
}
@media (max-width: 900px) {
  .canvases {
    grid-template-columns: 1fr;
  }
}
</style>
</head>
<body>
<header>
  <strong>DARTIS Kara–Su Maske Editörü</strong>
  <span id="itemTitle">Yükleniyor…</span>
  <span class="progress" id="progress"></span>
</header>

<main>
  <div class="toolbar">
    <button id="prevBtn">← Önceki</button>
    <button id="nextBtn">Sonraki →</button>

    <button id="waterBtn" class="water active">Su (W)</button>
    <button id="landBtn" class="land">Kara (L)</button>
    <button id="ignoreBtn" class="ignore">Belirsiz (I)</button>

    <label>
      Fırça:
      <input id="brushSize" type="range" min="2" max="120" value="28">
      <span id="brushValue">28</span> px
    </label>

    <button id="undoBtn">Geri al (Ctrl+Z)</button>
    <button id="redoBtn">Yinele (Ctrl+Y)</button>

    <button id="resetPrelabelBtn">Ön etikete dön</button>
    <button id="fillWaterBtn">Tümü su</button>
    <button id="fillLandBtn">Tümü kara</button>
    <button id="fillIgnoreBtn">Tümü belirsiz</button>

    <button id="saveDraftBtn">Taslak kaydet</button>
    <button id="saveCompleteBtn" class="complete">Kaydet ve tamamla</button>
    <button id="markIncompleteBtn" class="danger">Tamamlanmadı yap</button>
  </div>

  <div class="canvases">
    <section class="panel">
      <h3>Orijinal SAR</h3>
      <canvas id="originalCanvas" width="512" height="512"></canvas>
    </section>

    <section class="panel">
      <h3>Düzenlenen maske bindirmesi</h3>
      <canvas id="editCanvas" width="512" height="512"></canvas>
      <div class="legend">
        <span><span class="swatch" style="background:#00ffff"></span>Su = 255</span>
        <span><span class="swatch" style="background:#000"></span>Kara = 0</span>
        <span><span class="swatch" style="background:#ff00ff"></span>Belirsiz = 128</span>
      </div>
    </section>
  </div>

  <div class="status" id="status">
    Sol tuşla boya. Büyük alanlar için fırçayı büyüt. Kaydet ve tamamla demeden görüntü eğitim etiketine alınmaz.
  </div>
  <p class="small">
    Kısayollar: W=su, L=kara, I=belirsiz, A/D=önceki/sonraki,
    Ctrl+Z=geri al, Ctrl+Y=yinele, Ctrl+S=taslak kaydet.
  </p>
</main>

<script>
"use strict";

const originalCanvas = document.getElementById("originalCanvas");
const editCanvas = document.getElementById("editCanvas");
const originalCtx = originalCanvas.getContext("2d", {willReadFrequently: true});
const editCtx = editCanvas.getContext("2d", {willReadFrequently: true});

const maskCanvas = document.createElement("canvas");
maskCanvas.width = 512;
maskCanvas.height = 512;
const maskCtx = maskCanvas.getContext("2d", {willReadFrequently: true});

let items = [];
let currentIndex = 0;
let currentMode = "water";
let drawing = false;
let lastPoint = null;
let history = [];
let redoStack = [];
let originalImage = new Image();
let prelabelImage = new Image();

function setStatus(message, isError=false) {
  const box = document.getElementById("status");
  box.textContent = message;
  box.style.borderColor = isError ? "#c33" : "#444";
}

function selectMode(mode) {
  currentMode = mode;
  for (const id of ["waterBtn", "landBtn", "ignoreBtn"]) {
    document.getElementById(id).classList.remove("active");
  }
  document.getElementById(mode + "Btn").classList.add("active");
}

function brushValue() {
  return Number(document.getElementById("brushSize").value);
}

function maskGrayForMode() {
  if (currentMode === "water") return 255;
  if (currentMode === "land") return 0;
  return 128;
}

function saveHistory() {
  history.push(maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height));
  if (history.length > 30) history.shift();
  redoStack = [];
}

function undo() {
  if (history.length === 0) return;
  redoStack.push(maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height));
  const previous = history.pop();
  maskCtx.putImageData(previous, 0, 0);
  renderOverlay();
}

function redo() {
  if (redoStack.length === 0) return;
  history.push(maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height));
  const next = redoStack.pop();
  maskCtx.putImageData(next, 0, 0);
  renderOverlay();
}

function canvasPoint(event) {
  const rect = editCanvas.getBoundingClientRect();
  const x = (event.clientX - rect.left) * editCanvas.width / rect.width;
  const y = (event.clientY - rect.top) * editCanvas.height / rect.height;
  return {x, y};
}

function drawMaskLine(from, to) {
  const gray = maskGrayForMode();
  maskCtx.strokeStyle = `rgb(${gray},${gray},${gray})`;
  maskCtx.lineWidth = brushValue();
  maskCtx.lineCap = "round";
  maskCtx.lineJoin = "round";
  maskCtx.beginPath();
  maskCtx.moveTo(from.x, from.y);
  maskCtx.lineTo(to.x, to.y);
  maskCtx.stroke();
  renderOverlay();
}

function beginDraw(event) {
  event.preventDefault();
  saveHistory();
  drawing = true;
  lastPoint = canvasPoint(event);
  drawMaskLine(lastPoint, lastPoint);
  editCanvas.setPointerCapture(event.pointerId);
}

function continueDraw(event) {
  if (!drawing) return;
  event.preventDefault();
  const point = canvasPoint(event);
  drawMaskLine(lastPoint, point);
  lastPoint = point;
}

function endDraw(event) {
  if (!drawing) return;
  drawing = false;
  lastPoint = null;
  try {
    editCanvas.releasePointerCapture(event.pointerId);
  } catch (_) {}
}

function renderOverlay() {
  editCtx.clearRect(0, 0, editCanvas.width, editCanvas.height);
  editCtx.drawImage(originalImage, 0, 0, editCanvas.width, editCanvas.height);

  const mask = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
  const overlay = editCtx.getImageData(0, 0, editCanvas.width, editCanvas.height);

  for (let i = 0; i < mask.data.length; i += 4) {
    const value = mask.data[i];

    if (value >= 200) {
      overlay.data[i] = Math.round(overlay.data[i] * 0.50);
      overlay.data[i + 1] = Math.round(overlay.data[i + 1] * 0.50 + 127);
      overlay.data[i + 2] = Math.round(overlay.data[i + 2] * 0.50 + 127);
    } else if (value >= 64 && value < 200) {
      overlay.data[i] = Math.round(overlay.data[i] * 0.45 + 140);
      overlay.data[i + 1] = Math.round(overlay.data[i + 1] * 0.45);
      overlay.data[i + 2] = Math.round(overlay.data[i + 2] * 0.45 + 140);
    }
  }

  editCtx.putImageData(overlay, 0, 0);
}

function quantizedMaskDataUrl() {
  const data = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);

  for (let i = 0; i < data.data.length; i += 4) {
    const value = data.data[i];
    let quantized = 0;

    if (value >= 192) quantized = 255;
    else if (value >= 64) quantized = 128;

    data.data[i] = quantized;
    data.data[i + 1] = quantized;
    data.data[i + 2] = quantized;
    data.data[i + 3] = 255;
  }

  const exportCanvas = document.createElement("canvas");
  exportCanvas.width = maskCanvas.width;
  exportCanvas.height = maskCanvas.height;
  exportCanvas.getContext("2d").putImageData(data, 0, 0);
  return exportCanvas.toDataURL("image/png");
}

async function saveCurrent(completed) {
  const response = await fetch("/api/save", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      index: currentIndex,
      completed,
      mask_data_url: quantizedMaskDataUrl()
    })
  });

  const result = await response.json();

  if (!response.ok) {
    throw new Error(result.error || "Kaydetme başarısız.");
  }

  items[currentIndex].saved = true;
  items[currentIndex].completed = completed;
  updateHeader();
  setStatus(completed ? "Kaydedildi ve tamamlandı." : "Taslak kaydedildi.");
}

async function markIncomplete() {
  const response = await fetch("/api/status", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      index: currentIndex,
      completed: false
    })
  });

  const result = await response.json();

  if (!response.ok) {
    throw new Error(result.error || "Durum güncellenemedi.");
  }

  items[currentIndex].completed = false;
  updateHeader();
  setStatus("Görüntü tamamlanmadı olarak işaretlendi.");
}

function fillMask(value) {
  saveHistory();
  maskCtx.fillStyle = `rgb(${value},${value},${value})`;
  maskCtx.fillRect(0, 0, maskCanvas.width, maskCanvas.height);
  renderOverlay();
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = reject;
    image.src = url + "&cache=" + Date.now();
  });
}

async function loadCurrent() {
  if (items.length === 0) return;

  setStatus("Görüntü yükleniyor…");
  history = [];
  redoStack = [];

  const item = items[currentIndex];

  const [original, mask, prelabel] = await Promise.all([
    loadImage(`/api/image?index=${currentIndex}&kind=original`),
    loadImage(`/api/image?index=${currentIndex}&kind=mask`),
    loadImage(`/api/image?index=${currentIndex}&kind=prelabel`)
  ]);

  originalImage = original;
  prelabelImage = prelabel;

  originalCtx.clearRect(0, 0, originalCanvas.width, originalCanvas.height);
  originalCtx.drawImage(originalImage, 0, 0, originalCanvas.width, originalCanvas.height);

  maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
  maskCtx.drawImage(mask, 0, 0, maskCanvas.width, maskCanvas.height);

  renderOverlay();
  updateHeader();

  setStatus(
    item.completed
      ? "Bu görüntü tamamlanmış. Düzenleyip tekrar kaydedebilirsin."
      : item.saved
        ? "Taslak maske yüklendi."
        : "Başlangıç olarak modelin ön etiketi yüklendi. Kontrol edip düzelt."
  );
}

function updateHeader() {
  const item = items[currentIndex];
  const completedCount = items.filter(x => x.completed).length;
  const state = item.completed ? "✓ tamamlandı" : item.saved ? "taslak" : "yeni";

  document.getElementById("itemTitle").textContent =
    `${currentIndex + 1}/${items.length} — ${item.sample_id} — ${state}`;

  document.getElementById("progress").textContent =
    `Tamamlanan: ${completedCount}/${items.length}`;
}

async function changeIndex(delta) {
  currentIndex = (currentIndex + delta + items.length) % items.length;
  await loadCurrent();
}

async function resetToPrelabel() {
  saveHistory();
  maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
  maskCtx.drawImage(prelabelImage, 0, 0, maskCanvas.width, maskCanvas.height);
  renderOverlay();
  setStatus("Maske model ön etiketine döndürüldü. Henüz kaydedilmedi.");
}

async function initialize() {
  const response = await fetch("/api/items");
  const result = await response.json();

  if (!response.ok) {
    throw new Error(result.error || "Liste yüklenemedi.");
  }

  items = result.items;
  currentIndex = Math.max(0, Math.min(result.initial_index || 0, items.length - 1));
  await loadCurrent();
}

editCanvas.addEventListener("pointerdown", beginDraw);
editCanvas.addEventListener("pointermove", continueDraw);
editCanvas.addEventListener("pointerup", endDraw);
editCanvas.addEventListener("pointercancel", endDraw);

document.getElementById("brushSize").addEventListener("input", event => {
  document.getElementById("brushValue").textContent = event.target.value;
});

document.getElementById("waterBtn").onclick = () => selectMode("water");
document.getElementById("landBtn").onclick = () => selectMode("land");
document.getElementById("ignoreBtn").onclick = () => selectMode("ignore");
document.getElementById("undoBtn").onclick = undo;
document.getElementById("redoBtn").onclick = redo;
document.getElementById("prevBtn").onclick = () => changeIndex(-1);
document.getElementById("nextBtn").onclick = () => changeIndex(1);
document.getElementById("resetPrelabelBtn").onclick = resetToPrelabel;
document.getElementById("fillWaterBtn").onclick = () => fillMask(255);
document.getElementById("fillLandBtn").onclick = () => fillMask(0);
document.getElementById("fillIgnoreBtn").onclick = () => fillMask(128);
document.getElementById("saveDraftBtn").onclick = () => saveCurrent(false).catch(error => setStatus(error.message, true));
document.getElementById("saveCompleteBtn").onclick = () => saveCurrent(true).catch(error => setStatus(error.message, true));
document.getElementById("markIncompleteBtn").onclick = () => markIncomplete().catch(error => setStatus(error.message, true));

window.addEventListener("keydown", event => {
  const key = event.key.toLowerCase();

  if (event.ctrlKey && key === "z") {
    event.preventDefault();
    undo();
  } else if (event.ctrlKey && key === "y") {
    event.preventDefault();
    redo();
  } else if (event.ctrlKey && key === "s") {
    event.preventDefault();
    saveCurrent(false).catch(error => setStatus(error.message, true));
  } else if (key === "w") {
    selectMode("water");
  } else if (key === "l") {
    selectMode("land");
  } else if (key === "i") {
    selectMode("ignore");
  } else if (key === "a") {
    changeIndex(-1);
  } else if (key === "d") {
    changeIndex(1);
  }
});

initialize().catch(error => setStatus(error.message, true));
</script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "DARTIS aktif etiketleme kuyruğunu tarayıcıda "
            "düzenlemek için yerel maske sunucusu."
        )
    )

    parser.add_argument(
        "--queue",
        type=Path,
        default=DEFAULT_QUEUE,
    )

    parser.add_argument(
        "--status",
        type=Path,
        default=DEFAULT_STATUS,
    )

    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8765,
    )

    parser.add_argument(
        "--no-browser",
        action="store_true",
    )

    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)

    if not path.is_absolute():
        path = ROOT / path

    return path.resolve()


def read_queue(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Etiketleme kuyruğu bulunamadı: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        rows = list(
            csv.DictReader(file)
        )

    required = {
        "sample_id",
        "annotation_image_path",
        "prelabel_water_path",
        "manual_mask_path",
    }

    if not rows:
        raise RuntimeError(
            "Etiketleme kuyruğu boş."
        )

    missing = required - set(rows[0])

    if missing:
        raise RuntimeError(
            "Etiketleme kuyruğunda eksik sütunlar: "
            f"{sorted(missing)}"
        )

    return rows


def load_status(path: Path) -> dict:
    if not path.exists():
        return {
            "items": {},
            "last_index": 0,
        }

    try:
        data = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except Exception:
        return {
            "items": {},
            "last_index": 0,
        }

    if not isinstance(data, dict):
        return {
            "items": {},
            "last_index": 0,
        }

    data.setdefault(
        "items",
        {},
    )

    data.setdefault(
        "last_index",
        0,
    )

    return data


def save_status(
    path: Path,
    status: dict,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            status,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    temporary.replace(path)


def quantize_mask(
    image: Image.Image,
) -> Image.Image:
    gray = np.array(
        image.convert("L")
    )

    output = np.zeros_like(
        gray,
        dtype=np.uint8,
    )

    output[
        (gray >= 64)
        & (gray < 192)
    ] = 128

    output[
        gray >= 192
    ] = 255

    return Image.fromarray(
        output,
        mode="L",
    )


class AnnotationApplication:
    def __init__(
        self,
        queue_path: Path,
        status_path: Path,
    ) -> None:
        self.queue_path = queue_path
        self.status_path = status_path
        self.rows = read_queue(
            queue_path
        )
        self.status = load_status(
            status_path
        )
        self.lock = threading.Lock()

    def item_state(
        self,
        index: int,
    ) -> dict:
        sample_id = self.rows[
            index
        ]["sample_id"]

        return self.status[
            "items"
        ].get(
            sample_id,
            {
                "saved": False,
                "completed": False,
            },
        )

    def item_path(
        self,
        index: int,
        kind: str,
    ) -> Path:
        row = self.rows[index]
        state = self.item_state(
            index
        )

        if kind == "original":
            key = "annotation_image_path"
        elif kind == "prelabel":
            key = "prelabel_water_path"
        elif kind == "probability":
            key = (
                "ensemble_probability_path"
            )
        elif kind == "mask":
            if state.get(
                "saved",
                False,
            ):
                key = "manual_mask_path"
            else:
                key = (
                    "prelabel_water_path"
                )
        else:
            raise KeyError(
                f"Bilinmeyen görüntü türü: {kind}"
            )

        path = resolve_path(
            row[key]
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Dosya bulunamadı: {path}"
            )

        return path

    def save_mask(
        self,
        index: int,
        data_url: str,
        completed: bool,
    ) -> None:
        if not (
            0
            <= index
            < len(self.rows)
        ):
            raise IndexError(
                "Geçersiz görüntü indeksi."
            )

        prefix = (
            "data:image/png;base64,"
        )

        if not data_url.startswith(
            prefix
        ):
            raise ValueError(
                "Geçersiz PNG veri biçimi."
            )

        raw = base64.b64decode(
            data_url[
                len(prefix):
            ],
            validate=True,
        )

        image = Image.open(
            io.BytesIO(raw)
        )

        image.load()

        image = quantize_mask(
            image
        )

        target_path = resolve_path(
            self.rows[index][
                "manual_mask_path"
            ]
        )

        target_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary = target_path.with_suffix(
            ".tmp.png"
        )

        image.save(
            temporary,
            format="PNG",
        )

        temporary.replace(
            target_path
        )

        sample_id = self.rows[
            index
        ]["sample_id"]

        with self.lock:
            self.status[
                "items"
            ][sample_id] = {
                "saved": True,
                "completed": bool(
                    completed
                ),
                "mask_path": str(
                    target_path
                ),
            }

            self.status[
                "last_index"
            ] = index

            save_status(
                self.status_path,
                self.status,
            )

    def update_completed(
        self,
        index: int,
        completed: bool,
    ) -> None:
        sample_id = self.rows[
            index
        ]["sample_id"]

        with self.lock:
            state = self.status[
                "items"
            ].setdefault(
                sample_id,
                {
                    "saved": False,
                    "completed": False,
                },
            )

            state[
                "completed"
            ] = bool(completed)

            self.status[
                "last_index"
            ] = index

            save_status(
                self.status_path,
                self.status,
            )

    def items_payload(
        self,
    ) -> dict:
        items = []

        for index, row in enumerate(
            self.rows
        ):
            state = self.item_state(
                index
            )

            items.append(
                {
                    "index": index,
                    "sample_id": row[
                        "sample_id"
                    ],
                    "coastal_group": row.get(
                        "coastal_group",
                        "",
                    ),
                    "selection_reason": row.get(
                        "selection_reason",
                        "",
                    ),
                    "saved": bool(
                        state.get(
                            "saved",
                            False,
                        )
                    ),
                    "completed": bool(
                        state.get(
                            "completed",
                            False,
                        )
                    ),
                }
            )

        return {
            "items": items,
            "initial_index": int(
                self.status.get(
                    "last_index",
                    0,
                )
            ),
        }


def make_handler(
    application: AnnotationApplication,
):
    class Handler(
        BaseHTTPRequestHandler
    ):
        server_version = (
            "DartisAnnotation/1.0"
        )

        def log_message(
            self,
            format: str,
            *args,
        ) -> None:
            return

        def send_json(
            self,
            payload: dict,
            status: int = 200,
        ) -> None:
            raw = json.dumps(
                payload,
                ensure_ascii=False,
            ).encode("utf-8")

            self.send_response(
                status
            )
            self.send_header(
                "Content-Type",
                "application/json; "
                "charset=utf-8",
            )
            self.send_header(
                "Content-Length",
                str(len(raw)),
            )
            self.send_header(
                "Cache-Control",
                "no-store",
            )
            self.end_headers()
            self.wfile.write(raw)

        def send_file(
            self,
            path: Path,
        ) -> None:
            content_type = (
                mimetypes.guess_type(
                    str(path)
                )[0]
                or "application/octet-stream"
            )

            raw = path.read_bytes()

            self.send_response(
                HTTPStatus.OK
            )
            self.send_header(
                "Content-Type",
                content_type,
            )
            self.send_header(
                "Content-Length",
                str(len(raw)),
            )
            self.send_header(
                "Cache-Control",
                "no-store",
            )
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(
            self,
        ) -> None:
            parsed = urlparse(
                self.path
            )

            try:
                if parsed.path == "/":
                    raw = HTML.encode(
                        "utf-8"
                    )

                    self.send_response(
                        HTTPStatus.OK
                    )
                    self.send_header(
                        "Content-Type",
                        "text/html; "
                        "charset=utf-8",
                    )
                    self.send_header(
                        "Content-Length",
                        str(len(raw)),
                    )
                    self.end_headers()
                    self.wfile.write(raw)
                    return

                if (
                    parsed.path
                    == "/api/items"
                ):
                    self.send_json(
                        application.items_payload()
                    )
                    return

                if (
                    parsed.path
                    == "/api/image"
                ):
                    query = parse_qs(
                        parsed.query
                    )

                    index = int(
                        query.get(
                            "index",
                            ["0"],
                        )[0]
                    )

                    kind = query.get(
                        "kind",
                        ["original"],
                    )[0]

                    path = application.item_path(
                        index,
                        kind,
                    )

                    self.send_file(path)
                    return

                self.send_json(
                    {
                        "error": (
                            "Kaynak bulunamadı."
                        )
                    },
                    status=404,
                )
            except Exception as error:
                self.send_json(
                    {
                        "error": str(error)
                    },
                    status=500,
                )

        def do_POST(
            self,
        ) -> None:
            parsed = urlparse(
                self.path
            )

            try:
                content_length = int(
                    self.headers.get(
                        "Content-Length",
                        "0",
                    )
                )

                raw = self.rfile.read(
                    content_length
                )

                payload = json.loads(
                    raw.decode("utf-8")
                )

                if (
                    parsed.path
                    == "/api/save"
                ):
                    application.save_mask(
                        index=int(
                            payload[
                                "index"
                            ]
                        ),
                        data_url=str(
                            payload[
                                "mask_data_url"
                            ]
                        ),
                        completed=bool(
                            payload.get(
                                "completed",
                                False,
                            )
                        ),
                    )

                    self.send_json(
                        {"ok": True}
                    )
                    return

                if (
                    parsed.path
                    == "/api/status"
                ):
                    application.update_completed(
                        index=int(
                            payload[
                                "index"
                            ]
                        ),
                        completed=bool(
                            payload.get(
                                "completed",
                                False,
                            )
                        ),
                    )

                    self.send_json(
                        {"ok": True}
                    )
                    return

                self.send_json(
                    {
                        "error": (
                            "Kaynak bulunamadı."
                        )
                    },
                    status=404,
                )
            except Exception as error:
                self.send_json(
                    {
                        "error": str(error)
                    },
                    status=500,
                )

    return Handler


def main() -> None:
    args = parse_args()

    queue_path = resolve_path(
        args.queue
    )

    status_path = resolve_path(
        args.status
    )

    application = (
        AnnotationApplication(
            queue_path=queue_path,
            status_path=status_path,
        )
    )

    handler = make_handler(
        application
    )

    server = ThreadingHTTPServer(
        (
            args.host,
            args.port,
        ),
        handler,
    )

    url = (
        f"http://{args.host}:"
        f"{args.port}/"
    )

    print("=" * 78)
    print(
        "DARTIS KARA-SU ETİKETLEME SUNUCUSU"
    )
    print("=" * 78)
    print(
        "Kuyruk:",
        queue_path,
    )
    print(
        "Görüntü sayısı:",
        len(
            application.rows
        ),
    )
    print(
        "Adres:",
        url,
    )
    print(
        "Durdurmak için: Ctrl+C"
    )
    print()

    if not args.no_browser:
        threading.Timer(
            0.8,
            lambda: webbrowser.open(
                url
            ),
        ).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(
            "\nSunucu kapatıldı."
        )
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
