"""Build the landscape PDF report (Bahasa Indonesia) from training logs, test metrics and the
island AOI outputs. Report-only dependencies are pulled ad hoc:

    uv run --with matplotlib --with reportlab python scripts/report_pdf.py
"""

import io
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from PIL import Image  # noqa: E402
from rasterio.enums import Resampling  # noqa: E402
from reportlab.lib.colors import HexColor, white  # noqa: E402
from reportlab.lib.pagesizes import A4, landscape  # noqa: E402
from reportlab.lib.utils import ImageReader  # noqa: E402
from reportlab.pdfbase import pdfmetrics  # noqa: E402
from reportlab.pdfbase.ttfonts import TTFont  # noqa: E402
from reportlab.pdfgen import canvas  # noqa: E402

from s2lc.classes import CLASSES  # noqa: E402

OUT = Path("out/report/sentinel2-lc_report_v0.1.0.pdf")
ISLANDS = Path("out/islands_2026q3")
W, H = landscape(A4)
M = 36  # page margin, pt
INK, INK2, MUTED = HexColor("#0b0b0b"), HexColor("#52514e"), HexColor("#8a8984")
RULE, PANEL, ACCENT = HexColor("#e2e1dc"), HexColor("#f4f3ef"), HexColor("#2a78d6")
SERIES = "#2a78d6"

fontdir = Path(matplotlib.get_data_path()) / "fonts/ttf"
pdfmetrics.registerFont(TTFont("DV", str(fontdir / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DVB", str(fontdir / "DejaVuSans-Bold.ttf")))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#b9b8b2",
                     "axes.labelcolor": "#52514e", "xtick.color": "#52514e",
                     "ytick.color": "#52514e", "axes.spines.top": False,
                     "axes.spines.right": False})

AOIS = [  # folder, title, note
    ("jawa_karawang", "Jawa — Karawang–Bekasi", "Kawasan industri, permukiman, sawah"),
    ("sumatra_siak", "Sumatra — Siak–Pelalawan, Riau", "Sawit, hutan rawa gambut, HTI"),
    ("kalimantan_mahakam", "Kalimantan — Delta Mahakam", "Mangrove, tambak, sungai"),
    ("sulawesi_maros", "Sulawesi — Makassar–Maros", "Kota, sawah, hutan karst, pesisir"),
    ("papua_timika", "Papua — Dataran rendah Timika", "Hutan, mangrove, tailing tambang"),
]
# total seconds measured by run_aois; download/inference/post split estimated (see page note)
TOTAL_S = {"jawa_karawang": 266, "kalimantan_mahakam": 416, "papua_timika": 436,
           "sulawesi_maros": 263, "sumatra_siak": 239}
INFER_S, POST_S = 10, 5  # ~0.6 s per 1152 px block x 9 blocks on MPS, rounded up; ~2.1 s


def idn(v: float, nd: int = 1) -> str:
    """Indonesian number format: 1.234,5"""
    s = f"{v:,.{nd}f}"
    return s.replace(",", "_").replace(".", ",").replace("_", ".")


def hexc(rgb: tuple[int, int, int]) -> HexColor:
    return HexColor("#{:02x}{:02x}{:02x}".format(*rgb))


# ---------------------------------------------------------------- page chrome
page_no = 0


def page(c: canvas.Canvas, title: str, section: str = "") -> None:
    global page_no
    if page_no:
        c.showPage()
    page_no += 1
    c.setFillColor(INK)
    c.setFont("DVB", 17)
    c.drawString(M, H - M - 8, title)
    if section:
        c.setFont("DV", 9)
        c.setFillColor(ACCENT)
        c.drawString(M, H - M + 8, section.upper())
    c.setStrokeColor(RULE)
    c.setLineWidth(0.8)
    c.line(M, H - M - 18, W - M, H - M - 18)
    c.setFont("DV", 7.5)
    c.setFillColor(MUTED)
    c.drawString(M, 18, "sentinel2-lc v0.1.0 · Klasifikasi tutupan lahan kuartalan Sentinel-2 "
                        "untuk Indonesia · Contains modified Copernicus Sentinel data 2021, 2026")
    c.drawRightString(W - M, 18, str(page_no))


def text(c, x, y, s, size=9.5, font="DV", color=INK, lead=None, width=None) -> float:
    """Draw (optionally wrapped) text; returns the y below it."""
    c.setFont(font, size)
    c.setFillColor(color)
    lines = []
    for para in s.split("\n"):
        if width is None:
            lines.append(para)
            continue
        cur = ""
        for word in para.split(" "):
            t = (cur + " " + word).strip()
            if pdfmetrics.stringWidth(t, font, size) > width and cur:
                lines.append(cur)
                cur = word
            else:
                cur = t
        lines.append(cur)
    lead = lead or size * 1.35
    for ln in lines:
        c.drawString(x, y, ln)
        y -= lead
    return y


def box(c, x, y, w, h, label, sub="", fill=PANEL, stroke=RULE, fs=8.5, color=INK) -> None:
    c.setFillColor(fill)
    c.setStrokeColor(stroke)
    c.setLineWidth(0.8)
    c.roundRect(x, y, w, h, 5, stroke=1, fill=1)
    ty = y + h / 2 + (4 if sub else -3)
    c.setFillColor(color)
    while pdfmetrics.stringWidth(label, "DVB", fs) > w - 8:
        fs -= 0.25
    c.setFont("DVB", fs)
    c.drawCentredString(x + w / 2, ty, label)
    if sub:
        ss = fs - 1.5
        while max(pdfmetrics.stringWidth(ln, "DV", ss) for ln in sub.split("\n")) > w - 8:
            ss -= 0.25
        c.setFont("DV", ss)
        c.setFillColor(INK2 if color == INK else color)
        for k, line in enumerate(sub.split("\n")):
            c.drawCentredString(x + w / 2, ty - 11 - k * 9, line)


def arrow(c, x0, y0, x1, y1, color=INK2) -> None:
    c.setStrokeColor(color)
    c.setFillColor(color)
    c.setLineWidth(1.1)
    c.line(x0, y0, x1, y1)
    a = np.arctan2(y1 - y0, x1 - x0)
    p = c.beginPath()
    p.moveTo(x1, y1)
    p.lineTo(x1 - 6 * np.cos(a - 0.4), y1 - 6 * np.sin(a - 0.4))
    p.lineTo(x1 - 6 * np.cos(a + 0.4), y1 - 6 * np.sin(a + 0.4))
    p.close()
    c.drawPath(p, fill=1, stroke=0)


def fig_image(fig) -> ImageReader:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return ImageReader(buf)


def legend(c, x, y, cols=5, w=150) -> None:
    for i, (k, (nid, _nen, rgb)) in enumerate(CLASSES.items()):
        cx, cy = x + (i % cols) * w, y - (i // cols) * 15
        c.setFillColor(hexc(rgb))
        c.setStrokeColor(RULE)
        c.rect(cx, cy - 2, 10, 10, stroke=1, fill=1)
        text(c, cx + 15, cy, f"{k}  {nid}", size=8.5)


# ---------------------------------------------------------------- pages
def cover(c) -> None:
    page(c, "Laporan Teknis — Klasifikasi Tutupan Lahan Sentinel-2 Kuartalan", "sentinel2-lc")
    ev = json.loads(Path("runs/eval.json").read_text())[0]
    y = text(c, M, H - 90,
             "Layanan eksperimental: pengguna cukup memberi AOI (GeoJSON atau gambar di peta) "
             "serta "
             "tahun dan kuartal (Q1–Q4). Sistem mencari citra Sentinel-2 L2A, membuat komposit "
             "bebas awan, mengklasifikasikan 9 kelas tutupan lahan, lalu memotong hasil sesuai "
             "AOI. Dirancang untuk digabung ke layanan landcover_detection NARAGA sebagai model "
             "kedua (tanpa ortofoto).", size=10.5, width=W - 2 * M)
    tiles = [("Overall accuracy (uji)", idn(ev["oa"] * 100, 1) + " %"),
             ("mIoU (uji)", idn(ev["miou"], 3)),
             ("Parameter model", "32,55 juta"),
             ("Data latih", "334 lokasi"),
             ("Durasi training", "~3 j 40 m (M4)")]
    tw = (W - 2 * M - 4 * 12) / 5
    for i, (k, v) in enumerate(tiles):
        x = M + i * (tw + 12)
        c.setFillColor(PANEL)
        c.roundRect(x, y - 78, tw, 64, 6, stroke=0, fill=1)
        text(c, x + 12, y - 34, k, size=8.5, color=INK2)
        text(c, x + 12, y - 58, v, size=15 if len(v) < 14 else 12, font="DVB")
    y -= 110
    text(c, M, y, "Isi laporan", size=11, font="DVB")
    items = ["1. Diagram alir workflow (training dan inferensi)",
             "2. Arsitektur deep learning: U-Net + encoder ResNet50 (SSL4EO-S12)",
             "3. Proses training: hardware, durasi, grafik loss / IoU / overall accuracy",
             "4. Hasil uji pada wilayah hold-out",
             "5. Lima sampel inferensi Q3 2026: Jawa, Sumatra, Kalimantan, Sulawesi, Papua",
             "    (per pulau: halaman perbandingan citra vs hasil, lalu halaman statistik)"]
    for it in items:
        y = text(c, M + 10, y - 18, it, size=10, color=INK2) + 18 - 4
    text(c, M, 60, "Catatan: angka akurasi = kesepakatan dengan label referensi konsensus "
                   "(ESA WorldCover 2021, Esri/IO LULC 2021, peta sawit Descals 2021) pada wilayah "
                   "uji, bukan akurasi lapangan.", size=8.5, color=MUTED, width=W - 2 * M)


def workflow(c) -> None:
    page(c, "Diagram Alir Workflow", "1 · Workflow")
    # inference lane
    y0 = H - 150
    text(c, M, y0 + 50, "Inferensi (per permintaan pengguna)", size=10.5, font="DVB")
    steps = [("Input", "AOI GeoJSON / gambar\n+ tahun + kuartal"),
             ("Validasi", "dalam Indonesia\n≤ 1.000 km²\nkuartal selesai"),
             ("Cari citra", "STAC Earth Search\n≤ 16 scene terbersih"),
             ("Unduh & komposit", "10 band + SCL, 10 m\nmask awan · median"),
             ("Gap fill", "piksel kosong >1 %:\nisi kuartal sebelumnya"),
             ("Klasifikasi", "13 kanal · U-Net\ntile 512 px overlap 64"),
             ("Pasca-proses", "sieve 0,5 ha\npotong sesuai AOI"),
             ("Output", "COG · GeoJSON\nkomposit\n+ statistik kelas")]
    n = len(steps)
    gap = 14
    bw = (W - 2 * M - (n - 1) * gap) / n
    for i, (a, b) in enumerate(steps):
        x = M + i * (bw + gap)
        hl = i in (0, n - 1)
        box(c, x, y0 - 30, bw, 62, a, b, fill=HexColor("#e8f0fb") if hl else PANEL,
            stroke=ACCENT if hl else RULE)
        if i:
            arrow(c, x - gap + 1, y0 + 1, x - 2, y0 + 1)
    # training lane
    y1 = y0 - 150
    text(c, M, y1 + 50, "Training (offline, sekali)", size=10.5, font="DVB")
    tsteps = [("Sampling lokasi", "300 acak (7 wilayah)\n+ 34 hotspot"),
              ("Komposit 2021", "2 kuartal acak/lokasi\nkode = inferensi"),
              ("Label konsensus", "WorldCover ∩ Esri/IO\n∩ peta sawit 2021"),
              ("Split spasial", "sel 1°: 70/15/15 %\ntrain / val / uji"),
              ("Training U-Net", "25 epoch · AdamW\nCE berbobot kelas"),
              ("Evaluasi uji", "OA · IoU · F1\nper kelas"),
              ("Bobot model", "model.pt (32,55 M)\ndipakai inferensi")]
    n2 = len(tsteps)
    bw2 = (W - 2 * M - (n2 - 1) * gap) / n2
    for i, (a, b) in enumerate(tsteps):
        x = M + i * (bw2 + gap)
        box(c, x, y1 - 30, bw2, 62, a, b)
        if i:
            arrow(c, x - gap + 1, y1 + 1, x - 2, y1 + 1)
    # weights feed the classifier
    xs = M + (n2 - 1) * (bw2 + gap) + bw2 / 2
    xt = M + 5 * (bw + gap) + bw / 2
    c.setDash(3, 3)
    c.setStrokeColor(ACCENT)
    c.line(xs, y1 + 32, xs, y1 + 60)
    c.line(xs, y1 + 60, xt, y1 + 60)
    c.setDash()
    arrow(c, xt, y1 + 60, xt, y0 - 31, color=ACCENT)
    text(c, xt + 6, y1 + 66, "bobot", size=8, color=ACCENT)
    # data sources
    y2 = y1 - 110
    text(c, M, y2, "Sumber data (semua aman untuk komersial)", size=10.5, font="DVB")
    rows = [("Citra", "Sentinel-2 Collection-1 L2A COG — Element 84 Earth Search (AWS Open Data); "
                      "lisensi Copernicus free & open"),
            ("Label", "ESA WorldCover 2021 · Esri/Impact Observatory LULC 2021 · Descals et al. "
                      "oil palm extent 2021 — semuanya CC BY 4.0"),
            ("Bobot awal", "SSL4EO-S12 ResNet50 MoCo (Sentinel-2, 13 band) — CC BY 4.0"),
            ("Pustaka", "PyTorch, segmentation-models-pytorch, odc-stac, rasterio, FastAPI — "
                        "BSD / MIT / Apache-2.0")]
    for k, v in rows:
        y2 -= 18
        text(c, M, y2, k, size=9, font="DVB")
        text(c, M + 80, y2, v, size=9, color=INK2)


def architecture(c) -> None:
    page(c, "Arsitektur Deep Learning: U-Net + ResNet50", "2 · Arsitektur")
    # rows share a spatial resolution, so skip connections run straight across
    enc = [("Input", "13 × 256 × 256"), ("Stem conv 7×7", "64 × 128²"),
           ("Stage 1 · 3 blok", "256 × 64²"), ("Stage 2 · 4 blok", "512 × 32²"),
           ("Stage 3 · 6 blok", "1024 × 16²"), ("Stage 4 · 3 blok", "2048 × 8²")]
    dec = [("Decoder 5", "16 × 256²"), ("Decoder 4", "32 × 128²"), ("Decoder 3", "64 × 64²"),
           ("Decoder 2", "128 × 32²"), ("Decoder 1", "256 × 16²")]
    top, step, bw, bh = H - 110, 66, 118, 40
    ex, dx = M, M + 200
    blue, orange = HexColor("#e8f0fb"), HexColor("#fdf0e8")
    for i, (a, b) in enumerate(enc):
        y = top - i * step
        box(c, ex, y - bh, bw, bh, a, b, fill=blue if i else PANEL, stroke=ACCENT if i else RULE)
        if i:
            arrow(c, ex + bw / 2, y + step - bh - 1, ex + bw / 2, y + 1)
    for i, (a, b) in enumerate(dec):
        y = top - i * step
        box(c, dx, y - bh, bw, bh, a, b, fill=orange, stroke=HexColor("#eb6834"))
    for i in range(1, len(dec)):  # decoder flows upward
        y = top - i * step
        arrow(c, dx + bw / 2, y, dx + bw / 2, y + step - bh - 2)
    for i in range(1, 5):  # stem..stage3 -> decoder at the same resolution
        yy = top - i * step - bh / 2
        c.setDash(3, 3)
        c.setStrokeColor(MUTED)
        c.line(ex + bw + 1, yy, dx - 8, yy)
        c.setDash()
        arrow(c, dx - 9, yy, dx - 2, yy, color=MUTED)
    text(c, ex + bw + 8, top - step - bh / 2 + 5, "skip", size=7.5, color=MUTED)
    # bottleneck up into decoder 1
    yb = top - 5 * step - bh / 2
    c.setStrokeColor(INK2)
    c.line(ex + bw + 1, yb, dx + bw / 2, yb)
    arrow(c, dx + bw / 2, yb, dx + bw / 2, top - 4 * step - bh - 2)
    text(c, ex + bw + 8, yb + 5, "upsample ×2 + concat di tiap decoder", size=7.5, color=INK2)
    # head
    hx = dx + bw + 30
    box(c, hx, top - bh, 100, bh, "Head conv 3×3", "9 × 256²")
    arrow(c, dx + bw + 1, top - bh / 2, hx - 2, top - bh / 2)
    box(c, hx, top - step - bh, 100, bh, "Argmax + 1", "kelas 1–9 per piksel")
    arrow(c, hx + 50, top - bh - 1, hx + 50, top - step + 1)
    text(c, ex, top + 14, "Encoder ResNet50", size=9, font="DVB",
         color=ACCENT)
    text(c, dx, top + 14, "Decoder U-Net", size=9, font="DVB", color=HexColor("#c4521f"))
    text(c, ex, top - 6 * step + 8,
         "Ukuran = kanal × tinggi × lebar untuk tile 256 px. Inferensi memakai tile 512 px "
         "(overlap 64).", size=7.5, color=MUTED)
    # facts panel
    px = M + 470
    pw = W - M - px
    c.setFillColor(PANEL)
    c.roundRect(px, 70, pw, H - 170 + 40, 6, stroke=0, fill=1)
    y = H - 88
    y = text(c, px + 14, y, "Ringkasan model", size=11, font="DVB")
    rows = [("Arsitektur", "U-Net (smp 0.6)"),
            ("Encoder", "ResNet50 · 16 blok bottleneck"),
            ("Bobot awal", "SSL4EO-S12 MoCo, Sentinel-2 13 band"),
            ("Input", "13 kanal: 10 band (B02–B12) + NDVI, NDWI, NDBI"),
            ("Output", "9 kelas per piksel, 10 m"),
            ("", ""),
            ("Parameter total", "32.553.625 (~32,6 juta)"),
            ("  Encoder", "23.539.392"),
            ("  Decoder", "9.012.928"),
            ("  Head", "1.305"),
            ("", ""),
            ("Layer Conv2d", "64"),
            ("Layer BatchNorm2d", "63"),
            ("Kedalaman encoder", "50 layer (ResNet50)"),
            ("Ukuran bobot", "~130 MB (float32)")]
    for k, v in rows:
        y -= 4
        if not k:
            y -= 6
            continue
        sub = k.startswith("  ")
        text(c, px + 14 + (10 if sub else 0), y, k.strip(), size=8.8,
             font="DV" if sub else "DVB", color=INK2 if sub else INK)
        y = text(c, px + 130, y, v, size=8.8, color=INK2, width=pw - 144)


def training(c) -> None:
    page(c, "Proses Training", "3 · Training")
    rows = [re.findall(r"[\d.]+", ln) for ln in Path("runs/train.log").read_text(errors="ignore")
            .splitlines() if ln.startswith("epoch")]
    ep = np.array([int(r[0]) for r in rows])
    loss = np.array([float(r[2]) for r in rows])
    miou = np.array([float(r[3]) for r in rows])
    oa = np.array([float(r[4]) for r in rows])
    secs = np.array([float(r[5]) for r in rows])
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.0))
    comma = FuncFormatter(lambda v, _: f"{v:.2f}".replace(".", ","))
    for ax, y, title in [(axes[0], loss, "Training loss (cross-entropy)"),
                         (axes[1], miou, "Validasi mIoU"),
                         (axes[2], oa, "Validasi overall accuracy")]:
        ax.plot(ep, y, color=SERIES, lw=2, marker="o", ms=3.2, zorder=3)
        best = int(np.argmin(y) if ax is axes[0] else np.argmax(y))
        ax.scatter([ep[best]], [y[best]], s=46, color=SERIES, edgecolor="white", lw=1.5,
                   zorder=4)
        ax.annotate(f"{y[best]:.3f}".replace(".", ","), (ep[best], y[best]),
                    textcoords="offset points", xytext=(-8, 9),
                    ha="right", fontsize=8.5, color="#0b0b0b")
        ax.set_title(title, loc="left", fontsize=10, color="#0b0b0b", pad=8)
        ax.set_xlabel("Epoch")
        ax.grid(axis="y", color="#ecebe7", lw=0.8)
        ax.yaxis.set_major_formatter(comma)
        ax.set_xticks([1, 5, 10, 15, 20, 25])
    fig.tight_layout(w_pad=3)
    img = fig_image(fig)
    iw = W - 2 * M
    ih = iw * 3.0 / 11.2
    c.drawImage(img, M, H - 70 - ih, iw, ih)
    y = H - 70 - ih - 26
    col = (W - 2 * M - 24) / 3
    hw = [("Hardware", "MacBook Air 13\" M4 (Mac16,12)\nApple M4: CPU 10-core (4P + 6E), "
                       "GPU 10-core\nMemori 16 GB unified\nAkselerasi: PyTorch 2.15 · MPS (Metal)"
                       "\nmacOS 26.6"),
          ("Durasi", f"Training 25 epoch: ~{int(np.ceil(secs.sum() / 60)) // 60} j "
                     f"{int(np.ceil(secs.sum() / 60)) % 60} m "
                     f"(~{idn(np.ceil(secs.mean() / 6) / 10, 1)} menit/epoch)"
                     "\nPembuatan dataset: ~2 j 32 m (unduh citra, bandwidth ~53 MB/s)"
                     "\nUji coba awal (4 epoch): ~12 m\nEvaluasi uji: ~2 m"
                     "\nTotal semalam: ~6 j 20 m"),
          ("Konfigurasi", "Batch 16 · tile 256 px · 25 epoch\nAdamW, one-cycle lr maks 3e-4\n"
                          "Cross-entropy berbobot kelas (1/√frekuensi)\nAugmentasi flip + rotasi "
                          "90°\nCheckpoint = val mIoU terbaik (epoch 25)")]
    for i, (k, v) in enumerate(hw):
        x = M + i * (col + 12)
        c.setFillColor(PANEL)
        c.roundRect(x, y - 112, col, 120, 6, stroke=0, fill=1)
        yy = text(c, x + 12, y - 10, k, size=10.5, font="DVB")
        text(c, x + 12, yy - 4, v, size=8.8, color=INK2, lead=13, width=col - 24)
    text(c, W - M - 330, H - 64,
         "Validasi dihitung pada 1.536 tile hold-out (sel 1° terpisah).", size=7.5, color=MUTED)


def test_metrics(c) -> None:
    page(c, "Hasil Uji pada Wilayah Hold-out", "3b · Metrik uji")
    r = json.loads(Path("runs/eval.json").read_text())[0]
    names = [CLASSES[i][0] for i in range(1, 10)]
    names_en = [CLASSES[i][1] for i in range(1, 10)]
    f1s = [r["per_class"][n]["f1"] for n in names_en]
    f1_wo_bare = np.mean([f for n, f in zip(names_en, f1s, strict=True) if n != "Bare"])
    tiles = [("Overall accuracy", idn(r["oa"] * 100, 1) + " %"),
             ("mIoU", idn(r["miou"], 3)),
             ("Rata-rata F1 (tanpa lahan terbuka)", idn(f1_wo_bare, 3)),
             ("Piksel uji berlabel", idn(r["pixels"] / 1e6, 1) + " juta")]
    tw = (W - 2 * M - 3 * 12) / 4
    for i, (k, v) in enumerate(tiles):
        x = M + i * (tw + 12)
        c.setFillColor(PANEL)
        c.roundRect(x, H - 128, tw, 58, 6, stroke=0, fill=1)
        text(c, x + 12, H - 90, k, size=8.5, color=INK2)
        text(c, x + 12, H - 114, v, size=16, font="DVB")
    # per-class table
    y = H - 155
    cols = [M, M + 150, M + 225, M + 290, M + 345, M + 400]
    hdr = ["Kelas", "Precision", "Recall", "F1", "IoU", "Piksel uji"]
    conf = np.array(r["confusion"])
    iou = np.diag(conf) / (conf.sum(0) + conf.sum(1) - np.diag(conf))
    for x, h in zip(cols, hdr, strict=True):
        text(c, x, y, h, size=8.5, font="DVB", color=INK2)
    c.setStrokeColor(RULE)
    c.line(M, y - 5, M + 470, y - 5)
    for i, (n, ne) in enumerate(zip(names, names_en, strict=True)):
        y -= 19
        pc = r["per_class"][ne]
        c.setFillColor(hexc(CLASSES[i + 1][2]))
        c.rect(M, y - 2, 9, 9, stroke=0, fill=1)
        text(c, M + 14, y, n, size=8.8)
        for x, v in zip(cols[1:5], [pc["precision"], pc["recall"], pc["f1"], iou[i]],
                        strict=True):
            text(c, x, y, idn(v, 3), size=8.8, color=INK2)
        text(c, cols[5], y, idn(pc["support"], 0), size=8.8, color=INK2)
    # confusion heatmap (row-normalised recall)
    norm = conf / conf.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    short = ["Air", "Pohon", "Mangr.", "Sawit", "Tani", "Bangun", "Terbuka", "Semak",
             "L.basah"]
    ax.set_xticks(range(9), short, rotation=45, ha="right", fontsize=7.5)
    ax.set_yticks(range(9), short, fontsize=7.5)
    for i in range(9):
        for j in range(9):
            if norm[i, j] >= 0.01:
                ax.text(j, i, f"{norm[i, j] * 100:.0f}", ha="center", va="center", fontsize=6.5,
                        color="white" if norm[i, j] > 0.55 else "#0b0b0b")
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Referensi")
    ax.set_title("Confusion matrix (% per baris referensi)", loc="left", fontsize=9.5,
                 color="#0b0b0b")
    for s in ax.spines.values():
        s.set_visible(False)
    img = fig_image(fig)
    c.drawImage(img, M + 500, H - 150 - 300, 270, 300, preserveAspectRatio=True,
                anchor="n")
    notes = ("• Angka = kesepakatan dengan label konsensus pada piksel di mana 3 peta referensi "
             "setuju; akurasi lapangan pada seluruh piksel (termasuk transisi) akan lebih rendah."
             "\n• Lahan terbuka hanya 410 piksel uji: nilainya belum bermakna."
             "\n• Kebingungan utama: hutan ↔ sawit (~1,0 juta piksel hutan terprediksi sawit).")
    text(c, M, 92, notes, size=8.5, color=INK2, lead=12.5, width=470)


def render_aoi(name: str) -> tuple[ImageReader, ImageReader]:
    d = ISLANDS / name
    with rasterio.open(d / "composite.tif") as s:
        n = min(1400, s.width)
        rgb = s.read([1, 2, 3], out_shape=(3, n, n), resampling=Resampling.average)
    with rasterio.open(d / "landcover.tif") as s:
        lc = s.read(1, out_shape=(n, n), resampling=Resampling.nearest)
    rgb = (rgb.astype(np.float32) / 2500 * 255).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
    lut = np.zeros((256, 3), np.uint8)
    for k, v in CLASSES.items():
        lut[k] = v[2]
    lut[255] = (255, 255, 255)
    b1, b2 = io.BytesIO(), io.BytesIO()
    Image.fromarray(rgb).save(b1, "JPEG", quality=88)
    Image.fromarray(lut[lc]).save(b2, "PNG")
    b1.seek(0)
    b2.seek(0)
    return ImageReader(b1), ImageReader(b2)


def aoi_pages(c, idx: int, name: str, title: str, note: str) -> None:
    s = json.loads((ISLANDS / name / "summary.json").read_text())
    page(c, f"Sampel {idx}: {title}", f"4 · Sampel inferensi Q{s['imagery']['quarter']} "
                                     f"{s['imagery']['year']}")
    rgb, lc = render_aoi(name)
    side, gap = 312, 14
    y = H - 96 - side
    for k, (img, cap) in enumerate([(rgb, "Citra Sentinel-2 (komposit median, true colour)"),
                                    (lc, "Hasil klasifikasi tutupan lahan")]):
        x = M + k * (side + gap)
        c.drawImage(img, x, y, side, side)
        c.setStrokeColor(RULE)
        c.rect(x, y, side, side, stroke=1, fill=0)
        text(c, x, y + side + 6, cap, size=9, font="DVB", color=INK2)
    lx = M + 2 * (side + gap)
    c.setFillColor(PANEL)
    c.roundRect(lx, y, W - M - lx, side, 6, stroke=0, fill=1)
    yy = text(c, lx + 12, y + side - 20, "Simbologi kelas", size=10, font="DVB")
    for nid, nen, rgbc in CLASSES.values():
        yy -= 3
        c.setFillColor(hexc(rgbc))
        c.setStrokeColor(RULE)
        c.rect(lx + 12, yy - 2, 12, 12, stroke=1, fill=1)
        text(c, lx + 30, yy + 2, nid, size=8.8)
        yy = text(c, lx + 30, yy - 8, nen, size=7.2, color=MUTED) - 4
    c.setFillColor(white)
    c.setStrokeColor(RULE)
    c.rect(lx + 12, yy - 8, 12, 12, stroke=1, fill=1)
    text(c, lx + 30, yy - 4, "di luar AOI", size=8.8, color=INK2)
    x0, y0, x1, y1 = json.loads((ISLANDS / name / "aoi.geojson").read_text())[
        "features"][0]["geometry"]["coordinates"][0][0] + json.loads(
        (ISLANDS / name / "aoi.geojson").read_text())["features"][0]["geometry"][
        "coordinates"][0][2]
    text(c, M, y - 16, f"{note} · AOI {x0:.2f}–{x1:.2f} BT, {abs(y0):.2f}–{abs(y1):.2f} "
                       f"{'LS' if y1 < 0 else 'LS/LU'} · {s['crs']} · resolusi 10 m",
         size=8, color=MUTED)

    # ---- statistics page
    page(c, f"Statistik Sampel {idx}: {title}", "4 · Sampel inferensi — statistik")
    cls = sorted(s["classes"], key=lambda k: -k["percent"])
    fh = 0.3 * len(cls) + 0.7
    fig, ax = plt.subplots(figsize=(5.6, fh))
    ys = np.arange(len(cls))[::-1]
    colors = ["#{:02x}{:02x}{:02x}".format(*CLASSES[k["id"]][2]) for k in cls]
    ax.barh(ys, [k["percent"] for k in cls], color=colors, height=0.62,
            edgecolor="white", linewidth=1)
    for yv, k in zip(ys, cls, strict=True):
        lab = "<0,1 %" if k["percent"] < 0.1 else f"{k['percent']:.1f} %".replace(".", ",")
        ax.text(k["percent"] + 0.8, yv, lab, va="center", fontsize=8.5, color="#0b0b0b")
    ax.set_yticks(ys, [k["name"] for k in cls], fontsize=9)
    ax.set_xlim(0, max(k["percent"] for k in cls) * 1.18)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f} %"))
    ax.grid(axis="x", color="#ecebe7", lw=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Persentase tutupan lahan dalam AOI", loc="left", fontsize=10,
                 color="#0b0b0b")
    img = fig_image(fig)
    iw = 400
    ih = iw * fh / 5.6
    c.drawImage(img, M, H - 70 - ih, iw, ih)
    # class table under the chart
    y = H - 70 - ih - 22
    for x, h in zip([M, M + 190, M + 290], ["Kelas", "Luas (km²)", "Persentase"], strict=True):
        text(c, x, y, h, size=8.5, font="DVB", color=INK2)
    c.setStrokeColor(RULE)
    c.line(M, y - 5, M + 380, y - 5)
    for k in cls:
        y -= 15
        c.setFillColor(hexc(CLASSES[k["id"]][2]))
        c.rect(M, y - 2, 9, 9, stroke=0, fill=1)
        text(c, M + 14, y, k["name"], size=8.6)
        text(c, M + 190, y, idn(k["area_km2"], 2), size=8.6, color=INK2)
        text(c, M + 290, y, idn(k["percent"], 2) + " %", size=8.6, color=INK2)
    # facts panel
    px = M + 440
    pw = W - M - px
    c.setFillColor(PANEL)
    c.roundRect(px, 40, pw, H - 70 - 40, 6, stroke=0, fill=1)
    total = TOTAL_S[name]
    down = int(np.ceil((total - INFER_S - POST_S) / 10) * 10)
    facts = [("Luas total AOI", f"{idn(s['area_km2'], 1)} km²"),
             ("Periode", f"Q{s['imagery']['quarter']} {s['imagery']['year']} (Jul–Sep)"),
             ("Scene Sentinel-2 dipakai", f"{s['imagery']['scene_count']}"),
             ("Piksel bebas awan", f"{idn(s['clear_percent'], 1)} %"),
             ("Diisi kuartal sebelumnya", f"{idn(s['filled_percent'], 1)} %"),
             ("Tidak teramati (awan)", f"{idn(s['nodata_percent'], 1)} %"),
             ("", ""),
             ("Waktu unduh + komposit", f"~{down} s (~{idn(np.ceil(down / 6) / 10, 1)} mnt)"),
             ("Waktu inferensi model", f"~{INFER_S} s"),
             ("Waktu pasca-proses + tulis", f"~{POST_S} s"),
             ("Total (terukur)", f"{total} s ({idn(total / 60, 1)} menit)")]
    yy = H - 92
    yy = text(c, px + 14, yy, "Ringkasan proses", size=11, font="DVB") - 6
    for k, v in facts:
        if not k:
            yy -= 10
            continue
        text(c, px + 14, yy, k, size=9, color=INK2)
        text(c, px + 165, yy, v, size=9.5, font="DVB" if k.startswith("Total") else "DV")
        yy -= 20
    text(c, px + 14, 104,
         "Waktu diukur di MacBook Air M4 lewat jaringan rumah. Pembagian unduh / inferensi / "
         "pasca-proses adalah estimasi dibulatkan ke atas (~): inferensi ~0,6 s per blok "
         "1.152 px × 9 blok; sisanya dominan unduhan citra. Di server AWS us-west-2 (satu region "
         "dengan data) unduhan jauh lebih cepat.", size=7.5, color=MUTED, lead=10.5,
         width=pw - 28)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUT), pagesize=(W, H))
    c.setTitle("sentinel2-lc v0.1.0 — Laporan Teknis")
    c.setAuthor("Fairuz Akmal Pradana")
    cover(c)
    workflow(c)
    architecture(c)
    training(c)
    test_metrics(c)
    for i, (name, title, note) in enumerate(AOIS, 1):
        aoi_pages(c, i, name, title, note)
    c.save()
    print(OUT, page_no, "pages")


if __name__ == "__main__":
    main()
