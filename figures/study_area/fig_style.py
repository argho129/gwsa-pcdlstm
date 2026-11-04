"""
fig_style.py
Manuscript figure styling (Times New Roman, 800 dpi) and a panel composer.

Place in D:\\THESIS\\Arpon_Thesis_Final\\Codes\\Into the Models\\
and import at the TOP of every plotting script, before any figure is created:

    from fig_style import apply_style, save_fig
    apply_style()

Then replace every plt.savefig(...) with save_fig(fig, "name").
"""

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ----------------------------------------------------------------------
# 1. STYLE
# ----------------------------------------------------------------------

DPI = 800

# Times New Roman on Windows; metric-compatible fallbacks elsewhere.
SERIF_STACK = ["Times New Roman", "Liberation Serif", "TeX Gyre Termes",
               "Nimbus Roman", "DejaVu Serif"]


def _resolve_serif():
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in SERIF_STACK:
        if name in available:
            return name
    return "serif"


def apply_style(base=11, scale=1.0):
    """
    base  : font size (pt) of axis tick labels at FINAL printed size.
    scale : multiply every size, e.g. scale=1.6 for a panel that will be
            shrunk to ~60% width when placed in the composite figure.

    Rule of thumb for a journal: nothing below 7 pt after the figure is
    scaled into the page. Panel (a) in the current draft is roughly 6x too
    small once the map dominates the layout, so render it with scale~1.8.
    """
    f = _resolve_serif()
    if f == "serif":
        print("WARNING: no Times-like font found; falling back to generic serif.")

    s = base * scale
    matplotlib.rcParams.update({
        # --- fonts ---
        "font.family": "serif",
        "font.serif": SERIF_STACK,
        "mathtext.fontset": "stix",          # Times-like math, no Type-3 issues
        "axes.unicode_minus": False,          # ASCII hyphen, safer in PDFs

        "font.size":        s,
        "axes.titlesize":   s * 1.15,
        "axes.labelsize":   s * 1.05,
        "xtick.labelsize":  s,
        "ytick.labelsize":  s,
        "legend.fontsize":  s,
        "figure.titlesize": s * 1.30,

        # --- lines / ticks: thicken so they survive downscaling ---
        "axes.linewidth":   0.9 * scale,
        "lines.linewidth":  1.6 * scale,
        "lines.markersize": 5.0 * scale,
        "xtick.major.width": 0.9 * scale,
        "ytick.major.width": 0.9 * scale,
        "xtick.major.size":  3.5 * scale,
        "ytick.major.size":  3.5 * scale,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "grid.linewidth":  0.5 * scale,

        # --- output ---
        "figure.dpi":  120,          # on-screen only
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,          # embed TrueType, NOT Type 3
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })
    return f


def save_fig(fig, stem, outdir=".", also_pdf=True):
    """Write <stem>.png at 800 dpi (and a vector PDF, which most journals prefer)."""
    import os
    os.makedirs(outdir, exist_ok=True)
    png = os.path.join(outdir, stem + ".png")
    fig.savefig(png, dpi=DPI)
    if also_pdf:
        fig.savefig(os.path.join(outdir, stem + ".pdf"))
    print("wrote", png)
    return png


# ----------------------------------------------------------------------
# 2. COMPOSER  -- stitch panels (a), (b), (c) into one labelled figure
# ----------------------------------------------------------------------

def compose(panel_paths, out_path, width_mm=190, dpi=DPI,
            labels=("(a)", "(b)", "(c)"), label_pt=12, gap_mm=3):
    """
    Vertically stack already-rendered panel PNGs into a single manuscript
    figure of exact physical width, with (a)/(b)/(c) labels drawn in the
    same serif face.

    width_mm : 190 for a double-column figure, 90 for single column.

    NOTE: render each panel with apply_style(scale=...) chosen so that its
    text ends up the SAME apparent size after the panel is resized here.
    If panel (b) is rendered 2x wider than panel (a), give (a) half the
    scale of (b), otherwise the map labels will look tiny next to the
    line plots -- which is the main problem in the current draft.
    """
    from PIL import Image, ImageDraw, ImageFont
    import os

    W = int(round(width_mm / 25.4 * dpi))
    gap = int(round(gap_mm / 25.4 * dpi))

    ims = []
    for p in panel_paths:
        im = Image.open(p).convert("RGB")
        h = int(round(im.height * W / im.width))
        ims.append(im.resize((W, h), Image.LANCZOS))

    total_h = sum(im.height for im in ims) + gap * (len(ims) - 1)
    canvas = Image.new("RGB", (W, total_h), "white")

    # label font
    fpath = None
    for cand in [r"C:\Windows\Fonts\timesbd.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf"]:
        if os.path.exists(cand):
            fpath = cand
            break
    px = int(round(label_pt / 72 * dpi))
    font = ImageFont.truetype(fpath, px) if fpath else ImageFont.load_default()

    draw = ImageDraw.Draw(canvas)
    y = 0
    for im, lab in zip(ims, labels):
        canvas.paste(im, (0, y))
        draw.text((int(px * 0.3), y + int(px * 0.2)), lab, fill="black", font=font)
        y += im.height + gap

    canvas.save(out_path, dpi=(dpi, dpi))
    print(f"wrote {out_path}  {canvas.width}x{canvas.height}px = "
          f"{width_mm} mm at {dpi} dpi")
    return out_path


if __name__ == "__main__":
    apply_style()
    print("resolved serif:", _resolve_serif())