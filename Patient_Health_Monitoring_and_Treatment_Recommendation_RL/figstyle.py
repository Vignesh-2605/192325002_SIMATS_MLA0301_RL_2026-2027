"""Shared plotting style + diagram primitives (plain technical drawing)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Ellipse
from matplotlib.path import Path
import matplotlib.patches as mpatches

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10.5,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.5,
    "axes.edgecolor": "#444444",
    "axes.linewidth": 0.9,
    "axes.grid": True,
    "grid.color": "#d9d9d9",
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",
    "legend.frameon": True,
    "legend.framealpha": 1.0,
    "legend.edgecolor": "#bbbbbb",
    "legend.fontsize": 8.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

# muted, print-safe palette
INK = "#333333"
BLUE = "#2f6f9f"
LBLUE = "#dbe7f0"
GREEN = "#3f7d5a"
LGREEN = "#dcebe2"
ORANGE = "#c9762f"
LORANGE = "#f5e6d3"
RED = "#a8342f"
LRED = "#f2dcda"
GREY = "#6e6e6e"
LGREY = "#ececec"
PURPLE = "#6b5b8e"
LPURPLE = "#e5e0ee"


def box(ax, x, y, w, h, text, fc=LGREY, ec=INK, fs=8.5, weight="normal",
        style="round,pad=0.02,rounding_size=0.03", lw=1.0, ha="center"):
    """Rectangular process box with centred wrapped text."""
    p = FancyBboxPatch((x, y), w, h, boxstyle=style, linewidth=lw,
                       edgecolor=ec, facecolor=fc, mutation_scale=1)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=INK, fontweight=weight, linespacing=1.35)
    return (x, y, w, h)


def diamond(ax, cx, cy, w, h, text, fc=LORANGE, ec=INK, fs=8):
    pts = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2),
           (cx - w / 2, cy)]
    ax.add_patch(mpatches.Polygon(pts, closed=True, facecolor=fc,
                                  edgecolor=ec, linewidth=1.0))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=INK,
            linespacing=1.3)


def oval(ax, cx, cy, w, h, text, fc=LGREEN, ec=INK, fs=8.5, weight="bold"):
    ax.add_patch(Ellipse((cx, cy), w, h, facecolor=fc, edgecolor=ec,
                         linewidth=1.0))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            fontweight=weight, color=INK)


def arrow(ax, p0, p1, style="-|>", rad=0.0, color=INK, lw=1.0, ls="-",
          label=None, lpos=0.5, fs=7.5, lcolor=None, dx=0.0, dy=0.0):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=11,
                        connectionstyle=f"arc3,rad={rad}", color=color,
                        linewidth=lw, linestyle=ls, shrinkA=1, shrinkB=1,
                        zorder=1)
    ax.add_patch(a)
    if label:
        mx = p0[0] + (p1[0] - p0[0]) * lpos + dx
        my = p0[1] + (p1[1] - p0[1]) * lpos + dy
        ax.text(mx, my, label, fontsize=fs, ha="center", va="center",
                color=lcolor or color,
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec="none", alpha=0.92))


def canvas(w=9, h=6, xlim=(0, 10), ylim=(0, 10), title=None):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.axis("off")
    ax.set_aspect("auto")
    if title:
        ax.set_title(title, fontsize=11.5, fontweight="bold", pad=8)
    return fig, ax


def caption(fig, text):
    fig.text(0.5, -0.015, text, ha="center", va="top", fontsize=8.2,
             color="#555555", style="italic")


# ---- Unicode glyph constants (written via escapes for portability) ----
G_GAMMA = "\u03b3"; G_PI = "\u03c0"; G_SIGMA = "\u03a3"; G_ALPHA = "\u03b1"
G_BETA = "\u03b2"; G_DELTA = "\u03b4"; G_THETA = "\u03b8"; G_EPS = "\u03b5"
G_IN = "\u2208"; G_RARR = "\u2192"; G_LARR = "\u2190"; G_UARR = "\u2191"
G_STAR = "\u2217"; G_PRIME = "\u2032"; G_LEQ = "\u2264"; G_GEQ = "\u2265"
G_APPROX = "\u2248"; G_TIMES = "\u00d7"; G_BULL = "\u2022"; G_DASH = "\u2014"
G_INF = "\u221e"; G_PM = "\u00b1"; G_SQRT = "\u221a"; G_NEQ = "\u2260"
SUB = ["\u2080", "\u2081", "\u2082", "\u2083", "\u2084", "\u2085",
       "\u2086", "\u2087", "\u2088", "\u2089"]
SUP_T = "\u1d57"
