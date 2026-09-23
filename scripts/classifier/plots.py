"""Figure helpers for the classifier report (static PNGs).

Colour follows the data-viz reference palette: one series colour (blue), a
neutral grey for context, text in ink colours, a one-hue blue ramp for
magnitudes, recessive grids. Fifteen classes are too many for distinct hues,
so the class-coloured projection is drawn as small multiples instead.
"""
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
CONTEXT = "#c9c8c2"
SERIES = "#2a78d6"
BLUE_RAMP = LinearSegmentedColormap.from_list(
    "blue_ramp", [SURFACE, "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlesize": 13,
    "axes.labelsize": 11, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False, "grid.color": GRID, "grid.linewidth": 0.8,
})


def wrap(label, width=28):
    return "\n".join(textwrap.wrap(label, width))


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def class_distribution(counts, path, total):
    """counts: [(class, n)] sorted largest first. Horizontal bars, log scale."""
    labels = [c for c, _ in counts][::-1]
    values = [n for _, n in counts][::-1]
    fig, ax = plt.subplots(figsize=(9, 6.2))
    ax.barh(labels, values, color=SERIES, height=0.68)
    ax.set_xscale("log")
    ax.set_xlim(4, max(values) * 3.2)
    ax.grid(axis="x", which="major")
    ax.set_axisbelow(True)
    for y, v in enumerate(values):
        ax.text(v * 1.08, y, f"{v}  ({100 * v / total:.1f}%)", va="center", fontsize=9, color=INK_2)
    ax.set_xlabel("Schemes (log scale)")
    ax.set_ylabel("Category (silver label)")
    ax.set_title(f"Class distribution: {total:,} schemes, 15 categories")
    save(fig, path)


def length_by_class(lengths_by_class, path):
    """lengths_by_class: {class: [word counts]}; boxes sorted by median."""
    order = sorted(lengths_by_class, key=lambda c: np.median(lengths_by_class[c]))
    fig, ax = plt.subplots(figsize=(9, 6.6))
    box = ax.boxplot([lengths_by_class[c] for c in order], vert=False, widths=0.6, patch_artist=True,
                     showfliers=True, flierprops={"marker": "o", "markersize": 2.5, "markerfacecolor": CONTEXT,
                                                  "markeredgecolor": CONTEXT},
                     medianprops={"color": INK, "linewidth": 1.6},
                     whiskerprops={"color": INK_2}, capprops={"color": INK_2})
    for patch in box["boxes"]:
        patch.set(facecolor="#cde2fb", edgecolor=SERIES, linewidth=1.2)
    ax.set_yticks(range(1, len(order) + 1), [f"{c} (n={len(lengths_by_class[c])})" for c in order])
    ax.set_xscale("log")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    ax.set_xlabel("Input text length in words (name + description + benefits, log scale)")
    ax.set_title("Text length by class")
    save(fig, path)


def projection_small_multiples(xy, labels, classes, path, variance):
    """One panel per class: that class in blue over all schemes in grey."""
    cols, rows = 5, int(np.ceil(len(classes) / 5))
    fig, axes = plt.subplots(rows, cols, figsize=(16, 3.3 * rows), sharex=True, sharey=True)
    labels = np.asarray(labels)
    for ax, cls in zip(axes.flat, classes):
        mask = labels == cls
        ax.scatter(xy[~mask, 0], xy[~mask, 1], s=3, color=CONTEXT, alpha=0.5, linewidths=0)
        ax.scatter(xy[mask, 0], xy[mask, 1], s=9, color=SERIES, alpha=0.9, linewidths=0)
        ax.set_title(f"{wrap(cls, 30)} (n={mask.sum()})", fontsize=9.5, fontweight="bold")
        ax.tick_params(labelsize=8)
        ax.grid(True)
        ax.set_axisbelow(True)
    for ax in list(axes.flat)[len(classes):]:
        ax.set_visible(False)
    fig.supxlabel(f"PC 1 ({100 * variance[0]:.1f}% of variance)", color=INK_2)
    fig.supylabel(f"PC 2 ({100 * variance[1]:.1f}% of variance)", color=INK_2)
    fig.suptitle("bge-m3 embeddings (scheme name + description), PCA to 2D: each class against all schemes",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    save(fig, path)


def confusion(matrix, classes, path, title, normalised):
    """Rows = true class, columns = predicted. One-hue blue ramp."""
    fig, ax = plt.subplots(figsize=(11.5, 10))
    shown = matrix.astype(float)
    image = ax.imshow(shown, cmap=BLUE_RAMP, vmin=0, vmax=1 if normalised else shown.max())
    ticks = [wrap(c, 24) for c in classes]
    ax.set_xticks(range(len(classes)), ticks, rotation=60, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(classes)), ticks, fontsize=8.5)
    ax.set_xlabel("Predicted category")
    ax.set_ylabel("Silver label (true class for this evaluation)")
    ax.set_title(title)
    ax.spines[:].set_visible(False)
    threshold = 0.55 * (1 if normalised else shown.max())
    for i in range(len(classes)):
        for j in range(len(classes)):
            v = matrix[i, j]
            if v == 0:
                continue
            text = (f"{v:.2f}" if v >= 0.005 else "<0.01") if normalised else f"{int(v)}"
            ax.text(j, i, text, ha="center", va="center", fontsize=7.5,
                    color="#ffffff" if shown[i, j] > threshold else INK)
    bar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    bar.set_label("Share of the true class" if normalised else "Schemes", color=INK_2)
    save(fig, path)
