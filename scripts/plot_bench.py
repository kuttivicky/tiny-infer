"""Render the M6 figures from bench_data.json into docs/.

Palette is the validated categorical set (slots 1-3), checked with the
data-viz validator: all-pairs CVD dE 9.2, normal-vision 24.0. Aqua sits at
2.74:1 on the light surface, below 3:1, so wherever it appears it carries a
visible direct label rather than leaning on the legend alone.

Note there is no twin-axis figure here. Two y-scales on one plot let the
author choose where the curves cross, which invents a relationship the data
does not contain; paired panels on a shared x-axis show the same story
without the distortion.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = json.load(open("bench_data.json"))
OUTDIR = "docs"

# --- validated palette + chrome -------------------------------------------
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"       # blue, orange, aqua
SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK2,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 200,
})


def style(ax, ylabel=None, xlabel=None):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.8)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)   # solid hairline
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)


def title(ax, text, sub=None):
    ax.set_title(text, loc="left", fontsize=12, color=INK, pad=24 if sub else 8)
    if sub:
        ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=9,
                color=INK2, va="bottom")


def legend(ax, **kw):
    lg = ax.legend(frameon=False, labelcolor=INK2, **kw)
    return lg


FOOT = (f"{DATA['meta']['model']} · {DATA['meta']['gpu']} · "
        f"{DATA['meta']['dtype']}")


def footer(fig):
    fig.text(0.008, 0.006, FOOT, fontsize=7.5, color=MUTED, ha="left")


# ========================================================== 1. KV cache
d = DATA["next_token_cost"]
x = d["context_lens"]

fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.plot(x, d["naive_ms"], color=S2, linewidth=2, marker="o", markersize=8,
        markeredgecolor=SURFACE, markeredgewidth=2, label="no cache (recompute context)")
ax.plot(x, d["cached_ms"], color=S1, linewidth=2, marker="o", markersize=8,
        markeredgecolor=SURFACE, markeredgewidth=2, label="KV cache")
ax.annotate(f"{d['naive_ms'][-1]:.0f} ms", (x[-1], d["naive_ms"][-1]),
            textcoords="offset points", xytext=(-8, 4), ha="right",
            fontsize=9.5, color=S2)
ax.annotate(f"{d['cached_ms'][-1]:.0f} ms — flat", (x[-1], d["cached_ms"][-1]),
            textcoords="offset points", xytext=(-8, 12), ha="right",
            fontsize=9.5, color=S1)
ratio = d["naive_ms"][-1] / d["cached_ms"][-1]
ax.annotate(f"{ratio:.0f}x at {x[-1]} tokens",
            (x[-1], (d["naive_ms"][-1] + d["cached_ms"][-1]) / 2),
            textcoords="offset points", xytext=(-14, -4), ha="right",
            fontsize=10, color=INK2)
style(ax, "time to produce one more token (ms)", "context length already in flight (tokens)")
ax.set_ylim(bottom=0)
title(ax, "The KV cache turns O(n) per token into O(1)",
      "Cost of the NEXT token at a given context length. Without a cache, every step re-attends over everything.")
legend(ax, loc="upper left")
footer(fig)
fig.tight_layout(rect=[0, 0.02, 1, 1])
fig.savefig(f"{OUTDIR}/01_kv_cache.png", bbox_inches="tight")
print("wrote 01_kv_cache.png")

# ========================================================== 2. batch sweep
sw = DATA["batch_sweep"]
bs = [r["max_batch"] for r in sw]
tps = [r["tok_per_s"] for r in sw]
ttft = [r["ttft_p50_ms"] / 1000 for r in sw]

# Deliberately two panels on a shared x, never one plot with two y-scales.
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 5.6), sharex=True)

ax1.plot(bs, tps, color=S1, linewidth=2, marker="o", markersize=8,
         markeredgecolor=SURFACE, markeredgewidth=2)
for xi, yi in zip(bs, tps):
    ax1.annotate(f"{yi:.0f}", (xi, yi), textcoords="offset points",
                 xytext=(0, 11), ha="center", fontsize=9, color=INK2)
style(ax1, "throughput (tok/s)")
ax1.set_ylim(0, max(tps) * 1.22)
title(ax1, "Throughput rises and latency falls together",
      f"{len(DATA['batch_sweep'])} batch sizes, 8 requests x 48 tokens. Same work; only the scheduling differs.")

ax2.plot(bs, ttft, color=S2, linewidth=2, marker="o", markersize=8,
         markeredgecolor=SURFACE, markeredgewidth=2)
for xi, yi in zip(bs, ttft):
    ax2.annotate(f"{yi:.1f}s", (xi, yi), textcoords="offset points",
                 xytext=(0, 11), ha="center", fontsize=9, color=INK2)
style(ax2, "TTFT p50 (s)", "max batch size")
ax2.set_ylim(0, max(ttft) * 1.22)
ax2.set_xticks(bs)
footer(fig)
fig.tight_layout(rect=[0, 0.02, 1, 1])
fig.savefig(f"{OUTDIR}/02_batch_sweep.png", bbox_inches="tight")
print("wrote 02_batch_sweep.png")

# ========================================================== 3. the money plot
h = DATA["hetero"]
fig, (axa, axb) = plt.subplots(2, 1, figsize=(7.2, 6.2),
                               gridspec_kw={"height_ratios": [1, 1.35]})

names = ["static batching", "continuous batching"]
walls = [h["static"]["wall_s"], h["continuous"]["wall_s"]]
rates = [h["static"]["tok_per_s"], h["continuous"]["tok_per_s"]]
bars = axa.barh(names, walls, color=[S2, S1], height=0.52, zorder=2)
for b, w, r in zip(bars, walls, rates):
    axa.text(w + max(walls) * 0.015, b.get_y() + b.get_height() / 2,
             f"{w:.1f}s   ({r:.0f} tok/s)", va="center", fontsize=9.5, color=INK2)
axa.set_xlim(0, max(walls) * 1.30)
style(axa, None, "wall clock to finish all 8 requests (s)")
axa.grid(axis="y", linewidth=0)
axa.grid(axis="x", color=GRID, linewidth=0.8)
axa.invert_yaxis()
axa.tick_params(axis="y", labelsize=10, labelcolor=INK)
speedup = h["static"]["wall_s"] / h["continuous"]["wall_s"]
title(axa, f"Uneven output lengths: continuous batching finishes {speedup:.2f}x sooner",
      "8 requests asking for 8-96 tokens each. Static holds every slot until the longest finishes.")

for label, key, colr in (("static", "static", S2), ("continuous", "continuous", S1)):
    occ = [100 * u / b for u, b in
           zip(h[key]["useful_sizes"], h[key]["batch_sizes"])]
    axb.plot(range(len(occ)), occ, color=colr, linewidth=2, label=f"{label} batching")
    mean = sum(occ) / len(occ)
    axb.annotate(f"mean {mean:.0f}%", (len(occ) - 1, occ[-1]),
                 textcoords="offset points", xytext=(-8, 8 if colr == S1 else -16),
                 ha="right", fontsize=9, color=colr)
style(axb, "useful rows in batch (%)", "decode step")
axb.set_ylim(0, 108)
legend(axb, loc="lower left")
axb.set_title("Batch occupancy over time", loc="left", fontsize=11, color=INK, pad=8)
footer(fig)
fig.tight_layout(rect=[0, 0.02, 1, 1])
fig.savefig(f"{OUTDIR}/03_static_vs_continuous.png", bbox_inches="tight")
print("wrote 03_static_vs_continuous.png")

# ========================================================== 4. KV memory
km = DATA["kv_memory"]
rows = km["rows"]
conc = [r["concurrency"] for r in rows]

fig, (axl, axr) = plt.subplots(1, 2, figsize=(9.4, 4.2),
                               gridspec_kw={"width_ratios": [1.45, 1]})

axl.plot(conc, [r["contiguous_mb"] for r in rows], color=S2, linewidth=2,
         label="contiguous: reserved")
axl.plot(conc, [r["paged_mb"] for r in rows], color=S1, linewidth=2,
         label="paged: reserved")
axl.plot(conc, [r["used_mb"] for r in rows], color=S3, linewidth=2,
         label="actually used")
# Aqua is below 3:1 on this surface — direct-label it rather than rely on hue.
last = rows[-1]
axl.annotate("actually used", (conc[-1], last["used_mb"]),
             textcoords="offset points", xytext=(-6, -16), ha="right",
             fontsize=9, color=S3)
axl.annotate(f"{last['contiguous_mb']:.0f} MB", (conc[-1], last["contiguous_mb"]),
             textcoords="offset points", xytext=(-6, 7), ha="right",
             fontsize=9, color=S2)
axl.set_yscale("log")
style(axl, "KV memory (MB, log scale)", "concurrent sequences")
axl.set_xticks(conc)
title(axl, "Paged KV reserves only what it needs",
      f"Contiguous reserves max_seq={km['contiguous_max_seq']} slots per sequence regardless of length.")
legend(axl, loc="center left")

waste_c = 100 * (1 - last["used_tokens"] / last["contiguous_slots"])
waste_p = 100 * (1 - last["used_tokens"] / last["paged_slots"])
b2 = axr.bar(["contiguous", "paged"], [waste_c, waste_p],
             color=[S2, S1], width=0.5, zorder=2)
for b, w in zip(b2, [waste_c, waste_p]):
    axr.text(b.get_x() + b.get_width() / 2, w + 2.5, f"{w:.0f}%",
             ha="center", fontsize=13, color=INK)
style(axr, "reserved but unused (%)")
axr.set_ylim(0, 108)
axr.tick_params(axis="x", labelsize=10, labelcolor=INK)
axr.set_title(f"Waste at {conc[-1]} concurrent sequences", loc="left",
              fontsize=11, color=INK, pad=8)
footer(fig)
fig.tight_layout(rect=[0, 0.02, 1, 1])
fig.savefig(f"{OUTDIR}/04_kv_memory.png", bbox_inches="tight")
print("wrote 04_kv_memory.png")
