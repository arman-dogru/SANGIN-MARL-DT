"""Generate the manuscript's vector-layout diagrams as high-resolution PNGs."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"paper/figures"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    fig, ax = plt.subplots(figsize=(6.6, 2.5))
    ax.set(xlim=(0, 10), ylim=(0, 4)); ax.axis("off")
    points = [(1.1, 1, "Robots\nPANAMA actions"), (3.6, 1, "Drone-BS\nfinite CPU pool"),
              (6, 3, "LEO\nbackhaul"), (8.6, 1, "Gateway + cloud\nfinite CPU pool")]
    for x, y, label in points:
        ax.add_patch(FancyBboxPatch((x-.95, y-.55), 1.9, 1.1, boxstyle="round,pad=.06", fc="#edf2f5", ec="#18394a"))
        ax.text(x, y, label, ha="center", va="center")
    for left, right in (((2.1, 1), (2.55, 1)), ((4.25, 1.6), (5.2, 2.4)), ((6.8, 2.4), (8.0, 1.6))):
        ax.annotate("", xy=right, xytext=left, arrowprops=dict(arrowstyle="<->", color="#235a75", lw=1.5))
    ax.text(2.35, .25, "G2A", ha="center")
    ax.text(4.5, 2.7, "A2S", ha="center")
    ax.text(7.9, 2.8, "Gateway link", ha="center")
    ax.text(5.1, -.03, "Reports → allocation → computation → timestamped guidance and state", ha="center", fontsize=8)
    fig.tight_layout(pad=.3); fig.savefig(OUT/"system_model.png", dpi=300); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.6, 1.8)); ax.axis("off")
    labels = ["Sense +\nupload", "Batch wait +\nallocation", "Optional\nLEO upload", "Finite CPU\nservice", "Response +\nreturn", "Cache +\ndecision"]
    for i, label in enumerate(labels):
        ax.add_patch(FancyBboxPatch((i*1.15, .4), 1.0, .8, boxstyle="round,pad=.02", ec="#18394a", fc="#edf2f5"))
        ax.text(i*1.15+.5, .8, label, ha="center", va="center", fontsize=8)
        if i < len(labels)-1:
            ax.annotate("", xy=(i*1.15+1.14, .8), xytext=(i*1.15+1.02, .8), arrowprops=dict(arrowstyle="->", color="#235a75"))
    ax.set(xlim=(-.1, 7), ylim=(0, 1.6))
    ax.text(0, 1.4, r"Source time $t^{src}$", fontsize=8)
    ax.text(4.5, 1.4, r"Generation $t^{gen}$", fontsize=8, ha="center")
    ax.text(6.4, 1.4, r"Decision $t_n$", fontsize=8, ha="center")
    ax.text(3.3, .05, r"End-to-end AoI = $t_n-t^{src}$; source time survives all stages", ha="center", fontsize=8)
    fig.tight_layout(pad=.3); fig.savefig(OUT/"timing.png", dpi=300); plt.close(fig)


if __name__ == "__main__":
    main()
