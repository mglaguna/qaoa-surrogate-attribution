"""Shared style, validation and provenance helpers for the Energy Hub figures.

Every figure script:
  - reads only from energy_hub/data/ (canonical_5k/, attribution/, reference/);
  - validates protocol_version == EH_CANONICAL_V1 and the recorded config_hash
    before drawing anything;
  - never calls QAOA or IPOPT (figures are derived from the archived campaign
    outputs, or from a deterministic enumeration over the fixed instance).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIGURES_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = FIGURES_ROOT.parent
CODE_DIR = PACKAGE_ROOT / "code"
OUTPUTS_DIR = PACKAGE_ROOT / "data" / "canonical_5k"
ANALYSIS_DIR = OUTPUTS_DIR / "analysis"
INPUTS_DIR = PACKAGE_ROOT / "data" / "reference"
DATA_DIR = FIGURES_ROOT / "data"
FINAL_DIR = FIGURES_ROOT / "main"
EXPECTED_PROTOCOL_VERSION = "EH_CANONICAL_V1"

COLORS = {"S0": "#0072B2", "S1": "#D55E00", "S2": "#009E73", "Uniform": "#999999"}
LINESTYLES = {"S0": "-", "S1": "--", "S2": "-.", "Uniform": ":"}
MARKERS = {"S0": "o", "S1": "s", "S2": "^", "Uniform": "D"}
ARMS_ORDER = ["S0", "S1", "S2", "Uniform"]

MAX_WIDTH_IN = 190 / 25.4
MIN_FONTSIZE = 7.0


class FigureInputError(RuntimeError):
    pass


def validate_protocol_and_hash() -> dict:
    """Validate protocol_version and config_hash before any figure is drawn.

    Raises FigureInputError (not a bare assertion) so the master script can
    report a clean failure instead of a traceback if the canonical outputs
    are missing or from a different protocol version.
    """
    config_path = OUTPUTS_DIR / "canonical_config.json"
    manifest_path = OUTPUTS_DIR / "manifest.json"
    if not config_path.exists():
        raise FigureInputError(f"Missing {config_path}")
    if not manifest_path.exists():
        raise FigureInputError(f"Missing {manifest_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if config.get("protocol_version") != EXPECTED_PROTOCOL_VERSION:
        raise FigureInputError(
            f"canonical_config.json protocol_version={config.get('protocol_version')!r} "
            f"!= expected {EXPECTED_PROTOCOL_VERSION!r}"
        )
    if manifest.get("protocol_version") != EXPECTED_PROTOCOL_VERSION:
        raise FigureInputError(
            f"manifest.json protocol_version={manifest.get('protocol_version')!r} "
            f"!= expected {EXPECTED_PROTOCOL_VERSION!r}"
        )
    if config.get("config_hash") != manifest.get("config_hash"):
        raise FigureInputError("canonical_config.json and manifest.json config_hash disagree.")
    if manifest.get("final_status") != "PASS":
        raise FigureInputError(f"manifest.json final_status={manifest.get('final_status')!r}, expected PASS.")

    return {"config": config, "manifest": manifest}


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FigureInputError(f"Required figure input not found: {path}")
    return path


def setup_style():
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 600,
        }
    )


def panel_label(ax, label: str):
    ax.text(-0.12, 1.05, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top")


def light_grid(ax):
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_axisbelow(True)


def make_grayscale_preview(png_path: Path) -> Path:
    from PIL import Image

    out = png_path.with_name(png_path.stem + "_preview_gray.png")
    img = Image.open(png_path).convert("L")
    img.save(out)
    return out


def save_all(fig, name: str, config_hash: str, source_files: list[str], description: str, out_dir: Path | None = None) -> dict:
    """Save PNG/PDF/SVG + grayscale preview + a provenance record for one figure.

    out_dir defaults to FINAL_DIR (figures/main/); pass FIGURES_ROOT / "supplement"
    for a supplementary figure.
    """
    out_dir = out_dir if out_dir is not None else FINAL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{name}.png"
    pdf = out_dir / f"{name}.pdf"
    svg = out_dir / f"{name}.svg"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    gray = make_grayscale_preview(png)
    plt.close(fig)

    provenance = {
        "figure": name,
        "description": description,
        "protocol_version": EXPECTED_PROTOCOL_VERSION,
        "config_hash": config_hash,
        "source_files": source_files,
        "outputs": [str(p.relative_to(PACKAGE_ROOT)) for p in (png, pdf, svg, gray)],
    }
    (out_dir / f"{name}_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )
    return provenance
