#!/usr/bin/env python3
"""
Drought Analysis & Geoengineering Visualization Pipeline

End-to-end pipeline that:
  1. Runs GraphCast control forecast and plots weather maps
  2. Analyzes drought regions using the DroughtDetector (DSI-based)
  3. Applies each drought-mitigation seeding mask to the initial state
  4. Runs GraphCast forecast for each intervention
  5. Re-analyzes drought to measure effectiveness
  6. Identifies the most effective intervention
  7. Produces broadcast-style plots for every stage

All outputs saved to ~/treehacks/drought_analysis/

Usage:
    python drought_pipeline.py
    python drought_pipeline.py --intensity 0.7
    python drought_pipeline.py --mask-only  # skip GraphCast inference (fast)
"""

import dataclasses
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Local imports
sys.path.insert(0, os.path.dirname(__file__))
from drought_detector import DroughtDetector, DROUGHT_CLASSES
from geoengineering_masks import (
    GeoEngineeringSimulator,
    TargetRegion,
    DROUGHT_MASKS,
    MASK_REGISTRY,
    resolve_var,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphcast_data")
OUTPUT_DIR = os.path.expanduser("~/treehacks/drought_analysis")

# GraphCast GCS config
GCS_BUCKET = "dm_graphcast"
PARAMS_FILE = (
    "params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 "
    "- pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
)
DATASET_FILE = (
    "dataset/source-era5_date-2022-01-01_res-1.0_levels-13_steps-20.nc"
)
STATS_FILES = {
    "diffs_stddev_by_level": "stats/diffs_stddev_by_level.nc",
    "mean_by_level": "stats/mean_by_level.nc",
    "stddev_by_level": "stats/stddev_by_level.nc",
}

# Drought class colors (matches US Drought Monitor palette)
DROUGHT_COLORS = {
    0: (0.95, 0.95, 0.95, 0.0),   # None - transparent
    1: (1.0, 0.95, 0.0, 0.7),     # D0 Abnormally Dry - yellow
    2: (0.99, 0.82, 0.49, 0.85),   # D1 Moderate - tan/peach
    3: (0.93, 0.49, 0.23, 0.9),    # D2 Severe - orange
    4: (0.75, 0.15, 0.15, 0.95),   # D3 Extreme - dark red
}

DROUGHT_LABELS = {
    0: "None",
    1: "D0: Abnormally Dry",
    2: "D1: Moderate Drought",
    3: "D2: Severe Drought",
    4: "D3: Extreme Drought",
}

INTERVENTION_DISPLAY = {
    "glaciogenic_static":          "Glaciogenic\n(Static)",
    "hygroscopic_enhancement":     "Hygroscopic\n(Giant CCN)",
    "electric_ionization":         "Electric\n(Ionization)",
    "laser_induced_condensation":  "Laser\n(LIC)",
}

# Shared visual style
PROJ = ccrs.Robinson()
DATA_CRS = ccrs.PlateCarree()
FIG_BG = "#0a0a2e"
AX_BG = "#0d1b2a"


# ---------------------------------------------------------------------------
# Utility: Kelvin / Pressure conversions
# ---------------------------------------------------------------------------
def _kelvin_to_f(k):
    return (k - 273.15) * 9 / 5 + 32

def _pa_to_hpa(pa):
    return pa / 100.0


# ---------------------------------------------------------------------------
# Step 1: Load data + run GraphCast
# ---------------------------------------------------------------------------
def download_data():
    """Download GraphCast model + ERA5 data from GCS."""
    from google.cloud import storage

    def _dl(blob, dest):
        if os.path.exists(dest):
            print(f"  [cached] {os.path.basename(dest)}")
            return
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        print(f"  Downloading {blob} ...")
        client = storage.Client.create_anonymous_client()
        bucket = client.bucket(GCS_BUCKET)
        bucket.blob(blob).download_to_filename(dest)
        print(f"  Done ({os.path.getsize(dest) / 1e6:.1f} MB)")

    print("Downloading data ...")
    pp = os.path.join(CACHE_DIR, "params.npz"); _dl(PARAMS_FILE, pp)
    dp = os.path.join(CACHE_DIR, "example_data_20steps.nc"); _dl(DATASET_FILE, dp)
    sp = {}
    for n, b in STATS_FILES.items():
        p = os.path.join(CACHE_DIR, f"{n}.nc"); _dl(b, p); sp[n] = p
    return pp, dp, sp


def run_graphcast(params_path, dataset_path, stats_paths, seeded_inputs=None):
    """
    Run 20-step GraphCast forecast.

    If seeded_inputs is provided, uses those instead of the raw inputs.
    Returns (predictions, eval_targets, eval_inputs, base_datetime).
    """
    import haiku as hk
    import jax
    from graphcast import (
        autoregressive, casting, checkpoint, data_utils, graphcast, normalization,
    )

    print("Loading checkpoint ...")
    with open(params_path, "rb") as f:
        ckpt = checkpoint.load(f, graphcast.CheckPoint)
    model_config = ckpt.model_config
    task_config = ckpt.task_config
    params = ckpt.params
    state = {}

    stats = {k: xr.load_dataset(v).compute() for k, v in stats_paths.items()}

    print("Loading example dataset (20-step) ...")
    example_ds = xr.load_dataset(dataset_path).compute()
    if "batch" not in example_ds.dims:
        example_ds = example_ds.expand_dims("batch")

    n_steps = 20
    print(f"Extracting inputs/targets/forcings for {n_steps} steps ...")
    eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
        example_ds,
        target_lead_times=slice("6h", f"{6 * n_steps}h"),
        **dataclasses.asdict(task_config),
    )

    diffs_stddev = stats["diffs_stddev_by_level"]
    mean = stats["mean_by_level"]
    stddev = stats["stddev_by_level"]

    @hk.transform_with_state
    def run_forward(inputs, targets_template, forcings):
        predictor = graphcast.GraphCast(model_config, task_config)
        predictor = casting.Bfloat16Cast(predictor)
        predictor = normalization.InputsAndResiduals(
            predictor,
            diffs_stddev_by_level=diffs_stddev,
            mean_by_level=mean,
            stddev_by_level=stddev,
        )
        predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
        return predictor(inputs, targets_template=targets_template, forcings=forcings)

    run_fwd = jax.jit(run_forward.apply)
    rng = jax.random.PRNGKey(0)

    inputs_to_use = seeded_inputs if seeded_inputs is not None else eval_inputs

    print("JIT-compiling + running 20-step forecast ...")
    predictions, _ = run_fwd(
        params, state, rng,
        inputs_to_use,
        eval_targets * np.nan,
        eval_forcings,
    )
    jax.block_until_ready(predictions)
    print("Prediction complete!")

    base_datetime = example_ds.coords["datetime"].values[0, 0]
    return predictions, eval_targets, eval_inputs, base_datetime


def apply_mask_to_inputs(eval_inputs, mask_ds):
    """Apply a seeding mask to GraphCast eval_inputs."""
    seeded = eval_inputs.copy(deep=True)
    for var in mask_ds.data_vars:
        if var not in seeded.data_vars:
            continue
        mv = mask_ds[var].values
        iv = seeded[var].values
        if mv.shape == iv.shape:
            seeded[var].values = iv + mv
        else:
            # Apply to last time step of inputs (current state)
            try:
                min_t = min(iv.shape[1], mv.shape[1]) if iv.ndim >= 2 and mv.ndim >= 2 else 1
                if iv.ndim == mv.ndim:
                    seeded[var].values[:, -min_t:] = iv[:, -min_t:] + mv[:, :min_t]
            except (ValueError, IndexError):
                pass
    return seeded


# ---------------------------------------------------------------------------
# Step 2: Drought analysis
# ---------------------------------------------------------------------------
def analyze_drought(predictions, forecast_date=None, detector=None):
    """
    Run DroughtDetector on GraphCast predictions.

    If detector is provided (already trained), uses it directly.
    Otherwise trains a new one on the predictions.

    Returns:
        drought_grid: (lat, lon) array of drought class integers
        drought_counts: dict of {class_label: count}
        total_drought_cells: int (sum of all non-zero drought cells)
        lats, lons: coordinate arrays
        detector: the trained DroughtDetector (for reuse)
        cont_severity: probability-weighted continuous severity score
    """
    if forecast_date is None:
        forecast_date = datetime(2022, 1, 1)

    if detector is None:
        detector = DroughtDetector()
        metrics = detector.train(predictions=predictions, forecast_date=forecast_date)
        print(f"  Drought model trained. Accuracy: {metrics['accuracy']:.3f}")

    # Extract features and predict
    features, lats, lons = detector.extract_features(predictions)
    pred_classes = detector.model.predict(features)

    # Reshape to grid
    n_lats, n_lons = len(lats), len(lons)
    drought_grid = pred_classes.reshape(n_lats, n_lons)

    # Count cells by class
    counts = {}
    for cls_id, label in DROUGHT_CLASSES.items():
        counts[label] = int(np.sum(drought_grid == cls_id))

    total_drought = int(np.sum(drought_grid > 0))

    # Probability-weighted continuous severity (more sensitive to small changes)
    cont_severity = continuous_severity_score(features, detector)

    return drought_grid, counts, total_drought, lats, lons, detector, cont_severity


def drought_severity_score(drought_grid):
    """
    Compute a single severity score from a drought grid.
    Higher = more severe drought overall.
    Weights: D0=1, D1=2, D2=4, D3=8
    """
    weights = {0: 0, 1: 1, 2: 2, 3: 4, 4: 8}
    score = 0
    for cls_id, w in weights.items():
        score += w * np.sum(drought_grid == cls_id)
    return int(score)


def continuous_severity_score(features, detector):
    """
    Compute probability-weighted severity score using predict_proba.
    More sensitive than discrete class predictions for subtle perturbations.
    """
    CLASS_WEIGHTS = np.array([0, 1, 2, 4, 8], dtype=np.float64)
    proba = detector.model.predict_proba(features)
    model_classes = detector.model.classes_
    weights = np.array([CLASS_WEIGHTS[c] for c in model_classes])
    per_cell = proba @ weights
    return float(np.sum(per_cell))


def _ensure_time_first(ds):
    """Transpose dataset from (batch, time, ...) to (time, batch, ...) ordering."""
    new_ds = ds.copy()
    for var in new_ds.data_vars:
        dims = new_ds[var].dims
        if 'batch' in dims and 'time' in dims:
            batch_idx = dims.index('batch')
            time_idx = dims.index('time')
            if batch_idx < time_idx:
                new_order = list(dims)
                new_order[batch_idx] = 'time'
                new_order[time_idx] = 'batch'
                new_ds[var] = new_ds[var].transpose(*new_order)
    return new_ds


# ---------------------------------------------------------------------------
# Step 3: Plotting functions (broadcast-style, matching graphcast_weather_map)
# ---------------------------------------------------------------------------

def _style_ax(ax, title):
    """Apply the broadcast dark-style to an axis."""
    ax.set_global()
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="#cccccc")
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="#888888")
    ax.set_facecolor(AX_BG)
    ax.spines["geo"].set_edgecolor("#444466")
    ax.spines["geo"].set_linewidth(1)
    ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=6)


def plot_weather_map(predictions, base_datetime, output_path, title_prefix="Control"):
    """
    Plot a 4-panel weather map at the final forecast step (same style as
    graphcast_weather_map.py). Saves a static PNG.
    """
    lats = predictions.coords["lat"].values
    lons = predictions.coords["lon"].values
    levels = predictions.coords["level"].values

    # Use final time step
    t = -1
    t2m = _kelvin_to_f(predictions["2m_temperature"].values[t, 0])
    mslp = _pa_to_hpa(predictions["mean_sea_level_pressure"].values[t, 0])
    u10 = predictions["10m_u_component_of_wind"].values[t, 0]
    v10 = predictions["10m_v_component_of_wind"].values[t, 0]
    precip_mm = predictions["total_precipitation_6hr"].values[t, 0] * 1000

    idx_500 = np.argmin(np.abs(levels - 500))
    geopot = predictions["geopotential"].values[t, 0, idx_500] / (9.80665 * 10)

    time_vals = predictions.coords["time"].values
    fhr = int(time_vals[t] / np.timedelta64(1, "h"))

    fig = plt.figure(figsize=(20, 11), facecolor=FIG_BG)
    gs = fig.add_gridspec(2, 2, hspace=0.18, wspace=0.08,
                          left=0.03, right=0.97, top=0.88, bottom=0.05)
    axes = [fig.add_subplot(gs[i, j], projection=PROJ)
            for i in range(2) for j in range(2)]

    titles = [
        f"2m Temperature (\u00b0F) & 10m Wind",
        "Mean Sea Level Pressure (hPa)",
        "6hr Precipitation (mm)",
        "500 hPa Geopotential Height (dam)",
    ]
    for ax, ttl in zip(axes, titles):
        _style_ax(ax, ttl)

    # Temperature + wind
    temp_norm = mcolors.Normalize(vmin=-40, vmax=120)
    axes[0].contourf(lons, lats, t2m, levels=np.linspace(-40, 120, 33),
                     cmap="RdYlBu_r", norm=temp_norm, transform=DATA_CRS, extend="both")
    skip = 8
    lon_g, lat_g = np.meshgrid(lons, lats)
    axes[0].quiver(lon_g[::skip, ::skip], lat_g[::skip, ::skip],
                   u10[::skip, ::skip], v10[::skip, ::skip],
                   color="white", alpha=0.5, scale=300, width=0.002, transform=DATA_CRS)
    sm = plt.cm.ScalarMappable(norm=temp_norm, cmap="RdYlBu_r")
    cb = fig.colorbar(sm, ax=axes[0], orientation="horizontal", fraction=0.045, pad=0.02, aspect=35)
    cb.ax.tick_params(colors="white", labelsize=8); cb.set_label("\u00b0F", color="white", fontsize=9)

    # MSLP
    axes[1].contourf(lons, lats, mslp, levels=np.linspace(950, 1060, 23),
                     cmap="coolwarm", transform=DATA_CRS, extend="both")
    axes[1].contour(lons, lats, mslp, levels=np.arange(950, 1060, 4),
                    colors="white", linewidths=0.4, alpha=0.6, transform=DATA_CRS)
    sm2 = plt.cm.ScalarMappable(norm=mcolors.Normalize(vmin=950, vmax=1060), cmap="coolwarm")
    cb2 = fig.colorbar(sm2, ax=axes[1], orientation="horizontal", fraction=0.045, pad=0.02, aspect=35)
    cb2.ax.tick_params(colors="white", labelsize=8); cb2.set_label("hPa", color="white", fontsize=9)

    # Precipitation
    precip_colors = [
        (1, 1, 1, 0), (0.6, 0.8, 1, 0.4), (0.2, 0.5, 1, 0.7),
        (0.1, 0.2, 0.8, 0.85), (0.5, 0.1, 0.7, 0.9), (0.8, 0.1, 0.5, 0.95),
    ]
    pcmap = mcolors.LinearSegmentedColormap.from_list("precip", precip_colors, N=256)
    pnorm = mcolors.Normalize(vmin=0, vmax=25)
    axes[2].contourf(lons, lats, precip_mm,
                     levels=[0, 0.5, 1, 2, 5, 10, 15, 20, 25],
                     cmap=pcmap, norm=pnorm, transform=DATA_CRS, extend="max")
    axes[2].add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="#ccc")
    sm3 = plt.cm.ScalarMappable(norm=pnorm, cmap=pcmap)
    cb3 = fig.colorbar(sm3, ax=axes[2], orientation="horizontal", fraction=0.045, pad=0.02, aspect=35)
    cb3.ax.tick_params(colors="white", labelsize=8); cb3.set_label("mm/6hr", color="white", fontsize=9)

    # Geopotential
    axes[3].contourf(lons, lats, geopot, levels=30, cmap="viridis",
                     transform=DATA_CRS, extend="both")
    axes[3].contour(lons, lats, geopot, levels=15, colors="white",
                    linewidths=0.3, alpha=0.5, transform=DATA_CRS)
    sm4 = plt.cm.ScalarMappable(
        norm=mcolors.Normalize(vmin=np.nanmin(geopot), vmax=np.nanmax(geopot)), cmap="viridis")
    cb4 = fig.colorbar(sm4, ax=axes[3], orientation="horizontal", fraction=0.045, pad=0.02, aspect=35)
    cb4.ax.tick_params(colors="white", labelsize=8); cb4.set_label("dam", color="white", fontsize=9)

    vt = np.datetime64(base_datetime) + time_vals[t]
    fig.suptitle(f"{title_prefix} Forecast \u2022 +{fhr}h  |  Valid: {str(vt)[:16]} UTC",
                 color="white", fontsize=15, fontweight="bold", y=0.96)
    fig.text(0.01, 0.01, "GraphCast Small (1\u00b0) \u2022 Google DeepMind",
             color="#556677", fontsize=8)

    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_drought_map(drought_grid, lats, lons, output_path,
                     title="Drought Analysis", subtitle="", best_label=False):
    """
    Plot drought severity map in broadcast dark style.
    """
    fig = plt.figure(figsize=(16, 9), facecolor=FIG_BG)
    ax = fig.add_subplot(1, 1, 1, projection=PROJ)
    _style_ax(ax, "")

    # Build a custom colormap for drought classes
    from matplotlib.colors import ListedColormap, BoundaryNorm
    colors_list = [DROUGHT_COLORS[i] for i in range(5)]
    dcmap = ListedColormap(colors_list)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    dnorm = BoundaryNorm(bounds, dcmap.N)

    im = ax.pcolormesh(lons, lats, drought_grid, cmap=dcmap, norm=dnorm,
                       transform=DATA_CRS, shading="auto")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.7, edgecolor="white")
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="#aaa")

    # Legend
    legend_patches = [
        mpatches.Patch(color=DROUGHT_COLORS[i][:3], alpha=0.9, label=DROUGHT_LABELS[i])
        for i in range(5)
    ]
    leg = ax.legend(handles=legend_patches, loc="lower left", fontsize=9,
                    facecolor="#1a1a3e", edgecolor="#555",
                    labelcolor="white", framealpha=0.9)

    # Counts annotation
    total = drought_grid.size
    drought_total = int(np.sum(drought_grid > 0))
    pct = drought_total / total * 100

    counts_text = f"Total cells: {total:,}  |  Drought cells: {drought_total:,} ({pct:.1f}%)\n"
    for cls_id in [4, 3, 2, 1]:
        n = int(np.sum(drought_grid == cls_id))
        if n > 0:
            counts_text += f"  {DROUGHT_LABELS[cls_id]}: {n:,}\n"

    fig.text(0.98, 0.12, counts_text, ha="right", va="bottom",
             color="#aabbcc", fontsize=9, fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#1a1a3e", edgecolor="#555", alpha=0.85))

    title_str = title
    if best_label:
        title_str += "  \u2605 MOST EFFECTIVE"
    fig.suptitle(title_str, color="white", fontsize=15, fontweight="bold", y=0.97)
    if subtitle:
        fig.text(0.5, 0.92, subtitle, ha="center", color="#aabbcc", fontsize=11)

    fig.text(0.01, 0.01, "Drought Severity Index (DSI) \u2022 Random Forest Classifier",
             color="#556677", fontsize=8)

    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_comparison_dashboard(
    control_grid, intervention_grids, lats, lons, best_name, output_path,
):
    """
    Plot a multi-panel comparison of all drought interventions.
    Top-left: control. Remaining panels: each intervention.
    Best intervention is starred.
    """
    n = 1 + len(intervention_grids)  # control + interventions
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig = plt.figure(figsize=(7 * ncols, 5.5 * nrows + 1), facecolor=FIG_BG)
    gs = fig.add_gridspec(nrows, ncols, hspace=0.25, wspace=0.08,
                          left=0.02, right=0.98, top=0.90, bottom=0.06)

    from matplotlib.colors import ListedColormap, BoundaryNorm
    colors_list = [DROUGHT_COLORS[i] for i in range(5)]
    dcmap = ListedColormap(colors_list)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    dnorm = BoundaryNorm(bounds, dcmap.N)

    panels = [("Control (No Intervention)", control_grid, False)]
    for name, grid in intervention_grids.items():
        is_best = (name == best_name)
        display = INTERVENTION_DISPLAY.get(name, name.replace("_", " ").title())
        label = display.replace("\n", " ")
        if is_best:
            label += "  \u2605 BEST"
        panels.append((label, grid, is_best))

    for idx, (label, grid, is_best) in enumerate(panels):
        r, c = divmod(idx, ncols)
        ax = fig.add_subplot(gs[r, c], projection=PROJ)
        _style_ax(ax, "")

        ax.pcolormesh(lons, lats, grid, cmap=dcmap, norm=dnorm,
                      transform=DATA_CRS, shading="auto")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="white")

        # Title with color coding
        color = "#44ff44" if is_best else "white"
        drought_cells = int(np.sum(grid > 0))
        severity = drought_severity_score(grid)
        ax.set_title(f"{label}\nDrought cells: {drought_cells:,}  |  Severity: {severity:,}",
                     color=color, fontsize=10, fontweight="bold", pad=6)

        if is_best:
            for spine in ax.spines.values():
                spine.set_edgecolor("#44ff44")
                spine.set_linewidth(3)

    # Hide empty subplots
    for idx in range(len(panels), nrows * ncols):
        r, c = divmod(idx, ncols)
        ax = fig.add_subplot(gs[r, c])
        ax.set_visible(False)

    # Legend
    legend_patches = [
        mpatches.Patch(color=DROUGHT_COLORS[i][:3], alpha=0.9, label=DROUGHT_LABELS[i])
        for i in range(5)
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=5,
               fontsize=9, facecolor="#1a1a3e", edgecolor="#555",
               labelcolor="white", framealpha=0.9)

    fig.suptitle("Drought Intervention Comparison  \u2022  5-Day GraphCast Forecast",
                 color="white", fontsize=16, fontweight="bold", y=0.97)

    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_effectiveness_bar_chart(results, best_name, output_path):
    """
    Bar chart comparing drought reduction effectiveness of each intervention.
    Uses probability-weighted continuous severity for sensitive comparison.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), facecolor=FIG_BG)

    names = [n for n in results.keys() if n != "_control"]
    display_names = [INTERVENTION_DISPLAY.get(n, n).replace("\n", " ") for n in names]
    control_cont_sev = results["_control"]["continuous_severity"]

    # Continuous severity reduction percentages
    cont_sev_reduction = [results[n].get("cont_sev_reduction_pct", 0.0) for n in names]
    cell_reduction = [results[n]["cell_reduction_pct"] for n in names]

    colors = ["#44ff44" if n == best_name else "#4488cc" for n in names]

    # Panel 1: Drought cell reduction %
    ax1.set_facecolor(AX_BG)
    bars1 = ax1.barh(display_names, cell_reduction, color=colors, edgecolor="#666", height=0.6)
    ax1.set_xlabel("Drought Cell Reduction (%)", color="white", fontsize=11)
    ax1.set_title("Drought Area Reduction", color="white", fontsize=13, fontweight="bold")
    ax1.tick_params(colors="white", labelsize=9)
    for spine in ax1.spines.values():
        spine.set_edgecolor("#444466")
    for bar, val in zip(bars1, cell_reduction):
        ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 f"{val:.1f}%", va="center", color="white", fontsize=10, fontweight="bold")

    # Panel 2: Continuous severity reduction % (probability-weighted)
    ax2.set_facecolor(AX_BG)
    bars2 = ax2.barh(display_names, cont_sev_reduction, color=colors, edgecolor="#666", height=0.6)
    ax2.set_xlabel("Probability-Weighted Severity Reduction (%)", color="white", fontsize=11)
    ax2.set_title("Continuous Severity Reduction", color="white", fontsize=13, fontweight="bold")
    ax2.tick_params(colors="white", labelsize=9)
    for spine in ax2.spines.values():
        spine.set_edgecolor("#444466")
    for bar, val in zip(bars2, cont_sev_reduction):
        ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 f"{val:.2f}%", va="center", color="white", fontsize=10, fontweight="bold")

    fig.suptitle("Geoengineering Effectiveness Comparison",
                 color="white", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_seeded_weather_map(predictions, base_datetime, intervention_name,
                            drought_grid, lats, lons, output_path, is_best=False):
    """
    Plot a 4-panel weather map for a seeded forecast, with drought overlay
    on the precipitation panel.
    """
    lvls = predictions.coords["level"].values
    t = -1  # final timestep
    t2m = _kelvin_to_f(predictions["2m_temperature"].values[t, 0])
    mslp = _pa_to_hpa(predictions["mean_sea_level_pressure"].values[t, 0])
    precip_mm = predictions["total_precipitation_6hr"].values[t, 0] * 1000
    idx_500 = np.argmin(np.abs(lvls - 500))
    geopot = predictions["geopotential"].values[t, 0, idx_500] / (9.80665 * 10)

    time_vals = predictions.coords["time"].values
    fhr = int(time_vals[t] / np.timedelta64(1, "h"))

    fig = plt.figure(figsize=(20, 11), facecolor=FIG_BG)
    gs = fig.add_gridspec(2, 2, hspace=0.18, wspace=0.08,
                          left=0.03, right=0.97, top=0.88, bottom=0.05)
    axes = [fig.add_subplot(gs[i, j], projection=PROJ) for i in range(2) for j in range(2)]
    titles = [
        "2m Temperature (\u00b0F)",
        "Mean Sea Level Pressure (hPa)",
        "6hr Precipitation (mm) + Drought Overlay",
        "500 hPa Geopotential Height (dam)",
    ]
    for ax, ttl in zip(axes, titles):
        _style_ax(ax, ttl)

    # Temperature
    temp_norm = mcolors.Normalize(vmin=-40, vmax=120)
    axes[0].contourf(lons, lats, t2m, levels=np.linspace(-40, 120, 33),
                     cmap="RdYlBu_r", norm=temp_norm, transform=DATA_CRS, extend="both")
    sm = plt.cm.ScalarMappable(norm=temp_norm, cmap="RdYlBu_r")
    cb = fig.colorbar(sm, ax=axes[0], orientation="horizontal", fraction=0.045, pad=0.02, aspect=35)
    cb.ax.tick_params(colors="white", labelsize=8)

    # MSLP
    axes[1].contourf(lons, lats, mslp, levels=np.linspace(950, 1060, 23),
                     cmap="coolwarm", transform=DATA_CRS, extend="both")
    sm2 = plt.cm.ScalarMappable(norm=mcolors.Normalize(950, 1060), cmap="coolwarm")
    cb2 = fig.colorbar(sm2, ax=axes[1], orientation="horizontal", fraction=0.045, pad=0.02, aspect=35)
    cb2.ax.tick_params(colors="white", labelsize=8)

    # Precipitation + drought overlay
    precip_colors = [
        (1, 1, 1, 0), (0.6, 0.8, 1, 0.4), (0.2, 0.5, 1, 0.7),
        (0.1, 0.2, 0.8, 0.85), (0.5, 0.1, 0.7, 0.9), (0.8, 0.1, 0.5, 0.95),
    ]
    pcmap = mcolors.LinearSegmentedColormap.from_list("precip", precip_colors, N=256)
    pnorm = mcolors.Normalize(vmin=0, vmax=25)
    axes[2].contourf(lons, lats, precip_mm,
                     levels=[0, 0.5, 1, 2, 5, 10, 15, 20, 25],
                     cmap=pcmap, norm=pnorm, transform=DATA_CRS, extend="max")
    # Overlay drought contours
    drought_masked = np.ma.masked_where(drought_grid == 0, drought_grid)
    if np.any(drought_grid > 0):
        from matplotlib.colors import ListedColormap
        overlay_colors = ["#ffff00", "#ffcc00", "#ee6622", "#cc2222"]
        ocmap = ListedColormap(overlay_colors)
        onorm = mcolors.BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], ocmap.N)
        axes[2].pcolormesh(lons, lats, drought_masked, cmap=ocmap, norm=onorm,
                           transform=DATA_CRS, shading="auto", alpha=0.55)
    sm3 = plt.cm.ScalarMappable(norm=pnorm, cmap=pcmap)
    cb3 = fig.colorbar(sm3, ax=axes[2], orientation="horizontal", fraction=0.045, pad=0.02, aspect=35)
    cb3.ax.tick_params(colors="white", labelsize=8)

    # Geopotential
    axes[3].contourf(lons, lats, geopot, levels=30, cmap="viridis",
                     transform=DATA_CRS, extend="both")
    sm4 = plt.cm.ScalarMappable(
        norm=mcolors.Normalize(np.nanmin(geopot), np.nanmax(geopot)), cmap="viridis")
    cb4 = fig.colorbar(sm4, ax=axes[3], orientation="horizontal", fraction=0.045, pad=0.02, aspect=35)
    cb4.ax.tick_params(colors="white", labelsize=8)

    display = INTERVENTION_DISPLAY.get(intervention_name, intervention_name).replace("\n", " ")
    star = "  \u2605 MOST EFFECTIVE" if is_best else ""
    title_color = "#44ff44" if is_best else "white"
    vt = np.datetime64(base_datetime) + time_vals[t]
    fig.suptitle(f"Seeded Forecast: {display}{star}  \u2022  +{fhr}h  |  {str(vt)[:16]} UTC",
                 color=title_color, fontsize=15, fontweight="bold", y=0.96)
    fig.text(0.01, 0.01, "GraphCast + Geoengineering Mask",
             color="#556677", fontsize=8)

    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_pipeline(intensity=0.5, mask_only=False):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    forecast_date = datetime(2022, 1, 1)

    # Target: global (let drought detector find the regions)
    region = TargetRegion(lat_min=-60, lat_max=60, lon_min=0, lon_max=359)

    # ==================================================================
    # PHASE 1: Control forecast
    # ==================================================================
    print("\n" + "=" * 70)
    print("PHASE 1: Running control forecast")
    print("=" * 70)

    if mask_only:
        dataset_path = os.path.join(CACHE_DIR, "example_data_20steps.nc")
        if not os.path.exists(dataset_path):
            # Try treehacks local copy
            dataset_path = os.path.join(os.path.dirname(__file__), "graphcast_data", "example_data_20steps.nc")
        if not os.path.exists(dataset_path):
            dataset_path = os.path.join(CACHE_DIR, "example_data.nc")
        if not os.path.exists(dataset_path):
            dataset_path = os.path.join(os.path.dirname(__file__), "graphcast_data", "example_data.nc")
        print(f"Loading raw dataset (mask-only mode): {dataset_path}")
        control_preds = xr.load_dataset(dataset_path).compute()
        if "batch" not in control_preds.dims:
            control_preds = control_preds.expand_dims("batch")
        eval_inputs = control_preds
        base_datetime = control_preds.coords["datetime"].values[0, 0]
        # Transpose to (time, batch, ...) for DroughtDetector compatibility
        control_preds = _ensure_time_first(control_preds)
    else:
        pp, dp, sp = download_data()
        control_preds, eval_targets, eval_inputs, base_datetime = run_graphcast(pp, dp, sp)

    # ==================================================================
    # PHASE 2: Plot control weather map
    # ==================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: Plotting control weather map")
    print("=" * 70)
    plot_weather_map(
        control_preds, base_datetime,
        os.path.join(OUTPUT_DIR, "01_control_weather_map.png"),
        title_prefix="Control",
    )

    # ==================================================================
    # PHASE 3: Drought analysis on control
    # ==================================================================
    print("\n" + "=" * 70)
    print("PHASE 3: Analyzing drought on control forecast")
    print("=" * 70)
    ctrl_drought, ctrl_counts, ctrl_total, lats, lons, trained_detector, ctrl_cont_sev = analyze_drought(
        control_preds, forecast_date
    )
    ctrl_severity = drought_severity_score(ctrl_drought)
    print(f"  Control drought cells: {ctrl_total:,}")
    print(f"  Control severity score: {ctrl_severity:,}")
    print(f"  Control continuous severity: {ctrl_cont_sev:.2f}")

    plot_drought_map(
        ctrl_drought, lats, lons,
        os.path.join(OUTPUT_DIR, "02_control_drought_analysis.png"),
        title="Control Forecast \u2014 Drought Analysis",
        subtitle=f"5-Day GraphCast Forecast from {str(base_datetime)[:10]}",
    )

    # ==================================================================
    # PHASE 4: Apply each drought mask, run forecast, analyze
    # ==================================================================
    print("\n" + "=" * 70)
    print("PHASE 4: Testing drought interventions")
    print("=" * 70)

    # Generate masks using the raw dataset (eval_inputs shape)
    sim = GeoEngineeringSimulator(dataset=eval_inputs)

    intervention_results = {}
    intervention_grids = {}
    intervention_preds = {}

    for i, mask_name in enumerate(DROUGHT_MASKS):
        print(f"\n--- Intervention {i + 1}/{len(DROUGHT_MASKS)}: {mask_name} ---")

        # Generate mask
        result = sim.apply_intervention(mask_name, region, intensity)

        if mask_only:
            # In mask-only mode, apply mask directly to the dataset
            seeded_preds = _ensure_time_first(result.seeded_dataset)
        else:
            # Apply mask to eval_inputs and run GraphCast
            seeded_inputs = apply_mask_to_inputs(eval_inputs, result.mask_dataset)
            seeded_preds, _, _, _ = run_graphcast(pp, dp, sp, seeded_inputs=seeded_inputs)

        # Analyze drought on seeded predictions
        print(f"  Analyzing drought for {mask_name}...")
        s_drought, s_counts, s_total, _, _, _, s_cont_sev = analyze_drought(
            seeded_preds, forecast_date, detector=trained_detector
        )
        s_severity = drought_severity_score(s_drought)

        cont_sev_reduction = (1 - s_cont_sev / max(ctrl_cont_sev, 1e-9)) * 100
        print(f"  Drought cells: {s_total:,} (control: {ctrl_total:,})")
        print(f"  Severity: {s_severity:,} (control: {ctrl_severity:,})")
        print(f"  Continuous severity: {s_cont_sev:.2f} (reduction: {cont_sev_reduction:.2f}%)")

        intervention_results[mask_name] = {
            "drought_cells": s_total,
            "severity_score": s_severity,
            "continuous_severity": s_cont_sev,
            "counts": s_counts,
            "cell_reduction_pct": (1 - s_total / max(ctrl_total, 1)) * 100,
            "severity_reduction_pct": (1 - s_severity / max(ctrl_severity, 1)) * 100,
            "cont_sev_reduction_pct": cont_sev_reduction,
        }
        intervention_grids[mask_name] = s_drought
        intervention_preds[mask_name] = seeded_preds

    # Store control stats for comparison
    intervention_results["_control"] = {
        "drought_cells": ctrl_total,
        "severity_score": ctrl_severity,
        "continuous_severity": ctrl_cont_sev,
        "counts": ctrl_counts,
    }

    # ==================================================================
    # PHASE 5: Identify best intervention
    # ==================================================================
    print("\n" + "=" * 70)
    print("PHASE 5: Identifying most effective intervention")
    print("=" * 70)

    # Best = lowest continuous severity score (more sensitive than discrete)
    best_name = min(
        DROUGHT_MASKS,
        key=lambda n: intervention_results[n]["continuous_severity"]
    )
    best_res = intervention_results[best_name]

    display_best = INTERVENTION_DISPLAY.get(best_name, best_name).replace("\n", " ")
    print(f"\n  \u2605 MOST EFFECTIVE: {display_best}")
    print(f"    Drought cells: {best_res['drought_cells']:,}  "
          f"(reduction: {best_res['cell_reduction_pct']:.1f}%)")
    print(f"    Severity score: {best_res['severity_score']:,}  "
          f"(reduction: {best_res['severity_reduction_pct']:.1f}%)")
    print(f"    Continuous severity: {best_res['continuous_severity']:.2f}  "
          f"(reduction: {best_res['cont_sev_reduction_pct']:.2f}%)")

    # ==================================================================
    # PHASE 6: Generate all plots
    # ==================================================================
    print("\n" + "=" * 70)
    print("PHASE 6: Generating plots")
    print("=" * 70)

    # Individual drought map + weather map for each intervention
    for i, mask_name in enumerate(DROUGHT_MASKS):
        is_best = (mask_name == best_name)
        display = INTERVENTION_DISPLAY.get(mask_name, mask_name).replace("\n", " ")

        # Drought map
        plot_drought_map(
            intervention_grids[mask_name], lats, lons,
            os.path.join(OUTPUT_DIR, f"03_{mask_name}_drought.png"),
            title=f"Seeded: {display} \u2014 Drought Analysis",
            subtitle=f"Intensity: {intensity:.0%}  |  "
                     f"Cells: {intervention_results[mask_name]['drought_cells']:,}  |  "
                     f"Severity: {intervention_results[mask_name]['severity_score']:,}",
            best_label=is_best,
        )

        # Weather map with drought overlay
        plot_seeded_weather_map(
            intervention_preds[mask_name], base_datetime, mask_name,
            intervention_grids[mask_name], lats, lons,
            os.path.join(OUTPUT_DIR, f"04_{mask_name}_weather.png"),
            is_best=is_best,
        )

    # Comparison dashboard
    plot_comparison_dashboard(
        ctrl_drought, intervention_grids, lats, lons, best_name,
        os.path.join(OUTPUT_DIR, "05_comparison_dashboard.png"),
    )

    # Effectiveness bar chart
    plot_effectiveness_bar_chart(
        intervention_results, best_name,
        os.path.join(OUTPUT_DIR, "06_effectiveness_comparison.png"),
    )

    # ==================================================================
    # Save results JSON
    # ==================================================================
    results_json = {
        "control": {
            "drought_cells": ctrl_total,
            "severity_score": ctrl_severity,
            "continuous_severity": round(ctrl_cont_sev, 4),
            "counts": ctrl_counts,
        },
        "interventions": {
            name: {
                "drought_cells": r["drought_cells"],
                "severity_score": r["severity_score"],
                "continuous_severity": round(r["continuous_severity"], 4),
                "cell_reduction_pct": round(r["cell_reduction_pct"], 2),
                "severity_reduction_pct": round(r["severity_reduction_pct"], 2),
                "cont_sev_reduction_pct": round(r["cont_sev_reduction_pct"], 4),
                "counts": r["counts"],
            }
            for name, r in intervention_results.items() if name != "_control"
        },
        "best_intervention": best_name,
        "intensity": intensity,
        "forecast_date": str(forecast_date),
    }
    json_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"  Saved: {json_path}")

    # ==================================================================
    # Final summary
    # ==================================================================
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}/\n")
    print(f"{'Intervention':<30} {'Drought Cells':>14} {'Cont. Severity':>15} {'Cell Red.':>10} {'Cont. Red.':>11}")
    print("-" * 80)
    print(f"{'Control (baseline)':<30} {ctrl_total:>14,} {ctrl_cont_sev:>15.2f} {'---':>10} {'---':>11}")
    for name in DROUGHT_MASKS:
        r = intervention_results[name]
        star = " *" if name == best_name else ""
        display = INTERVENTION_DISPLAY.get(name, name).replace("\n", " ")
        print(f"{display + star:<30} {r['drought_cells']:>14,} {r['continuous_severity']:>15.2f} "
              f"{r['cell_reduction_pct']:>9.1f}% {r['cont_sev_reduction_pct']:>10.4f}%")
    print("=" * 80)
    print(f"\n\u2605 Most Effective: {display_best}")

    return results_json


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Drought analysis + geoengineering pipeline")
    parser.add_argument("--intensity", type=float, default=0.5, help="Mask intensity 0.0-1.0")
    parser.add_argument("--mask-only", action="store_true",
                        help="Skip GraphCast inference (fast mode, uses raw data as predictions)")
    args = parser.parse_args()

    run_pipeline(intensity=args.intensity, mask_only=args.mask_only)
