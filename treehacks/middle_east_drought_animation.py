#!/usr/bin/env python3
"""
Middle East Drought Geoengineering Animation Pipeline

Produces broadcast-style animated .mov files showing how each geoengineering
intervention affects precipitation over the Middle East region (Arabian
Peninsula, Iraq, Iran, Horn of Africa).

Outputs (to ~/treehacks/drought_analysis_mideast/):
  - control_vs_glaciogenic_static.mov
  - control_vs_hygroscopic_enhancement.mov
  - control_vs_electric_ionization.mov
  - control_vs_laser_induced_condensation.mov
  - all_interventions_comparison.mov
  - results_mideast.json

Usage:
    python middle_east_drought_animation.py
    python middle_east_drought_animation.py --mask-only
    python middle_east_drought_animation.py --intensity 0.7
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import xarray as xr

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Local imports
sys.path.insert(0, os.path.dirname(__file__))
from drought_pipeline import (
    analyze_drought,
    drought_severity_score,
    continuous_severity_score,
    _ensure_time_first,
    DROUGHT_COLORS,
    DROUGHT_LABELS,
    INTERVENTION_DISPLAY,
)
from geoengineering_masks import (
    GeoEngineeringSimulator,
    TargetRegion,
    DROUGHT_MASKS,
)
from drought_detector import DroughtDetector

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphcast_data")
OUTPUT_DIR = os.path.expanduser("~/treehacks/drought_analysis_mideast")

# Middle East bounding box
ME_REGION = TargetRegion(lat_min=10, lat_max=42, lon_min=25, lon_max=65)

# Map extent (slightly wider than the region for visual context)
MAP_EXTENT = [20, 70, 8, 45]  # [lon_min, lon_max, lat_min, lat_max]

# Visual style (matching graphcast_weather_map.py)
DATA_CRS = ccrs.PlateCarree()
FIG_BG = "#0a0a2e"
AX_BG = "#0d1b2a"

# Animation codec
ANIM_FPS = 3
ANIM_DPI = 150
ANIM_CODEC = "prores_ks"
ANIM_PIX_FMT = "yuva444p10le"

# Country labels for geographic context
COUNTRY_LABELS = {
    "Saudi Arabia": (24.0, 45.0),
    "Iran": (33.0, 53.0),
    "Iraq": (33.5, 43.5),
    "Yemen": (15.5, 48.0),
    "Oman": (21.0, 57.0),
    "UAE": (24.0, 54.5),
    "Egypt": (26.5, 30.0),
    "Ethiopia": (9.5, 39.0),
    "Somalia": (5.0, 46.0),
    "Turkey": (39.5, 35.0),
    "Syria": (35.0, 38.5),
    "Pakistan": (30.0, 68.0),
    "Afghanistan": (34.0, 66.0),
}

# Precipitation colormap (matching existing pipeline)
PRECIP_COLORS = [
    (1, 1, 1, 0),
    (0.6, 0.8, 1, 0.4),
    (0.2, 0.5, 1, 0.7),
    (0.1, 0.2, 0.8, 0.85),
    (0.5, 0.1, 0.7, 0.9),
    (0.8, 0.1, 0.5, 0.95),
]
PRECIP_CMAP = mcolors.LinearSegmentedColormap.from_list("precip", PRECIP_COLORS, N=256)
PRECIP_NORM = mcolors.Normalize(vmin=0, vmax=25)
PRECIP_LEVELS = [0, 0.5, 1, 2, 5, 10, 15, 20, 25]

# Drought overlay colors
DROUGHT_OVERLAY_COLORS = ["#ffff00", "#ffcc00", "#ee6622", "#cc2222"]
DROUGHT_OVERLAY_CMAP = ListedColormap(DROUGHT_OVERLAY_COLORS)
DROUGHT_OVERLAY_NORM = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], DROUGHT_OVERLAY_CMAP.N)

# Diverging colormap for difference maps
DIFF_CMAP = "RdYlGn"
DIFF_VMAX = 5.0  # mm


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_and_prepare(intensity=0.5, mask_only=False):
    """
    Load dataset, run control + 4 interventions focused on the Middle East.

    Returns:
        control_preds: xarray Dataset (time-first)
        intervention_preds: dict of {name: xarray Dataset}
        intervention_results: dict of {name: InterventionResult}
        base_datetime: numpy datetime64
        trained_detector: DroughtDetector
    """
    forecast_date = datetime(2022, 1, 1)

    # --- Load dataset ---
    if mask_only:
        dataset_path = os.path.join(CACHE_DIR, "example_data_20steps.nc")
        if not os.path.exists(dataset_path):
            dataset_path = os.path.join(
                os.path.dirname(__file__), "graphcast_data", "example_data_20steps.nc"
            )
        if not os.path.exists(dataset_path):
            dataset_path = os.path.join(CACHE_DIR, "example_data.nc")
        if not os.path.exists(dataset_path):
            dataset_path = os.path.join(
                os.path.dirname(__file__), "graphcast_data", "example_data.nc"
            )
        print(f"Loading dataset (mask-only mode): {dataset_path}")
        raw_ds = xr.load_dataset(dataset_path).compute()
        if "batch" not in raw_ds.dims:
            raw_ds = raw_ds.expand_dims("batch")
        eval_inputs = raw_ds
        base_datetime = raw_ds.coords["datetime"].values[0, 0]
        control_preds = _ensure_time_first(raw_ds)
    else:
        from drought_pipeline import download_data, run_graphcast
        pp, dp, sp = download_data()
        control_preds, _, eval_inputs, base_datetime = run_graphcast(pp, dp, sp)

    # --- Train drought detector on control ---
    print("Training drought detector on control forecast...")
    trained_detector = DroughtDetector()
    trained_detector.train(predictions=control_preds, forecast_date=forecast_date)

    # --- Apply each intervention ---
    sim = GeoEngineeringSimulator(dataset=eval_inputs)
    intervention_preds = {}
    intervention_results = {}

    for mask_name in DROUGHT_MASKS:
        print(f"\nApplying intervention: {mask_name}")
        result = sim.apply_intervention(mask_name, ME_REGION, intensity)
        intervention_results[mask_name] = result

        if mask_only:
            seeded_preds = _ensure_time_first(result.seeded_dataset)
        else:
            from drought_pipeline import apply_mask_to_inputs
            seeded_inputs = apply_mask_to_inputs(eval_inputs, result.mask_dataset)
            seeded_preds, _, _, _ = run_graphcast(pp, dp, sp, seeded_inputs=seeded_inputs)

        intervention_preds[mask_name] = seeded_preds

    return (
        control_preds,
        intervention_preds,
        intervention_results,
        base_datetime,
        trained_detector,
        forecast_date,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _extract_precip_mm(ds, t_idx):
    """Extract precipitation in mm for a single timestep."""
    precip = ds["total_precipitation_6hr"].values[t_idx, 0]
    return precip * 1000.0


def _compute_drought_grid(ds, detector):
    """Compute drought grid from predictions."""
    features, lats, lons = detector.extract_features(ds)
    pred_classes = detector.model.predict(features)
    return pred_classes.reshape(len(lats), len(lons)), lats, lons


def _zone_precip_stats(control_precip, seeded_precip, lats, lons):
    """
    Compute precipitation change statistics within the ME intervention zone.
    """
    lat_mask = (lats >= ME_REGION.lat_min) & (lats <= ME_REGION.lat_max)
    lon_mask = (lons >= ME_REGION.lon_min) & (lons <= ME_REGION.lon_max)
    zone_mask = np.outer(lat_mask, lon_mask)

    ctrl_zone = control_precip[zone_mask]
    seed_zone = seeded_precip[zone_mask]

    ctrl_mean = np.mean(ctrl_zone)
    seed_mean = np.mean(seed_zone)

    if ctrl_mean > 1e-9:
        pct_change = ((seed_mean - ctrl_mean) / ctrl_mean) * 100
    else:
        pct_change = 0.0

    return pct_change, ctrl_mean, seed_mean


def _add_country_labels(ax):
    """Add country name labels to a map axis."""
    for name, (lat, lon) in COUNTRY_LABELS.items():
        ax.text(
            lon, lat, name,
            transform=DATA_CRS,
            fontsize=7,
            color="#aabbcc",
            ha="center",
            va="center",
            alpha=0.7,
            fontweight="bold",
        )


def _add_intervention_zone_rect(ax):
    """Draw a dashed cyan rectangle around the intervention zone."""
    rect = mpatches.FancyBboxPatch(
        (ME_REGION.lon_min, ME_REGION.lat_min),
        ME_REGION.lon_max - ME_REGION.lon_min,
        ME_REGION.lat_max - ME_REGION.lat_min,
        boxstyle="round,pad=0",
        linewidth=1.5,
        edgecolor="cyan",
        facecolor="none",
        linestyle="--",
        transform=DATA_CRS,
        zorder=10,
    )
    ax.add_patch(rect)
    ax.text(
        ME_REGION.lon_min + 1,
        ME_REGION.lat_max - 1.5,
        "Intervention Zone",
        transform=DATA_CRS,
        fontsize=7,
        color="cyan",
        fontweight="bold",
        alpha=0.9,
        zorder=11,
    )


def _style_me_ax(ax, title):
    """Style a cartopy axis for Middle East zoom."""
    ax.set_extent(MAP_EXTENT, crs=DATA_CRS)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="#cccccc")
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="#888888")
    ax.set_facecolor(AX_BG)
    ax.spines["geo"].set_edgecolor("#444466")
    ax.spines["geo"].set_linewidth(1)
    ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=6)


def _draw_drought_overlay(ax, drought_grid, lats, lons):
    """Draw drought contours as semi-transparent overlay."""
    drought_masked = np.ma.masked_where(drought_grid == 0, drought_grid)
    if np.any(drought_grid > 0):
        ax.pcolormesh(
            lons, lats, drought_masked,
            cmap=DROUGHT_OVERLAY_CMAP,
            norm=DROUGHT_OVERLAY_NORM,
            transform=DATA_CRS,
            shading="auto",
            alpha=0.45,
            zorder=3,
        )


# ---------------------------------------------------------------------------
# Side-by-side animation (control vs. intervention)
# ---------------------------------------------------------------------------
def create_intervention_animation(
    control_preds,
    seeded_preds,
    intervention_name,
    base_datetime,
    detector,
    output_path,
):
    """
    Create a 2-panel side-by-side animation: Control vs Seeded precipitation.

    Left panel: control precip + drought overlay
    Right panel: seeded precip + drought overlay + intervention zone + stats HUD
    """
    n_steps = control_preds.sizes["time"]
    lats = control_preds.coords["lat"].values
    lons = control_preds.coords["lon"].values
    time_vals = control_preds.coords["time"].values
    forecast_hours = [int(t / np.timedelta64(1, "h")) for t in time_vals]

    display_name = INTERVENTION_DISPLAY.get(intervention_name, intervention_name)
    display_name_flat = display_name.replace("\n", " ")

    # Pre-extract all timesteps
    ctrl_precip_all = [_extract_precip_mm(control_preds, t) for t in range(n_steps)]
    seed_precip_all = [_extract_precip_mm(seeded_preds, t) for t in range(n_steps)]

    # Compute drought grids
    ctrl_drought, _, _ = _compute_drought_grid(control_preds, detector)
    seed_drought, _, _ = _compute_drought_grid(seeded_preds, detector)

    # Drought stats for HUD
    ctrl_drought_pct = np.sum(ctrl_drought > 0) / ctrl_drought.size * 100
    seed_drought_pct = np.sum(seed_drought > 0) / seed_drought.size * 100
    drought_change = seed_drought_pct - ctrl_drought_pct

    # --- Figure setup ---
    fig = plt.figure(figsize=(20, 9), facecolor=FIG_BG)

    # Gridspec: title row, two panels, lower-third bar, colorbar
    gs = fig.add_gridspec(
        3, 2,
        height_ratios=[1, 12, 0.8],
        hspace=0.08,
        wspace=0.06,
        left=0.03,
        right=0.97,
        top=0.92,
        bottom=0.06,
    )

    ax_left = fig.add_subplot(gs[1, 0], projection=DATA_CRS)
    ax_right = fig.add_subplot(gs[1, 1], projection=DATA_CRS)

    _style_me_ax(ax_left, "Control Precip (mm)")
    _style_me_ax(ax_right, f"Seeded Precip (mm)")

    # Main title / subtitle (updated per frame)
    title_text = fig.suptitle(
        f"Control vs {display_name_flat}",
        color="white", fontsize=16, fontweight="bold", y=0.97,
    )
    subtitle_text = fig.text(
        0.5, 0.935, "",
        ha="center", color="#aabbcc", fontsize=11,
    )

    # Shared colorbar
    sm_precip = plt.cm.ScalarMappable(norm=PRECIP_NORM, cmap=PRECIP_CMAP)
    cb = fig.colorbar(
        sm_precip, ax=[ax_left, ax_right],
        orientation="horizontal",
        fraction=0.035, pad=0.04, aspect=50,
    )
    cb.ax.tick_params(colors="white", labelsize=8)
    cb.set_label("Precipitation (mm / 6hr)", color="white", fontsize=9)

    # Lower-third bar
    fig.text(
        0.5, 0.015,
        "GraphCast  |  Geoengineering Simulation  |  Middle East Regional Analysis",
        ha="center", va="bottom",
        color="#aabbcc", fontsize=9,
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="#0d1b2a",
            edgecolor="#334455",
            alpha=0.95,
        ),
    )

    # Stats HUD placeholder (on right panel)
    hud_text = ax_right.text(
        0.98, 0.02, "",
        transform=ax_right.transAxes,
        fontsize=9,
        color="white",
        ha="right", va="bottom",
        fontfamily="monospace",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="#0d1b2a",
            edgecolor="#556677",
            alpha=0.85,
        ),
        zorder=20,
    )

    # Forecast hour chyron
    chyron_text = fig.text(
        0.98, 0.94, "",
        ha="right", va="top",
        color="white", fontsize=12, fontweight="bold",
        fontfamily="monospace",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="#cc3333",
            edgecolor="white",
            alpha=0.9,
        ),
    )

    # --- Animation ---
    def draw_frame(i):
        # Clear previous collections
        for ax in [ax_left, ax_right]:
            for coll in list(ax.collections):
                coll.remove()

        fhr = forecast_hours[i]
        vt = np.datetime64(base_datetime) + time_vals[i]
        vt_str = str(vt)[:16].replace("T", " ")
        init_str = str(np.datetime64(base_datetime))[:16].replace("T", " ")

        subtitle_text.set_text(f"Init: {init_str} UTC  \u2192  Valid: {vt_str} UTC")
        chyron_text.set_text(f" +{fhr:03d}h ")

        ctrl_p = ctrl_precip_all[i]
        seed_p = seed_precip_all[i]

        # Left: control precip
        ax_left.contourf(
            lons, lats, ctrl_p,
            levels=PRECIP_LEVELS,
            cmap=PRECIP_CMAP, norm=PRECIP_NORM,
            transform=DATA_CRS, extend="max",
        )
        _draw_drought_overlay(ax_left, ctrl_drought, lats, lons)
        _add_country_labels(ax_left)

        # Right: seeded precip
        ax_right.contourf(
            lons, lats, seed_p,
            levels=PRECIP_LEVELS,
            cmap=PRECIP_CMAP, norm=PRECIP_NORM,
            transform=DATA_CRS, extend="max",
        )
        _draw_drought_overlay(ax_right, seed_drought, lats, lons)
        _add_intervention_zone_rect(ax_right)
        _add_country_labels(ax_right)

        # Stats HUD
        pct_change, _, _ = _zone_precip_stats(ctrl_p, seed_p, lats, lons)
        hud_text.set_text(
            f"Zone Precip: {pct_change:+.1f}%\n"
            f"Drought: {drought_change:+.1f}%"
        )

        print(f"  [{intervention_name}] Frame {i + 1}/{n_steps}: +{fhr}h", flush=True)
        return []

    print(f"Rendering {n_steps} frames for {display_name_flat}...")
    anim = animation.FuncAnimation(fig, draw_frame, frames=n_steps, blit=False)

    writer = animation.FFMpegWriter(
        fps=ANIM_FPS, codec=ANIM_CODEC,
        extra_args=["-pix_fmt", ANIM_PIX_FMT],
    )
    anim.save(output_path, writer=writer, dpi=ANIM_DPI)
    plt.close(fig)
    print(f"  Saved: {output_path} ({os.path.getsize(output_path) / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# 2x2 summary animation (all interventions)
# ---------------------------------------------------------------------------
def create_summary_animation(
    control_preds,
    intervention_preds,
    base_datetime,
    best_name,
    output_path,
):
    """
    Create a 2x2 animation showing precip difference (seeded - control)
    for all 4 interventions. Best intervention gets a green border.
    """
    n_steps = control_preds.sizes["time"]
    lats = control_preds.coords["lat"].values
    lons = control_preds.coords["lon"].values
    time_vals = control_preds.coords["time"].values
    forecast_hours = [int(t / np.timedelta64(1, "h")) for t in time_vals]

    # Pre-extract all control precip
    ctrl_precip_all = [_extract_precip_mm(control_preds, t) for t in range(n_steps)]

    # Pre-extract all seeded precip
    seed_precip_all = {}
    for name in DROUGHT_MASKS:
        seed_precip_all[name] = [
            _extract_precip_mm(intervention_preds[name], t) for t in range(n_steps)
        ]

    # --- Figure setup ---
    fig = plt.figure(figsize=(20, 9), facecolor=FIG_BG)

    gs = fig.add_gridspec(
        2, 2,
        hspace=0.22,
        wspace=0.1,
        left=0.04,
        right=0.92,
        top=0.88,
        bottom=0.08,
    )

    axes = []
    for idx, name in enumerate(DROUGHT_MASKS):
        r, c = divmod(idx, 2)
        ax = fig.add_subplot(gs[r, c], projection=DATA_CRS)
        display = INTERVENTION_DISPLAY.get(name, name).replace("\n", " ")
        _style_me_ax(ax, display)

        # Green border for best intervention
        if name == best_name:
            for spine in ax.spines.values():
                spine.set_edgecolor("#44ff44")
                spine.set_linewidth(3)

        axes.append(ax)

    # Title
    title_text = fig.suptitle(
        "Precipitation Change: All Interventions",
        color="white", fontsize=16, fontweight="bold", y=0.96,
    )
    subtitle_text = fig.text(
        0.5, 0.93, "",
        ha="center", color="#aabbcc", fontsize=11,
    )

    # Shared colorbar for diverging diff
    diff_norm = mcolors.TwoSlopeNorm(vmin=-DIFF_VMAX, vcenter=0, vmax=DIFF_VMAX)
    sm_diff = plt.cm.ScalarMappable(norm=diff_norm, cmap=DIFF_CMAP)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.015, 0.6])
    cb = fig.colorbar(sm_diff, cax=cbar_ax, orientation="vertical")
    cb.ax.tick_params(colors="white", labelsize=8)
    cb.set_label("\u0394 Precip (mm)", color="white", fontsize=9)

    # Lower-third
    fig.text(
        0.5, 0.015,
        "GraphCast  |  Geoengineering Simulation  |  Middle East Comparison",
        ha="center", va="bottom",
        color="#aabbcc", fontsize=9,
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="#0d1b2a",
            edgecolor="#334455",
            alpha=0.95,
        ),
    )

    # Chyron
    chyron_text = fig.text(
        0.92, 0.92, "",
        ha="right", va="top",
        color="white", fontsize=12, fontweight="bold",
        fontfamily="monospace",
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="#cc3333",
            edgecolor="white",
            alpha=0.9,
        ),
    )

    def draw_frame(i):
        for ax in axes:
            for coll in list(ax.collections):
                coll.remove()
            for txt in list(ax.texts):
                txt.remove()

        fhr = forecast_hours[i]
        vt = np.datetime64(base_datetime) + time_vals[i]
        vt_str = str(vt)[:16].replace("T", " ")
        init_str = str(np.datetime64(base_datetime))[:16].replace("T", " ")
        subtitle_text.set_text(f"Init: {init_str} UTC  \u2192  Valid: {vt_str} UTC")
        chyron_text.set_text(f" +{fhr:03d}h ")

        ctrl_p = ctrl_precip_all[i]

        for idx, name in enumerate(DROUGHT_MASKS):
            ax = axes[idx]
            seed_p = seed_precip_all[name][i]
            diff = seed_p - ctrl_p

            ax.contourf(
                lons, lats, diff,
                levels=np.linspace(-DIFF_VMAX, DIFF_VMAX, 21),
                cmap=DIFF_CMAP, norm=diff_norm,
                transform=DATA_CRS, extend="both",
            )
            _add_country_labels(ax)
            _add_intervention_zone_rect(ax)

            # Per-panel zone stats
            pct_change, _, _ = _zone_precip_stats(ctrl_p, seed_p, lats, lons)
            star = " \u2605" if name == best_name else ""
            ax.text(
                0.98, 0.02, f"Zone: {pct_change:+.1f}%{star}",
                transform=ax.transAxes,
                fontsize=8, color="white",
                ha="right", va="bottom",
                fontfamily="monospace",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="#0d1b2a",
                    edgecolor="#556677",
                    alpha=0.85,
                ),
                zorder=20,
            )

        print(f"  [summary] Frame {i + 1}/{n_steps}: +{fhr}h", flush=True)
        return []

    print(f"Rendering {n_steps} frames for summary comparison...")
    anim = animation.FuncAnimation(fig, draw_frame, frames=n_steps, blit=False)

    writer = animation.FFMpegWriter(
        fps=ANIM_FPS, codec=ANIM_CODEC,
        extra_args=["-pix_fmt", ANIM_PIX_FMT],
    )
    anim.save(output_path, writer=writer, dpi=ANIM_DPI)
    plt.close(fig)
    print(f"  Saved: {output_path} ({os.path.getsize(output_path) / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run_middle_east_pipeline(intensity=0.5, mask_only=False):
    """Orchestrate the full Middle East drought animation pipeline."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    forecast_date = datetime(2022, 1, 1)

    # ==================================================================
    # Load data and run interventions
    # ==================================================================
    print("\n" + "=" * 70)
    print("MIDDLE EAST DROUGHT ANIMATION PIPELINE")
    print("=" * 70)

    (
        control_preds,
        intervention_preds,
        intervention_results,
        base_datetime,
        trained_detector,
        forecast_date,
    ) = load_and_prepare(intensity=intensity, mask_only=mask_only)

    # ==================================================================
    # Drought analysis for each variant
    # ==================================================================
    print("\n" + "=" * 70)
    print("DROUGHT ANALYSIS")
    print("=" * 70)

    ctrl_drought, ctrl_counts, ctrl_total, lats, lons, _, ctrl_cont_sev = analyze_drought(
        control_preds, forecast_date, detector=trained_detector
    )
    ctrl_severity = drought_severity_score(ctrl_drought)
    print(f"  Control: {ctrl_total:,} drought cells, severity {ctrl_severity:,}")

    analysis_results = {}
    for name in DROUGHT_MASKS:
        s_drought, s_counts, s_total, _, _, _, s_cont_sev = analyze_drought(
            intervention_preds[name], forecast_date, detector=trained_detector
        )
        s_severity = drought_severity_score(s_drought)
        cont_sev_reduction = (1 - s_cont_sev / max(ctrl_cont_sev, 1e-9)) * 100

        display = INTERVENTION_DISPLAY.get(name, name).replace("\n", " ")
        print(f"  {display}: {s_total:,} drought cells, severity {s_severity:,}, "
              f"cont. reduction {cont_sev_reduction:.2f}%")

        analysis_results[name] = {
            "drought_cells": s_total,
            "severity_score": s_severity,
            "continuous_severity": s_cont_sev,
            "cell_reduction_pct": round((1 - s_total / max(ctrl_total, 1)) * 100, 2),
            "severity_reduction_pct": round((1 - s_severity / max(ctrl_severity, 1)) * 100, 2),
            "cont_sev_reduction_pct": round(cont_sev_reduction, 4),
            "counts": s_counts,
        }

    # Identify best intervention
    best_name = min(
        DROUGHT_MASKS,
        key=lambda n: analysis_results[n]["continuous_severity"]
    )
    best_display = INTERVENTION_DISPLAY.get(best_name, best_name).replace("\n", " ")
    print(f"\n  \u2605 MOST EFFECTIVE: {best_display}")

    # ==================================================================
    # Generate side-by-side animations
    # ==================================================================
    print("\n" + "=" * 70)
    print("GENERATING ANIMATIONS")
    print("=" * 70)

    for name in DROUGHT_MASKS:
        out_path = os.path.join(OUTPUT_DIR, f"control_vs_{name}.mov")
        create_intervention_animation(
            control_preds,
            intervention_preds[name],
            name,
            base_datetime,
            trained_detector,
            out_path,
        )

    # ==================================================================
    # Generate 2x2 summary animation
    # ==================================================================
    summary_path = os.path.join(OUTPUT_DIR, "all_interventions_comparison.mov")
    create_summary_animation(
        control_preds,
        intervention_preds,
        base_datetime,
        best_name,
        summary_path,
    )

    # ==================================================================
    # Save results JSON
    # ==================================================================
    results_json = {
        "region": "Middle East",
        "bounding_box": {
            "lat_min": ME_REGION.lat_min,
            "lat_max": ME_REGION.lat_max,
            "lon_min": ME_REGION.lon_min,
            "lon_max": ME_REGION.lon_max,
        },
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
                "cell_reduction_pct": r["cell_reduction_pct"],
                "severity_reduction_pct": r["severity_reduction_pct"],
                "cont_sev_reduction_pct": r["cont_sev_reduction_pct"],
                "counts": r["counts"],
            }
            for name, r in analysis_results.items()
        },
        "best_intervention": best_name,
        "intensity": intensity,
        "forecast_date": str(forecast_date),
        "outputs": {
            "animations": [
                f"control_vs_{name}.mov" for name in DROUGHT_MASKS
            ] + ["all_interventions_comparison.mov"],
        },
    }
    json_path = os.path.join(OUTPUT_DIR, "results_mideast.json")
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\n  Saved: {json_path}")

    # ==================================================================
    # Summary table
    # ==================================================================
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}/\n")
    print(f"{'Intervention':<30} {'Drought Cells':>14} {'Cont. Severity':>15} "
          f"{'Cell Red.':>10} {'Cont. Red.':>11}")
    print("-" * 80)
    print(f"{'Control (baseline)':<30} {ctrl_total:>14,} {ctrl_cont_sev:>15.2f} "
          f"{'---':>10} {'---':>11}")
    for name in DROUGHT_MASKS:
        r = analysis_results[name]
        star = " *" if name == best_name else ""
        display = INTERVENTION_DISPLAY.get(name, name).replace("\n", " ")
        print(f"{display + star:<30} {r['drought_cells']:>14,} "
              f"{r['continuous_severity']:>15.2f} "
              f"{r['cell_reduction_pct']:>9.1f}% "
              f"{r['cont_sev_reduction_pct']:>10.4f}%")
    print("=" * 80)
    print(f"\n\u2605 Most Effective: {best_display}")
    print(f"\nGenerated files:")
    for name in DROUGHT_MASKS:
        print(f"  - control_vs_{name}.mov")
    print(f"  - all_interventions_comparison.mov")
    print(f"  - results_mideast.json")

    return results_json


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Middle East drought geoengineering animation pipeline"
    )
    parser.add_argument(
        "--intensity", type=float, default=0.5,
        help="Mask intensity 0.0-1.0 (default: 0.5)",
    )
    parser.add_argument(
        "--mask-only", action="store_true",
        help="Skip GraphCast inference (fast mode, applies masks directly)",
    )
    args = parser.parse_args()

    run_middle_east_pipeline(intensity=args.intensity, mask_only=args.mask_only)
