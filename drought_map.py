#!/usr/bin/env python3
"""Drought Detection Visualization Map.

Creates an animated map showing drought risk predictions from the drought detector,
along with relevant weather variables (precipitation, temperature).

Usage:
    python drought_map.py predictions_10day_real.nc drought_results_real.json
"""

import argparse
import json
import os

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import xarray as xr

OUTPUT_FILE = os.path.expanduser("~/drought_forecast.mov")


def load_data(predictions_path: str, results_path: str):
    """Load predictions and drought results."""
    predictions = xr.open_dataset(predictions_path)
    with open(results_path, "r") as f:
        results = json.load(f)
    return predictions, results


def create_drought_grid(results: dict, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Convert detection list to a 2D grid of drought classes."""
    grid = np.zeros((len(lats), len(lons)), dtype=np.float32)

    # Create lookup for lat/lon indices
    lat_to_idx = {float(lat): i for i, lat in enumerate(lats)}
    lon_to_idx = {float(lon): i for i, lon in enumerate(lons)}

    for detection in results["detections"]:
        lat = detection["lat"]
        lon = detection["lon"]
        if lat in lat_to_idx and lon in lon_to_idx:
            i = lat_to_idx[lat]
            j = lon_to_idx[lon]
            grid[i, j] = detection["drought_class"]

    return grid


def create_animation(predictions: xr.Dataset, results: dict, output_path: str):
    """Create an animated drought forecast map."""
    n_steps = predictions.sizes["time"]
    lats = predictions.coords["lat"].values
    lons = predictions.coords["lon"].values

    # Extract data (squeeze batch dim)
    precip = predictions["total_precipitation_6hr"].values
    if precip.ndim == 4:
        precip = precip[:, 0, :, :]  # (time, lat, lon)

    temp_2m = predictions["2m_temperature"].values
    if temp_2m.ndim == 4:
        temp_2m = temp_2m[:, 0, :, :]

    humidity = predictions["specific_humidity"].values
    if humidity.ndim == 5:
        humidity = humidity[:, 0, -1, :, :]  # Surface level
    elif humidity.ndim == 4:
        humidity = humidity[:, -1, :, :]

    # Compute cumulative precipitation over time
    precip_cumsum = np.cumsum(precip, axis=0) * 1000  # Convert to mm

    # Convert temperature to Celsius
    temp_c = temp_2m - 273.15

    # Create drought grid from results
    drought_grid = create_drought_grid(results, lats, lons)

    # Compute forecast hours
    time_vals = predictions.coords["time"].values
    forecast_hours = [int(t / np.timedelta64(1, "h")) for t in time_vals]

    # -----------------------------------------------------------------------
    # Color maps
    # -----------------------------------------------------------------------
    # Drought severity: white -> yellow -> orange -> red -> dark red
    drought_colors = [
        (0.95, 0.95, 0.95, 0.0),   # 0: None (transparent)
        (1.0, 0.95, 0.4, 0.7),     # 1: Abnormally dry (yellow)
        (1.0, 0.7, 0.2, 0.8),      # 2: Moderate (orange)
        (0.9, 0.3, 0.1, 0.85),     # 3: Severe (red-orange)
        (0.6, 0.0, 0.0, 0.95),     # 4: Extreme (dark red)
    ]
    drought_cmap = mcolors.ListedColormap(drought_colors)
    drought_norm = mcolors.BoundaryNorm([0, 0.5, 1.5, 2.5, 3.5, 4.5], drought_cmap.N)

    # Precipitation: blues
    precip_cmap = plt.cm.Blues
    precip_norm = mcolors.Normalize(vmin=0, vmax=max(50, np.nanmax(precip_cumsum)))

    # Temperature: cool to warm
    temp_norm = mcolors.Normalize(vmin=-30, vmax=45)
    temp_cmap = plt.cm.RdYlBu_r

    # -----------------------------------------------------------------------
    # Figure layout: 2x2 panels
    # -----------------------------------------------------------------------
    proj = ccrs.Robinson()
    data_crs = ccrs.PlateCarree()

    fig = plt.figure(figsize=(20, 11), facecolor="#1a1a2e")

    gs = fig.add_gridspec(2, 2, hspace=0.18, wspace=0.08,
                          left=0.03, right=0.97, top=0.88, bottom=0.05)

    axes = [
        fig.add_subplot(gs[0, 0], projection=proj),
        fig.add_subplot(gs[0, 1], projection=proj),
        fig.add_subplot(gs[1, 0], projection=proj),
        fig.add_subplot(gs[1, 1], projection=proj),
    ]

    panel_titles = [
        "Drought Risk Classification",
        "Cumulative Precipitation (mm)",
        "2m Temperature (\u00b0C)",
        "Drought Risk + Precipitation Overlay",
    ]

    def style_ax(ax, title):
        ax.set_global()
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="#cccccc")
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="#666666")
        ax.add_feature(cfeature.LAKES, facecolor="#1a1a3e", edgecolor="#666666", linewidth=0.3)
        ax.set_facecolor("#0d1b2a")
        ax.spines["geo"].set_edgecolor("#444466")
        ax.spines["geo"].set_linewidth(1)
        ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=6)

    for ax, title in zip(axes, panel_titles):
        style_ax(ax, title)

    # Main title
    title_text = fig.suptitle("", color="white", fontsize=16, fontweight="bold", y=0.96)
    subtitle_text = fig.text(0.5, 0.92, "", ha="center", color="#aabbcc", fontsize=11)

    # Colorbars
    # Drought colorbar with labels
    sm_drought = plt.cm.ScalarMappable(norm=drought_norm, cmap=drought_cmap)
    cb_drought = fig.colorbar(sm_drought, ax=axes[0], orientation="horizontal",
                               fraction=0.045, pad=0.02, aspect=35,
                               ticks=[0.25, 1, 2, 3, 4])
    cb_drought.ax.set_xticklabels(["None", "Abnormal", "Moderate", "Severe", "Extreme"],
                                   color="white", fontsize=8)
    cb_drought.ax.tick_params(colors="white", labelsize=8)

    # Precipitation colorbar
    sm_precip = plt.cm.ScalarMappable(norm=precip_norm, cmap=precip_cmap)
    cb_precip = fig.colorbar(sm_precip, ax=axes[1], orientation="horizontal",
                              fraction=0.045, pad=0.02, aspect=35)
    cb_precip.ax.tick_params(colors="white", labelsize=8)
    cb_precip.set_label("mm", color="white", fontsize=9)

    # Temperature colorbar
    sm_temp = plt.cm.ScalarMappable(norm=temp_norm, cmap=temp_cmap)
    cb_temp = fig.colorbar(sm_temp, ax=axes[2], orientation="horizontal",
                            fraction=0.045, pad=0.02, aspect=35)
    cb_temp.ax.tick_params(colors="white", labelsize=8)
    cb_temp.set_label("\u00b0C", color="white", fontsize=9)

    # Overlay colorbar (same as drought)
    sm_overlay = plt.cm.ScalarMappable(norm=drought_norm, cmap=drought_cmap)
    cb_overlay = fig.colorbar(sm_overlay, ax=axes[3], orientation="horizontal",
                               fraction=0.045, pad=0.02, aspect=35,
                               ticks=[0.25, 1, 2, 3, 4])
    cb_overlay.ax.set_xticklabels(["None", "Abnormal", "Moderate", "Severe", "Extreme"],
                                   color="white", fontsize=8)
    cb_overlay.ax.tick_params(colors="white", labelsize=8)

    # Summary stats text
    summary = results["summary"]
    stats_text = fig.text(
        0.5, 0.01,
        f"Extreme: {summary['drought_cells']['extreme']} | "
        f"Severe: {summary['drought_cells']['severe']} | "
        f"Moderate: {summary['drought_cells']['moderate']} | "
        f"Abnormally Dry: {summary['drought_cells']['abnormally_dry']} | "
        f"No Drought: {summary['no_drought_cells']}",
        ha="center", color="#88aacc", fontsize=10
    )

    # Branding
    fig.text(0.01, 0.01, "Drought Detector + GraphCast",
             color="#556677", fontsize=8, ha="left")

    # -----------------------------------------------------------------------
    # Animation
    # -----------------------------------------------------------------------
    def draw_frame(i):
        for ax in axes:
            for coll in list(ax.collections):
                coll.remove()
            for line in list(ax.lines):
                line.remove()

        fhr = forecast_hours[i]
        days = fhr / 24
        title_text.set_text(f"Drought Forecast \u2022 Hour {fhr:03d} ({days:.1f} days)")
        subtitle_text.set_text(
            f"Period: {results['forecast_period']['start'][:10]} to {results['forecast_period']['end'][:10]}"
        )

        # Panel 1: Drought classification (static, but redraw for animation)
        axes[0].contourf(
            lons, lats, drought_grid,
            levels=[0, 0.5, 1.5, 2.5, 3.5, 4.5],
            cmap=drought_cmap, norm=drought_norm,
            transform=data_crs,
        )
        axes[0].add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="#cccccc")

        # Panel 2: Cumulative precipitation
        axes[1].contourf(
            lons, lats, precip_cumsum[i],
            levels=np.linspace(0, max(50, np.nanmax(precip_cumsum)), 20),
            cmap=precip_cmap, norm=precip_norm,
            transform=data_crs, extend="max",
        )
        axes[1].add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="#333333")

        # Panel 3: Temperature
        axes[2].contourf(
            lons, lats, temp_c[i],
            levels=np.linspace(-30, 45, 31),
            cmap=temp_cmap, norm=temp_norm,
            transform=data_crs, extend="both",
        )
        axes[2].add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="#cccccc")

        # Panel 4: Overlay - drought risk with precipitation contours
        axes[3].contourf(
            lons, lats, drought_grid,
            levels=[0, 0.5, 1.5, 2.5, 3.5, 4.5],
            cmap=drought_cmap, norm=drought_norm,
            transform=data_crs,
        )
        # Add precipitation contours on top
        axes[3].contour(
            lons, lats, precip_cumsum[i],
            levels=[5, 10, 20, 30, 50],
            colors=["#4488ff", "#2266dd", "#1144aa", "#0033aa", "#002288"],
            linewidths=[0.5, 0.7, 0.9, 1.1, 1.3],
            transform=data_crs,
        )
        axes[3].add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="#cccccc")

        print(f"  Frame {i + 1}/{n_steps}: +{fhr}h", flush=True)
        return []

    print(f"Rendering {n_steps} frames ...")
    anim = animation.FuncAnimation(fig, draw_frame, frames=n_steps, blit=False)

    # Try to use ffmpeg, fall back to saving individual frames
    try:
        writer = animation.FFMpegWriter(fps=4, codec="prores_ks",
                                         extra_args=["-pix_fmt", "yuva444p10le"])
        print(f"Saving animation to {output_path} ...")
        anim.save(output_path, writer=writer, dpi=120)
    except Exception as e:
        print(f"FFmpeg not available ({e}), saving as GIF instead...")
        gif_path = output_path.replace(".mov", ".gif")
        anim.save(gif_path, writer="pillow", fps=4, dpi=100)
        output_path = gif_path

    plt.close(fig)
    print(f"Done! File saved: {output_path}")
    print(f"  Size: {os.path.getsize(output_path) / 1e6:.1f} MB")
    return output_path


def create_static_map(predictions: xr.Dataset, results: dict, output_path: str):
    """Create a static drought risk map (single image)."""
    lats = predictions.coords["lat"].values
    lons = predictions.coords["lon"].values

    # Get cumulative precipitation
    precip = predictions["total_precipitation_6hr"].values
    if precip.ndim == 4:
        precip = precip[:, 0, :, :]
    precip_total = np.sum(precip, axis=0) * 1000  # mm

    # Create drought grid
    drought_grid = create_drought_grid(results, lats, lons)

    # Drought colormap
    drought_colors = [
        (0.95, 0.95, 0.95, 0.0),
        (1.0, 0.95, 0.4, 0.7),
        (1.0, 0.7, 0.2, 0.8),
        (0.9, 0.3, 0.1, 0.85),
        (0.6, 0.0, 0.0, 0.95),
    ]
    drought_cmap = mcolors.ListedColormap(drought_colors)
    drought_norm = mcolors.BoundaryNorm([0, 0.5, 1.5, 2.5, 3.5, 4.5], drought_cmap.N)

    # Create figure
    proj = ccrs.Robinson()
    data_crs = ccrs.PlateCarree()

    fig = plt.figure(figsize=(16, 9), facecolor="#1a1a2e")
    ax = fig.add_subplot(1, 1, 1, projection=proj)

    ax.set_global()
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor="#cccccc")
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="#666666")
    ax.set_facecolor("#0d1b2a")
    ax.spines["geo"].set_edgecolor("#444466")

    # Plot drought risk
    cf = ax.contourf(
        lons, lats, drought_grid,
        levels=[0, 0.5, 1.5, 2.5, 3.5, 4.5],
        cmap=drought_cmap, norm=drought_norm,
        transform=data_crs,
    )

    # Add precipitation contours
    ax.contour(
        lons, lats, precip_total,
        levels=[5, 10, 20, 30, 50, 100],
        colors=["#4488ff", "#2266dd", "#1144aa", "#0033aa", "#002288", "#001166"],
        linewidths=[0.5, 0.7, 0.9, 1.1, 1.3, 1.5],
        transform=data_crs,
    )
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor="#cccccc")

    # Title
    fig.suptitle(
        f"10-Day Drought Forecast",
        color="white", fontsize=18, fontweight="bold", y=0.95
    )
    fig.text(
        0.5, 0.90,
        f"Period: {results['forecast_period']['start'][:10]} to {results['forecast_period']['end'][:10]}",
        ha="center", color="#aabbcc", fontsize=12
    )

    # Colorbar
    cbar = fig.colorbar(cf, ax=ax, orientation="horizontal",
                        fraction=0.046, pad=0.08, aspect=40,
                        ticks=[0.25, 1, 2, 3, 4])
    cbar.ax.set_xticklabels(["None", "Abnormally Dry", "Moderate", "Severe", "Extreme"],
                            color="white", fontsize=10)
    cbar.ax.tick_params(colors="white")

    # Summary
    summary = results["summary"]
    fig.text(
        0.5, 0.02,
        f"Extreme: {summary['drought_cells']['extreme']} | "
        f"Severe: {summary['drought_cells']['severe']} | "
        f"Moderate: {summary['drought_cells']['moderate']} | "
        f"Abnormally Dry: {summary['drought_cells']['abnormally_dry']}",
        ha="center", color="#88aacc", fontsize=11
    )

    # Save
    static_path = output_path.replace(".mov", ".png")
    plt.savefig(static_path, dpi=150, facecolor=fig.get_facecolor(),
                bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print(f"Static map saved: {static_path}")
    return static_path


def main():
    parser = argparse.ArgumentParser(description="Visualize drought predictions")
    parser.add_argument("predictions", help="Path to predictions NetCDF file")
    parser.add_argument("results", help="Path to drought results JSON file")
    parser.add_argument("--output", "-o", default=OUTPUT_FILE,
                        help="Output file path")
    parser.add_argument("--static", action="store_true",
                        help="Create static image instead of animation")

    args = parser.parse_args()

    print("=" * 60)
    print("Drought Forecast Visualization")
    print("=" * 60)

    print(f"Loading {args.predictions}...")
    predictions, results = load_data(args.predictions, args.results)
    print(f"  Grid: {predictions.sizes['lat']} x {predictions.sizes['lon']}")
    print(f"  Time steps: {predictions.sizes['time']}")
    print(f"  Detections: {len(results['detections'])}")
    print()

    if args.static:
        output = create_static_map(predictions, results, args.output)
    else:
        output = create_animation(predictions, results, args.output)

    predictions.close()
    return output


if __name__ == "__main__":
    main()
