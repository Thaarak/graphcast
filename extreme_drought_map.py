#!/usr/bin/env python3
"""Extreme Drought Visualization.

Creates visualizations highlighting extreme drought areas, showing their
exact locations, concentration, and regional distribution.

Usage:
    python extreme_drought_map.py drought_results_real.json predictions_10day_real.nc
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import xarray as xr

# Output file paths
OUTPUT_SCATTER = os.path.expanduser("~/extreme_drought_scatter.png")
OUTPUT_SUMMARY = os.path.expanduser("~/extreme_drought_summary.png")
OUTPUT_HOTSPOTS = os.path.expanduser("~/extreme_drought_hotspots.png")

# Region definitions for classification
REGIONS = {
    "Sahara/N.Africa": {"lat_range": (15, 35), "lon_range": (0, 40)},
    "W.Africa/Sahel": {"lat_range": (5, 20), "lon_range": (330, 360)},
    "Middle East": {"lat_range": (15, 40), "lon_range": (40, 70)},
    "S.Asia/India": {"lat_range": (5, 30), "lon_range": (70, 100)},
    "E.Africa": {"lat_range": (-15, 15), "lon_range": (30, 55)},
    "Australia": {"lat_range": (-35, -15), "lon_range": (115, 155)},
    "S.America": {"lat_range": (-35, 0), "lon_range": (280, 320)},
    "Other": {"lat_range": (-90, 90), "lon_range": (0, 360)},
}

# Hotspot zoom regions for drought-prone areas
# extent format: (lat_min, lat_max, lon_min, lon_max)
HOTSPOTS = [
    {"name": "Sahara Desert", "lat": 25, "lon": 10, "extent": (10, 35, -15, 45)},
    {"name": "West Africa / Sahel", "lat": 15, "lon": -10, "extent": (0, 25, -20, 20)},
    {"name": "Middle East", "lat": 28, "lon": 50, "extent": (15, 40, 30, 70)},
    {"name": "Horn of Africa", "lat": 5, "lon": 45, "extent": (-10, 20, 30, 60)},
]


def load_extreme_cells(results_path: str) -> Tuple[List[Dict], Dict]:
    """Extract only extreme drought cells (class 4).

    Args:
        results_path: Path to drought results JSON file.

    Returns:
        Tuple of (list of extreme cell dicts, full results dict).
    """
    with open(results_path, "r") as f:
        results = json.load(f)

    extreme_cells = [
        d for d in results["detections"]
        if d["drought_class"] == 4
    ]

    return extreme_cells, results


def classify_region(lat: float, lon: float) -> str:
    """Classify a cell into a region based on coordinates."""
    # Normalize longitude to 0-360
    if lon < 0:
        lon += 360

    # Check specific regions first (order matters - more specific first)
    # Sahara/North Africa
    if 15 <= lat <= 35 and 0 <= lon <= 40:
        return "Sahara/N.Africa"

    # West Africa / Sahel (wraps around 0/360)
    if 5 <= lat <= 20 and (lon >= 330 or lon <= 15):
        return "W.Africa/Sahel"

    # Middle East
    if 15 <= lat <= 40 and 40 <= lon <= 70:
        return "Middle East"

    # South Asia / India
    if 5 <= lat <= 30 and 70 <= lon <= 100:
        return "S.Asia/India"

    # East Africa
    if -15 <= lat <= 15 and 30 <= lon <= 55:
        return "E.Africa"

    # Australia
    if -35 <= lat <= -15 and 115 <= lon <= 155:
        return "Australia"

    # South America (dry regions)
    if -35 <= lat <= 0 and 280 <= lon <= 320:
        return "S.America"

    return "Other"


def get_region_counts(extreme_cells: List[Dict]) -> Dict[str, int]:
    """Count cells by region."""
    counts = {}
    for cell in extreme_cells:
        region = classify_region(cell["lat"], cell["lon"])
        counts[region] = counts.get(region, 0) + 1
    return counts


def create_drought_grid(results: Dict, lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Convert detection list to a 2D grid of drought classes."""
    grid = np.zeros((len(lats), len(lons)), dtype=np.float32)

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


def create_scatter_map(extreme_cells: List[Dict], results: Dict, output_path: str):
    """Create global map with markers at exact extreme locations.

    Args:
        extreme_cells: List of extreme drought cell dictionaries.
        results: Full results dictionary.
        output_path: Path to save the output image.
    """
    print("Creating scatter map...")

    # Extract coordinates and probabilities
    lats = np.array([c["lat"] for c in extreme_cells])
    lons = np.array([c["lon"] for c in extreme_cells])
    probs = np.array([c["probability"] for c in extreme_cells])

    # Set up figure with dark theme
    proj = ccrs.Robinson()
    data_crs = ccrs.PlateCarree()

    fig = plt.figure(figsize=(18, 10), facecolor="#1a1a2e")
    ax = fig.add_subplot(1, 1, 1, projection=proj)

    ax.set_global()
    ax.set_facecolor("#0d1b2a")
    ax.spines["geo"].set_edgecolor("#444466")
    ax.spines["geo"].set_linewidth(1.5)

    # Add background features
    ax.add_feature(cfeature.OCEAN, facecolor="#0d1b2a")
    ax.add_feature(cfeature.LAND, facecolor="#1a2a3e", alpha=0.7)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="#cccccc")
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="#666666")

    # Marker size based on probability (scaled for visibility)
    sizes = 50 + (probs - 0.36) / (1.0 - 0.36) * 150  # Range: 50-200

    # Plot extreme drought cells
    scatter = ax.scatter(
        lons, lats,
        c=probs,
        s=sizes,
        cmap="YlOrRd",
        vmin=0.36,
        vmax=1.0,
        edgecolors="black",
        linewidths=0.8,
        transform=data_crs,
        zorder=10,
        alpha=0.85,
    )

    # Colorbar
    cbar = fig.colorbar(
        scatter, ax=ax, orientation="horizontal",
        fraction=0.04, pad=0.06, aspect=40,
        label="Drought Probability"
    )
    cbar.ax.tick_params(colors="white", labelsize=10)
    cbar.set_label("Drought Probability", color="white", fontsize=12)

    # Title
    fig.suptitle(
        "Extreme Drought Cell Locations",
        color="white", fontsize=20, fontweight="bold", y=0.95
    )

    # Subtitle with stats
    fig.text(
        0.5, 0.89,
        f"{len(extreme_cells)} extreme cells out of {results['summary']['total_cells']:,} total ({100*len(extreme_cells)/results['summary']['total_cells']:.2f}%)",
        ha="center", color="#aabbcc", fontsize=13
    )

    # Period info
    fig.text(
        0.5, 0.85,
        f"Forecast: {results['forecast_period']['start'][:10]} to {results['forecast_period']['end'][:10]}",
        ha="center", color="#88aacc", fontsize=11
    )

    # Legend for marker sizes
    legend_elements = [
        plt.scatter([], [], c='#ff6b35', s=50, edgecolors='black', label='Prob ~0.36'),
        plt.scatter([], [], c='#ff3333', s=125, edgecolors='black', label='Prob ~0.68'),
        plt.scatter([], [], c='#8b0000', s=200, edgecolors='black', label='Prob ~1.0'),
    ]
    legend = ax.legend(
        handles=legend_elements, loc='lower left',
        title='Probability Scale', framealpha=0.7,
        facecolor='#2a2a4e', edgecolor='#666666',
        fontsize=9, title_fontsize=10
    )
    legend.get_title().set_color('white')
    for text in legend.get_texts():
        text.set_color('white')

    # Branding
    fig.text(0.01, 0.01, "GraphCast Extreme Drought Visualization",
             color="#556677", fontsize=9, ha="left")

    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(),
                bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def create_regional_summary(extreme_cells: List[Dict], results: Dict, output_path: str):
    """Create 4-panel summary with bar charts and statistics.

    Args:
        extreme_cells: List of extreme drought cell dictionaries.
        results: Full results dictionary.
        output_path: Path to save the output image.
    """
    print("Creating regional summary...")

    proj = ccrs.Robinson()
    data_crs = ccrs.PlateCarree()

    fig = plt.figure(figsize=(18, 14), facecolor="#1a1a2e")

    # Create grid: 2 rows, 2 columns
    gs = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.15,
                          left=0.05, right=0.95, top=0.88, bottom=0.08)

    # Panel 1: Global scatter (top left)
    ax1 = fig.add_subplot(gs[0, 0], projection=proj)
    ax1.set_global()
    ax1.set_facecolor("#0d1b2a")
    ax1.add_feature(cfeature.LAND, facecolor="#1a2a3e", alpha=0.7)
    ax1.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="#888888")
    ax1.spines["geo"].set_edgecolor("#444466")

    lats = np.array([c["lat"] for c in extreme_cells])
    lons = np.array([c["lon"] for c in extreme_cells])
    probs = np.array([c["probability"] for c in extreme_cells])

    ax1.scatter(
        lons, lats, c=probs, s=80, cmap="YlOrRd",
        vmin=0.36, vmax=1.0, edgecolors="black", linewidths=0.6,
        transform=data_crs, zorder=10, alpha=0.85
    )
    ax1.set_title("Global Distribution of Extreme Cells",
                  color="white", fontsize=13, fontweight="bold", pad=8)

    # Panel 2: Bar chart by region (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#1a2a3e")

    region_counts = get_region_counts(extreme_cells)
    # Sort by count descending
    sorted_regions = sorted(region_counts.items(), key=lambda x: x[1], reverse=True)
    regions = [r[0] for r in sorted_regions]
    counts = [r[1] for r in sorted_regions]

    colors = ["#ff4444", "#ff6b35", "#ff9500", "#ffcc00", "#88aacc"][:len(regions)]
    bars = ax2.barh(regions, counts, color=colors, edgecolor="white", linewidth=0.5)

    # Add count labels
    for bar, count in zip(bars, counts):
        ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                 str(count), va='center', ha='left', color='white', fontsize=11)

    ax2.set_xlabel("Number of Extreme Cells", color="white", fontsize=11)
    ax2.set_title("Extreme Cells by Region", color="white", fontsize=13,
                  fontweight="bold", pad=10)
    ax2.tick_params(colors="white", labelsize=10)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["bottom"].set_color("#666666")
    ax2.spines["left"].set_color("#666666")
    ax2.set_xlim(0, max(counts) * 1.15)

    # Panel 3: Top 5 cluster info (bottom left)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor("#1a2a3e")
    ax3.axis("off")

    # Cluster analysis: find clusters by proximity
    def find_clusters(cells, lat_tol=5, lon_tol=10):
        clusters = []
        used = set()
        for i, cell in enumerate(cells):
            if i in used:
                continue
            cluster = [cell]
            used.add(i)
            for j, other in enumerate(cells):
                if j in used:
                    continue
                if (abs(cell["lat"] - other["lat"]) <= lat_tol and
                    abs(cell["lon"] - other["lon"]) <= lon_tol):
                    cluster.append(other)
                    used.add(j)
            if len(cluster) >= 2:
                clusters.append(cluster)
        return clusters

    clusters = find_clusters(extreme_cells)
    clusters.sort(key=len, reverse=True)

    cluster_text = "Top 5 Clusters (by cell count)\n" + "=" * 35 + "\n\n"
    for i, cluster in enumerate(clusters[:5]):
        avg_lat = np.mean([c["lat"] for c in cluster])
        avg_lon = np.mean([c["lon"] for c in cluster])
        avg_prob = np.mean([c["probability"] for c in cluster])
        region = classify_region(avg_lat, avg_lon)
        cluster_text += f"{i+1}. {region}\n"
        cluster_text += f"   Center: ({avg_lat:.1f}°, {avg_lon:.1f}°)\n"
        cluster_text += f"   Cells: {len(cluster)}, Avg Prob: {avg_prob:.3f}\n\n"

    ax3.text(0.05, 0.95, cluster_text, transform=ax3.transAxes,
             fontsize=11, color="white", fontfamily="monospace",
             verticalalignment="top")
    ax3.set_title("Cluster Locations", color="white", fontsize=13,
                  fontweight="bold", pad=10, loc="left")

    # Panel 4: Probability distribution histogram (bottom right)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor("#1a2a3e")

    ax4.hist(probs, bins=20, range=(0.35, 1.0), color="#ff6b35",
             edgecolor="white", linewidth=0.5, alpha=0.8)

    # Add statistics
    mean_prob = np.mean(probs)
    median_prob = np.median(probs)
    ax4.axvline(mean_prob, color="#44ff44", linestyle="--", linewidth=2, label=f"Mean: {mean_prob:.3f}")
    ax4.axvline(median_prob, color="#4488ff", linestyle="--", linewidth=2, label=f"Median: {median_prob:.3f}")

    ax4.set_xlabel("Drought Probability", color="white", fontsize=11)
    ax4.set_ylabel("Number of Cells", color="white", fontsize=11)
    ax4.set_title("Probability Distribution", color="white", fontsize=13,
                  fontweight="bold", pad=10)
    ax4.tick_params(colors="white", labelsize=10)
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)
    ax4.spines["bottom"].set_color("#666666")
    ax4.spines["left"].set_color("#666666")
    ax4.legend(facecolor="#2a2a4e", edgecolor="#666666", fontsize=10)
    for text in ax4.get_legend().get_texts():
        text.set_color("white")

    # Main title
    fig.suptitle(
        "Extreme Drought Regional Summary",
        color="white", fontsize=18, fontweight="bold", y=0.96
    )

    # Subtitle with stats
    total_clustered = sum(len(c) for c in clusters)
    fig.text(
        0.5, 0.91,
        f"{len(extreme_cells)} extreme cells | {len(clusters)} clusters containing {total_clustered} cells | "
        f"Prob range: {min(probs):.2f} - {max(probs):.2f}",
        ha="center", color="#aabbcc", fontsize=12
    )

    # Branding
    fig.text(0.01, 0.01, "GraphCast Extreme Drought Analysis",
             color="#556677", fontsize=9, ha="left")

    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(),
                bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def create_hotspot_zooms(extreme_cells: List[Dict], predictions: xr.Dataset,
                         results: Dict, output_path: str):
    """Create zoomed maps of top 4 hotspot regions.

    Args:
        extreme_cells: List of extreme drought cell dictionaries.
        predictions: xarray Dataset with prediction data.
        results: Full results dictionary.
        output_path: Path to save the output image.
    """
    print("Creating hotspot zoom maps...")

    data_crs = ccrs.PlateCarree()

    # Get drought grid for background
    lats_grid = predictions.coords["lat"].values
    lons_grid = predictions.coords["lon"].values
    drought_grid = create_drought_grid(results, lats_grid, lons_grid)

    # Muted drought colormap for background
    drought_colors = [
        (0.1, 0.1, 0.2, 0.0),    # 0: None (transparent)
        (0.5, 0.5, 0.3, 0.3),    # 1: Abnormally dry (muted yellow)
        (0.5, 0.4, 0.2, 0.4),    # 2: Moderate (muted orange)
        (0.5, 0.25, 0.15, 0.5),  # 3: Severe (muted red)
        (0.4, 0.0, 0.0, 0.0),    # 4: Extreme (will be overlaid with markers)
    ]
    drought_cmap = mcolors.ListedColormap(drought_colors)
    drought_norm = mcolors.BoundaryNorm([0, 0.5, 1.5, 2.5, 3.5, 4.5], drought_cmap.N)

    fig = plt.figure(figsize=(18, 14), facecolor="#1a1a2e")

    gs = fig.add_gridspec(2, 2, hspace=0.15, wspace=0.1,
                          left=0.03, right=0.97, top=0.88, bottom=0.05)

    for idx, hotspot in enumerate(HOTSPOTS):
        row = idx // 2
        col = idx % 2

        # Use PlateCarree for regional maps
        ax = fig.add_subplot(gs[row, col], projection=data_crs)

        lat_min, lat_max, lon_min, lon_max = hotspot["extent"]
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=data_crs)

        ax.set_facecolor("#0d1b2a")
        ax.add_feature(cfeature.LAND, facecolor="#1a2a3e", alpha=0.6)
        ax.add_feature(cfeature.OCEAN, facecolor="#0d1b2a")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor="#aaaaaa")
        ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="#666666")
        ax.add_feature(cfeature.LAKES, facecolor="#0d1b2a", edgecolor="#666666", linewidth=0.3)

        # Add gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="#444466",
                          alpha=0.7, linestyle="--")
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"color": "white", "fontsize": 8}
        gl.ylabel_style = {"color": "white", "fontsize": 8}

        # Plot background drought levels (muted)
        ax.contourf(
            lons_grid, lats_grid, drought_grid,
            levels=[0, 0.5, 1.5, 2.5, 3.5, 4.5],
            cmap=drought_cmap, norm=drought_norm,
            transform=data_crs, alpha=0.5
        )

        # Filter extreme cells for this region
        # Handle longitude normalization (data may be 0-360 or -180 to 180)
        def in_lon_range(lon, lon_min, lon_max):
            # Normalize to -180 to 180 for comparison
            if lon > 180:
                lon = lon - 360
            return lon_min <= lon <= lon_max

        region_cells = [
            c for c in extreme_cells
            if lat_min <= c["lat"] <= lat_max and in_lon_range(c["lon"], lon_min, lon_max)
        ]

        if region_cells:
            cell_lats = np.array([c["lat"] for c in region_cells])
            # Normalize longitudes for plotting (convert 0-360 to -180-180 if needed)
            cell_lons = np.array([c["lon"] - 360 if c["lon"] > 180 else c["lon"]
                                  for c in region_cells])
            cell_probs = np.array([c["probability"] for c in region_cells])

            sizes = 80 + (cell_probs - 0.36) / (1.0 - 0.36) * 200

            ax.scatter(
                cell_lons, cell_lats,
                c=cell_probs, s=sizes, cmap="YlOrRd",
                vmin=0.36, vmax=1.0,
                edgecolors="black", linewidths=1,
                transform=data_crs, zorder=10, alpha=0.9
            )

        # Title with cell count
        ax.set_title(
            f"{hotspot['name']} ({len(region_cells)} cells)",
            color="white", fontsize=12, fontweight="bold", pad=8
        )

        ax.spines["geo"].set_edgecolor("#666666")
        ax.spines["geo"].set_linewidth(1)

    # Main title
    fig.suptitle(
        "Extreme Drought Hotspot Regions",
        color="white", fontsize=18, fontweight="bold", y=0.96
    )

    fig.text(
        0.5, 0.91,
        f"Zoomed views of the 4 major hotspot regions | Marker size = probability",
        ha="center", color="#aabbcc", fontsize=12
    )

    # Add legend for marker scale
    legend_ax = fig.add_axes([0.4, 0.01, 0.2, 0.02])
    legend_ax.axis("off")

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap="YlOrRd", norm=mcolors.Normalize(vmin=0.36, vmax=1.0))
    cbar = fig.colorbar(sm, ax=legend_ax, orientation="horizontal",
                        fraction=1.0, pad=0, aspect=30)
    cbar.ax.tick_params(colors="white", labelsize=9)
    cbar.set_label("Drought Probability", color="white", fontsize=10)

    # Branding
    fig.text(0.01, 0.01, "GraphCast Extreme Drought Analysis",
             color="#556677", fontsize=9, ha="left")

    plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(),
                bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    """Generate all 3 visualization outputs."""
    parser = argparse.ArgumentParser(
        description="Visualize extreme drought areas with detailed regional analysis"
    )
    parser.add_argument("results", help="Path to drought results JSON file")
    parser.add_argument("predictions", help="Path to predictions NetCDF file")
    parser.add_argument("--scatter", default=OUTPUT_SCATTER,
                        help="Output path for scatter map")
    parser.add_argument("--summary", default=OUTPUT_SUMMARY,
                        help="Output path for regional summary")
    parser.add_argument("--hotspots", default=OUTPUT_HOTSPOTS,
                        help="Output path for hotspot zooms")

    args = parser.parse_args()

    print("=" * 60)
    print("Extreme Drought Visualization")
    print("=" * 60)

    # Load data
    print(f"\nLoading {args.results}...")
    extreme_cells, results = load_extreme_cells(args.results)
    print(f"  Found {len(extreme_cells)} extreme drought cells")
    print(f"  Total cells: {results['summary']['total_cells']:,}")
    print(f"  Percentage: {100*len(extreme_cells)/results['summary']['total_cells']:.3f}%")

    print(f"\nLoading {args.predictions}...")
    predictions = xr.open_dataset(args.predictions)
    print(f"  Grid: {predictions.sizes['lat']} x {predictions.sizes['lon']}")

    # Print region breakdown
    region_counts = get_region_counts(extreme_cells)
    print("\nExtreme cells by region:")
    for region, count in sorted(region_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {region}: {count}")

    print("\n" + "-" * 60)
    print("Generating visualizations...")
    print("-" * 60 + "\n")

    # Generate all three visualizations
    create_scatter_map(extreme_cells, results, args.scatter)
    create_regional_summary(extreme_cells, results, args.summary)
    create_hotspot_zooms(extreme_cells, predictions, results, args.hotspots)

    predictions.close()

    print("\n" + "=" * 60)
    print("Done! Output files:")
    print(f"  1. Scatter map: {args.scatter}")
    print(f"  2. Regional summary: {args.summary}")
    print(f"  3. Hotspot zooms: {args.hotspots}")
    print("=" * 60)


if __name__ == "__main__":
    main()
