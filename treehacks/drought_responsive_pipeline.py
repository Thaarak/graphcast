#!/usr/bin/env python3
"""
Drought-Responsive Geoengineering Pipeline

Implements a two-pass approach for physically-grounded geoengineering simulation:
  1. Run control (unmodified) GraphCast forecast
  2. Analyze control forecast for drought regions using DroughtDetector
  3. Create interventions targeting detected drought regions
  4. Apply interventions to INITIAL CONDITIONS only (first 2 time steps)
  5. Re-run GraphCast with modified initial state
  6. Compare control vs intervention results

The key innovation is applying perturbations to initial conditions rather than
post-processing outputs. This allows GraphCast's learned physics to propagate
the intervention effects through all forecast steps naturally.

Usage:
    python drought_responsive_pipeline.py
    python drought_responsive_pipeline.py --intensity 0.7
    python drought_responsive_pipeline.py --intervention hygroscopic_enhancement
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr

# Visualization
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

# Local imports
sys.path.insert(0, os.path.dirname(__file__))
from drought_detector import DroughtDetector, DROUGHT_CLASSES
from geoengineering_masks import (
    TargetRegion,
    SeedingConfig,
    InitialConditionMask,
    create_drought_targeted_mask,
    drought_grid_to_region,
    MASK_REGISTRY,
    DROUGHT_MASKS,
    HygroscopicEnhancementMask,
)
from graphcast_geoengineering import (
    GraphCastRunner,
    download_all,
    HAS_GRAPHCAST,
    CACHE_DIR,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.expanduser("~/treehacks/drought_responsive_output")

# Animation settings
ANIM_FPS = 3
ANIM_DPI = 150
ANIM_CODEC = "prores_ks"
ANIM_PIX_FMT = "yuva444p10le"

# Visual style
FIG_BG = "#0a0a2e"
AX_BG = "#0d1b2a"
DATA_CRS = ccrs.PlateCarree() if HAS_CARTOPY else None

# Precipitation visualization
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

# Drought overlay
DROUGHT_OVERLAY_COLORS = ["#ffff00", "#ffcc00", "#ee6622", "#cc2222"]
DROUGHT_OVERLAY_CMAP = mcolors.ListedColormap(DROUGHT_OVERLAY_COLORS)
DROUGHT_OVERLAY_NORM = mcolors.BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], 4)

# Difference colormap
DIFF_CMAP = "RdYlGn"
DIFF_VMAX = 5.0


# ---------------------------------------------------------------------------
# Result Container
# ---------------------------------------------------------------------------
@dataclass
class PipelineResult:
    """Container for drought-responsive pipeline results."""
    control_predictions: xr.Dataset
    seeded_predictions: xr.Dataset
    control_drought_grid: np.ndarray
    seeded_drought_grid: np.ndarray
    target_region: TargetRegion
    intervention_type: str
    intensity: float
    lats: np.ndarray
    lons: np.ndarray
    base_datetime: np.datetime64
    forecast_date: datetime

    # Computed metrics
    control_drought_cells: int = 0
    seeded_drought_cells: int = 0
    drought_reduction_pct: float = 0.0
    control_severity_score: float = 0.0
    seeded_severity_score: float = 0.0
    severity_reduction_pct: float = 0.0

    def __post_init__(self):
        """Compute metrics after initialization."""
        self.control_drought_cells = int(np.sum(self.control_drought_grid > 0))
        self.seeded_drought_cells = int(np.sum(self.seeded_drought_grid > 0))

        if self.control_drought_cells > 0:
            self.drought_reduction_pct = (
                (self.control_drought_cells - self.seeded_drought_cells)
                / self.control_drought_cells
            ) * 100
        else:
            self.drought_reduction_pct = 0.0

        # Weighted severity score: extreme=4, severe=3, moderate=2, abnormal=1
        def severity_score(grid):
            return float(
                4 * np.sum(grid == 4) +
                3 * np.sum(grid == 3) +
                2 * np.sum(grid == 2) +
                1 * np.sum(grid == 1)
            )

        self.control_severity_score = severity_score(self.control_drought_grid)
        self.seeded_severity_score = severity_score(self.seeded_drought_grid)

        if self.control_severity_score > 0:
            self.severity_reduction_pct = (
                (self.control_severity_score - self.seeded_severity_score)
                / self.control_severity_score
            ) * 100
        else:
            self.severity_reduction_pct = 0.0

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "intervention_type": self.intervention_type,
            "intensity": self.intensity,
            "target_region": {
                "lat_min": self.target_region.lat_min,
                "lat_max": self.target_region.lat_max,
                "lon_min": self.target_region.lon_min,
                "lon_max": self.target_region.lon_max,
            },
            "forecast_date": self.forecast_date.isoformat(),
            "control": {
                "drought_cells": self.control_drought_cells,
                "severity_score": self.control_severity_score,
            },
            "seeded": {
                "drought_cells": self.seeded_drought_cells,
                "severity_score": self.seeded_severity_score,
            },
            "improvement": {
                "drought_reduction_pct": round(self.drought_reduction_pct, 2),
                "severity_reduction_pct": round(self.severity_reduction_pct, 2),
            },
        }


# ---------------------------------------------------------------------------
# Main Pipeline Class
# ---------------------------------------------------------------------------
class DroughtResponsivePipeline:
    """
    Two-pass drought-responsive geoengineering pipeline.

    This pipeline implements the physically-grounded approach:
      1. Run control forecast to identify drought regions
      2. Target interventions at detected droughts
      3. Apply perturbations to initial conditions
      4. Re-run GraphCast to propagate effects through physics

    Unlike post-processing approaches, this allows the intervention
    effects to evolve naturally through GraphCast's learned dynamics.
    """

    def __init__(
        self,
        params_path: Optional[str] = None,
        stats_paths: Optional[dict] = None,
        dataset_path: Optional[str] = None,
        n_steps: int = 20,
    ):
        """
        Initialize the pipeline.

        Args:
            params_path: Path to GraphCast parameters (downloads if None)
            stats_paths: Dict of paths to normalization stats (downloads if None)
            dataset_path: Path to ERA5 dataset (downloads if None)
            n_steps: Number of forecast steps (default 20 = 5 days)
        """
        self.n_steps = n_steps
        self.runner: Optional[GraphCastRunner] = None
        self.detector = DroughtDetector()

        # Will be populated after initialization
        self.params_path = params_path
        self.stats_paths = stats_paths
        self.dataset_path = dataset_path

        if HAS_GRAPHCAST:
            self._initialize_runner()

    def _initialize_runner(self):
        """Initialize the GraphCast runner."""
        if self.params_path is None:
            print("Downloading GraphCast data...")
            self.params_path, self.dataset_path, self.stats_paths = download_all()

        print("\nInitializing GraphCast runner...")
        self.runner = GraphCastRunner(
            self.params_path,
            self.stats_paths,
            self.dataset_path,
            n_steps=self.n_steps,
        )

    def run(
        self,
        intensity: float = 0.5,
        intervention_type: str = "hygroscopic_enhancement",
        min_drought_severity: int = 2,
        buffer_degrees: float = 5.0,
        forecast_date: Optional[datetime] = None,
        perturbation_scale: float = None,
    ) -> PipelineResult:
        """
        Run the full two-pass drought-responsive pipeline.

        Args:
            intensity: Intervention intensity (0.0 to 1.0)
            intervention_type: Type of seeding mask to apply
            min_drought_severity: Minimum drought class to target (2=moderate)
            buffer_degrees: Buffer around drought regions
            forecast_date: Date context for drought detection
            perturbation_scale: Multiplier for perturbation magnitudes (default: 20.0).
                               Higher values create stronger effects.

        Returns:
            PipelineResult with control vs seeded predictions and metrics
        """
        if not HAS_GRAPHCAST:
            raise RuntimeError(
                "GraphCast not installed. Install with: "
                "pip install dm-haiku jax graphcast google-cloud-storage"
            )

        if self.runner is None:
            self._initialize_runner()

        if forecast_date is None:
            forecast_date = datetime(2022, 1, 1)

        print("\n" + "=" * 70)
        print("DROUGHT-RESPONSIVE GEOENGINEERING PIPELINE")
        print("=" * 70)
        print(f"Intervention: {intervention_type}")
        print(f"Intensity: {intensity}")
        print(f"Min drought severity: {min_drought_severity}")
        print(f"Buffer: {buffer_degrees} degrees")

        # ===================================================================
        # STEP 1: Run control (unmodified) forecast
        # ===================================================================
        print("\n" + "-" * 50)
        print("STEP 1: Running control forecast")
        print("-" * 50)
        control_preds = self.runner.run_control()

        # ===================================================================
        # STEP 2: Analyze control forecast for drought regions
        # ===================================================================
        print("\n" + "-" * 50)
        print("STEP 2: Analyzing drought in control forecast")
        print("-" * 50)
        drought_grid, lats, lons = self._analyze_drought(control_preds, forecast_date)

        drought_cells = np.sum(drought_grid > 0)
        print(f"  Drought cells detected: {drought_cells}")
        print(f"    Extreme: {np.sum(drought_grid == 4)}")
        print(f"    Severe: {np.sum(drought_grid == 3)}")
        print(f"    Moderate: {np.sum(drought_grid == 2)}")
        print(f"    Abnormal: {np.sum(drought_grid == 1)}")

        # ===================================================================
        # STEP 3: Create drought-targeted mask for initial conditions
        # ===================================================================
        print("\n" + "-" * 50)
        print("STEP 3: Creating drought-targeted initial condition mask")
        print("-" * 50)
        mask, target_region = create_drought_targeted_mask(
            drought_grid=drought_grid,
            lats=lats,
            lons=lons,
            eval_inputs=self.runner.eval_inputs,
            intensity=intensity,
            min_severity=min_drought_severity,
            buffer_degrees=buffer_degrees,
            intervention_type=intervention_type,
            perturbation_scale=perturbation_scale,
        )

        # ===================================================================
        # STEP 4: Apply mask to initial conditions only (first 2 time steps)
        # ===================================================================
        print("\n" + "-" * 50)
        print("STEP 4: Applying mask to initial conditions")
        print("-" * 50)
        seeded_inputs = self._apply_initial_condition_mask(
            self.runner.eval_inputs,
            mask,
        )

        # Verify mask only affects initial conditions
        self._verify_mask_application(self.runner.eval_inputs, seeded_inputs, mask)

        # ===================================================================
        # STEP 5: Re-run GraphCast with modified initial state
        # ===================================================================
        print("\n" + "-" * 50)
        print("STEP 5: Running seeded forecast with modified initial conditions")
        print("-" * 50)
        seeded_preds = self.runner.run_seeded_with_inputs(seeded_inputs)

        # ===================================================================
        # STEP 6: Analyze results and compare
        # ===================================================================
        print("\n" + "-" * 50)
        print("STEP 6: Analyzing seeded forecast")
        print("-" * 50)
        seeded_drought_grid, _, _ = self._analyze_drought(seeded_preds, forecast_date)

        seeded_cells = np.sum(seeded_drought_grid > 0)
        print(f"  Seeded drought cells: {seeded_cells}")
        print(f"    Extreme: {np.sum(seeded_drought_grid == 4)}")
        print(f"    Severe: {np.sum(seeded_drought_grid == 3)}")
        print(f"    Moderate: {np.sum(seeded_drought_grid == 2)}")
        print(f"    Abnormal: {np.sum(seeded_drought_grid == 1)}")

        # Create result
        result = PipelineResult(
            control_predictions=control_preds,
            seeded_predictions=seeded_preds,
            control_drought_grid=drought_grid,
            seeded_drought_grid=seeded_drought_grid,
            target_region=target_region,
            intervention_type=intervention_type,
            intensity=intensity,
            lats=lats,
            lons=lons,
            base_datetime=self.runner.base_datetime,
            forecast_date=forecast_date,
        )

        # Print summary
        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)
        print(f"Control drought cells:  {result.control_drought_cells:,}")
        print(f"Seeded drought cells:   {result.seeded_drought_cells:,}")
        print(f"Drought reduction:      {result.drought_reduction_pct:+.1f}%")
        print(f"Control severity score: {result.control_severity_score:,.0f}")
        print(f"Seeded severity score:  {result.seeded_severity_score:,.0f}")
        print(f"Severity reduction:     {result.severity_reduction_pct:+.1f}%")
        print("=" * 70)

        return result

    def _analyze_drought(
        self,
        predictions: xr.Dataset,
        forecast_date: datetime,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Analyze predictions for drought regions.

        Returns:
            Tuple of (drought_grid, lats, lons)
        """
        # Train detector if not already trained
        if self.detector.model is None:
            print("  Training drought detector...")
            self.detector.train(predictions=predictions, forecast_date=forecast_date)

        # Extract features and predict
        features, lats, lons = self.detector.extract_features(predictions)
        pred_classes = self.detector.model.predict(features)

        # Reshape to grid
        drought_grid = pred_classes.reshape(len(lats), len(lons))

        return drought_grid, lats, lons

    def _apply_initial_condition_mask(
        self,
        eval_inputs: xr.Dataset,
        mask: InitialConditionMask,
    ) -> xr.Dataset:
        """
        Apply an InitialConditionMask to eval_inputs.

        Args:
            eval_inputs: Original eval_inputs from GraphCast
            mask: InitialConditionMask to apply

        Returns:
            Modified eval_inputs with perturbations applied to first 2 time steps
        """
        # Generate the mask values
        mask_ds = mask.generate_mask(eval_inputs)

        # Apply: X_seeded = X_raw + M
        seeded_inputs = eval_inputs.copy(deep=True)

        for var in mask_ds.data_vars:
            if var in seeded_inputs.data_vars:
                mask_vals = mask_ds[var].values
                if np.any(mask_vals != 0):
                    seeded_inputs[var].values = (
                        seeded_inputs[var].values + mask_vals
                    )
                    nonzero = np.count_nonzero(mask_vals)
                    max_pert = np.max(np.abs(mask_vals))
                    print(f"    {var}: {nonzero} cells perturbed, max |delta| = {max_pert:.6f}")

        return seeded_inputs

    def _verify_mask_application(
        self,
        original_inputs: xr.Dataset,
        seeded_inputs: xr.Dataset,
        mask: InitialConditionMask,
    ):
        """Verify that mask was applied correctly to initial conditions."""
        print("  Verifying mask application...")

        # Check a few key variables
        for var_name in ["temperature", "specific_humidity"]:
            if var_name not in original_inputs.data_vars:
                continue

            orig = original_inputs[var_name].values
            seed = seeded_inputs[var_name].values
            diff = seed - orig

            # Check time steps 0 and 1 have perturbations
            if orig.ndim >= 2:
                # Assuming shape (batch, time, ...)
                if "time" in original_inputs[var_name].dims:
                    t_idx = list(original_inputs[var_name].dims).index("time")
                    if t_idx == 1:  # (batch, time, ...)
                        t0_diff = diff[:, 0]
                        t1_diff = diff[:, 1] if diff.shape[1] > 1 else None
                    else:  # (time, ...)
                        t0_diff = diff[0]
                        t1_diff = diff[1] if diff.shape[0] > 1 else None

                    t0_nonzero = np.count_nonzero(t0_diff)
                    t1_nonzero = np.count_nonzero(t1_diff) if t1_diff is not None else 0

                    if t0_nonzero > 0 or t1_nonzero > 0:
                        print(f"    {var_name}: t=0 has {t0_nonzero} changes, "
                              f"t=1 has {t1_nonzero} changes")

        # Verify precipitation is NOT modified
        if "total_precipitation_6hr" in original_inputs.data_vars:
            orig_precip = original_inputs["total_precipitation_6hr"].values
            seed_precip = seeded_inputs["total_precipitation_6hr"].values
            precip_diff = np.max(np.abs(seed_precip - orig_precip))
            if precip_diff < 1e-10:
                print("    total_precipitation_6hr: correctly NOT modified")
            else:
                print(f"    WARNING: total_precipitation_6hr modified by {precip_diff}")

    def run_all_interventions(
        self,
        intensity: float = 0.5,
        min_drought_severity: int = 2,
        forecast_date: Optional[datetime] = None,
        perturbation_scale: float = None,
    ) -> dict[str, PipelineResult]:
        """
        Run all drought intervention types and compare.

        Args:
            intensity: Intervention intensity
            min_drought_severity: Minimum drought class to target
            forecast_date: Date context

        Returns:
            Dict of {intervention_type: PipelineResult}
        """
        results = {}

        for intervention_type in DROUGHT_MASKS:
            print(f"\n{'#' * 70}")
            print(f"# INTERVENTION: {intervention_type}")
            print(f"{'#' * 70}")

            result = self.run(
                intensity=intensity,
                intervention_type=intervention_type,
                min_drought_severity=min_drought_severity,
                forecast_date=forecast_date,
                perturbation_scale=perturbation_scale,
            )
            results[intervention_type] = result

        # Print comparison summary
        print("\n" + "=" * 80)
        print("ALL INTERVENTIONS COMPARISON")
        print("=" * 80)
        print(f"{'Intervention':<35} {'Drought Cells':>15} {'Severity':>12} {'Reduction':>12}")
        print("-" * 80)

        # Control baseline (use first result)
        first_result = next(iter(results.values()))
        print(f"{'Control (baseline)':<35} {first_result.control_drought_cells:>15,} "
              f"{first_result.control_severity_score:>12,.0f} {'---':>12}")

        best_name = None
        best_reduction = -float("inf")

        for name, result in results.items():
            if result.severity_reduction_pct > best_reduction:
                best_reduction = result.severity_reduction_pct
                best_name = name

            print(f"{name:<35} {result.seeded_drought_cells:>15,} "
                  f"{result.seeded_severity_score:>12,.0f} "
                  f"{result.severity_reduction_pct:>+11.1f}%")

        print("=" * 80)
        print(f"\nBest intervention: {best_name} ({best_reduction:+.1f}% severity reduction)")

        return results


# ---------------------------------------------------------------------------
# Visualization Functions
# ---------------------------------------------------------------------------
def create_comparison_movie(
    result: PipelineResult,
    output_path: str,
):
    """
    Create a side-by-side comparison animation of control vs seeded forecasts.
    """
    if not HAS_CARTOPY:
        print("Cartopy not available - skipping animation")
        return

    control = result.control_predictions
    seeded = result.seeded_predictions
    lats = result.lats
    lons = result.lons

    n_steps = control.sizes["time"]
    time_vals = control.coords["time"].values
    forecast_hours = [int(t / np.timedelta64(1, "h")) for t in time_vals]

    # Extract precipitation
    ctrl_precip = control["total_precipitation_6hr"].values[:, 0] * 1000  # to mm
    seed_precip = seeded["total_precipitation_6hr"].values[:, 0] * 1000

    # Figure setup
    fig = plt.figure(figsize=(20, 9), facecolor=FIG_BG)
    gs = fig.add_gridspec(
        2, 2, height_ratios=[1, 1],
        hspace=0.15, wspace=0.1,
        left=0.05, right=0.92, top=0.88, bottom=0.08,
    )

    ax_ctrl = fig.add_subplot(gs[0, 0], projection=DATA_CRS)
    ax_seed = fig.add_subplot(gs[0, 1], projection=DATA_CRS)
    ax_diff = fig.add_subplot(gs[1, :], projection=DATA_CRS)

    for ax in [ax_ctrl, ax_seed, ax_diff]:
        ax.set_global()
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="#ccc")
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="#888")
        ax.set_facecolor(AX_BG)
        ax.spines["geo"].set_edgecolor("#444466")

    ax_ctrl.set_title("Control Precipitation (mm)", color="white", fontsize=11, fontweight="bold")
    ax_seed.set_title(f"Seeded ({result.intervention_type})", color="white", fontsize=11, fontweight="bold")
    ax_diff.set_title("Precipitation Change (seeded - control)", color="white", fontsize=11, fontweight="bold")

    # Colorbars
    sm_precip = plt.cm.ScalarMappable(norm=PRECIP_NORM, cmap=PRECIP_CMAP)
    cb1 = fig.colorbar(sm_precip, ax=[ax_ctrl, ax_seed], orientation="horizontal",
                       fraction=0.03, pad=0.02, aspect=40)
    cb1.ax.tick_params(colors="white", labelsize=8)
    cb1.set_label("mm/6hr", color="white", fontsize=9)

    diff_norm = mcolors.TwoSlopeNorm(vmin=-DIFF_VMAX, vcenter=0, vmax=DIFF_VMAX)
    sm_diff = plt.cm.ScalarMappable(norm=diff_norm, cmap=DIFF_CMAP)
    cb2 = fig.colorbar(sm_diff, ax=ax_diff, orientation="horizontal",
                       fraction=0.04, pad=0.04, aspect=50)
    cb2.ax.tick_params(colors="white", labelsize=8)
    cb2.set_label("delta mm/6hr", color="white", fontsize=9)

    # Title
    title_text = fig.suptitle("", color="white", fontsize=14, fontweight="bold", y=0.96)
    subtitle_text = fig.text(0.5, 0.92, "", ha="center", color="#aabbcc", fontsize=10)

    # Stats HUD
    hud_text = fig.text(
        0.02, 0.02,
        f"Control drought: {result.control_drought_cells:,} cells\n"
        f"Seeded drought: {result.seeded_drought_cells:,} cells\n"
        f"Reduction: {result.drought_reduction_pct:+.1f}%",
        color="white", fontsize=9, fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#0d1b2a",
                  edgecolor="#556677", alpha=0.9),
        va="bottom",
    )

    def draw_frame(i):
        for ax in [ax_ctrl, ax_seed, ax_diff]:
            for coll in list(ax.collections):
                coll.remove()

        fhr = forecast_hours[i]
        vt = np.datetime64(result.base_datetime) + time_vals[i]
        vt_str = str(vt)[:16].replace("T", " ")

        title_text.set_text(f"Drought-Responsive Pipeline: {result.intervention_type} | +{fhr}h")
        subtitle_text.set_text(f"Valid: {vt_str} UTC")

        # Use pcolormesh instead of contourf to avoid cartopy/shapely rendering bugs
        ax_ctrl.pcolormesh(lons, lats, ctrl_precip[i],
                           cmap=PRECIP_CMAP, norm=PRECIP_NORM,
                           transform=DATA_CRS, shading="auto")
        ax_seed.pcolormesh(lons, lats, seed_precip[i],
                           cmap=PRECIP_CMAP, norm=PRECIP_NORM,
                           transform=DATA_CRS, shading="auto")

        diff = seed_precip[i] - ctrl_precip[i]
        ax_diff.pcolormesh(lons, lats, diff,
                           cmap=DIFF_CMAP, norm=diff_norm,
                           transform=DATA_CRS, shading="auto")

        # Draw intervention region
        rect = mpatches.Rectangle(
            (result.target_region.lon_min, result.target_region.lat_min),
            result.target_region.lon_max - result.target_region.lon_min,
            result.target_region.lat_max - result.target_region.lat_min,
            linewidth=2, edgecolor="cyan", facecolor="none",
            linestyle="--", transform=DATA_CRS, zorder=10,
        )
        ax_seed.add_patch(rect)

        print(f"  Frame {i + 1}/{n_steps}: +{fhr}h", flush=True)
        return []

    print(f"Rendering {n_steps} frames...")
    anim = animation.FuncAnimation(fig, draw_frame, frames=n_steps, blit=False)

    writer = animation.FFMpegWriter(
        fps=ANIM_FPS, codec=ANIM_CODEC,
        extra_args=["-pix_fmt", ANIM_PIX_FMT],
    )
    anim.save(output_path, writer=writer, dpi=ANIM_DPI)
    plt.close(fig)
    print(f"Saved: {output_path} ({os.path.getsize(output_path) / 1e6:.1f} MB)")


def save_results_json(result: PipelineResult, output_path: str):
    """Save pipeline results to JSON."""
    with open(output_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Drought-responsive geoengineering pipeline with two-pass approach"
    )
    parser.add_argument(
        "--intensity", type=float, default=0.5,
        help="Intervention intensity (0.0 to 1.0, default: 0.5)",
    )
    parser.add_argument(
        "--intervention", "-i",
        choices=DROUGHT_MASKS + ["all"],
        default="hygroscopic_enhancement",
        help="Intervention type (default: hygroscopic_enhancement)",
    )
    parser.add_argument(
        "--min-severity", type=int, default=2,
        choices=[1, 2, 3, 4],
        help="Minimum drought severity to target (1=abnormal, 4=extreme, default: 2=moderate)",
    )
    parser.add_argument(
        "--buffer", type=float, default=5.0,
        help="Buffer zone around drought regions in degrees (default: 5.0)",
    )
    parser.add_argument(
        "--output", "-o", default=OUTPUT_DIR,
        help=f"Output directory (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--perturbation-scale", type=float, default=None,
        help="Multiplier for perturbation magnitudes (default: 20.0). "
             "Higher values = stronger effects for AI models.",
    )
    parser.add_argument(
        "--no-movie", action="store_true",
        help="Skip movie generation",
    )

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Initialize pipeline
    pipeline = DroughtResponsivePipeline()

    if args.intervention == "all":
        # Run all interventions
        results = pipeline.run_all_interventions(
            intensity=args.intensity,
            min_drought_severity=args.min_severity,
            perturbation_scale=args.perturbation_scale,
        )

        # Save all results
        all_results = {
            name: result.to_dict()
            for name, result in results.items()
        }
        json_path = os.path.join(args.output, "all_interventions_results.json")
        with open(json_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nSaved: {json_path}")

        # Generate movies for each
        if not args.no_movie:
            for name, result in results.items():
                movie_path = os.path.join(args.output, f"comparison_{name}.mov")
                create_comparison_movie(result, movie_path)

    else:
        # Run single intervention
        result = pipeline.run(
            intensity=args.intensity,
            intervention_type=args.intervention,
            min_drought_severity=args.min_severity,
            buffer_degrees=args.buffer,
            perturbation_scale=args.perturbation_scale,
        )

        # Save results
        json_path = os.path.join(args.output, f"results_{args.intervention}.json")
        save_results_json(result, json_path)

        # Generate movie
        if not args.no_movie:
            movie_path = os.path.join(args.output, f"comparison_{args.intervention}.mov")
            create_comparison_movie(result, movie_path)

    print(f"\nAll outputs saved to: {args.output}/")


if __name__ == "__main__":
    main()
