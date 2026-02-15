#!/usr/bin/env python3
"""
GraphCast + Geoengineering Integration

Runs GraphCast forecasts with and without geoengineering interventions,
producing side-by-side comparison animations showing the predicted effects
of each seeding modality.

Pipeline:
  1. Load GraphCast model + ERA5 data (reuses graphcast_weather_map.py infra)
  2. Extract eval_inputs from the raw dataset
  3. For each intervention mask, apply M to eval_inputs: X_seeded = X_raw + M
  4. Run GraphCast forward pass on both control and seeded inputs
  5. Render a comparison animation (control vs seeded)

Usage:
    python graphcast_geoengineering.py --intervention hygroscopic_enhancement \\
        --lat-min 25 --lat-max 45 --lon-min 230 --lon-max 260 --intensity 0.5

    python graphcast_geoengineering.py --intervention all --intensity 0.7
"""

import dataclasses
import os
from typing import Optional

import numpy as np
import xarray as xr

# GraphCast imports (may not be installed in all environments)
try:
    from google.cloud import storage
    import haiku as hk
    import jax
    from graphcast import (
        autoregressive,
        casting,
        checkpoint,
        data_utils,
        graphcast,
        normalization,
    )
    HAS_GRAPHCAST = True
except ImportError:
    HAS_GRAPHCAST = False

# Visualization
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

# Local imports
from geoengineering_masks import (
    GeoEngineeringSimulator,
    TargetRegion,
    PlumeConfig,
    SeedingConfig,
    MASK_REGISTRY,
    DROUGHT_MASKS,
    HURRICANE_MASKS,
    InterventionResult,
)

# ---------------------------------------------------------------------------
# GCS config (same as graphcast_weather_map.py)
# ---------------------------------------------------------------------------
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
CACHE_DIR = os.path.expanduser("~/graphcast_data")
OUTPUT_DIR = os.path.expanduser("~/treehacks/geoengineering_output")


# ---------------------------------------------------------------------------
# Data download (reuses pattern from graphcast_weather_map.py)
# ---------------------------------------------------------------------------
def _download(blob_name: str, dest_path: str) -> None:
    if os.path.exists(dest_path):
        print(f"  [cached] {os.path.basename(dest_path)}")
        return
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"  Downloading {blob_name} ...")
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(GCS_BUCKET)
    bucket.blob(blob_name).download_to_filename(dest_path)
    print(f"  Done ({os.path.getsize(dest_path) / 1e6:.1f} MB)")


def download_all():
    print("Downloading data ...")
    params_path = os.path.join(CACHE_DIR, "params.npz")
    _download(PARAMS_FILE, params_path)

    dataset_path = os.path.join(CACHE_DIR, "example_data_20steps.nc")
    _download(DATASET_FILE, dataset_path)

    stats_paths = {}
    for name, blob in STATS_FILES.items():
        p = os.path.join(CACHE_DIR, f"{name}.nc")
        _download(blob, p)
        stats_paths[name] = p
    return params_path, dataset_path, stats_paths


# ---------------------------------------------------------------------------
# GraphCast Model Runner
# ---------------------------------------------------------------------------
class GraphCastRunner:
    """
    Manages GraphCast model loading and inference.

    Supports running forecasts with modified initial states to simulate
    geoengineering interventions.
    """

    def __init__(self, params_path: str, stats_paths: dict, dataset_path: str, n_steps: int = 20):
        if not HAS_GRAPHCAST:
            raise RuntimeError(
                "GraphCast dependencies not installed. "
                "Install: pip install dm-haiku jax graphcast google-cloud-storage"
            )

        self.n_steps = n_steps

        # Load checkpoint
        print("Loading checkpoint ...")
        with open(params_path, "rb") as f:
            ckpt = checkpoint.load(f, graphcast.CheckPoint)
        self.model_config = ckpt.model_config
        self.task_config = ckpt.task_config
        self.params = ckpt.params
        self.state = {}

        # Load stats
        self.stats = {k: xr.load_dataset(v).compute() for k, v in stats_paths.items()}

        # Load dataset
        print(f"Loading dataset ({n_steps}-step) ...")
        self.raw_ds = xr.load_dataset(dataset_path).compute()
        if "batch" not in self.raw_ds.dims:
            self.raw_ds = self.raw_ds.expand_dims("batch")

        # Extract inputs / targets / forcings
        print(f"Extracting inputs/targets/forcings for {n_steps} steps ...")
        self.eval_inputs, self.eval_targets, self.eval_forcings = (
            data_utils.extract_inputs_targets_forcings(
                self.raw_ds,
                target_lead_times=slice("6h", f"{6 * n_steps}h"),
                **dataclasses.asdict(self.task_config),
            )
        )

        # Build model
        diffs_stddev = self.stats["diffs_stddev_by_level"]
        mean = self.stats["mean_by_level"]
        stddev = self.stats["stddev_by_level"]

        @hk.transform_with_state
        def run_forward(inputs, targets_template, forcings):
            predictor = graphcast.GraphCast(self.model_config, self.task_config)
            predictor = casting.Bfloat16Cast(predictor)
            predictor = normalization.InputsAndResiduals(
                predictor,
                diffs_stddev_by_level=diffs_stddev,
                mean_by_level=mean,
                stddev_by_level=stddev,
            )
            predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
            return predictor(inputs, targets_template=targets_template, forcings=forcings)

        self._run_forward_jitted = jax.jit(run_forward.apply)
        self._rng = jax.random.PRNGKey(0)

        # Get base datetime for labels
        self.base_datetime = self.raw_ds.coords["datetime"].values[0, 0]

    def run_control(self) -> xr.Dataset:
        """Run the control (unmodified) forecast."""
        print("Running CONTROL forecast ...")
        predictions, _ = self._run_forward_jitted(
            self.params, self.state, self._rng,
            self.eval_inputs,
            self.eval_targets * np.nan,
            self.eval_forcings,
        )
        jax.block_until_ready(predictions)
        print("Control forecast complete.")
        return predictions

    def run_seeded(self, mask_ds: xr.Dataset) -> xr.Dataset:
        """
        Run a forecast with geoengineering mask applied to the inputs.

        The mask M is applied to eval_inputs: X_seeded = X_raw + M.
        Only variables present in both the mask and eval_inputs are modified.

        Args:
            mask_ds: xr.Dataset of perturbation values from a SeedingMask

        Returns:
            Predictions xr.Dataset from GraphCast
        """
        print("Running SEEDED forecast ...")

        # Apply mask to inputs
        seeded_inputs = self.eval_inputs.copy(deep=True)
        for var in mask_ds.data_vars:
            if var in seeded_inputs.data_vars:
                mask_vals = mask_ds[var].values
                input_vals = seeded_inputs[var].values

                # Align shapes - mask may have different time dimension
                # eval_inputs typically has 2 time steps (t-1, t)
                if mask_vals.shape == input_vals.shape:
                    seeded_inputs[var].values = input_vals + mask_vals
                else:
                    # Apply mask to the last time step of inputs (current state)
                    # Reshape mask to fit
                    try:
                        if input_vals.ndim == mask_vals.ndim:
                            # Trim or pad time dimension
                            min_t = min(input_vals.shape[0], mask_vals.shape[0])
                            seeded_inputs[var].values[-min_t:] = (
                                input_vals[-min_t:] + mask_vals[:min_t]
                            )
                        else:
                            # Broadcast - apply to last time step only
                            seeded_inputs[var].values[-1] = (
                                input_vals[-1] + mask_vals[0]
                                if mask_vals.ndim == input_vals.ndim
                                else input_vals[-1] + mask_vals
                            )
                    except (ValueError, IndexError):
                        # Shape mismatch - skip this variable
                        print(f"  Warning: shape mismatch for {var}, skipping")
                        continue

        predictions, _ = self._run_forward_jitted(
            self.params, self.state, self._rng,
            seeded_inputs,
            self.eval_targets * np.nan,
            self.eval_forcings,
        )
        jax.block_until_ready(predictions)
        print("Seeded forecast complete.")
        return predictions

    def run_seeded_with_inputs(self, seeded_inputs: xr.Dataset) -> xr.Dataset:
        """
        Run forecast with pre-modified initial conditions.

        This method is designed for the two-pass drought-responsive pipeline
        where initial conditions have already been modified with an
        InitialConditionMask before calling this method.

        Unlike run_seeded() which takes a mask and applies it, this method
        takes already-modified inputs directly. This allows for more
        sophisticated perturbation strategies like drought-targeted
        interventions that analyze the control forecast first.

        Args:
            seeded_inputs: Pre-modified eval_inputs xr.Dataset.
                          Should have shape (batch=1, time=2, ...) matching
                          the expected GraphCast input structure.

        Returns:
            Predictions xr.Dataset from GraphCast autoregressive inference.
        """
        print("Running SEEDED forecast with pre-modified inputs ...")

        predictions, _ = self._run_forward_jitted(
            self.params, self.state, self._rng,
            seeded_inputs,
            self.eval_targets * np.nan,
            self.eval_forcings,
        )
        jax.block_until_ready(predictions)
        print("Seeded forecast complete.")
        return predictions

    def get_eval_inputs(self) -> xr.Dataset:
        """Return the current eval_inputs for external modification."""
        return self.eval_inputs.copy(deep=True)


# ---------------------------------------------------------------------------
# Comparison Visualization
# ---------------------------------------------------------------------------

def _kelvin_to_f(k):
    return (k - 273.15) * 9 / 5 + 32

def _pa_to_hpa(pa):
    return pa / 100.0


def create_comparison_animation(
    control_preds: xr.Dataset,
    seeded_preds: xr.Dataset,
    intervention_name: str,
    base_datetime,
    output_path: str,
):
    """
    Create a side-by-side animation comparing control vs seeded forecasts.

    Left column: Control forecast
    Right column: Seeded (geoengineered) forecast
    Bottom: Difference map (seeded - control)
    """
    if not HAS_CARTOPY:
        print("Cartopy not available - skipping animation. Install with: pip install cartopy")
        return

    n_steps = control_preds.sizes["time"]
    lats = control_preds.coords["lat"].values
    lons = control_preds.coords["lon"].values

    # Extract 2m temperature and precipitation (the most visible variables)
    t2m_ctrl = control_preds["2m_temperature"].values[:, 0]  # (time, lat, lon)
    t2m_seed = seeded_preds["2m_temperature"].values[:, 0]

    precip_ctrl = control_preds["total_precipitation_6hr"].values[:, 0]
    precip_seed = seeded_preds["total_precipitation_6hr"].values[:, 0]

    # Convert units
    t2m_ctrl_f = _kelvin_to_f(t2m_ctrl)
    t2m_seed_f = _kelvin_to_f(t2m_seed)
    t2m_diff = t2m_seed_f - t2m_ctrl_f

    precip_ctrl_mm = precip_ctrl * 1000
    precip_seed_mm = precip_seed * 1000
    precip_diff_mm = precip_seed_mm - precip_ctrl_mm

    # Forecast hours
    time_vals = control_preds.coords["time"].values
    forecast_hours = [int(t / np.timedelta64(1, "h")) for t in time_vals]
    valid_times = [np.datetime64(base_datetime) + t for t in time_vals]

    # Setup figure: 3 rows x 2 cols
    proj = ccrs.Robinson()
    data_crs = ccrs.PlateCarree()

    fig = plt.figure(figsize=(20, 16), facecolor="#0a0a2e")
    gs = fig.add_gridspec(3, 2, hspace=0.22, wspace=0.08,
                          left=0.03, right=0.97, top=0.90, bottom=0.04)

    axes = np.array([
        [fig.add_subplot(gs[r, c], projection=proj) for c in range(2)]
        for r in range(3)
    ])

    panel_titles = [
        ["Control: Temperature (F)", f"Seeded ({intervention_name}): Temperature (F)"],
        ["Control: Precipitation (mm/6hr)", f"Seeded ({intervention_name}): Precipitation (mm/6hr)"],
        ["Temperature Difference (F)", "Precipitation Difference (mm/6hr)"],
    ]

    temp_norm = mcolors.Normalize(vmin=-40, vmax=120)
    temp_cmap = plt.cm.RdYlBu_r

    precip_colors = [
        (1, 1, 1, 0), (0.6, 0.8, 1, 0.4), (0.2, 0.5, 1, 0.7),
        (0.1, 0.2, 0.8, 0.85), (0.5, 0.1, 0.7, 0.9), (0.8, 0.1, 0.5, 0.95),
    ]
    precip_cmap = mcolors.LinearSegmentedColormap.from_list("precip", precip_colors, N=256)
    precip_norm = mcolors.Normalize(vmin=0, vmax=25)

    diff_temp_norm = mcolors.TwoSlopeNorm(vmin=-5, vcenter=0, vmax=5)
    diff_precip_norm = mcolors.TwoSlopeNorm(vmin=-10, vcenter=0, vmax=10)

    def style_ax(ax, title):
        ax.set_global()
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="#ccc")
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="#888")
        ax.set_facecolor("#0d1b2a")
        ax.spines["geo"].set_edgecolor("#444466")
        ax.set_title(title, color="white", fontsize=10, fontweight="bold", pad=6)

    for r in range(3):
        for c in range(2):
            style_ax(axes[r, c], panel_titles[r][c])

    title_text = fig.suptitle("", color="white", fontsize=16, fontweight="bold", y=0.97)
    subtitle_text = fig.text(0.5, 0.935, "", ha="center", color="#aabbcc", fontsize=11)

    # Colorbars
    for ax, norm, cmap, label in [
        (axes[0, 0], temp_norm, temp_cmap, "F"),
        (axes[0, 1], temp_norm, temp_cmap, "F"),
        (axes[1, 0], precip_norm, precip_cmap, "mm/6hr"),
        (axes[1, 1], precip_norm, precip_cmap, "mm/6hr"),
        (axes[2, 0], diff_temp_norm, "coolwarm", "delta F"),
        (axes[2, 1], diff_precip_norm, "BrBG", "delta mm/6hr"),
    ]:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        cb = fig.colorbar(sm, ax=ax, orientation="horizontal",
                          fraction=0.04, pad=0.02, aspect=35)
        cb.ax.tick_params(colors="white", labelsize=7)
        cb.set_label(label, color="white", fontsize=8)

    fig.text(0.01, 0.01,
             f"GraphCast Small (1 deg) + {intervention_name} Geoengineering Mask",
             color="#556677", fontsize=8)

    def draw_frame(i):
        for r in range(3):
            for c in range(2):
                for coll in list(axes[r, c].collections):
                    coll.remove()

        fhr = forecast_hours[i]
        vt_str = str(valid_times[i])[:16].replace("T", " ")
        title_text.set_text(f"Geoengineering Comparison: {intervention_name} | Hour {fhr:03d}")
        subtitle_text.set_text(
            f"Init: {str(np.datetime64(base_datetime))[:16].replace('T', ' ')} UTC"
            f"    ->    Valid: {vt_str} UTC"
        )

        # Row 0: Temperature
        axes[0, 0].contourf(lons, lats, t2m_ctrl_f[i], levels=np.linspace(-40, 120, 33),
                             cmap=temp_cmap, norm=temp_norm, transform=data_crs, extend="both")
        axes[0, 1].contourf(lons, lats, t2m_seed_f[i], levels=np.linspace(-40, 120, 33),
                             cmap=temp_cmap, norm=temp_norm, transform=data_crs, extend="both")

        # Row 1: Precipitation
        axes[1, 0].contourf(lons, lats, precip_ctrl_mm[i],
                             levels=[0, 0.5, 1, 2, 5, 10, 15, 20, 25],
                             cmap=precip_cmap, norm=precip_norm, transform=data_crs, extend="max")
        axes[1, 1].contourf(lons, lats, precip_seed_mm[i],
                             levels=[0, 0.5, 1, 2, 5, 10, 15, 20, 25],
                             cmap=precip_cmap, norm=precip_norm, transform=data_crs, extend="max")

        # Row 2: Differences
        axes[2, 0].contourf(lons, lats, t2m_diff[i], levels=np.linspace(-5, 5, 21),
                             cmap="coolwarm", norm=diff_temp_norm, transform=data_crs, extend="both")
        axes[2, 1].contourf(lons, lats, precip_diff_mm[i], levels=np.linspace(-10, 10, 21),
                             cmap="BrBG", norm=diff_precip_norm, transform=data_crs, extend="both")

        print(f"  Frame {i + 1}/{n_steps}: +{fhr}h", flush=True)
        return []

    print(f"Rendering {n_steps} frames ...")
    anim = animation.FuncAnimation(fig, draw_frame, frames=n_steps, blit=False)

    writer = animation.FFMpegWriter(fps=3, codec="prores_ks",
                                     extra_args=["-pix_fmt", "yuva444p10le"])
    print(f"Saving animation to {output_path} ...")
    anim.save(output_path, writer=writer, dpi=120)
    plt.close(fig)
    print(f"Done! File saved: {output_path}")
    print(f"  Size: {os.path.getsize(output_path) / 1e6:.1f} MB")


def create_static_comparison(
    control_preds: xr.Dataset,
    seeded_preds: xr.Dataset,
    intervention_name: str,
    base_datetime,
    time_idx: int = -1,
    output_path: Optional[str] = None,
):
    """
    Create a static PNG comparison at a single time step.

    Useful for quick inspection without needing ffmpeg.
    """
    lats = control_preds.coords["lat"].values
    lons = control_preds.coords["lon"].values

    t2m_ctrl = _kelvin_to_f(control_preds["2m_temperature"].values[time_idx, 0])
    t2m_seed = _kelvin_to_f(seeded_preds["2m_temperature"].values[time_idx, 0])
    t2m_diff = t2m_seed - t2m_ctrl

    precip_ctrl = control_preds["total_precipitation_6hr"].values[time_idx, 0] * 1000
    precip_seed = seeded_preds["total_precipitation_6hr"].values[time_idx, 0] * 1000
    precip_diff = precip_seed - precip_ctrl

    if HAS_CARTOPY:
        proj = ccrs.Robinson()
        data_crs = ccrs.PlateCarree()
        fig, axes = plt.subplots(2, 3, figsize=(24, 12),
                                  subplot_kw={"projection": proj},
                                  facecolor="#0a0a2e")
    else:
        fig, axes = plt.subplots(2, 3, figsize=(24, 12), facecolor="#0a0a2e")

    # Row 0: Temperature (control, seeded, difference)
    temp_data = [t2m_ctrl, t2m_seed, t2m_diff]
    temp_titles = ["Control Temp (F)", f"Seeded Temp (F)", "Temp Difference (F)"]
    temp_cmaps = ["RdYlBu_r", "RdYlBu_r", "coolwarm"]

    for c in range(3):
        ax = axes[0, c]
        if HAS_CARTOPY:
            ax.set_global()
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="#ccc")
            ax.set_facecolor("#0d1b2a")
            if c < 2:
                im = ax.contourf(lons, lats, temp_data[c], levels=30,
                                  cmap=temp_cmaps[c], transform=data_crs)
            else:
                im = ax.contourf(lons, lats, temp_data[c], levels=np.linspace(-5, 5, 21),
                                  cmap=temp_cmaps[c], transform=data_crs, extend="both")
        else:
            im = ax.imshow(temp_data[c], cmap=temp_cmaps[c], aspect="auto")
        ax.set_title(temp_titles[c], color="white", fontsize=11, fontweight="bold")
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.046, pad=0.06)
        cb.ax.tick_params(colors="white", labelsize=8)

    # Row 1: Precipitation (control, seeded, difference)
    precip_data = [precip_ctrl, precip_seed, precip_diff]
    precip_titles = ["Control Precip (mm/6hr)", "Seeded Precip (mm/6hr)", "Precip Difference (mm/6hr)"]
    precip_cmaps = ["Blues", "Blues", "BrBG"]

    for c in range(3):
        ax = axes[1, c]
        if HAS_CARTOPY:
            ax.set_global()
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="#ccc")
            ax.set_facecolor("#0d1b2a")
            if c < 2:
                im = ax.contourf(lons, lats, precip_data[c],
                                  levels=[0, 0.5, 1, 2, 5, 10, 15, 20, 25],
                                  cmap=precip_cmaps[c], transform=data_crs, extend="max")
            else:
                im = ax.contourf(lons, lats, precip_data[c], levels=np.linspace(-10, 10, 21),
                                  cmap=precip_cmaps[c], transform=data_crs, extend="both")
        else:
            im = ax.imshow(precip_data[c], cmap=precip_cmaps[c], aspect="auto")
        ax.set_title(precip_titles[c], color="white", fontsize=11, fontweight="bold")
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.046, pad=0.06)
        cb.ax.tick_params(colors="white", labelsize=8)

    time_vals = control_preds.coords["time"].values
    fhr = int(time_vals[time_idx] / np.timedelta64(1, "h"))
    fig.suptitle(
        f"GraphCast Geoengineering: {intervention_name} | +{fhr}h Forecast",
        color="white", fontsize=14, fontweight="bold", y=0.98,
    )
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Comparison saved: {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Offline Mode (no GraphCast) - Mask-only visualization
# ---------------------------------------------------------------------------

def run_mask_only_mode(dataset_path: str, intervention: str, region: TargetRegion,
                       intensity: float, output_dir: str):
    """
    Run in mask-only mode when GraphCast is not installed.

    Generates and visualizes the seeding masks applied to the raw ERA5 data
    without running GraphCast inference. Useful for inspecting mask geometry
    and perturbation magnitudes.
    """
    print("=" * 60)
    print("MASK-ONLY MODE (GraphCast not installed)")
    print("Generating seeding masks on raw ERA5 data...")
    print("=" * 60)

    sim = GeoEngineeringSimulator(dataset_path=dataset_path)

    os.makedirs(output_dir, exist_ok=True)

    if intervention == "all":
        interventions = list(MASK_REGISTRY.keys())
    elif intervention == "all_drought":
        interventions = DROUGHT_MASKS
    elif intervention == "all_hurricane":
        interventions = HURRICANE_MASKS
    else:
        interventions = [intervention]

    results = {}
    for name in interventions:
        result = sim.apply_intervention(name, region, intensity)
        result.save(output_dir)
        results[name] = result

    # Generate summary plot
    from geoengineering_masks import plot_all_interventions_summary
    plot_all_interventions_summary(
        results,
        output_path=os.path.join(output_dir, "masks_summary.png"),
    )

    # Print summary table
    print("\n" + "=" * 70)
    print("GEOENGINEERING MASK GENERATION COMPLETE")
    print("=" * 70)
    print(f"\n{'Type':<30} {'Category':<12} {'Purpose':<30}")
    print("-" * 70)
    purpose_map = {
        "glaciogenic_static": "Cold cloud precip enhancement",
        "hygroscopic_enhancement": "Warm cloud rain triggering",
        "electric_ionization": "Accelerated coalescence",
        "laser_induced_condensation": "Sub-saturated nucleation",
        "glaciogenic_dynamic": "Storm circulation disruption",
        "hygroscopic_suppression": "Rainband weakening (Twomey)",
    }
    for name, result in results.items():
        purpose = purpose_map.get(name, "")
        print(f"{name:<30} {result.category:<12} {purpose:<30}")
    print("=" * 70)
    print(f"\nOutput files saved to: {output_dir}")
    print("\nTo run with full GraphCast inference, install:")
    print("  pip install dm-haiku jax graphcast google-cloud-storage")

    return results


# ---------------------------------------------------------------------------
# Full Pipeline Mode (with GraphCast inference)
# ---------------------------------------------------------------------------

def run_full_pipeline(intervention: str, region: TargetRegion,
                      intensity: float, output_dir: str,
                      animate: bool = True):
    """
    Run the full pipeline: masks + GraphCast inference + comparison.
    """
    print("=" * 60)
    print("FULL PIPELINE MODE (GraphCast + Geoengineering)")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # Download data
    params_path, dataset_path, stats_paths = download_all()

    # Initialize runner
    runner = GraphCastRunner(params_path, stats_paths, dataset_path)

    # Run control forecast
    control_preds = runner.run_control()

    # Determine which interventions to run
    if intervention == "all":
        interventions = list(MASK_REGISTRY.keys())
    elif intervention == "all_drought":
        interventions = DROUGHT_MASKS
    elif intervention == "all_hurricane":
        interventions = HURRICANE_MASKS
    else:
        interventions = [intervention]

    # Create simulator for mask generation
    sim = GeoEngineeringSimulator(dataset=runner.eval_inputs)

    for name in interventions:
        print(f"\n{'='*60}")
        print(f"INTERVENTION: {name}")
        print(f"{'='*60}")

        # Generate mask
        result = sim.apply_intervention(name, region, intensity)

        # Run seeded forecast
        seeded_preds = runner.run_seeded(result.mask_dataset)

        # Save seeded predictions
        seeded_path = os.path.join(output_dir, f"predictions_seeded_{name}.nc")
        seeded_preds.to_netcdf(seeded_path)

        # Generate comparison
        if animate:
            anim_path = os.path.join(output_dir, f"comparison_{name}.mov")
            create_comparison_animation(
                control_preds, seeded_preds, name,
                runner.base_datetime, anim_path,
            )
        else:
            png_path = os.path.join(output_dir, f"comparison_{name}.png")
            create_static_comparison(
                control_preds, seeded_preds, name,
                runner.base_datetime, time_idx=-1,
                output_path=png_path,
            )

    # Save control predictions
    control_path = os.path.join(output_dir, "predictions_control.nc")
    control_preds.to_netcdf(control_path)

    print(f"\nAll outputs saved to: {output_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run GraphCast with geoengineering intervention masks"
    )
    parser.add_argument(
        "--intervention", "-i",
        choices=list(MASK_REGISTRY.keys()) + ["all_drought", "all_hurricane", "all"],
        default="all",
        help="Intervention type",
    )
    parser.add_argument("--lat-min", type=float, default=20, help="Min latitude")
    parser.add_argument("--lat-max", type=float, default=50, help="Max latitude")
    parser.add_argument("--lon-min", type=float, default=230, help="Min longitude")
    parser.add_argument("--lon-max", type=float, default=270, help="Max longitude")
    parser.add_argument("--intensity", type=float, default=0.5, help="0.0-1.0")
    parser.add_argument("--output", "-o", default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--no-animate", action="store_true",
                        help="Generate PNG instead of .mov")
    parser.add_argument("--mask-only", action="store_true",
                        help="Only generate masks (skip GraphCast inference)")
    parser.add_argument("--dataset", default=None,
                        help="Path to custom dataset (for mask-only mode)")

    args = parser.parse_args()
    region = TargetRegion(args.lat_min, args.lat_max, args.lon_min, args.lon_max)

    if args.mask_only or not HAS_GRAPHCAST:
        dataset = args.dataset or os.path.join(CACHE_DIR, "example_data_20steps.nc")
        if not os.path.exists(dataset):
            dataset = os.path.join(CACHE_DIR, "example_data.nc")
        if not os.path.exists(dataset):
            print("ERROR: No dataset found. Download first by running graphcast_demo.py")
            return
        run_mask_only_mode(dataset, args.intervention, region, args.intensity, args.output)
    else:
        run_full_pipeline(
            args.intervention, region, args.intensity,
            args.output, animate=not args.no_animate,
        )


if __name__ == "__main__":
    main()
