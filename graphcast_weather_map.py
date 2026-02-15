"""GraphCast Animated Weather Map.

Downloads GraphCast_small and 5-day example data, runs a 20-step (120h)
autoregressive forecast, then renders an animated weather map saved as .mov.
"""

import dataclasses
import os

from google.cloud import storage
import haiku as hk
import jax
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import xarray as xr

from graphcast import (
    autoregressive,
    casting,
    checkpoint,
    data_utils,
    graphcast,
    normalization,
)

# ---------------------------------------------------------------------------
# GCS config
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
OUTPUT_FILE = os.path.expanduser("~/graphcast_forecast.mov")


# ---------------------------------------------------------------------------
# Download helpers
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
# Model inference
# ---------------------------------------------------------------------------
def run_graphcast(params_path, dataset_path, stats_paths):
    """Run 20-step GraphCast prediction and return predictions + targets."""
    # Load checkpoint
    print("Loading checkpoint ...")
    with open(params_path, "rb") as f:
        ckpt = checkpoint.load(f, graphcast.CheckPoint)
    model_config = ckpt.model_config
    task_config = ckpt.task_config
    params = ckpt.params
    state = {}

    # Load stats
    stats = {k: xr.load_dataset(v).compute() for k, v in stats_paths.items()}

    # Load example data
    print("Loading example dataset (20-step) ...")
    example_ds = xr.load_dataset(dataset_path).compute()
    if "batch" not in example_ds.dims:
        example_ds = example_ds.expand_dims("batch")

    # Extract inputs / targets / forcings for all 20 steps
    n_steps = 20
    print(f"Extracting inputs/targets/forcings for {n_steps} steps ...")
    eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
        example_ds,
        target_lead_times=slice("6h", f"{6 * n_steps}h"),
        **dataclasses.asdict(task_config),
    )

    # Build haiku model
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

    run_forward_jitted = jax.jit(run_forward.apply)
    rng = jax.random.PRNGKey(0)

    print("JIT-compiling + running 20-step forecast (may take several minutes on CPU) ...")
    predictions, _ = run_forward_jitted(
        params, state, rng,
        eval_inputs,
        eval_targets * np.nan,
        eval_forcings,
    )
    jax.block_until_ready(predictions)
    print("Prediction complete!")

    # Get datetime for labels
    base_datetime = example_ds.coords["datetime"].values[0, 0]  # batch=0, first time
    return predictions, eval_targets, base_datetime, task_config


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _kelvin_to_f(k):
    return (k - 273.15) * 9 / 5 + 32


def _pa_to_hpa(pa):
    return pa / 100.0


def _wind_speed(u, v):
    return np.sqrt(u**2 + v**2)


def create_animation(predictions, targets, base_datetime, output_path):
    """Create a broadcast-style animated weather map."""
    n_steps = predictions.sizes["time"]
    lats = predictions.coords["lat"].values
    lons = predictions.coords["lon"].values

    # Pre-extract all data (squeeze batch dim)
    t2m_pred = predictions["2m_temperature"].values[:, 0]  # (time, lat, lon)
    mslp_pred = predictions["mean_sea_level_pressure"].values[:, 0]
    u10_pred = predictions["10m_u_component_of_wind"].values[:, 0]
    v10_pred = predictions["10m_v_component_of_wind"].values[:, 0]
    precip_pred = predictions["total_precipitation_6hr"].values[:, 0]

    # For geopotential at ~500hPa: find closest level
    levels = predictions.coords["level"].values
    idx_500 = np.argmin(np.abs(levels - 500))
    geopot_pred = predictions["geopotential"].values[:, 0, idx_500]  # (time, lat, lon)

    # Convert units
    t2m_f = _kelvin_to_f(t2m_pred)
    mslp_hpa = _pa_to_hpa(mslp_pred)
    wspd = _wind_speed(u10_pred, v10_pred)
    geopot_dam = geopot_pred / (9.80665 * 10)  # geopotential -> decameters
    precip_mm = precip_pred * 1000  # m -> mm

    # Compute forecast hours
    time_vals = predictions.coords["time"].values  # timedelta64
    forecast_hours = [int(t / np.timedelta64(1, "h")) for t in time_vals]

    # Compute valid datetimes
    valid_times = [np.datetime64(base_datetime) + t for t in time_vals]

    # Wind subsampling for quiver
    skip = 8
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # -----------------------------------------------------------------------
    # Color maps & normalization
    # -----------------------------------------------------------------------
    # Temperature: deep blue (-40F) -> cyan -> green -> yellow -> orange -> red (120F)
    temp_norm = mcolors.Normalize(vmin=-40, vmax=120)
    temp_cmap = plt.cm.RdYlBu_r

    # Precipitation: transparent white -> blue -> purple
    precip_colors = [
        (1, 1, 1, 0),       # transparent
        (0.6, 0.8, 1, 0.4), # light blue
        (0.2, 0.5, 1, 0.7), # blue
        (0.1, 0.2, 0.8, 0.85), # dark blue
        (0.5, 0.1, 0.7, 0.9),  # purple
        (0.8, 0.1, 0.5, 0.95), # magenta
    ]
    precip_cmap = mcolors.LinearSegmentedColormap.from_list("precip", precip_colors, N=256)
    precip_norm = mcolors.Normalize(vmin=0, vmax=25)

    # Wind speed
    wind_norm = mcolors.Normalize(vmin=0, vmax=30)

    # -----------------------------------------------------------------------
    # Figure layout: 2x2 panels
    # -----------------------------------------------------------------------
    proj = ccrs.Robinson()
    data_crs = ccrs.PlateCarree()

    fig = plt.figure(figsize=(20, 11), facecolor="#0a0a2e")

    gs = fig.add_gridspec(2, 2, hspace=0.18, wspace=0.08,
                          left=0.03, right=0.97, top=0.90, bottom=0.05)

    axes = [
        fig.add_subplot(gs[0, 0], projection=proj),
        fig.add_subplot(gs[0, 1], projection=proj),
        fig.add_subplot(gs[1, 0], projection=proj),
        fig.add_subplot(gs[1, 1], projection=proj),
    ]

    panel_titles = [
        "2m Temperature (\u00b0F) & 10m Wind",
        "Mean Sea Level Pressure (hPa)",
        "6hr Precipitation (mm)",
        "500 hPa Geopotential Height (dam)",
    ]

    def style_ax(ax, title):
        ax.set_global()
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="#cccccc")
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="#888888")
        ax.set_facecolor("#0d1b2a")
        ax.spines["geo"].set_edgecolor("#444466")
        ax.spines["geo"].set_linewidth(1)
        ax.set_title(title, color="white", fontsize=11, fontweight="bold", pad=6)

    for ax, title in zip(axes, panel_titles):
        style_ax(ax, title)

    # Main title (updated per frame)
    title_text = fig.suptitle("", color="white", fontsize=16, fontweight="bold", y=0.97)
    subtitle_text = fig.text(0.5, 0.935, "", ha="center", color="#aabbcc", fontsize=11)

    # Colorbars
    sm_temp = plt.cm.ScalarMappable(norm=temp_norm, cmap=temp_cmap)
    cb_temp = fig.colorbar(sm_temp, ax=axes[0], orientation="horizontal",
                           fraction=0.045, pad=0.02, aspect=35)
    cb_temp.ax.tick_params(colors="white", labelsize=8)
    cb_temp.set_label("\u00b0F", color="white", fontsize=9)

    sm_mslp = plt.cm.ScalarMappable(
        norm=mcolors.Normalize(vmin=np.nanmin(mslp_hpa), vmax=np.nanmax(mslp_hpa)),
        cmap="coolwarm")
    cb_mslp = fig.colorbar(sm_mslp, ax=axes[1], orientation="horizontal",
                           fraction=0.045, pad=0.02, aspect=35)
    cb_mslp.ax.tick_params(colors="white", labelsize=8)
    cb_mslp.set_label("hPa", color="white", fontsize=9)

    sm_precip = plt.cm.ScalarMappable(norm=precip_norm, cmap=precip_cmap)
    cb_precip = fig.colorbar(sm_precip, ax=axes[2], orientation="horizontal",
                             fraction=0.045, pad=0.02, aspect=35)
    cb_precip.ax.tick_params(colors="white", labelsize=8)
    cb_precip.set_label("mm / 6hr", color="white", fontsize=9)

    sm_gpt = plt.cm.ScalarMappable(
        norm=mcolors.Normalize(vmin=np.nanmin(geopot_dam), vmax=np.nanmax(geopot_dam)),
        cmap="viridis")
    cb_gpt = fig.colorbar(sm_gpt, ax=axes[3], orientation="horizontal",
                          fraction=0.045, pad=0.02, aspect=35)
    cb_gpt.ax.tick_params(colors="white", labelsize=8)
    cb_gpt.set_label("dam", color="white", fontsize=9)

    # Branding
    fig.text(0.01, 0.01, "GraphCast Small (1\u00b0) \u2022 Google DeepMind",
             color="#556677", fontsize=8, ha="left")

    # -----------------------------------------------------------------------
    # Animation
    # -----------------------------------------------------------------------
    def draw_frame(i):
        for ax in axes:
            # Clear previous filled contours but keep map features
            for coll in list(ax.collections):
                coll.remove()
            for line in list(ax.lines):
                line.remove()
            for text in list(ax.texts):
                text.remove()

        fhr = forecast_hours[i]
        vt = valid_times[i]
        vt_str = str(vt)[:16].replace("T", " ")
        title_text.set_text(f"GraphCast Forecast \u2022 Hour {fhr:03d}")
        subtitle_text.set_text(
            f"Init: {str(np.datetime64(base_datetime))[:16].replace('T', ' ')} UTC"
            f"    \u2192    Valid: {vt_str} UTC"
        )

        # Panel 1: Temperature + wind
        axes[0].contourf(
            lons, lats, t2m_f[i],
            levels=np.linspace(-40, 120, 33),
            cmap=temp_cmap, norm=temp_norm,
            transform=data_crs, extend="both",
        )
        axes[0].quiver(
            lon_grid[::skip, ::skip], lat_grid[::skip, ::skip],
            u10_pred[i, ::skip, ::skip], v10_pred[i, ::skip, ::skip],
            color="white", alpha=0.5, scale=300, width=0.002,
            transform=data_crs,
        )

        # Panel 2: MSLP filled + contour lines
        axes[1].contourf(
            lons, lats, mslp_hpa[i],
            levels=np.linspace(950, 1060, 23),
            cmap="coolwarm",
            transform=data_crs, extend="both",
        )
        cs = axes[1].contour(
            lons, lats, mslp_hpa[i],
            levels=np.arange(950, 1060, 4),
            colors="white", linewidths=0.4, alpha=0.6,
            transform=data_crs,
        )

        # Panel 3: Precipitation
        axes[2].contourf(
            lons, lats, precip_mm[i],
            levels=[0, 0.5, 1, 2, 5, 10, 15, 20, 25],
            cmap=precip_cmap, norm=precip_norm,
            transform=data_crs, extend="max",
        )
        # Add coastlines again on top for contrast
        axes[2].add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="#cccccc")

        # Panel 4: 500hPa geopotential
        axes[3].contourf(
            lons, lats, geopot_dam[i],
            levels=30,
            cmap="viridis",
            transform=data_crs, extend="both",
        )
        axes[3].contour(
            lons, lats, geopot_dam[i],
            levels=15,
            colors="white", linewidths=0.3, alpha=0.5,
            transform=data_crs,
        )

        print(f"  Frame {i + 1}/{n_steps}: +{fhr}h", flush=True)
        return []

    print(f"Rendering {n_steps} frames ...")
    anim = animation.FuncAnimation(fig, draw_frame, frames=n_steps, blit=False)

    writer = animation.FFMpegWriter(fps=3, codec="prores_ks",
                                     extra_args=["-pix_fmt", "yuva444p10le"])
    print(f"Saving animation to {output_path} ...")
    anim.save(output_path, writer=writer, dpi=150)
    plt.close(fig)
    print(f"Done! File saved: {output_path}")
    print(f"  Size: {os.path.getsize(output_path) / 1e6:.1f} MB")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("GraphCast 5-Day Animated Weather Map")
    print("=" * 60)
    print(f"JAX backend: {jax.default_backend()}")
    print()

    params_path, dataset_path, stats_paths = download_all()
    print()

    predictions, targets, base_dt, task_config = run_graphcast(
        params_path, dataset_path, stats_paths
    )
    print()

    create_animation(predictions, targets, base_dt, OUTPUT_FILE)


if __name__ == "__main__":
    main()
