"""GraphCast Hello World Demo.

Downloads GraphCast_small model and example ERA5 data from Google Cloud Storage,
runs a 4-step (24h) autoregressive weather prediction, and prints a summary.
"""

import dataclasses
import functools
import io
import math
import os
import tempfile

from google.cloud import storage
import haiku as hk
import jax
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
# GCS paths
# ---------------------------------------------------------------------------
GCS_BUCKET = "dm_graphcast"

PARAMS_FILE = (
    "params/GraphCast_small - ERA5 1979-2015 - resolution 1.0 "
    "- pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
)
DATASET_FILE = (
    "dataset/source-era5_date-2022-01-01_res-1.0_levels-13_steps-04.nc"
)
STATS_FILES = {
    "diffs_stddev_by_level": "stats/diffs_stddev_by_level.nc",
    "mean_by_level": "stats/mean_by_level.nc",
    "stddev_by_level": "stats/stddev_by_level.nc",
}

# Local cache directory
CACHE_DIR = os.path.expanduser("~/graphcast_data")


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------
def _download_from_gcs(blob_name: str, dest_path: str) -> None:
    """Download a blob from GCS (anonymous) if not already cached."""
    if os.path.exists(dest_path):
        print(f"  [cached] {os.path.basename(dest_path)}")
        return
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"  Downloading {blob_name} ...")
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(dest_path)
    size_mb = os.path.getsize(dest_path) / 1e6
    print(f"  Done ({size_mb:.1f} MB)")


def download_all():
    """Download model params, example dataset, and normalization stats."""
    print("Downloading data from GCS ...")
    params_path = os.path.join(CACHE_DIR, "params.npz")
    _download_from_gcs(PARAMS_FILE, params_path)

    dataset_path = os.path.join(CACHE_DIR, "example_data.nc")
    _download_from_gcs(DATASET_FILE, dataset_path)

    stats_paths = {}
    for name, blob_name in STATS_FILES.items():
        path = os.path.join(CACHE_DIR, f"{name}.nc")
        _download_from_gcs(blob_name, path)
        stats_paths[name] = path

    return params_path, dataset_path, stats_paths


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_checkpoint(params_path: str):
    """Load model checkpoint (params + configs)."""
    print("Loading model checkpoint ...")
    with open(params_path, "rb") as f:
        ckpt = checkpoint.load(f, graphcast.CheckPoint)
    print(f"  Model config: resolution={ckpt.model_config.resolution}, "
          f"mesh_size={ckpt.model_config.mesh_size}, "
          f"latent_size={ckpt.model_config.latent_size}")
    print(f"  Task config: {len(ckpt.task_config.pressure_levels)} pressure levels, "
          f"input_duration={ckpt.task_config.input_duration}")
    return ckpt


def load_stats(stats_paths: dict[str, str]) -> dict[str, xr.Dataset]:
    """Load normalization statistics."""
    print("Loading normalization statistics ...")
    stats = {}
    for name, path in stats_paths.items():
        stats[name] = xr.load_dataset(path).compute()
    return stats


def load_example_data(dataset_path: str) -> xr.Dataset:
    """Load example ERA5 weather data."""
    print("Loading example dataset ...")
    ds = xr.load_dataset(dataset_path).compute()
    print(f"  Variables: {list(ds.data_vars)[:10]}...")
    print(f"  Dimensions: {dict(ds.dims)}")
    return ds


# ---------------------------------------------------------------------------
# Build and run model
# ---------------------------------------------------------------------------
def build_and_run(ckpt, stats, example_ds):
    """Build GraphCast predictor and run autoregressive prediction."""
    model_config = ckpt.model_config
    task_config = ckpt.task_config
    params = ckpt.params
    state = {}

    # Add batch dimension if missing
    if "batch" not in example_ds.dims:
        example_ds = example_ds.expand_dims("batch")

    # Extract inputs, targets, forcings
    print("Extracting inputs, targets, and forcings ...")
    # The example dataset has steps for 4 lead times (6h, 12h, 18h, 24h)
    eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
        example_ds,
        target_lead_times=slice("6h", f"{6 * 4}h"),
        **dataclasses.asdict(task_config),
    )
    print(f"  Inputs time steps: {eval_inputs.dims.get('time', 'N/A')}")
    print(f"  Target time steps: {eval_targets.dims.get('time', 'N/A')}")
    print(f"  Forcings time steps: {eval_forcings.dims.get('time', 'N/A')}")

    # Build the wrapped predictor inside a haiku transform
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

    # JIT compile the forward pass
    print("JIT-compiling model (this may take a few minutes on first run) ...")
    run_forward_jitted = jax.jit(run_forward.apply)

    # Run prediction
    rng = jax.random.PRNGKey(0)
    print("Running 4-step autoregressive prediction ...")
    predictions, _ = run_forward_jitted(
        params, state, rng,
        eval_inputs,
        eval_targets * np.nan,  # template only - values are ignored
        eval_forcings,
    )

    # Block until computation is done (JAX is async)
    jax.block_until_ready(predictions)
    print("Prediction complete!")

    return predictions, eval_targets


def print_summary(predictions, targets):
    """Print a summary of predicted weather variables."""
    print("\n" + "=" * 60)
    print("PREDICTION SUMMARY")
    print("=" * 60)
    print(f"Dimensions: {dict(predictions.dims)}")
    print(f"Time steps: {predictions.coords['time'].values}")
    print()

    for var_name in sorted(predictions.data_vars):
        pred_data = predictions[var_name].values
        tgt_data = targets[var_name].values
        print(f"  {var_name}:")
        print(f"    shape      = {pred_data.shape}")
        print(f"    pred range = [{np.nanmin(pred_data):.4f}, {np.nanmax(pred_data):.4f}]")
        print(f"    true range = [{np.nanmin(tgt_data):.4f}, {np.nanmax(tgt_data):.4f}]")

    print("\n" + "=" * 60)
    print("GraphCast demo completed successfully!")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("GraphCast Hello World Demo")
    print("=" * 60)
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    print()

    # 1. Download data
    params_path, dataset_path, stats_paths = download_all()
    print()

    # 2. Load everything
    ckpt = load_checkpoint(params_path)
    stats = load_stats(stats_paths)
    example_ds = load_example_data(dataset_path)
    print()

    # 3. Build model and predict
    predictions, targets = build_and_run(ckpt, stats, example_ds)

    # 4. Print results
    print_summary(predictions, targets)


if __name__ == "__main__":
    main()
