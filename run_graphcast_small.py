#!/usr/bin/env python3
"""Run weather forecast simulation using GraphCast_small model.

This script downloads the GraphCast_small checkpoint (1° resolution, 13 pressure
levels), loads matching example data and normalization statistics, runs inference
using rollout.chunked_prediction, and saves predictions to a NetCDF file.

Usage:
    python run_graphcast_small.py
"""

import dataclasses
import functools

from google.cloud import storage
from graphcast import autoregressive
from graphcast import casting
from graphcast import checkpoint
from graphcast import data_utils
from graphcast import graphcast
from graphcast import normalization
from graphcast import rollout
import haiku as hk
import jax
import numpy as np
import xarray


# GraphCast_small checkpoint filename
PARAMS_FILE = (
    "GraphCast_small - ERA5 1979-2015 - resolution 1.0 - "
    "pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
)

# Dataset matching 1.0° resolution and 13 pressure levels
DATASET_FILE = "source-era5_date-2022-01-01_res-1.0_levels-13_steps-04.nc"

# GCS bucket and prefix
GCS_BUCKET = "dm_graphcast"
DIR_PREFIX = "graphcast/"


def load_checkpoint(gcs_bucket):
    """Load the GraphCast_small model checkpoint from GCS."""
    print(f"Loading checkpoint: {PARAMS_FILE}")
    with gcs_bucket.blob(f"{DIR_PREFIX}params/{PARAMS_FILE}").open("rb") as f:
        ckpt = checkpoint.load(f, graphcast.CheckPoint)
    print(f"Model description: {ckpt.description}")
    return ckpt


def load_dataset(gcs_bucket):
    """Load the example dataset from GCS."""
    print(f"Loading dataset: {DATASET_FILE}")
    with gcs_bucket.blob(f"{DIR_PREFIX}dataset/{DATASET_FILE}").open("rb") as f:
        example_batch = xarray.load_dataset(f, decode_timedelta=True).compute()
    print(f"Dataset dimensions: {dict(example_batch.sizes)}")
    return example_batch


def load_normalization_stats(gcs_bucket):
    """Load normalization statistics from GCS."""
    print("Loading normalization statistics...")
    with gcs_bucket.blob(f"{DIR_PREFIX}stats/diffs_stddev_by_level.nc").open("rb") as f:
        diffs_stddev_by_level = xarray.load_dataset(f).compute()
    with gcs_bucket.blob(f"{DIR_PREFIX}stats/mean_by_level.nc").open("rb") as f:
        mean_by_level = xarray.load_dataset(f).compute()
    with gcs_bucket.blob(f"{DIR_PREFIX}stats/stddev_by_level.nc").open("rb") as f:
        stddev_by_level = xarray.load_dataset(f).compute()
    return diffs_stddev_by_level, mean_by_level, stddev_by_level


def construct_wrapped_graphcast(
    model_config: graphcast.ModelConfig,
    task_config: graphcast.TaskConfig,
    diffs_stddev_by_level: xarray.Dataset,
    mean_by_level: xarray.Dataset,
    stddev_by_level: xarray.Dataset,
):
    """Constructs and wraps the GraphCast Predictor."""
    # Core one-step predictor
    predictor = graphcast.GraphCast(model_config, task_config)

    # Cast to/from BFloat16 for efficiency
    predictor = casting.Bfloat16Cast(predictor)

    # Apply normalization to inputs and predict residuals for targets
    predictor = normalization.InputsAndResiduals(
        predictor,
        diffs_stddev_by_level=diffs_stddev_by_level,
        mean_by_level=mean_by_level,
        stddev_by_level=stddev_by_level,
    )

    # Wrap to enable autoregressive multi-step prediction
    predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
    return predictor


def main():
    print("=" * 60)
    print("GraphCast_small Weather Forecast Simulation")
    print("=" * 60)
    print()

    # Initialize GCS client (anonymous access)
    print("Connecting to Google Cloud Storage...")
    gcs_client = storage.Client.create_anonymous_client()
    gcs_bucket = gcs_client.get_bucket(GCS_BUCKET)

    # Load checkpoint
    ckpt = load_checkpoint(gcs_bucket)
    params = ckpt.params
    state = {}
    model_config = ckpt.model_config
    task_config = ckpt.task_config

    print(f"Model config: resolution={model_config.resolution}, "
          f"mesh_size={model_config.mesh_size}, "
          f"latent_size={model_config.latent_size}")
    print()

    # Load example dataset
    example_batch = load_dataset(gcs_bucket)

    # Load normalization statistics
    diffs_stddev_by_level, mean_by_level, stddev_by_level = load_normalization_stats(gcs_bucket)

    # Extract inputs, targets, and forcings
    # Use all available time steps minus the 2 input steps
    eval_steps = example_batch.sizes["time"] - 2
    print(f"\nExtracting inputs/targets/forcings for {eval_steps} prediction steps...")

    eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
        example_batch,
        target_lead_times=slice("6h", f"{eval_steps * 6}h"),
        **dataclasses.asdict(task_config),
    )

    print(f"Eval Inputs:   {dict(eval_inputs.sizes)}")
    print(f"Eval Targets:  {dict(eval_targets.sizes)}")
    print(f"Eval Forcings: {dict(eval_forcings.sizes)}")

    # Build the wrapped predictor with haiku transform
    @hk.transform_with_state
    def run_forward(model_config, task_config, inputs, targets_template, forcings):
        predictor = construct_wrapped_graphcast(
            model_config,
            task_config,
            diffs_stddev_by_level,
            mean_by_level,
            stddev_by_level,
        )
        return predictor(inputs, targets_template=targets_template, forcings=forcings)

    # Create jitted function with configs baked in
    def with_configs(fn):
        return functools.partial(fn, model_config=model_config, task_config=task_config)

    def with_params(fn):
        return functools.partial(fn, params=params, state=state)

    def drop_state(fn):
        return lambda **kw: fn(**kw)[0]

    run_forward_jitted = drop_state(
        with_params(jax.jit(with_configs(run_forward.apply)))
    )

    # Verify resolution matches
    data_resolution = 360.0 / eval_inputs.sizes["lon"]
    assert model_config.resolution in (0, data_resolution), (
        f"Model resolution ({model_config.resolution}) doesn't match "
        f"data resolution ({data_resolution})."
    )

    # Run autoregressive prediction
    print("\n" + "=" * 60)
    print("Running autoregressive prediction...")
    print("(First run includes JIT compilation time)")
    print("=" * 60)

    predictions = rollout.chunked_prediction(
        run_forward_jitted,
        rng=jax.random.PRNGKey(0),
        inputs=eval_inputs,
        targets_template=eval_targets * np.nan,
        forcings=eval_forcings,
    )

    print(f"\nPredictions shape: {dict(predictions.sizes)}")
    print(f"Predicted variables: {list(predictions.data_vars.keys())}")

    # Save predictions to NetCDF
    output_file = "predictions.nc"
    print(f"\nSaving predictions to {output_file}...")
    predictions.to_netcdf(output_file)
    print(f"Successfully saved predictions to {output_file}")

    # Print summary statistics for a few key variables
    print("\n" + "=" * 60)
    print("Prediction Summary Statistics")
    print("=" * 60)
    key_vars = ["2m_temperature", "10m_u_component_of_wind", "mean_sea_level_pressure"]
    for var in key_vars:
        if var in predictions.data_vars:
            data = predictions[var].values
            print(f"{var}:")
            print(f"  min={np.nanmin(data):.2f}, max={np.nanmax(data):.2f}, "
                  f"mean={np.nanmean(data):.2f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
