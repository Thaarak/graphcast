#!/usr/bin/env python3
"""Run 10-day weather forecast using GraphCast_small model.

This script runs GraphCast_small for a 10-day (40 timestep) forecast.
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


# GraphCast_small checkpoint
PARAMS_FILE = (
    "GraphCast_small - ERA5 1979-2015 - resolution 1.0 - "
    "pressure levels 13 - mesh 2to5 - precipitation input and output.npz"
)

# 40-step dataset for 10-day forecast
DATASET_FILE = "source-era5_date-2022-01-01_res-1.0_levels-13_steps-40.nc"

GCS_BUCKET = "dm_graphcast"
DIR_PREFIX = "graphcast/"


def load_checkpoint(gcs_bucket):
    print(f"Loading checkpoint: {PARAMS_FILE}")
    with gcs_bucket.blob(f"{DIR_PREFIX}params/{PARAMS_FILE}").open("rb") as f:
        ckpt = checkpoint.load(f, graphcast.CheckPoint)
    print(f"Model description: {ckpt.description}")
    return ckpt


def load_dataset(gcs_bucket):
    print(f"Loading dataset: {DATASET_FILE}")
    with gcs_bucket.blob(f"{DIR_PREFIX}dataset/{DATASET_FILE}").open("rb") as f:
        example_batch = xarray.load_dataset(f, decode_timedelta=True).compute()
    print(f"Dataset dimensions: {dict(example_batch.sizes)}")
    return example_batch


def load_normalization_stats(gcs_bucket):
    print("Loading normalization statistics...")
    with gcs_bucket.blob(f"{DIR_PREFIX}stats/diffs_stddev_by_level.nc").open("rb") as f:
        diffs_stddev_by_level = xarray.load_dataset(f).compute()
    with gcs_bucket.blob(f"{DIR_PREFIX}stats/mean_by_level.nc").open("rb") as f:
        mean_by_level = xarray.load_dataset(f).compute()
    with gcs_bucket.blob(f"{DIR_PREFIX}stats/stddev_by_level.nc").open("rb") as f:
        stddev_by_level = xarray.load_dataset(f).compute()
    return diffs_stddev_by_level, mean_by_level, stddev_by_level


def construct_wrapped_graphcast(
    model_config, task_config,
    diffs_stddev_by_level, mean_by_level, stddev_by_level,
):
    predictor = graphcast.GraphCast(model_config, task_config)
    predictor = casting.Bfloat16Cast(predictor)
    predictor = normalization.InputsAndResiduals(
        predictor,
        diffs_stddev_by_level=diffs_stddev_by_level,
        mean_by_level=mean_by_level,
        stddev_by_level=stddev_by_level,
    )
    predictor = autoregressive.Predictor(predictor, gradient_checkpointing=True)
    return predictor


def main():
    print("=" * 60)
    print("GraphCast 10-Day Weather Forecast")
    print("=" * 60)
    print()

    # Connect to GCS
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
          f"mesh_size={model_config.mesh_size}")
    print()

    # Load 40-step dataset
    example_batch = load_dataset(gcs_bucket)

    # Load normalization stats
    diffs_stddev_by_level, mean_by_level, stddev_by_level = load_normalization_stats(gcs_bucket)

    # Extract inputs/targets/forcings for all prediction steps
    # 40 total steps - 2 input steps = 38 prediction steps (9.5 days)
    eval_steps = example_batch.sizes["time"] - 2
    print(f"\nPreparing {eval_steps}-step prediction ({eval_steps * 6} hours = {eval_steps * 6 / 24:.1f} days)...")

    eval_inputs, eval_targets, eval_forcings = data_utils.extract_inputs_targets_forcings(
        example_batch,
        target_lead_times=slice("6h", f"{eval_steps * 6}h"),
        **dataclasses.asdict(task_config),
    )

    print(f"Inputs:   {dict(eval_inputs.sizes)}")
    print(f"Targets:  {dict(eval_targets.sizes)}")
    print(f"Forcings: {dict(eval_forcings.sizes)}")

    # Build predictor
    @hk.transform_with_state
    def run_forward(model_config, task_config, inputs, targets_template, forcings):
        predictor = construct_wrapped_graphcast(
            model_config, task_config,
            diffs_stddev_by_level, mean_by_level, stddev_by_level,
        )
        return predictor(inputs, targets_template=targets_template, forcings=forcings)

    def with_configs(fn):
        return functools.partial(fn, model_config=model_config, task_config=task_config)

    def with_params(fn):
        return functools.partial(fn, params=params, state=state)

    def drop_state(fn):
        return lambda **kw: fn(**kw)[0]

    run_forward_jitted = drop_state(
        with_params(jax.jit(with_configs(run_forward.apply)))
    )

    # Run prediction
    print("\n" + "=" * 60)
    print("Running 10-day autoregressive prediction...")
    print("(This will take a while - JIT compilation + 38 steps)")
    print("=" * 60)

    predictions = rollout.chunked_prediction(
        run_forward_jitted,
        rng=jax.random.PRNGKey(0),
        inputs=eval_inputs,
        targets_template=eval_targets * np.nan,
        forcings=eval_forcings,
    )

    print(f"\nPredictions: {dict(predictions.sizes)}")
    print(f"Variables: {list(predictions.data_vars.keys())}")

    # Save
    output_file = "predictions_10day_real.nc"
    print(f"\nSaving to {output_file}...")
    predictions.to_netcdf(output_file)
    print(f"Saved!")

    # Summary stats
    print("\n" + "=" * 60)
    print("Prediction Summary")
    print("=" * 60)
    for var in ["2m_temperature", "total_precipitation_6hr", "mean_sea_level_pressure"]:
        if var in predictions.data_vars:
            data = predictions[var].values
            print(f"{var}: min={np.nanmin(data):.2f}, max={np.nanmax(data):.2f}, mean={np.nanmean(data):.2f}")

    print("\nDone! Now run: python drought_detector.py predictions_10day_real.nc")


if __name__ == "__main__":
    main()
