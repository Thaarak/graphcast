# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GraphCast is Google DeepMind's weather forecasting system implementing two AI models:
- **GraphCast**: Deterministic medium-range weather forecasting using graph neural networks
- **GenCast**: Diffusion-based ensemble forecasting for probabilistic predictions

Both models operate on global weather data at various resolutions (0.25° and 1.0°) using an icosahedral mesh representation.

## Build & Installation

```bash
pip install -e .
```

Dependencies are defined in `setup.py`. Key requirements: JAX, Haiku, Jraph, XArray, NumPy, SciPy.

## Running Tests

Tests use `absl.testing.absltest`. Run individual tests:
```bash
python graphcast/checkpoint_test.py
python graphcast/xarray_jax_test.py
python graphcast/data_utils_test.py
```

## Architecture

### Core Prediction Flow
1. **Input Processing** (`data_utils.py`, `model_utils.py`): Convert xarray weather data to flat node/edge features on the mesh
2. **Graph Neural Network** (`deep_typed_graph_net.py`, `typed_graph_net.py`): Process on `TypedGraph` structures with heterogeneous node/edge types
3. **Autoregressive Wrapper** (`autoregressive.py`): Chain single-step predictions for multi-step forecasts (differentiable)
4. **Rollout** (`rollout.py`): Non-differentiable inference loop for longer trajectories

### Mesh System
- `icosahedral_mesh.py`: Defines multi-resolution icosahedral mesh hierarchy
- `grid_mesh_connectivity.py`: Bidirectional mapping between lat/lon grids and triangular mesh
- `typed_graph.py`: Core `TypedGraph` dataclass for heterogeneous graphs

### Model-Specific Components
**GraphCast** (`graphcast.py`):
- Encoder-processor-decoder architecture
- Single deterministic prediction per step
- `casting.py`: BFloat16 precision wrapper

**GenCast** (`gencast.py`, `denoiser.py`):
- Diffusion model with denoising network
- `sparse_transformer.py`: Attention mechanism on mesh
- `dpm_solver_plus_plus_2s.py`: DPM-Solver++ sampler for inference
- `nan_cleaning.py`: Handles NaN values in sea surface temperature

### JAX/XArray Integration
- `xarray_jax.py`: Makes xarray compatible with JAX transformations (jit, grad, vmap)
- `xarray_tree.py`: Tree operations for xarray structures
- `normalization.py`: Statistical normalization using historical data

### Supporting Modules
- `losses.py`: Loss functions with latitude weighting
- `solar_radiation.py`: TOA solar radiation forcing computation
- `checkpoint.py`: Serialization for model weights and configs

## Data Sources

- Training data: ERA5 reanalysis from ECMWF (via WeatherBench2 Zarr)
- Operational fine-tuning: HRES-fc0 data
- Pretrained weights and stats: `gs://dm_graphcast/` Google Cloud Bucket

## Demo Notebooks

- `graphcast_demo.ipynb`: GraphCast inference example
- `gencast_mini_demo.ipynb`: GenCast with small model (runnable in free Colab)
- `gencast_demo_cloud_vm.ipynb`: GenCast on Cloud TPU/GPU
