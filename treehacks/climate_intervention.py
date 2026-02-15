"""
Climate Intervention Pipeline for GraphCast Weather Forecasting

This module provides tools for:
1. Weather anomaly detection (hurricanes, droughts)
2. Intervention optimization (cloud seeding strategies)
3. Running control vs. seeded forecasts
4. Visualization and analysis of intervention impact

Developed for Treehacks 2026 - Morro Project
"""

import os
import json
import shutil
import numpy as np
import xarray as xr
from typing import NamedTuple, Optional, List, Dict, Any, Callable
from datetime import datetime, timedelta

# JAX imports
import jax
import jax.numpy as jnp
import optax

# Earth2Studio imports
from earth2studio.models.px import GraphCastOperational
from earth2studio.data import GFS, ARCO, DataSource
from earth2studio.io import ZarrBackend
from earth2studio.run import deterministic as run_deterministic


# =============================================================================
# SECTION 1: WEATHER SENTINEL (Anomaly Detector)
# =============================================================================

class WeatherSentinel:
    """
    Detects weather anomalies (hurricanes, droughts) from forecast data.

    Scans atmospheric fields for cyclogenesis signatures and drought conditions
    using physics-based thresholds.
    """

    def __init__(self):
        # Physics Thresholds
        self.HURRICANE_WIND_THRESHOLD = 33.0    # m/s (Category 1 hurricane)
        self.LOW_PRESSURE_THRESHOLD = 98000.0   # Pa (980 hPa)
        self.DROUGHT_HUMIDITY_THRESHOLD = 0.005 # kg/kg (Very dry specific humidity)

    def detect_hurricane_genesis(
        self,
        u_wind: np.ndarray,
        v_wind: np.ndarray,
        msl: np.ndarray,
        lat_coords: np.ndarray,
        lon_coords: np.ndarray
    ) -> tuple[bool, List[Dict[str, Any]]]:
        """
        Scans forecast arrays for cyclogenesis signatures and ranks them.

        Args:
            u_wind: 2D array of u-wind component (Lat, Lon)
            v_wind: 2D array of v-wind component (Lat, Lon)
            msl: 2D array of mean sea level pressure (Lat, Lon)
            lat_coords: 1D array of latitude coordinates
            lon_coords: 1D array of longitude coordinates

        Returns:
            Tuple of (detected: bool, formations: list of ranked storm candidates)
        """
        # Calculate Wind Speed Magnitude
        wind_speed = np.sqrt(u_wind**2 + v_wind**2)

        # Create Boolean Masks
        high_wind_mask = wind_speed > self.HURRICANE_WIND_THRESHOLD
        low_pressure_mask = msl < self.LOW_PRESSURE_THRESHOLD

        # Intersection: High Wind AND Low Pressure
        storm_candidates_mask = np.logical_and(high_wind_mask, low_pressure_mask)

        formations = []
        if np.any(storm_candidates_mask):
            rows, cols = np.where(storm_candidates_mask)

            for r, c in zip(rows, cols):
                w_val = float(wind_speed[r, c])
                p_val = float(msl[r, c])

                # Heuristic score weighted by wind excess and pressure deficit
                wind_excess = (w_val - self.HURRICANE_WIND_THRESHOLD) / self.HURRICANE_WIND_THRESHOLD
                pressure_deficit = (self.LOW_PRESSURE_THRESHOLD - p_val) / self.LOW_PRESSURE_THRESHOLD
                score = (wind_excess * 0.7) + (pressure_deficit * 0.3)

                formations.append({
                    "latitude": float(lat_coords[r]),
                    "longitude": float(lon_coords[c]),
                    "wind_speed_ms": w_val,
                    "pressure_pa": p_val,
                    "score": score
                })

            # Sort by score descending (most severe first)
            formations.sort(key=lambda x: x["score"], reverse=True)

        return len(formations) > 0, formations

    def detect_drought_conditions(
        self,
        specific_humidity: np.ndarray,
        lat_coords: np.ndarray,
        lon_coords: np.ndarray
    ) -> tuple[bool, List[Dict[str, Any]]]:
        """
        Scans for drought conditions based on specific humidity.

        Args:
            specific_humidity: 2D array of specific humidity at 700hPa (Lat, Lon)
            lat_coords: 1D array of latitude coordinates
            lon_coords: 1D array of longitude coordinates

        Returns:
            Tuple of (detected: bool, drought_regions: list of affected areas)
        """
        drought_mask = specific_humidity < self.DROUGHT_HUMIDITY_THRESHOLD

        drought_regions = []
        if np.any(drought_mask):
            rows, cols = np.where(drought_mask)

            # Group into regions (simplified - just count cells)
            drought_regions.append({
                "cells_affected": len(rows),
                "center_lat": float(np.mean(lat_coords[rows])),
                "center_lon": float(np.mean(lon_coords[cols])),
                "min_humidity": float(np.min(specific_humidity[drought_mask])),
                "mean_humidity": float(np.mean(specific_humidity[drought_mask]))
            })

        return len(drought_regions) > 0, drought_regions


# =============================================================================
# SECTION 2: INTERVENTION ARCHITECT (Optimizer)
# =============================================================================

class CloudSeedingConfig(NamedTuple):
    """Configuration for cloud seeding intervention."""
    target_lat: float
    target_lon: float
    radius_km: float = 200.0
    seeding_intensity: float = 1e-4  # kg/kg (moisture addition)
    pressure_levels: list = [1, 2, 3]  # Indices for ~850hPa - 500hPa


class InterventionArchitect:
    """
    Optimizes cloud seeding intervention strategies using differentiable physics.

    Uses JAX for automatic differentiation to find optimal moisture injection
    patterns that minimize storm intensity.
    """

    def __init__(self, forward_model_fn: Callable):
        """
        Args:
            forward_model_fn: A JAX-jitted function that takes (inputs) -> (forecast).
                              This is your GraphCast model's forward pass.
        """
        self.model = forward_model_fn
        self.optimizer = optax.adam(learning_rate=0.01)

    def _create_spatial_mask(
        self,
        lat_grid: jnp.ndarray,
        lon_grid: jnp.ndarray,
        config: CloudSeedingConfig
    ) -> jnp.ndarray:
        """Creates a binary mask restricting seeding to the target region."""
        dist = jnp.sqrt((lat_grid - config.target_lat)**2 + (lon_grid - config.target_lon)**2)
        radius_deg = config.radius_km / 111.0  # ~111km per degree
        mask = jnp.where(dist < radius_deg, 1.0, 0.0)
        return mask

    def loss_function(
        self,
        perturbation: jnp.ndarray,
        initial_state: jnp.ndarray,
        targets: Dict[str, Any]
    ) -> jnp.ndarray:
        """
        Objective function: minimize maximum wind speed in forecast.

        Args:
            perturbation: The seeding pattern to optimize
            initial_state: Current atmospheric state
            targets: Dictionary with target location info

        Returns:
            Loss value (max wind speed)
        """
        # Apply perturbation (the "seeding")
        seeded_state = initial_state + perturbation

        # Run forecast (differentiable physics)
        forecast = self.model(seeded_state)

        # Calculate loss (max wind speed)
        # Assuming channel 0 is u-wind and 1 is v-wind
        u = forecast[:, 0, :, :]
        v = forecast[:, 1, :, :]
        wind_speed = jnp.sqrt(u**2 + v**2)

        # Minimize maximum wind speed
        max_wind = jnp.max(wind_speed)

        # L2 regularization on perturbation magnitude
        reg_term = 0.01 * jnp.sum(perturbation**2)

        return max_wind + reg_term

    def generate_intervention(
        self,
        initial_state: jnp.ndarray,
        event_json_path: str,
        lat_grid: jnp.ndarray,
        lon_grid: jnp.ndarray,
        num_steps: int = 50
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Generates optimized intervention pattern.

        Args:
            initial_state: Current atmospheric state (Batch, Channels, Lat, Lon)
            event_json_path: Path to JSON file with event configuration
            lat_grid: 2D latitude grid
            lon_grid: 2D longitude grid
            num_steps: Number of optimization steps

        Returns:
            Tuple of (modified_state, seeding_mask)
        """
        # Load event config
        with open(event_json_path, 'r') as f:
            event = json.load(f)

        config = CloudSeedingConfig(
            target_lat=event['latitude'],
            target_lon=event['longitude'],
            radius_km=200.0,
            seeding_intensity=1e-4
        )

        # Create spatial constraint mask
        spatial_mask = self._create_spatial_mask(lat_grid, lon_grid, config)

        # Initialize perturbation (seeding pattern)
        perturbation = jnp.zeros_like(initial_state)
        # Only perturb humidity channel (index 2)
        perturbation = perturbation.at[:, 2, :, :].set(
            spatial_mask * config.seeding_intensity
        )

        # Initialize optimizer state
        opt_state = self.optimizer.init(perturbation)

        # Define gradient function
        grad_fn = jax.grad(self.loss_function)

        # Optimization loop
        print(f"Optimizing intervention over {num_steps} steps...")
        for step in range(num_steps):
            grads = grad_fn(perturbation, initial_state, {'config': config})

            # Apply spatial constraint
            grads = grads.at[:, 2, :, :].set(grads[:, 2, :, :] * spatial_mask)

            updates, opt_state = self.optimizer.update(grads, opt_state, perturbation)
            perturbation = optax.apply_updates(perturbation, updates)

            if step % 10 == 0:
                loss = self.loss_function(perturbation, initial_state, {'config': config})
                print(f"  Step {step}: Loss = {loss:.4f}")

        # Apply final perturbation
        modified_state = initial_state + perturbation
        seeding_mask = perturbation

        return modified_state, seeding_mask


# =============================================================================
# SECTION 3: DATA SOURCE WRAPPERS
# =============================================================================

class SeededDataSource(DataSource):
    """
    Wraps a data source to inject cloud seeding intervention.

    Intercepts data fetches and adds moisture to specific humidity channels
    in a circular region around the target location.
    """

    def __init__(
        self,
        base_source: DataSource,
        event_config: Dict[str, Any],
        seeding_intensity: float = 2e-4,
        radius_deg: float = 5.0
    ):
        """
        Args:
            base_source: The underlying data source (e.g., GFS, ARCO)
            event_config: Dictionary with 'latitude' and 'longitude' keys
            seeding_intensity: Moisture injection intensity (kg/kg)
            radius_deg: Radius of seeding area in degrees
        """
        self.base_source = base_source
        self.event_config = event_config
        self.seeding_intensity = seeding_intensity
        self.radius_deg = radius_deg
        self.seeding_mask_values = None

    def __call__(self, time, variable):
        print(f"\n[SeededDataSource] Fetching base data for {time}...")
        base_data = self.base_source(time, variable)

        # Check for empty data
        if base_data.sizes.get('time', 0) == 0:
            print("Warning: Received empty data from base source!")
            return base_data

        # Initialize mask on first call
        if self.seeding_mask_values is None:
            print("[SeededDataSource] Generating seeding mask...")
            mask = xr.zeros_like(base_data)

            # Extract coordinates
            lats = base_data.coords['lat'].values
            lons = base_data.coords['lon'].values
            lat_grid, lon_grid = np.meshgrid(lats, lons, indexing='ij')

            # Target location
            target_lat = self.event_config['latitude']
            target_lon = self.event_config['longitude']

            # Create circular spatial mask
            dist = np.sqrt((lat_grid - target_lat)**2 + (lon_grid - target_lon)**2)
            spatial_mask = np.where(dist < self.radius_deg, 1.0, 0.0)

            # Apply to humidity channels
            for i, var_name in enumerate(variable):
                if var_name.startswith('q') and ('700' in var_name or '850' in var_name):
                    print(f"   -> Injecting moisture into channel: {var_name}")
                    if mask.ndim == 4:  # (Time, Var, Lat, Lon)
                        mask.values[:, i, :, :] = spatial_mask * self.seeding_intensity
                    elif mask.ndim == 3:  # (Var, Lat, Lon)
                        mask.values[i, :, :] = spatial_mask * self.seeding_intensity

            self.seeding_mask_values = mask.values.copy()

        # Apply intervention
        seeded_data = base_data.copy(deep=True)
        seeded_data.values = base_data.values + self.seeding_mask_values

        return seeded_data


class CycloneDataSource(DataSource):
    """
    Wraps a data source to inject a synthetic cyclone for testing.

    Used to create control runs with a known storm for comparison.
    """

    def __init__(self, base_source: DataSource, event_config: Dict[str, Any]):
        """
        Args:
            base_source: The underlying data source
            event_config: Dictionary with 'latitude' and 'longitude' keys
        """
        self.base_source = base_source
        self.event_config = event_config

    def __call__(self, time, variable):
        print(f"\n[CycloneDataSource] Fetching base data...")
        base_data = self.base_source(time, variable)

        # Check for empty data
        if base_data.sizes.get('time', 0) == 0:
            return base_data

        # Inject synthetic cyclone
        vals = base_data.values
        lats = base_data.coords['lat'].values
        lons = base_data.coords['lon'].values

        target_lat = self.event_config['latitude']
        target_lon = self.event_config['longitude']

        # Find indices for anomaly center
        lat_idx = np.abs(lats - target_lat).argmin()
        lon_idx = np.abs(lons - target_lon).argmin()

        # Define cyclone radius (~5 degrees / 20 grid points)
        radius = 20
        y, x = np.ogrid[-radius:radius, -radius:radius]
        mask = x**2 + y**2 <= radius**2

        # Safe slicing indices
        sl_lat = slice(lat_idx - radius, lat_idx + radius)
        sl_lon = slice(lon_idx - radius, lon_idx + radius)

        print(f"[CycloneDataSource] Injecting synthetic cyclone at {target_lat}N, {target_lon}E...")

        for i, var_name in enumerate(variable):
            # High winds (hurricane force)
            if var_name in ['u10m', 'v10m']:
                wind_perturbation = np.random.uniform(-40, 40, mask.shape) * mask
                if vals.ndim == 4:  # (Time, Var, Lat, Lon)
                    vals[0, i, sl_lat, sl_lon] += wind_perturbation
                elif vals.ndim == 3:  # (Var, Lat, Lon)
                    vals[i, sl_lat, sl_lon] += wind_perturbation

            # Low pressure (hurricane signature)
            elif var_name == 'msl':
                pressure_drop = -5000 * (1 - np.sqrt(x**2 + y**2) / radius) * mask
                if vals.ndim == 4:
                    vals[0, i, sl_lat, sl_lon] += pressure_drop
                elif vals.ndim == 3:
                    vals[i, sl_lat, sl_lon] += pressure_drop

        return base_data


class AmplifiedSeededDataSource(SeededDataSource):
    """
    Seeded data source with amplified intervention parameters.

    Uses larger radius (10 deg) and higher intensity (1e-3 kg/kg) for
    studying maximum intervention effects.
    """

    def __init__(self, base_source: DataSource, event_config: Dict[str, Any]):
        super().__init__(
            base_source=base_source,
            event_config=event_config,
            seeding_intensity=1e-3,  # 5x normal
            radius_deg=10.0          # 2x normal
        )


# =============================================================================
# SECTION 4: FORECAST PIPELINE
# =============================================================================

class ForecastPipeline:
    """
    Orchestrates running control and intervention forecasts.
    """

    def __init__(self, output_dir: str = "outputs"):
        """
        Args:
            output_dir: Directory for saving forecast outputs
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.model = None
        self.sentinel = WeatherSentinel()

    def load_model(self):
        """Load the GraphCast model."""
        print("Loading GraphCast model...")
        package = GraphCastOperational.load_default_package()
        self.model = GraphCastOperational.load_model(package)
        print("Model loaded successfully.")
        return self.model

    def run_control_forecast(
        self,
        start_time: datetime,
        nsteps: int = 40,
        data_source: Optional[DataSource] = None
    ) -> str:
        """
        Run a control forecast (no intervention).

        Args:
            start_time: Forecast start time
            nsteps: Number of 6-hour forecast steps (40 = 10 days)
            data_source: Data source to use (default: ARCO for historical)

        Returns:
            Path to output Zarr store
        """
        if self.model is None:
            self.load_model()

        if data_source is None:
            data_source = ARCO()

        output_path = os.path.join(self.output_dir, "control_forecast.zarr")
        if os.path.exists(output_path):
            shutil.rmtree(output_path)

        io = ZarrBackend(output_path)
        time_obj = [start_time]

        print(f"Running control forecast from {start_time} for {nsteps} steps...")
        run_deterministic(time_obj, nsteps=nsteps, prognostic=self.model, data=data_source, io=io)
        print(f"Control forecast saved to {output_path}")

        return output_path

    def run_seeded_forecast(
        self,
        start_time: datetime,
        event_config: Dict[str, Any],
        nsteps: int = 40,
        amplified: bool = False
    ) -> str:
        """
        Run a seeded (intervention) forecast.

        Args:
            start_time: Forecast start time
            event_config: Event configuration with 'latitude' and 'longitude'
            nsteps: Number of 6-hour forecast steps
            amplified: Use amplified intervention parameters

        Returns:
            Path to output Zarr store
        """
        if self.model is None:
            self.load_model()

        base_source = ARCO()

        if amplified:
            seeded_source = AmplifiedSeededDataSource(base_source, event_config)
            output_name = "seeded_forecast_amplified.zarr"
        else:
            seeded_source = SeededDataSource(base_source, event_config)
            output_name = "seeded_forecast.zarr"

        output_path = os.path.join(self.output_dir, output_name)
        if os.path.exists(output_path):
            shutil.rmtree(output_path)

        io = ZarrBackend(output_path)
        time_obj = [start_time]

        print(f"Running seeded forecast from {start_time} for {nsteps} steps...")
        run_deterministic(time_obj, nsteps=nsteps, prognostic=self.model, data=seeded_source, io=io)
        print(f"Seeded forecast saved to {output_path}")

        return output_path

    def detect_storm_center(
        self,
        data_path: str,
        search_bounds: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Detect storm center from forecast data using minimum pressure.

        Args:
            data_path: Path to Zarr forecast store
            search_bounds: Dict with lat_min, lat_max, lon_min, lon_max

        Returns:
            Event configuration dictionary
        """
        store = ZarrBackend(data_path)

        lat = store.root["lat"][:]
        lon = store.root["lon"][:]
        msl = store.root["msl"][0, 0, :, :]  # First timestep

        # Define search region
        lat_mask = (lat >= search_bounds['lat_min']) & (lat <= search_bounds['lat_max'])
        lon_mask = (lon >= search_bounds['lon_min']) & (lon <= search_bounds['lon_max'])

        lat_indices = np.where(lat_mask)[0]
        lon_indices = np.where(lon_mask)[0]

        # Subset and find minimum
        msl_subset = msl[np.ix_(lat_indices, lon_indices)]
        min_idx = np.unravel_index(np.argmin(msl_subset), msl_subset.shape)

        center_lat = float(lat[lat_indices[min_idx[0]]])
        center_lon = float(lon[lon_indices[min_idx[1]]])
        min_pressure = float(msl[lat_indices[min_idx[0]], lon_indices[min_idx[1]]])

        event_config = {
            "type": "Detected Storm",
            "latitude": center_lat,
            "longitude": center_lon,
            "min_pressure_pa": min_pressure,
            "timestamp": datetime.utcnow().isoformat()
        }

        print(f"Detected storm center: {center_lat:.2f}N, {center_lon:.2f}E")
        print(f"Minimum pressure: {min_pressure:.2f} Pa")

        return event_config


# =============================================================================
# SECTION 5: ANALYSIS UTILITIES
# =============================================================================

def compute_wind_reduction(
    control_path: str,
    seeded_path: str
) -> Dict[str, Any]:
    """
    Compute wind speed reduction between control and seeded forecasts.

    Args:
        control_path: Path to control forecast Zarr store
        seeded_path: Path to seeded forecast Zarr store

    Returns:
        Dictionary with reduction statistics
    """
    store_ctrl = ZarrBackend(control_path)
    store_seed = ZarrBackend(seeded_path)

    # Extract wind components
    u_ctrl = store_ctrl.root["u10m"][0]
    v_ctrl = store_ctrl.root["v10m"][0]
    u_seed = store_seed.root["u10m"][0]
    v_seed = store_seed.root["v10m"][0]

    # Calculate wind speed
    wind_ctrl = np.sqrt(u_ctrl**2 + v_ctrl**2)
    wind_seed = np.sqrt(u_seed**2 + v_seed**2)

    # Compute max wind per timestep
    max_wind_ctrl = np.max(wind_ctrl, axis=(1, 2))
    max_wind_seed = np.max(wind_seed, axis=(1, 2))

    # Calculate reduction
    reduction = max_wind_ctrl - max_wind_seed
    peak_reduction = np.max(reduction)
    peak_time_idx = np.argmax(reduction)

    # Average reduction
    avg_reduction = np.mean(reduction[reduction > 0])

    results = {
        "peak_reduction_ms": float(peak_reduction),
        "peak_reduction_hour": int(peak_time_idx * 6),
        "average_reduction_ms": float(avg_reduction),
        "max_wind_ctrl_timeseries": max_wind_ctrl.tolist(),
        "max_wind_seed_timeseries": max_wind_seed.tolist(),
        "reduction_timeseries": reduction.tolist()
    }

    print(f"Peak wind speed reduction: {peak_reduction:.2f} m/s at T+{peak_time_idx * 6} hours")
    print(f"Average reduction: {avg_reduction:.2f} m/s")

    return results


def extract_storm_track(
    data_path: str,
    search_bounds: Dict[str, float]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract storm track from forecast by following minimum pressure.

    Args:
        data_path: Path to Zarr forecast store
        search_bounds: Dict with lat_min, lat_max, lon_min, lon_max

    Returns:
        Tuple of (latitudes, longitudes) arrays for storm track
    """
    store = ZarrBackend(data_path)

    lat = store.root["lat"][:]
    lon = store.root["lon"][:]
    msl = store.root["msl"][0]  # (Time, Lat, Lon)

    lat_mask = (lat >= search_bounds['lat_min']) & (lat <= search_bounds['lat_max'])
    lon_mask = (lon >= search_bounds['lon_min']) & (lon <= search_bounds['lon_max'])

    lat_indices = np.where(lat_mask)[0]
    lon_indices = np.where(lon_mask)[0]

    track_lats = []
    track_lons = []

    for t in range(msl.shape[0]):
        msl_sub = msl[t][np.ix_(lat_mask, lon_mask)]
        local_min_idx = np.unravel_index(np.argmin(msl_sub), msl_sub.shape)

        track_lats.append(lat[lat_indices[local_min_idx[0]]])
        track_lons.append(lon[lon_indices[local_min_idx[1]]])

    return np.array(track_lats), np.array(track_lons)


# =============================================================================
# SECTION 6: MAIN ENTRY POINT
# =============================================================================

def run_hurricane_intervention_study(
    event_name: str = "Hurricane Katrina",
    start_time: datetime = datetime(2005, 8, 27, 12, 0),
    nsteps: int = 40,
    output_dir: str = "outputs"
) -> Dict[str, Any]:
    """
    Run a complete hurricane intervention study.

    Args:
        event_name: Name of the event for logging
        start_time: Forecast start time
        nsteps: Number of 6-hour steps (40 = 10 days)
        output_dir: Output directory

    Returns:
        Dictionary with paths and analysis results
    """
    print(f"=" * 60)
    print(f"HURRICANE INTERVENTION STUDY: {event_name}")
    print(f"Start Time: {start_time}")
    print(f"Forecast Length: {nsteps * 6} hours ({nsteps * 6 / 24:.1f} days)")
    print(f"=" * 60)

    pipeline = ForecastPipeline(output_dir=output_dir)

    # Step 1: Run control forecast
    print("\n[Step 1/4] Running control forecast...")
    control_path = pipeline.run_control_forecast(start_time, nsteps=nsteps)

    # Step 2: Detect storm center
    print("\n[Step 2/4] Detecting storm center...")
    search_bounds = {
        'lat_min': 15, 'lat_max': 45,
        'lon_min': 260, 'lon_max': 300
    }
    event_config = pipeline.detect_storm_center(control_path, search_bounds)
    event_config['type'] = event_name

    # Save event config
    config_path = os.path.join(output_dir, "event_config.json")
    with open(config_path, 'w') as f:
        json.dump(event_config, f, indent=2)

    # Step 3: Run seeded forecast
    print("\n[Step 3/4] Running seeded forecast...")
    seeded_path = pipeline.run_seeded_forecast(start_time, event_config, nsteps=nsteps)

    # Step 4: Analyze results
    print("\n[Step 4/4] Analyzing intervention impact...")
    reduction_stats = compute_wind_reduction(control_path, seeded_path)

    # Extract storm tracks
    ctrl_track = extract_storm_track(control_path, search_bounds)
    seed_track = extract_storm_track(seeded_path, search_bounds)

    results = {
        "event_config": event_config,
        "control_path": control_path,
        "seeded_path": seeded_path,
        "reduction_stats": reduction_stats,
        "control_track": {"lat": ctrl_track[0].tolist(), "lon": ctrl_track[1].tolist()},
        "seeded_track": {"lat": seed_track[0].tolist(), "lon": seed_track[1].tolist()}
    }

    # Save results
    results_path = os.path.join(output_dir, "intervention_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print("STUDY COMPLETE")
    print(f"Results saved to: {results_path}")
    print(f"{'=' * 60}")

    return results


if __name__ == "__main__":
    # Example: Run Hurricane Katrina intervention study
    results = run_hurricane_intervention_study(
        event_name="Hurricane Katrina",
        start_time=datetime(2005, 8, 27, 12, 0),
        nsteps=40,  # 10 days
        output_dir="outputs"
    )
