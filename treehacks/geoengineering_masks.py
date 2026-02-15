#!/usr/bin/env python3
"""
Geoengineering Seeding Masks for GraphCast

Implements physically-grounded perturbation masks that modify GraphCast's
initial state vector to simulate geoengineering interventions.

Based on: "Computational Geoengineering: Microphysical Mechanisms and State
Vector Parameterization for AI-Driven Weather Modification"

Supported intervention types:
  Drought Prevention:
    - Hygroscopic Enhancement (giant particles, warm cloud coalescence)
    - Electric/Ionization (electrostatic coalescence)
    - Laser-Induced Condensation (photochemical nucleation)
    - Glaciogenic Static (ice nucleation in cold clouds)

  Hurricane/Storm Disruption:
    - Glaciogenic Dynamic (latent heat release, circulation modification)
    - Hygroscopic Suppression (sub-micrometer, Twomey effect)

The mask M is an additive perturbation: X_seeded = X_raw + M
where X is the ERA5 state vector ingested by GraphCast.

Usage:
    python geoengineering_masks.py dataset.nc --intervention hygroscopic_enhancement \\
        --lat-min 30 --lat-max 35 --lon-min 250 --lon-max 260 --intensity 0.5 \\
        --output seeded_dataset.nc
"""

import argparse
import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr


# ---------------------------------------------------------------------------
# Physical Constants
# ---------------------------------------------------------------------------
L_FUSION = 3.34e5          # Latent heat of fusion (J/kg)
L_VAPORIZATION = 2.26e6    # Latent heat of vaporization (J/kg)
C_P = 1004.0               # Specific heat capacity of dry air (J/kg/K)
R_D = 287.05               # Gas constant for dry air (J/kg/K)
G = 9.80665                # Gravitational acceleration (m/s^2)
T_FREEZE = 273.15          # Freezing point of water (K)

# Glaciogenic activation temperature range
T_ACT_MIN = 258.15         # -15C in Kelvin
T_ACT_MAX = 268.15         # -5C in Kelvin

# Pressure level ranges (hPa) for different interventions
CLOUD_BASE_LEVELS = (925, 850)         # 925-850 hPa
MID_TROPOSPHERE = (700, 400)           # 700-400 hPa
UPPER_TROPOSPHERE = (300, 150)         # 300-150 hPa
STORM_CORE_LEVELS = (500, 300)         # 500-300 hPa
CIRRUS_LEVELS = (200, 100)             # 200-100 hPa

# GraphCast small model pressure levels (13 levels, hPa)
GRAPHCAST_SMALL_LEVELS = [
    50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000
]

# Variable name aliases: maps physical concepts to possible ERA5/GraphCast names.
# GraphCast_small uses these exact names from ERA5:
#   temperature, specific_humidity, geopotential, vertical_velocity,
#   u_component_of_wind, v_component_of_wind, total_precipitation_6hr,
#   2m_temperature, mean_sea_level_pressure, 10m_u/v_component_of_wind
# Note: clwc/ciwc (cloud water) are NOT in GraphCast_small. When targeting
# cloud water, we proxy through specific_humidity and temperature perturbations.
VAR_ALIASES = {
    "temperature":    ["temperature", "t"],
    "humidity":       ["specific_humidity", "q"],
    "geopotential":   ["geopotential", "z"],
    "vertical_vel":   ["vertical_velocity", "w"],
    "u_wind":         ["u_component_of_wind", "u"],
    "v_wind":         ["v_component_of_wind", "v"],
    "precip":         ["total_precipitation_6hr"],
    "mslp":           ["mean_sea_level_pressure", "msl"],
    "t2m":            ["2m_temperature"],
    "u10":            ["10m_u_component_of_wind"],
    "v10":            ["10m_v_component_of_wind"],
    "clwc":           ["specific_cloud_liquid_water_content", "clwc"],
    "ciwc":           ["specific_cloud_ice_water_content", "ciwc"],
}


def resolve_var(ds: xr.Dataset, alias: str) -> Optional[str]:
    """Resolve a variable alias to the actual name present in the dataset."""
    candidates = VAR_ALIASES.get(alias, [alias])
    for name in candidates:
        if name in ds.data_vars:
            return name
    return None


# ---------------------------------------------------------------------------
# Data Classes for Configuration
# ---------------------------------------------------------------------------
@dataclass
class TargetRegion:
    """Geographic bounding box for the seeding intervention."""
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    def contains(self, lat: np.ndarray, lon: np.ndarray):
        """Return boolean mask of grid cells within the target region."""
        lat_mask = (lat >= self.lat_min) & (lat <= self.lat_max)
        lon_mask = (lon >= self.lon_min) & (lon <= self.lon_max)
        # Handle 2D vs 1D coordinates
        if lat_mask.ndim == 1 and lon_mask.ndim == 1:
            return np.outer(lat_mask, lon_mask)
        return lat_mask & lon_mask


@dataclass
class PlumeConfig:
    """Gaussian plume dispersion parameters."""
    source_lat: float = 0.0
    source_lon: float = 0.0
    sigma_y_km: float = 50.0      # Cross-wind dispersion (km)
    sigma_z_hpa: float = 50.0     # Vertical dispersion (hPa)
    release_altitude_hpa: float = 700.0  # Seeding altitude
    pasquill_class: str = "D"     # Stability class (A-F, D = neutral)


@dataclass
class SeedingConfig:
    """Configuration for a seeding intervention."""
    intervention_type: str         # e.g., "glaciogenic_static"
    target_region: TargetRegion
    intensity: float = 0.5         # 0.0 to 1.0 scale factor
    plume: Optional[PlumeConfig] = None
    lag_steps: int = 1             # Temporal lag in 6-hour time steps
    ramp_steps: int = 2            # Steps over which to ramp up the mask


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def gaussian_plume_2d(
    lats: np.ndarray,
    lons: np.ndarray,
    center_lat: float,
    center_lon: float,
    sigma_y_km: float = 50.0,
) -> np.ndarray:
    """
    Compute a 2D Gaussian plume field (horizontal).

    Uses great-circle approximation for distance. Returns a normalized
    weight field (0-1) representing the effective seeding fraction F_seed
    at each grid cell.

    Args:
        lats: 1D array of latitudes
        lons: 1D array of longitudes
        center_lat: Plume source latitude
        center_lon: Plume source longitude
        sigma_y_km: Horizontal dispersion width (km)

    Returns:
        2D array (lat x lon) of plume weights [0, 1]
    """
    # Convert sigma from km to degrees (approximate)
    sigma_lat = sigma_y_km / 111.0
    sigma_lon = sigma_y_km / (111.0 * np.cos(np.radians(center_lat)))

    lat_diff = lats[:, None] - center_lat
    lon_diff = lons[None, :] - center_lon

    plume = np.exp(-0.5 * (lat_diff**2 / sigma_lat**2 + lon_diff**2 / sigma_lon**2))
    return plume


def gaussian_plume_vertical(
    levels: np.ndarray,
    center_level: float,
    sigma_z: float = 100.0,
) -> np.ndarray:
    """
    Compute vertical Gaussian weighting across pressure levels.

    Args:
        levels: 1D array of pressure levels (hPa)
        center_level: Center pressure level (hPa)
        sigma_z: Vertical dispersion (hPa)

    Returns:
        1D array of vertical weights [0, 1]
    """
    return np.exp(-0.5 * ((levels - center_level) / sigma_z) ** 2)


def level_band_mask(levels: np.ndarray, p_min: float, p_max: float) -> np.ndarray:
    """
    Create a mask for pressure levels within [p_min, p_max] (hPa).

    Note: In pressure coordinates, higher pressure = lower altitude.
    p_min is the TOP (lower hPa), p_max is the BOTTOM (higher hPa).
    """
    return (levels >= p_min) & (levels <= p_max)


def temporal_ramp(n_steps: int, lag_steps: int, ramp_steps: int) -> np.ndarray:
    """
    Generate a temporal ramp function for gradual mask application.

    The mask is zero during the lag period, then ramps linearly to full
    intensity over ramp_steps.

    Args:
        n_steps: Total number of time steps
        lag_steps: Number of initial steps with zero mask
        ramp_steps: Number of steps to ramp from 0 to 1

    Returns:
        1D array of temporal weights [0, 1] of length n_steps
    """
    weights = np.zeros(n_steps)
    for t in range(n_steps):
        if t < lag_steps:
            weights[t] = 0.0
        elif t < lag_steps + ramp_steps:
            weights[t] = (t - lag_steps + 1) / ramp_steps
        else:
            weights[t] = 1.0
    return weights


def get_variable(ds: xr.Dataset, names: list[str]) -> Optional[xr.DataArray]:
    """Try to get a variable from the dataset by multiple possible names."""
    for name in names:
        if name in ds.data_vars:
            return ds[name]
    return None


def _collapse_to_3d(
    arr: np.ndarray,
    n_levels: int,
    n_lats: int,
    n_lons: int,
) -> np.ndarray:
    """
    Collapse a multi-dimensional variable to (level, lat, lon) by averaging
    over time and batch dimensions.

    Handles shapes: (time, batch, level, lat, lon), (time, level, lat, lon),
    (batch, level, lat, lon), (level, lat, lon).
    """
    target_shape = (n_levels, n_lats, n_lons)
    if arr.shape == target_shape:
        return arr.copy()

    # Average over leading dimensions until we get to 3D
    x = arr.copy()
    while x.ndim > 3:
        x = np.nanmean(x, axis=0)

    # If we collapsed too far (e.g., surface variable with no level dim)
    if x.ndim == 2:
        return np.broadcast_to(x[None, :, :], target_shape).copy()

    return x


def safe_get_levels(ds: xr.Dataset) -> Optional[np.ndarray]:
    """Extract pressure level coordinates from the dataset."""
    for name in ["level", "pressure_level", "plev"]:
        if name in ds.coords:
            return ds.coords[name].values
    return None


# ---------------------------------------------------------------------------
# Base Mask Class
# ---------------------------------------------------------------------------

class SeedingMask:
    """
    Base class for all geoengineering seeding masks.

    A seeding mask is an additive perturbation M applied to the GraphCast
    input state: X_seeded = X_raw + M.

    Subclasses implement generate_perturbations() to define the physical
    modifications for each intervention type.
    """

    # Human-readable name
    name: str = "base"
    # Category: "drought" or "hurricane"
    category: str = "unknown"
    # Physical lag time in minutes
    lag_minutes: float = 0.0

    def __init__(self, config: SeedingConfig):
        self.config = config
        self.region = config.target_region
        self.intensity = np.clip(config.intensity, 0.0, 1.0)

    def generate_mask(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Generate the complete seeding mask for a GraphCast dataset.

        Args:
            ds: Input xarray Dataset (GraphCast format with ERA5 variables).
                Typical shapes:
                  3D vars: (batch, time, level, lat, lon)
                  Surface vars: (batch, time, lat, lon)

        Returns:
            xr.Dataset of same shape with perturbation values (M).
            Variables not perturbed are filled with zeros.
        """
        # Create a zeroed copy for the mask
        mask = xr.zeros_like(ds).copy()

        # Get coordinates
        lats = ds.coords["lat"].values
        lons = ds.coords["lon"].values
        levels = safe_get_levels(ds)

        # Compute spatial geometry (where to seed)
        if self.config.plume is not None:
            geo_weight = gaussian_plume_2d(
                lats, lons,
                self.config.plume.source_lat,
                self.config.plume.source_lon,
                self.config.plume.sigma_y_km,
            )
        else:
            # Use rectangular target region with soft edges
            geo_weight = self.region.contains(lats, lons).astype(np.float64)
            # Apply Gaussian smoothing at edges (3-cell taper)
            geo_weight = _smooth_region_edges(geo_weight, taper_cells=3)

        # Compute temporal ramp
        n_times = ds.sizes.get("time", 1)
        t_weights = temporal_ramp(n_times, self.config.lag_steps, self.config.ramp_steps)

        # Let subclass compute the physical perturbations
        perturbations = self.generate_perturbations(ds, lats, lons, levels)

        # Apply spatial weighting, temporal ramping, and intensity scaling
        for var_name, delta in perturbations.items():
            if var_name not in mask.data_vars:
                continue

            target = mask[var_name].values  # full shape from dataset
            dims = list(ds[var_name].dims)
            delta_val = delta * self.intensity

            has_batch = "batch" in dims
            has_time = "time" in dims
            has_level = "level" in dims

            # Determine the time axis index
            t_axis = dims.index("time") if has_time else None
            n_t = target.shape[t_axis] if t_axis is not None else 1

            # Build the perturbation for a single time step.
            # delta_val can be (level, lat, lon) or (lat, lon).
            # geo_weight is always (lat, lon).
            if has_level and delta_val.ndim == 3:
                # delta is (level, lat, lon), geo is (lat, lon)
                step_pert = delta_val * geo_weight[None, :, :]  # (level, lat, lon)
            elif delta_val.ndim == 2:
                # delta is (lat, lon)
                step_pert = delta_val * geo_weight  # (lat, lon)
            else:
                # Fallback: try element-wise
                step_pert = delta_val * geo_weight

            # Apply to each time step with temporal ramp
            for t in range(n_t):
                ramp = t_weights[t] if t < len(t_weights) else t_weights[-1]
                if ramp == 0:
                    continue

                scaled_pert = step_pert * ramp

                # Navigate to the right slice based on actual dim ordering
                # Typical: (batch, time, level, lat, lon) or (batch, time, lat, lon)
                if has_batch and has_time and has_level:
                    # (batch, time, level, lat, lon)
                    target[0, t] += scaled_pert
                elif has_batch and has_time:
                    # (batch, time, lat, lon) - surface variable
                    if scaled_pert.ndim == 2:
                        target[0, t] += scaled_pert
                    else:
                        # delta had levels but variable doesn't - average over levels
                        target[0, t] += np.mean(scaled_pert, axis=0)
                elif has_time and has_level:
                    # (time, level, lat, lon)
                    target[t] += scaled_pert
                elif has_time:
                    # (time, lat, lon)
                    if scaled_pert.ndim == 2:
                        target[t] += scaled_pert
                    else:
                        target[t] += np.mean(scaled_pert, axis=0)
                else:
                    # Static variable, apply once
                    if scaled_pert.ndim == target.ndim:
                        target += scaled_pert
                    break

            mask[var_name].values = target

        return mask

    def generate_perturbations(
        self,
        ds: xr.Dataset,
        lats: np.ndarray,
        lons: np.ndarray,
        levels: Optional[np.ndarray],
    ) -> dict[str, np.ndarray]:
        """
        Compute the raw perturbation values for each variable.

        Must be implemented by subclasses. Returns a dict mapping
        variable names to perturbation arrays. Spatial/temporal weighting
        is applied by generate_mask().

        Returns:
            Dict of {variable_name: perturbation_array}
        """
        raise NotImplementedError


def _smooth_region_edges(mask: np.ndarray, taper_cells: int = 3) -> np.ndarray:
    """Apply a simple distance-based taper to the edges of a binary mask."""
    from scipy.ndimage import distance_transform_edt
    if not np.any(mask > 0):
        return mask
    # Distance from the edge of the active region
    dist_outside = distance_transform_edt(mask == 0)
    dist_inside = distance_transform_edt(mask > 0)
    # Taper zone: cells within taper_cells of the boundary
    taper = np.ones_like(mask, dtype=np.float64)
    edge_zone = (dist_inside > 0) & (dist_inside <= taper_cells)
    taper[edge_zone] = dist_inside[edge_zone] / taper_cells
    taper[mask == 0] = 0.0
    return taper


class InitialConditionMask(SeedingMask):
    """
    Mask that only perturbs initial conditions (first 2 time steps).

    This mask is designed for the two-pass drought-responsive pipeline where
    interventions are applied to the initial state vector before GraphCast
    runs its autoregressive prediction. Unlike other masks that apply
    perturbations throughout the forecast, this mask only modifies:
      - Time step 0 (t-1, history)
      - Time step 1 (t, current state)

    Key differences from base SeedingMask:
      - No temporal ramping (full intensity immediately)
      - Only first 2 time steps are modified
      - Skips total_precipitation_6hr (output-only variable, not a GraphCast input)
      - Supports perturbation_scale to amplify effects for AI weather models

    The physical rationale: by perturbing the initial atmospheric state,
    the effects propagate through GraphCast's learned physics across all
    subsequent autoregressive forecast steps.

    Note on perturbation_scale: AI weather models like GraphCast are trained
    on real atmospheric data with large natural variability. Small perturbations
    (< 5% of natural std) get smoothed out. Use perturbation_scale=10-50 to
    create perturbations that are ~10-50% of natural variability, which is
    large enough for the model to respond while still being physically plausible
    for a geoengineering scenario.
    """

    name = "initial_condition"
    category = "drought"
    lag_minutes = 0.0

    # Variables to skip (outputs only, not used as GraphCast inputs)
    SKIP_VARS = ["total_precipitation_6hr"]

    # Default perturbation scale for AI weather models
    # Amplifies base perturbations to be significant relative to natural variability
    DEFAULT_PERTURBATION_SCALE = 20.0

    def __init__(
        self,
        config: SeedingConfig,
        base_mask_class: type = None,
        perturbation_scale: float = None,
    ):
        """
        Initialize InitialConditionMask.

        Args:
            config: SeedingConfig with target region and intensity
            base_mask_class: Optional mask class to use for perturbation values.
                           If None, uses HygroscopicEnhancementMask by default.
            perturbation_scale: Multiplier for perturbation magnitudes.
                              Default is 20.0 to create ~10-20% of natural std
                              perturbations that AI models can respond to.
                              Set to 1.0 for physically realistic (but small) effects.
        """
        super().__init__(config)
        self.base_mask_class = base_mask_class
        self.perturbation_scale = (
            perturbation_scale if perturbation_scale is not None
            else self.DEFAULT_PERTURBATION_SCALE
        )

    def generate_mask(self, eval_inputs: xr.Dataset) -> xr.Dataset:
        """
        Generate mask that only perturbs initial conditions.

        Args:
            eval_inputs: GraphCast eval_inputs with shape (batch, time=2, ...)
                        where time=2 represents the two initial time steps.

        Returns:
            xr.Dataset of same shape with perturbation values.
            Only time steps 0 and 1 are non-zero.
        """
        # Create a zeroed copy for the mask
        mask = xr.zeros_like(eval_inputs).copy()

        # Get coordinates
        lats = eval_inputs.coords["lat"].values
        lons = eval_inputs.coords["lon"].values
        levels = safe_get_levels(eval_inputs)

        # Compute spatial geometry
        if self.config.plume is not None:
            geo_weight = gaussian_plume_2d(
                lats, lons,
                self.config.plume.source_lat,
                self.config.plume.source_lon,
                self.config.plume.sigma_y_km,
            )
        else:
            geo_weight = self.region.contains(lats, lons).astype(np.float64)
            geo_weight = _smooth_region_edges(geo_weight, taper_cells=3)

        # Get perturbations from generate_perturbations (can be overridden)
        perturbations = self.generate_perturbations(eval_inputs, lats, lons, levels)

        # Apply spatial weighting, intensity, and perturbation scale - NO temporal ramping
        # Full intensity applied immediately to both initial time steps
        # perturbation_scale amplifies effects for AI weather models
        for var_name, delta in perturbations.items():
            if var_name not in mask.data_vars:
                continue
            if var_name in self.SKIP_VARS:
                continue

            target = mask[var_name].values
            dims = list(eval_inputs[var_name].dims)
            # Apply both intensity and perturbation_scale
            delta_val = delta * self.intensity * self.perturbation_scale

            has_batch = "batch" in dims
            has_time = "time" in dims
            has_level = "level" in dims

            # Get number of time steps (should be 2 for initial conditions)
            t_axis = dims.index("time") if has_time else None
            n_t = target.shape[t_axis] if t_axis is not None else 1

            # Build perturbation for a single time step
            if has_level and delta_val.ndim == 3:
                step_pert = delta_val * geo_weight[None, :, :]
            elif delta_val.ndim == 2:
                step_pert = delta_val * geo_weight
            else:
                step_pert = delta_val * geo_weight

            # Apply to first 2 time steps with FULL intensity (no ramping)
            for t in range(min(n_t, 2)):
                if has_batch and has_time and has_level:
                    target[0, t] += step_pert
                elif has_batch and has_time:
                    if step_pert.ndim == 2:
                        target[0, t] += step_pert
                    else:
                        target[0, t] += np.mean(step_pert, axis=0)
                elif has_time and has_level:
                    target[t] += step_pert
                elif has_time:
                    if step_pert.ndim == 2:
                        target[t] += step_pert
                    else:
                        target[t] += np.mean(step_pert, axis=0)
                else:
                    if step_pert.ndim == target.ndim:
                        target += step_pert
                    break

            mask[var_name].values = target

        return mask

    def generate_perturbations(
        self,
        ds: xr.Dataset,
        lats: np.ndarray,
        lons: np.ndarray,
        levels: Optional[np.ndarray],
    ) -> dict[str, np.ndarray]:
        """
        Generate perturbation values for initial condition modification.

        If a base_mask_class was provided, delegates to that class.
        Otherwise, returns perturbations targeting humidity and temperature
        for drought mitigation.
        """
        if self.base_mask_class is not None:
            # Use the base mask class to generate perturbations
            base_mask = self.base_mask_class(self.config)
            return base_mask.generate_perturbations(ds, lats, lons, levels)

        # Default: basic drought mitigation perturbations
        perturbations = {}
        if levels is None:
            return perturbations

        n_levels, n_lats, n_lons = len(levels), len(lats), len(lons)
        base_shape = (n_levels, n_lats, n_lons)

        # Target cloud-forming levels (925-600 hPa)
        lev_mask = level_band_mask(levels, 600, 925)

        # Humidity increase: seed moisture for precipitation
        q_var = resolve_var(ds, "humidity")
        if q_var:
            delta_q = np.zeros(base_shape)
            for k in range(n_levels):
                if lev_mask[k]:
                    delta_q[k] = 2e-4  # Increase humidity
            perturbations[q_var] = delta_q

        # Temperature: slight warming to enhance convection
        t_var = resolve_var(ds, "temperature")
        if t_var:
            delta_t = np.zeros(base_shape)
            for k in range(n_levels):
                if lev_mask[k]:
                    delta_t[k] = 0.3  # +0.3K warming
            perturbations[t_var] = delta_t

        # Vertical velocity: enhance updrafts
        w_var = resolve_var(ds, "vertical_vel")
        if w_var:
            delta_w = np.zeros(base_shape)
            for k in range(n_levels):
                if lev_mask[k]:
                    delta_w[k] = -0.05  # Upward (negative in Pa/s)
            perturbations[w_var] = delta_w

        # 2m temperature: surface warming
        t2m_var = resolve_var(ds, "t2m")
        if t2m_var:
            perturbations[t2m_var] = np.full((n_lats, n_lons), 0.3)

        return perturbations


# ---------------------------------------------------------------------------
# Drought Prevention Masks
# ---------------------------------------------------------------------------

class GlaciogenicStaticMask(SeedingMask):
    """
    Glaciogenic Static Seeding: Microphysical Optimization.

    Targets ice-deficient supercooled clouds (T between -5C and -15C).
    Triggers the Bergeron-Findeisen process by introducing artificial ice
    nuclei. Since GraphCast_small lacks explicit cloud water variables,
    the effect is represented as: humidity increase (enhanced moisture)
    and temperature increase (latent heat + convective forcing).

    Note: Perturbation magnitudes are scaled for AI weather models.
    The approach focuses on creating favorable conditions for precipitation
    rather than simulating exact microphysical processes.

    Primary perturbations (on GraphCast_small variables):
        - temperature: + (enhanced warming to drive convection)
        - specific_humidity: + (enhanced moisture for precipitation)
        - vertical_velocity: - (enhanced updrafts)

    Category: Drought Prevention (cold-cloud precipitation enhancement)
    Lag time: 15-20 minutes
    """

    name = "glaciogenic_static"
    category = "drought"
    lag_minutes = 17.5

    def generate_perturbations(self, ds, lats, lons, levels):
        perturbations = {}
        if levels is None:
            return perturbations

        n_levels, n_lats, n_lons = len(levels), len(lats), len(lons)
        base_shape = (n_levels, n_lats, n_lons)

        # Get temperature to find target regions
        t_var = resolve_var(ds, "temperature")
        if t_var is None:
            return perturbations

        temp_vals = ds[t_var].values
        temp_mean = _collapse_to_3d(temp_vals, n_levels, n_lats, n_lons)

        # Target mid-troposphere levels (850-400 hPa) for glaciogenic seeding
        lev_mask = level_band_mask(levels, 400, 850)

        # ENHANCED: Temperature increase to enhance ice crystal growth
        # Original: ~0.017K. New: 2.0K (significant warming)
        delta_t = np.zeros(base_shape)
        for k in range(n_levels):
            if lev_mask[k]:
                delta_t[k] = 2.0  # +2K warming
        perturbations[t_var] = delta_t

        # ENHANCED: Humidity increase to provide moisture for ice crystals
        # Original: -5e-5 (removing moisture). New: +8e-4 (adding moisture)
        q_var = resolve_var(ds, "humidity")
        if q_var:
            delta_q = np.zeros(base_shape)
            for k in range(n_levels):
                if lev_mask[k]:
                    delta_q[k] = 8e-4  # Add moisture
            perturbations[q_var] = delta_q

        # ENHANCED: Vertical velocity to enhance updrafts
        w_var = resolve_var(ds, "vertical_vel")
        if w_var:
            delta_w = np.zeros(base_shape)
            for k in range(n_levels):
                if lev_mask[k]:
                    delta_w[k] = -0.25  # Strong upward motion
            perturbations[w_var] = delta_w

        # 2m temperature: surface warming
        t2m_var = resolve_var(ds, "t2m")
        if t2m_var:
            perturbations[t2m_var] = np.full((n_lats, n_lons), 1.2)  # +1.2K

        # MSLP decrease to enhance convergence
        mslp_var = resolve_var(ds, "mslp")
        if mslp_var:
            perturbations[mslp_var] = np.full((n_lats, n_lons), -150.0)  # -1.5 hPa

        return perturbations


class HygroscopicEnhancementMask(SeedingMask):
    """
    Hygroscopic Enhancement Seeding: Giant Particle Regime (1-10 um).

    Targets warm clouds stuck in colloidal stability. Giant CCN broaden
    the droplet size distribution to trigger collision-coalescence,
    enhancing precipitation efficiency by 15-30%.

    Primary perturbations (on GraphCast_small variables):
        - specific_humidity: + at cloud base (enhanced moisture convergence)
        - temperature: + at cloud base (latent heat + enhanced convection)
        - vertical_velocity: upward (enhanced updrafts from seeding)

    Note: Perturbation magnitudes are scaled for AI weather models.
    Original physically-realistic values were too small (~1% of natural std)
    for neural network models to respond to meaningfully.

    Category: Drought Prevention (warm cloud precipitation enhancement)
    Lag time: 20-30 minutes
    """

    name = "hygroscopic_enhancement"
    category = "drought"
    lag_minutes = 25.0

    def generate_perturbations(self, ds, lats, lons, levels):
        perturbations = {}
        if levels is None:
            return perturbations

        n_levels, n_lats, n_lons = len(levels), len(lats), len(lons)
        base_shape = (n_levels, n_lats, n_lons)

        # Target cloud base levels (925-850 hPa) AND mid-levels for deeper effect
        lev_mask_low = level_band_mask(levels, CLOUD_BASE_LEVELS[1], CLOUD_BASE_LEVELS[0])
        lev_mask_mid = level_band_mask(levels, 600, 700)

        # ENHANCED: Humidity increase to promote cloud formation and precipitation
        # Original: 2e-4 kg/kg (too small). New: 1e-3 kg/kg (5x larger base)
        dq_enhanced = 1e-3  # kg/kg - significant moisture injection

        q_var = resolve_var(ds, "humidity")
        if q_var:
            delta_q = np.zeros(base_shape)
            for k in range(n_levels):
                if lev_mask_low[k]:
                    delta_q[k] = dq_enhanced  # Add moisture at cloud base
                elif lev_mask_mid[k]:
                    delta_q[k] = dq_enhanced * 0.5  # Also enhance mid-levels
            perturbations[q_var] = delta_q

        # ENHANCED: Temperature increase to drive convection
        # Original: ~0.34K. New: 1.5K (stronger convective forcing)
        t_var = resolve_var(ds, "temperature")
        if t_var:
            delta_t = np.zeros(base_shape)
            for k in range(n_levels):
                if lev_mask_low[k]:
                    delta_t[k] = 1.5  # +1.5K warming at cloud base
                elif lev_mask_mid[k]:
                    delta_t[k] = 0.8  # +0.8K at mid-levels
            perturbations[t_var] = delta_t

        # ENHANCED: Updraft enhancement (negative = upward in pressure coords)
        # Original: -0.05 Pa/s. New: -0.3 Pa/s (6x stronger updrafts)
        w_var = resolve_var(ds, "vertical_vel")
        if w_var:
            delta_w = np.zeros(base_shape)
            for k in range(n_levels):
                if lev_mask_low[k]:
                    delta_w[k] = -0.3  # Strong upward motion
                elif lev_mask_mid[k]:
                    delta_w[k] = -0.2  # Continued uplift
            perturbations[w_var] = delta_w

        # 2m temperature: surface warming to enhance boundary layer instability
        t2m_var = resolve_var(ds, "t2m")
        if t2m_var:
            perturbations[t2m_var] = np.full((n_lats, n_lons), 1.0)  # +1K surface

        # Mean sea level pressure: slight decrease to enhance convergence
        mslp_var = resolve_var(ds, "mslp")
        if mslp_var:
            perturbations[mslp_var] = np.full((n_lats, n_lons), -100.0)  # -1 hPa

        return perturbations


class ElectricIonizationMask(SeedingMask):
    """
    Electric/Ionization Seeding: Electrostatic Coalescence.

    Accelerates the autoconversion rate (cloud water -> rain) by enhancing
    collision efficiency via electrostatic attraction from corona discharge.
    Research indicates ~2-3x enhancement in collision efficiency.

    Note: Perturbation magnitudes are scaled for AI weather models.
    Focuses on creating conditions favorable for enhanced precipitation.

    Primary perturbations (on GraphCast_small variables):
        - specific_humidity: + (enhanced moisture availability)
        - temperature: + (enhanced convective instability)
        - vertical_velocity: - (enhanced updrafts)

    Category: Drought Prevention (precipitation acceleration)
    Lag time: < 10 minutes
    """

    name = "electric_ionization"
    category = "drought"
    lag_minutes = 8.0

    def generate_perturbations(self, ds, lats, lons, levels):
        perturbations = {}
        n_lats, n_lons = len(lats), len(lons)

        if levels is not None:
            n_levels = len(levels)
            base_shape = (n_levels, n_lats, n_lons)

            # Target low-to-mid level cloud layers (925-600 hPa)
            lev_mask_low = level_band_mask(levels, 700, 925)
            lev_mask_mid = level_band_mask(levels, 500, 700)

            # ENHANCED: Humidity increase for enhanced precipitation
            q_var = resolve_var(ds, "humidity")
            if q_var:
                delta_q = np.zeros(base_shape)
                for k in range(n_levels):
                    if lev_mask_low[k]:
                        delta_q[k] = 1.2e-3  # Strong moisture injection
                    elif lev_mask_mid[k]:
                        delta_q[k] = 6e-4
                perturbations[q_var] = delta_q

            # ENHANCED: Temperature increase for instability
            t_var = resolve_var(ds, "temperature")
            if t_var:
                delta_t = np.zeros(base_shape)
                for k in range(n_levels):
                    if lev_mask_low[k]:
                        delta_t[k] = 1.8  # +1.8K
                    elif lev_mask_mid[k]:
                        delta_t[k] = 1.0  # +1.0K
                perturbations[t_var] = delta_t

            # ENHANCED: Vertical velocity for updrafts
            w_var = resolve_var(ds, "vertical_vel")
            if w_var:
                delta_w = np.zeros(base_shape)
                for k in range(n_levels):
                    if lev_mask_low[k]:
                        delta_w[k] = -0.35  # Strong upward
                    elif lev_mask_mid[k]:
                        delta_w[k] = -0.25
                perturbations[w_var] = delta_w

        # 2m temperature: surface warming
        t2m_var = resolve_var(ds, "t2m")
        if t2m_var:
            perturbations[t2m_var] = np.full((n_lats, n_lons), 1.3)

        # MSLP decrease
        mslp_var = resolve_var(ds, "mslp")
        if mslp_var:
            perturbations[mslp_var] = np.full((n_lats, n_lons), -120.0)

        return perturbations


class LaserInducedCondensationMask(SeedingMask):
    """
    Laser-Induced Condensation (LIC): Photochemical Nucleation.

    Femtosecond laser filamentation produces HNO3, enabling binary
    H2O-HNO3 condensation at RH as low as 70%. This effectively
    forces condensation in sub-saturated air.

    Note: Perturbation magnitudes are scaled for AI weather models.
    Represents the most aggressive intervention with strongest forcing.

    Primary perturbations (on GraphCast_small variables):
        - specific_humidity: + (enhanced moisture for condensation)
        - temperature: + (thermal forcing + latent heat)
        - vertical_velocity: - (strong convective enhancement)

    Category: Drought Prevention (precise, targeted nucleation)
    Lag time: Immediate
    """

    name = "laser_induced_condensation"
    category = "drought"
    lag_minutes = 1.0

    def generate_perturbations(self, ds, lats, lons, levels):
        perturbations = {}
        if levels is None:
            return perturbations

        n_levels, n_lats, n_lons = len(levels), len(lats), len(lons)
        base_shape = (n_levels, n_lats, n_lons)

        # Target multiple levels for maximum effect (925-500 hPa)
        lev_mask_low = level_band_mask(levels, 700, 925)
        lev_mask_mid = level_band_mask(levels, 500, 700)

        # ENHANCED: Humidity increase - strongest of all interventions
        q_var = resolve_var(ds, "humidity")
        if q_var:
            delta_q = np.zeros(base_shape)
            for k in range(n_levels):
                if lev_mask_low[k]:
                    delta_q[k] = 1.5e-3  # Very strong moisture injection
                elif lev_mask_mid[k]:
                    delta_q[k] = 1.0e-3
            perturbations[q_var] = delta_q

        # ENHANCED: Temperature - strong thermal forcing
        t_var = resolve_var(ds, "temperature")
        if t_var:
            delta_t = np.zeros(base_shape)
            for k in range(n_levels):
                if lev_mask_low[k]:
                    delta_t[k] = 2.5  # +2.5K (strongest warming)
                elif lev_mask_mid[k]:
                    delta_t[k] = 1.5  # +1.5K
            perturbations[t_var] = delta_t

        # ENHANCED: Vertical velocity - strongest updraft enhancement
        w_var = resolve_var(ds, "vertical_vel")
        if w_var:
            delta_w = np.zeros(base_shape)
            for k in range(n_levels):
                if lev_mask_low[k]:
                    delta_w[k] = -0.4  # Very strong upward
                elif lev_mask_mid[k]:
                    delta_w[k] = -0.3
            perturbations[w_var] = delta_w

        # 2m temperature: strong surface warming
        t2m_var = resolve_var(ds, "t2m")
        if t2m_var:
            perturbations[t2m_var] = np.full((n_lats, n_lons), 1.5)

        # MSLP: strongest pressure decrease
        mslp_var = resolve_var(ds, "mslp")
        if mslp_var:
            perturbations[mslp_var] = np.full((n_lats, n_lons), -200.0)  # -2 hPa

        return perturbations


# ---------------------------------------------------------------------------
# Hurricane / Storm Disruption Masks
# ---------------------------------------------------------------------------

class GlaciogenicDynamicMask(SeedingMask):
    """
    Glaciogenic Dynamic Seeding: Circulation Modification.

    Massive overseeding at mid-levels (500-300 hPa) to freeze supercooled
    water, releasing latent heat that creates artificial secondary updrafts.
    These compete with the main storm for inflow energy.

    For 1 g/kg frozen: dT ~ 0.33K. Dynamic seeding targets 3-5 g/kg,
    yielding dT ~ 1.0-1.7K -- a massive perturbation in NWP terms.

    Primary perturbations (on GraphCast_small variables):
        - temperature: ++ (massive latent heat of fusion)
        - specific_humidity: -- (water budget consumed by freezing)
        - vertical_velocity: + (updraft invigoration from buoyancy)
        - geopotential: + (pressure surface lifting)
        - u_component_of_wind: + (outward divergence weakens convergence)

    Category: Hurricane/Storm Disruption
    Lag time: 10-15 minutes
    """

    name = "glaciogenic_dynamic"
    category = "hurricane"
    lag_minutes = 12.5

    def generate_perturbations(self, ds, lats, lons, levels):
        perturbations = {}
        if levels is None:
            return perturbations

        n_levels, n_lats, n_lons = len(levels), len(lats), len(lons)
        base_shape = (n_levels, n_lats, n_lons)

        # Target storm core levels (500-300 hPa)
        lev_mask = level_band_mask(levels, STORM_CORE_LEVELS[1], STORM_CORE_LEVELS[0])

        # Massive latent heat: freeze 3 g/kg of supercooled water
        dq_frozen = 3e-3  # kg/kg
        dT = L_FUSION * dq_frozen / C_P  # ~1.0 K

        # Temperature
        t_var = resolve_var(ds, "temperature")
        if t_var:
            delta_t = np.zeros(base_shape)
            for k in range(n_levels):
                if lev_mask[k]:
                    delta_t[k] = dT
            perturbations[t_var] = delta_t

        # Humidity reduction
        q_var = resolve_var(ds, "humidity")
        if q_var:
            delta_q = np.zeros(base_shape)
            for k in range(n_levels):
                if lev_mask[k]:
                    delta_q[k] = -dq_frozen
            perturbations[q_var] = delta_q

        # Updraft invigoration
        w_var = resolve_var(ds, "vertical_vel")
        if w_var:
            delta_w = np.zeros(base_shape)
            for k in range(n_levels):
                if lev_mask[k]:
                    delta_w[k] = -0.2  # Pa/s (upward)
            perturbations[w_var] = delta_w

        # Geopotential increase (warm air lifts pressure surfaces)
        z_var = resolve_var(ds, "geopotential")
        if z_var:
            delta_z = np.zeros(base_shape)
            for k in range(n_levels):
                if lev_mask[k]:
                    delta_z[k] = dT * R_D / G * 100  # ~30 m^2/s^2
            perturbations[z_var] = delta_z

        # Wind divergence at storm flanks (weakens convergence)
        u_var = resolve_var(ds, "u_wind")
        if u_var:
            delta_u = np.zeros(base_shape)
            for k in range(n_levels):
                if lev_mask[k]:
                    delta_u[k] = 1.0  # m/s outward
            perturbations[u_var] = delta_u

        return perturbations


class HygroscopicSuppressionMask(SeedingMask):
    """
    Hygroscopic Suppression Seeding: Sub-micrometer Regime (< 0.8 um).

    Introduces massive numbers of small CCN into hurricane outer rainbands
    to suppress the warm rain process (Twomey Effect). More, smaller
    droplets = brighter cloud + no precipitation.

    Primary perturbations (on GraphCast_small variables):
        - specific_humidity: + (moisture trapped, can't precipitate)
        - temperature: - (evaporative cooling from suppressed precipitation)
        - total_precipitation_6hr: - (precipitation suppressed)
        - mean_sea_level_pressure: + (weaker storm -> higher pressure)

    Category: Hurricane/Storm Disruption (warm rain suppression)
    Lag time: 15-20 minutes
    """

    name = "hygroscopic_suppression"
    category = "hurricane"
    lag_minutes = 17.5

    def generate_perturbations(self, ds, lats, lons, levels):
        perturbations = {}
        n_lats, n_lons = len(lats), len(lons)

        if levels is not None:
            n_levels = len(levels)
            base_shape = (n_levels, n_lats, n_lons)

            # Target outer rainband levels (925-700 hPa)
            lev_mask = level_band_mask(levels, 700, 925)

            # Humidity increase: moisture trapped as tiny non-precipitating drops
            q_var = resolve_var(ds, "humidity")
            if q_var:
                delta_q = np.zeros(base_shape)
                for k in range(n_levels):
                    if lev_mask[k]:
                        delta_q[k] = 5e-5  # trapped moisture
                perturbations[q_var] = delta_q

            # Temperature decrease: no latent heat release (suppressed precip)
            t_var = resolve_var(ds, "temperature")
            if t_var:
                delta_t = np.zeros(base_shape)
                for k in range(n_levels):
                    if lev_mask[k]:
                        delta_t[k] = -0.15  # slight cooling
                perturbations[t_var] = delta_t

        # Suppress precipitation (the key storm-weakening effect)
        p_var = resolve_var(ds, "precip")
        if p_var:
            precip = ds[p_var].values
            precip_mean = np.nanmean(precip, axis=0)
            while precip_mean.ndim > 2:
                precip_mean = precip_mean[0]
            perturbations[p_var] = -np.abs(precip_mean) * 0.5

        # MSLP increase: weaker storm -> higher central pressure
        mslp_var = resolve_var(ds, "mslp")
        if mslp_var:
            perturbations[mslp_var] = np.full((n_lats, n_lons), 200.0)  # +2 hPa

        return perturbations


# ---------------------------------------------------------------------------
# Mask Registry
# ---------------------------------------------------------------------------

MASK_REGISTRY: dict[str, type[SeedingMask]] = {
    # Drought prevention
    "glaciogenic_static": GlaciogenicStaticMask,
    "hygroscopic_enhancement": HygroscopicEnhancementMask,
    "electric_ionization": ElectricIonizationMask,
    "laser_induced_condensation": LaserInducedCondensationMask,
    # Hurricane / storm disruption
    "glaciogenic_dynamic": GlaciogenicDynamicMask,
    "hygroscopic_suppression": HygroscopicSuppressionMask,
}

DROUGHT_MASKS = [
    "glaciogenic_static",
    "hygroscopic_enhancement",
    "electric_ionization",
    "laser_induced_condensation",
]

HURRICANE_MASKS = [
    "glaciogenic_dynamic",
    "hygroscopic_suppression",
]


# ---------------------------------------------------------------------------
# Drought-Targeted Mask Factory
# ---------------------------------------------------------------------------

def create_drought_targeted_mask(
    drought_grid: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    eval_inputs: xr.Dataset,
    intensity: float = 0.5,
    min_severity: int = 2,
    buffer_degrees: float = 5.0,
    intervention_type: str = "hygroscopic_enhancement",
    perturbation_scale: float = None,
) -> tuple[InitialConditionMask, TargetRegion]:
    """
    Create an InitialConditionMask targeting drought regions from a drought grid.

    This factory function analyzes a drought severity grid (from DroughtDetector)
    and creates an intervention mask focused on the affected areas with a buffer
    zone for upstream moisture enhancement.

    Args:
        drought_grid: 2D array of drought severity classes (0=none, 1-4=severity)
                     Shape should be (n_lats, n_lons)
        lats: 1D array of latitudes corresponding to drought_grid rows
        lons: 1D array of longitudes corresponding to drought_grid columns
        eval_inputs: GraphCast eval_inputs xr.Dataset for mask generation
        intensity: Intervention intensity (0.0 to 1.0)
        min_severity: Minimum drought class to target (default 2 = moderate)
        buffer_degrees: Buffer zone around drought regions in degrees (default 5.0)
        intervention_type: Type of seeding intervention to use. Options:
                          "hygroscopic_enhancement", "glaciogenic_static",
                          "electric_ionization", "laser_induced_condensation"
        perturbation_scale: Multiplier for perturbation magnitudes (default: 20.0).
                           Higher values create stronger effects that AI models respond to.

    Returns:
        Tuple of (InitialConditionMask, TargetRegion):
          - InitialConditionMask ready for application to eval_inputs
          - TargetRegion describing the intervention area

    Example:
        >>> drought_grid, lats, lons = detector.compute_drought_grid(predictions)
        >>> mask, region = create_drought_targeted_mask(
        ...     drought_grid, lats, lons, eval_inputs,
        ...     intensity=0.7, min_severity=2
        ... )
        >>> seeded_inputs = eval_inputs + mask.generate_mask(eval_inputs)
    """
    # Create drought mask for cells meeting minimum severity
    drought_mask = drought_grid >= min_severity

    if not np.any(drought_mask):
        # No drought detected at requested severity - return a no-op mask
        print(f"  Warning: No drought cells found with severity >= {min_severity}")
        # Create a minimal region at the center of the grid
        center_lat = float(np.mean(lats))
        center_lon = float(np.mean(lons))
        target_region = TargetRegion(
            lat_min=center_lat - 1,
            lat_max=center_lat + 1,
            lon_min=center_lon - 1,
            lon_max=center_lon + 1,
        )
        config = SeedingConfig(
            intervention_type=intervention_type,
            target_region=target_region,
            intensity=0.0,  # Zero intensity since no drought
            lag_steps=0,
            ramp_steps=0,
        )
        base_mask_class = MASK_REGISTRY.get(intervention_type, HygroscopicEnhancementMask)
        return InitialConditionMask(
            config,
            base_mask_class=base_mask_class,
            perturbation_scale=perturbation_scale,
        ), target_region

    # Find bounding box of drought cells
    lat_has_drought = np.any(drought_mask, axis=1)
    lon_has_drought = np.any(drought_mask, axis=0)

    lat_indices = np.where(lat_has_drought)[0]
    lon_indices = np.where(lon_has_drought)[0]

    if len(lat_indices) == 0 or len(lon_indices) == 0:
        raise ValueError("Drought mask is non-empty but no indices found")

    # Get bounding box with buffer
    lat_min_idx, lat_max_idx = lat_indices[0], lat_indices[-1]
    lon_min_idx, lon_max_idx = lon_indices[0], lon_indices[-1]

    lat_min = float(lats[lat_min_idx]) - buffer_degrees
    lat_max = float(lats[lat_max_idx]) + buffer_degrees
    lon_min = float(lons[lon_min_idx]) - buffer_degrees
    lon_max = float(lons[lon_max_idx]) + buffer_degrees

    # Clamp to valid ranges
    lat_min = max(lat_min, -90.0)
    lat_max = min(lat_max, 90.0)
    lon_min = max(lon_min, 0.0)
    lon_max = min(lon_max, 360.0)

    target_region = TargetRegion(
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
    )

    # Count drought cells for logging
    total_drought_cells = int(np.sum(drought_mask))
    severe_cells = int(np.sum(drought_grid >= 3))
    extreme_cells = int(np.sum(drought_grid >= 4))

    print(f"  Drought-targeted region created:")
    print(f"    Bounding box: lat [{lat_min:.1f}, {lat_max:.1f}], "
          f"lon [{lon_min:.1f}, {lon_max:.1f}]")
    print(f"    Drought cells: {total_drought_cells} total, "
          f"{severe_cells} severe+, {extreme_cells} extreme")
    print(f"    Buffer: {buffer_degrees} degrees")
    print(f"    Intervention: {intervention_type}")

    # Create the config
    config = SeedingConfig(
        intervention_type=intervention_type,
        target_region=target_region,
        intensity=intensity,
        lag_steps=0,  # No lag for initial conditions
        ramp_steps=0,  # No ramping - full intensity immediately
    )

    # Get the base mask class for this intervention type
    base_mask_class = MASK_REGISTRY.get(intervention_type, HygroscopicEnhancementMask)

    # Create the InitialConditionMask with perturbation scaling
    mask = InitialConditionMask(
        config,
        base_mask_class=base_mask_class,
        perturbation_scale=perturbation_scale,
    )

    return mask, target_region


def drought_grid_to_region(
    drought_grid: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    min_severity: int = 2,
    buffer_degrees: float = 5.0,
) -> Optional[TargetRegion]:
    """
    Convert a drought grid to a TargetRegion encompassing drought cells.

    Simpler helper that just returns the region without creating a mask.

    Args:
        drought_grid: 2D array of drought severity classes
        lats: Latitude coordinates
        lons: Longitude coordinates
        min_severity: Minimum severity to consider
        buffer_degrees: Buffer around drought cells

    Returns:
        TargetRegion or None if no drought cells found
    """
    drought_mask = drought_grid >= min_severity

    if not np.any(drought_mask):
        return None

    lat_has_drought = np.any(drought_mask, axis=1)
    lon_has_drought = np.any(drought_mask, axis=0)

    lat_indices = np.where(lat_has_drought)[0]
    lon_indices = np.where(lon_has_drought)[0]

    lat_min = float(lats[lat_indices[0]]) - buffer_degrees
    lat_max = float(lats[lat_indices[-1]]) + buffer_degrees
    lon_min = float(lons[lon_indices[0]]) - buffer_degrees
    lon_max = float(lons[lon_indices[-1]]) + buffer_degrees

    lat_min = max(lat_min, -90.0)
    lat_max = min(lat_max, 90.0)
    lon_min = max(lon_min, 0.0)
    lon_max = min(lon_max, 360.0)

    return TargetRegion(lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max)


# ---------------------------------------------------------------------------
# GeoEngineering Simulator (Orchestrator)
# ---------------------------------------------------------------------------

class GeoEngineeringSimulator:
    """
    Orchestrator for applying geoengineering seeding masks to GraphCast data.

    This class manages loading datasets, applying one or more intervention
    masks, validating physical conservation, and producing modified datasets
    ready for GraphCast inference.

    Usage:
        sim = GeoEngineeringSimulator("dataset.nc")
        result = sim.apply_intervention(
            "hygroscopic_enhancement",
            target_region=TargetRegion(30, 35, 250, 260),
            intensity=0.5,
        )
        result.seeded_dataset.to_netcdf("seeded.nc")

    Or apply all drought / hurricane masks at once:
        results = sim.apply_all_drought_masks(
            target_region=TargetRegion(30, 35, 250, 260),
            intensity=0.5,
        )
    """

    def __init__(self, dataset_path: Optional[str] = None, dataset: Optional[xr.Dataset] = None):
        """
        Initialize the simulator.

        Args:
            dataset_path: Path to a NetCDF file with GraphCast input data.
            dataset: Alternatively, provide an xarray Dataset directly.
        """
        if dataset is not None:
            self.raw_dataset = dataset
        elif dataset_path is not None:
            self.raw_dataset = xr.load_dataset(dataset_path).compute()
        else:
            raise ValueError("Must provide either dataset_path or dataset")

        self._inspect_dataset()

    def _inspect_dataset(self):
        """Log dataset structure for debugging."""
        ds = self.raw_dataset
        self.has_batch = "batch" in ds.dims
        self.has_levels = safe_get_levels(ds) is not None
        self.variables = list(ds.data_vars)
        self.n_times = ds.sizes.get("time", 0)
        self.levels = safe_get_levels(ds)

        print(f"Dataset loaded:")
        print(f"  Dimensions: {dict(ds.dims)}")
        print(f"  Variables: {self.variables[:10]}{'...' if len(self.variables) > 10 else ''}")
        print(f"  Pressure levels: {self.levels}")

    def apply_intervention(
        self,
        intervention_type: str,
        target_region: TargetRegion,
        intensity: float = 0.5,
        plume: Optional[PlumeConfig] = None,
        lag_steps: int = 1,
        ramp_steps: int = 2,
    ) -> "InterventionResult":
        """
        Apply a single geoengineering intervention mask.

        Args:
            intervention_type: One of the keys in MASK_REGISTRY
            target_region: Geographic bounding box
            intensity: Perturbation intensity (0.0 to 1.0)
            plume: Optional Gaussian plume configuration
            lag_steps: Temporal lag in 6-hour steps
            ramp_steps: Steps over which to ramp mask to full intensity

        Returns:
            InterventionResult with the seeded dataset and diagnostics
        """
        if intervention_type not in MASK_REGISTRY:
            raise ValueError(
                f"Unknown intervention: {intervention_type}. "
                f"Available: {list(MASK_REGISTRY.keys())}"
            )

        config = SeedingConfig(
            intervention_type=intervention_type,
            target_region=target_region,
            intensity=intensity,
            plume=plume,
            lag_steps=lag_steps,
            ramp_steps=ramp_steps,
        )

        mask_cls = MASK_REGISTRY[intervention_type]
        mask_obj = mask_cls(config)

        print(f"\nApplying {mask_obj.name} mask (category: {mask_obj.category})...")
        print(f"  Region: lat [{target_region.lat_min}, {target_region.lat_max}], "
              f"lon [{target_region.lon_min}, {target_region.lon_max}]")
        print(f"  Intensity: {intensity:.2f}")
        print(f"  Physical lag: {mask_obj.lag_minutes:.1f} min")

        # Generate the mask
        mask_ds = mask_obj.generate_mask(self.raw_dataset)

        # Apply: X_seeded = X_raw + M
        seeded_ds = self.raw_dataset + mask_ds

        # Compute diagnostics
        diagnostics = self._compute_diagnostics(mask_ds, intervention_type)

        print(f"  Perturbation summary:")
        for var, stats in diagnostics["variable_stats"].items():
            print(f"    {var}: min={stats['min']:.6f}, max={stats['max']:.6f}, "
                  f"mean={stats['mean']:.6f}")

        return InterventionResult(
            intervention_type=intervention_type,
            category=mask_obj.category,
            raw_dataset=self.raw_dataset,
            mask_dataset=mask_ds,
            seeded_dataset=seeded_ds,
            diagnostics=diagnostics,
        )

    def apply_all_drought_masks(
        self,
        target_region: TargetRegion,
        intensity: float = 0.5,
        **kwargs,
    ) -> dict[str, "InterventionResult"]:
        """Apply all drought-prevention intervention masks."""
        results = {}
        for name in DROUGHT_MASKS:
            results[name] = self.apply_intervention(
                name, target_region, intensity, **kwargs
            )
        return results

    def apply_all_hurricane_masks(
        self,
        target_region: TargetRegion,
        intensity: float = 0.5,
        **kwargs,
    ) -> dict[str, "InterventionResult"]:
        """Apply all hurricane-disruption intervention masks."""
        results = {}
        for name in HURRICANE_MASKS:
            results[name] = self.apply_intervention(
                name, target_region, intensity, **kwargs
            )
        return results

    def _compute_diagnostics(self, mask_ds: xr.Dataset, intervention_type: str) -> dict:
        """Compute diagnostic statistics for a mask."""
        stats = {}
        for var in mask_ds.data_vars:
            vals = mask_ds[var].values
            if np.any(vals != 0):
                nonzero = vals[vals != 0]
                stats[var] = {
                    "min": float(np.nanmin(nonzero)),
                    "max": float(np.nanmax(nonzero)),
                    "mean": float(np.nanmean(nonzero)),
                    "n_perturbed_cells": int(np.count_nonzero(vals)),
                    "total_cells": int(vals.size),
                }
        return {
            "intervention_type": intervention_type,
            "variable_stats": stats,
        }


@dataclass
class InterventionResult:
    """Container for the result of a geoengineering intervention."""
    intervention_type: str
    category: str
    raw_dataset: xr.Dataset
    mask_dataset: xr.Dataset
    seeded_dataset: xr.Dataset
    diagnostics: dict

    def save(self, output_dir: str):
        """Save the seeded dataset and diagnostics to disk."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        seeded_path = out / f"seeded_{self.intervention_type}.nc"
        mask_path = out / f"mask_{self.intervention_type}.nc"
        diag_path = out / f"diagnostics_{self.intervention_type}.json"

        self.seeded_dataset.to_netcdf(str(seeded_path))
        self.mask_dataset.to_netcdf(str(mask_path))
        with open(diag_path, "w") as f:
            json.dump(self.diagnostics, f, indent=2)

        print(f"Saved: {seeded_path}")
        print(f"Saved: {mask_path}")
        print(f"Saved: {diag_path}")

    def delta_summary(self) -> dict:
        """Compute difference summary between raw and seeded datasets."""
        summary = {}
        for var in self.raw_dataset.data_vars:
            raw = self.raw_dataset[var].values
            seeded = self.seeded_dataset[var].values
            diff = seeded - raw
            if np.any(diff != 0):
                summary[var] = {
                    "max_delta": float(np.nanmax(np.abs(diff))),
                    "mean_delta": float(np.nanmean(diff)),
                    "rms_delta": float(np.sqrt(np.nanmean(diff**2))),
                }
        return summary


# ---------------------------------------------------------------------------
# Visualization Helpers
# ---------------------------------------------------------------------------

def plot_mask_comparison(
    result: InterventionResult,
    variable: str,
    time_idx: int = 0,
    level_idx: Optional[int] = None,
    output_path: Optional[str] = None,
):
    """
    Plot a side-by-side comparison of raw vs seeded state for one variable.

    Args:
        result: InterventionResult from the simulator
        variable: Variable name to plot
        time_idx: Which time step to visualize
        level_idx: Which pressure level index (for 3D variables)
        output_path: If provided, save the figure instead of showing
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        has_cartopy = True
    except ImportError:
        has_cartopy = False

    raw = result.raw_dataset
    seeded = result.seeded_dataset
    mask = result.mask_dataset

    if variable not in raw.data_vars:
        print(f"Variable '{variable}' not in dataset. Available: {list(raw.data_vars)}")
        return

    lats = raw.coords["lat"].values
    lons = raw.coords["lon"].values

    # Extract data slice
    def extract_slice(ds):
        v = ds[variable].values
        # Navigate dimensions: time, batch, level, lat, lon
        if v.ndim >= 2:
            if "time" in ds[variable].dims:
                v = v[time_idx]
            if "batch" in ds[variable].dims:
                v = v[0]
            if level_idx is not None and v.ndim == 3:
                v = v[level_idx]
            elif v.ndim == 3:
                # Average over levels
                v = np.nanmean(v, axis=0)
        return v

    raw_slice = extract_slice(raw)
    seeded_slice = extract_slice(seeded)
    mask_slice = extract_slice(mask)

    if has_cartopy:
        proj = ccrs.Robinson()
        data_crs = ccrs.PlateCarree()
        fig, axes = plt.subplots(1, 3, figsize=(20, 5),
                                  subplot_kw={"projection": proj},
                                  facecolor="#0a0a2e")
    else:
        fig, axes = plt.subplots(1, 3, figsize=(20, 5), facecolor="#0a0a2e")

    titles = ["Original", "Seeded", "Perturbation (Mask)"]
    data = [raw_slice, seeded_slice, mask_slice]

    for ax, title, d in zip(axes, titles, data):
        if has_cartopy:
            ax.set_global()
            ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="#ccc")
            ax.set_facecolor("#0d1b2a")
            im = ax.contourf(lons, lats, d, levels=30, transform=data_crs,
                             cmap="RdYlBu_r" if title != "Perturbation (Mask)" else "coolwarm")
        else:
            im = ax.imshow(d, aspect="auto", cmap="RdYlBu_r" if title != "Perturbation (Mask)" else "coolwarm")
        ax.set_title(title, color="white", fontsize=12, fontweight="bold")
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.046, pad=0.08)
        cb.ax.tick_params(colors="white", labelsize=8)

    fig.suptitle(
        f"{result.intervention_type.replace('_', ' ').title()} — {variable}",
        color="white", fontsize=14, fontweight="bold", y=1.02,
    )
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Figure saved: {output_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_all_interventions_summary(
    results: dict[str, InterventionResult],
    output_path: Optional[str] = None,
):
    """
    Create a summary dashboard showing all intervention masks.

    Args:
        results: Dict of {intervention_name: InterventionResult}
        output_path: If provided, save the figure
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(results)
    fig, axes = plt.subplots(2, max(n, 1), figsize=(6 * n, 10), facecolor="#0a0a2e")
    if n == 1:
        axes = axes.reshape(2, 1)

    for i, (name, result) in enumerate(results.items()):
        mask = result.mask_dataset
        lats = mask.coords["lat"].values
        lons = mask.coords["lon"].values

        # Find the first perturbed variable with 2D data
        plot_var = None
        for var in mask.data_vars:
            vals = mask[var].values
            if np.any(vals != 0):
                plot_var = var
                break

        if plot_var is None:
            continue

        vals = mask[plot_var].values
        # Get a 2D slice
        while vals.ndim > 2:
            vals = vals[0]

        # Top row: mask magnitude
        im1 = axes[0, i].imshow(vals, aspect="auto", cmap="coolwarm",
                                 extent=[lons[0], lons[-1], lats[-1], lats[0]])
        axes[0, i].set_title(f"{name}\n(mask: {plot_var})", color="white", fontsize=10)
        fig.colorbar(im1, ax=axes[0, i], fraction=0.046, pad=0.04)
        axes[0, i].tick_params(colors="white")

        # Bottom row: diagnostics bar chart
        diag = result.diagnostics["variable_stats"]
        vars_perturbed = list(diag.keys())
        max_deltas = [diag[v]["max"] for v in vars_perturbed]

        if vars_perturbed:
            bars = axes[1, i].barh(vars_perturbed, max_deltas, color="#4488cc")
            axes[1, i].set_title("Max perturbation", color="white", fontsize=10)
            axes[1, i].tick_params(colors="white")
            axes[1, i].set_facecolor("#0d1b2a")

    for ax_row in axes:
        for ax in ax_row:
            ax.set_facecolor("#0d1b2a")
            for spine in ax.spines.values():
                spine.set_edgecolor("#444466")

    fig.suptitle("Geoengineering Intervention Masks — Summary",
                 color="white", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Summary figure saved: {output_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Apply geoengineering seeding masks to GraphCast input data"
    )
    parser.add_argument(
        "dataset", help="Path to GraphCast input NetCDF dataset"
    )
    parser.add_argument(
        "--intervention", "-i",
        choices=list(MASK_REGISTRY.keys()) + ["all_drought", "all_hurricane", "all"],
        default="all",
        help="Intervention type to apply",
    )
    parser.add_argument("--lat-min", type=float, default=-90, help="Minimum latitude")
    parser.add_argument("--lat-max", type=float, default=90, help="Maximum latitude")
    parser.add_argument("--lon-min", type=float, default=0, help="Minimum longitude")
    parser.add_argument("--lon-max", type=float, default=360, help="Maximum longitude")
    parser.add_argument(
        "--intensity", type=float, default=0.5,
        help="Perturbation intensity (0.0-1.0)",
    )
    parser.add_argument(
        "--output", "-o", default="./geoengineering_output",
        help="Output directory for seeded datasets",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate comparison plots",
    )

    args = parser.parse_args()

    region = TargetRegion(args.lat_min, args.lat_max, args.lon_min, args.lon_max)

    sim = GeoEngineeringSimulator(dataset_path=args.dataset)

    if args.intervention == "all":
        drought_results = sim.apply_all_drought_masks(region, args.intensity)
        hurricane_results = sim.apply_all_hurricane_masks(region, args.intensity)
        results = {**drought_results, **hurricane_results}
    elif args.intervention == "all_drought":
        results = sim.apply_all_drought_masks(region, args.intensity)
    elif args.intervention == "all_hurricane":
        results = sim.apply_all_hurricane_masks(region, args.intensity)
    else:
        result = sim.apply_intervention(args.intervention, region, args.intensity)
        results = {args.intervention: result}

    # Save results
    for name, result in results.items():
        result.save(args.output)

    # Generate plots if requested
    if args.plot:
        plot_all_interventions_summary(
            results,
            output_path=str(Path(args.output) / "summary.png"),
        )

    # Print summary table
    print("\n" + "=" * 70)
    print("GEOENGINEERING MASK SUMMARY")
    print("=" * 70)
    print(f"{'Intervention':<30} {'Category':<12} {'Vars Perturbed':<8} {'Max |delta|':<12}")
    print("-" * 70)
    for name, result in results.items():
        n_vars = len(result.diagnostics["variable_stats"])
        max_delta = max(
            (s["max"] for s in result.diagnostics["variable_stats"].values()),
            default=0,
        )
        print(f"{name:<30} {result.category:<12} {n_vars:<8} {max_delta:<12.6f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
