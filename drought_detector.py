#!/usr/bin/env python3
"""
ML-Based Drought Detection from GraphCast Predictions

This module analyzes GraphCast weather predictions and classifies drought risk
using a composite Drought Severity Index (DSI) that combines:
- Standardized Precipitation Index (SPI)
- Temperature stress (high temps increase drought risk)
- Humidity deficit (low humidity increases drought risk)

Cold regions (mean temp < 5°C) are excluded from drought classification
since drought is not a meaningful concept for frozen/polar regions.

Usage:
    python drought_detector.py predictions.nc --output drought_results.json
"""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import shapely.geometry as sgeom
import cartopy.io.shapereader as shpreader


# Drought class definitions based on composite DSI thresholds
DROUGHT_CLASSES = {
    0: "none",
    1: "abnormally_dry",
    2: "moderate",
    3: "severe",
    4: "extreme",
}

# Composite Drought Severity Index (DSI) thresholds
# DSI combines precipitation deficit, heat stress, and humidity deficit
# Tightened extreme threshold to -2.5 (was -2.0) to reduce false positives
DSI_THRESHOLDS = {
    "none": (-0.5, float("inf")),
    "abnormally_dry": (-1.0, -0.5),
    "moderate": (-1.5, -1.0),
    "severe": (-2.5, -1.5),
    "extreme": (float("-inf"), -2.5),
}

# Temperature thresholds (Kelvin)
TEMP_MIN_FOR_DROUGHT = 278.15  # 5°C - below this, no drought classification
TEMP_HEAT_STRESS_START = 298.15  # 25°C - heat stress begins
TEMP_HEAT_STRESS_MAX = 318.15  # 45°C - maximum heat stress

# Humidity thresholds (specific humidity kg/kg)
HUMIDITY_NORMAL = 0.010  # ~10 g/kg - typical mid-latitude value
HUMIDITY_DRY = 0.005  # ~5 g/kg - dry conditions
HUMIDITY_VERY_DRY = 0.002  # ~2 g/kg - very dry/arid conditions

# Aridity classification based on expected precipitation (mm per 10-day period)
# These thresholds help identify "unnatural" drought vs naturally dry regions
ARIDITY_THRESHOLDS = {
    "hyper_arid": 2.0,      # < 2mm/10-day: Desert (Sahara, Arabian) - drought not meaningful
    "arid": 5.0,            # 2-5mm/10-day: Semi-desert - drought less significant
    "semi_arid": 15.0,      # 5-15mm/10-day: Steppe/savanna - drought noteworthy
    "sub_humid": 30.0,      # 15-30mm/10-day: Mediterranean/seasonal - drought significant
    "humid": float("inf"),  # > 30mm/10-day: Humid regions - drought very significant
}

# Anomaly significance weights by aridity class
# Higher weight = more significant when drought occurs there
ARIDITY_WEIGHTS = {
    "hyper_arid": 0.0,   # Ignore - always dry
    "arid": 0.3,         # Low significance
    "semi_arid": 0.7,    # Moderate significance
    "sub_humid": 1.0,    # High significance
    "humid": 1.2,        # Very high significance (unusual drought)
}

# Legacy SPI thresholds (kept for reference)
SPI_THRESHOLDS = {
    "none": (-0.5, float("inf")),
    "abnormally_dry": (-1.0, -0.5),
    "moderate": (-1.5, -1.0),
    "severe": (-2.0, -1.5),
    "extreme": (float("-inf"), -2.0),
}


class Climatology:
    """
    Handles ERA5-based climatology data for SPI calculations.

    Uses synthetic climatology when real ERA5 data is not available,
    based on typical global precipitation patterns.
    """

    def __init__(self, climatology_path: Optional[str] = None):
        """
        Initialize climatology data.

        Args:
            climatology_path: Path to ERA5 climatology NetCDF file.
                            If None, uses synthetic climatology.
        """
        self.climatology_path = climatology_path
        self._data = None
        self._load_climatology()

    def _load_climatology(self):
        """Load or generate climatology data."""
        if self.climatology_path and Path(self.climatology_path).exists():
            self._data = xr.open_dataset(self.climatology_path)
        else:
            # Generate synthetic climatology based on typical patterns
            # In production, this should be replaced with actual ERA5 data
            self._generate_synthetic_climatology()

    def _generate_synthetic_climatology(self):
        """
        Generate synthetic climatology for demonstration.

        This creates realistic precipitation statistics based on:
        - Latitude (tropics are wetter)
        - Season (summer typically wetter in mid-latitudes)
        """
        lats = np.arange(-90, 91, 1.0)
        lons = np.arange(0, 360, 1.0)
        months = np.arange(1, 13)

        # Create meshgrid for calculations
        lat_grid, month_grid = np.meshgrid(lats, months, indexing='ij')

        # Base precipitation pattern (mm/day)
        # Higher in tropics, lower at poles
        base_precip = 5.0 * np.exp(-((lat_grid) ** 2) / (2 * 30**2))

        # Add seasonal variation (simplified)
        seasonal_factor = 1.0 + 0.3 * np.sin(2 * np.pi * (month_grid - 4) / 12)

        # Mean precipitation (mm per 10-day period)
        mean_precip = base_precip * seasonal_factor * 10

        # Standard deviation (typically ~50% of mean for precipitation)
        std_precip = mean_precip * 0.5
        std_precip = np.maximum(std_precip, 1.0)  # Minimum std of 1mm

        self._data = xr.Dataset({
            "precip_mean": (["lat", "month"], mean_precip),
            "precip_std": (["lat", "month"], std_precip),
        }, coords={
            "lat": lats,
            "month": months,
        })

    def get_statistics(self, lat: float, lon: float, month: int) -> tuple:
        """
        Get precipitation statistics for a location and month.

        Args:
            lat: Latitude
            lon: Longitude
            month: Month (1-12)

        Returns:
            Tuple of (mean, std) precipitation in mm/10-day period
        """
        # Find nearest latitude
        lat_idx = np.abs(self._data.lat.values - lat).argmin()

        mean = float(self._data.precip_mean.isel(lat=lat_idx, month=month-1).values)
        std = float(self._data.precip_std.isel(lat=lat_idx, month=month-1).values)

        return mean, std


class LandMask:
    """
    Land/ocean mask using Natural Earth data via cartopy.

    Filters out ocean grid cells where drought is not meaningful.
    """

    def __init__(self):
        """Load land geometries from Natural Earth."""
        self._land_geoms = None
        self._load_land_mask()

    def _load_land_mask(self):
        """Load land polygons from Natural Earth."""
        try:
            land_shp = shpreader.natural_earth(
                resolution='110m', category='physical', name='land'
            )
            self._land_geoms = list(shpreader.Reader(land_shp).geometries())
            print(f"  Land mask loaded: {len(self._land_geoms)} polygons")
        except Exception as e:
            print(f"  Warning: Could not load land mask: {e}")
            self._land_geoms = None

    def is_land(self, lat: float, lon: float) -> bool:
        """
        Check if a coordinate is on land.

        Args:
            lat: Latitude (-90 to 90)
            lon: Longitude (0-360 or -180 to 180)

        Returns:
            True if on land, False if ocean
        """
        if self._land_geoms is None:
            return True  # Assume land if mask unavailable

        # Normalize longitude to -180 to 180 for shapely
        if lon > 180:
            lon = lon - 360

        point = sgeom.Point(lon, lat)
        for geom in self._land_geoms:
            if geom.contains(point):
                return True
        return False


class DroughtDetector:
    """
    ML-based drought detection from GraphCast predictions.

    Uses a Random Forest classifier trained on SPI-derived labels
    to predict drought risk from weather forecast features.
    """

    def __init__(self, climatology_path: Optional[str] = None, model_path: Optional[str] = None):
        """
        Initialize the drought detector.

        Args:
            climatology_path: Path to ERA5 climatology data (optional)
            model_path: Path to pre-trained model (optional)
        """
        self.climatology = Climatology(climatology_path)
        self.land_mask = LandMask()
        self.model: Optional[RandomForestClassifier] = None
        self.feature_names = [
            "precip_total",
            "precip_mean",
            "precip_variance",
            "temp_max",
            "temp_mean",
            "humidity_mean",
            "humidity_trend",
            "pressure_mean",
        ]

        if model_path and Path(model_path).exists():
            self.load_model(model_path)

    def extract_features(self, predictions: xr.Dataset) -> tuple:
        """
        Extract ML features from GraphCast predictions.

        Args:
            predictions: xarray Dataset with GraphCast predictions

        Returns:
            Tuple of (features array, lat coords, lon coords)
        """
        # Get coordinates
        lats = predictions.lat.values
        lons = predictions.lon.values
        n_lats = len(lats)
        n_lons = len(lons)

        # Extract variables (handle batch dimension if present)
        precip = predictions["total_precipitation_6hr"].values
        if precip.ndim == 4:  # (time, batch, lat, lon)
            precip = precip[:, 0, :, :]  # Take first batch

        temp_2m = predictions["2m_temperature"].values
        if temp_2m.ndim == 4:
            temp_2m = temp_2m[:, 0, :, :]

        # Specific humidity at surface (1000 hPa level, index -1)
        humidity = predictions["specific_humidity"].values
        if humidity.ndim == 5:  # (time, batch, level, lat, lon)
            humidity = humidity[:, 0, -1, :, :]  # Surface level
        elif humidity.ndim == 4:  # (time, level, lat, lon)
            humidity = humidity[:, -1, :, :]

        pressure = predictions["mean_sea_level_pressure"].values
        if pressure.ndim == 4:
            pressure = pressure[:, 0, :, :]

        n_times = precip.shape[0]

        # Initialize feature array: (n_cells, n_features)
        n_cells = n_lats * n_lons
        features = np.zeros((n_cells, len(self.feature_names)), dtype=np.float32)

        # Calculate features for each grid cell
        idx = 0
        for i in range(n_lats):
            for j in range(n_lons):
                # Precipitation features
                precip_series = precip[:, i, j]
                features[idx, 0] = np.sum(precip_series)  # precip_total
                features[idx, 1] = np.mean(precip_series)  # precip_mean
                features[idx, 2] = np.var(precip_series)  # precip_variance

                # Temperature features
                temp_series = temp_2m[:, i, j]
                features[idx, 3] = np.max(temp_series)  # temp_max
                features[idx, 4] = np.mean(temp_series)  # temp_mean

                # Humidity features
                humidity_series = humidity[:, i, j]
                features[idx, 5] = np.mean(humidity_series)  # humidity_mean

                # Humidity trend (linear regression slope)
                if n_times > 1:
                    x = np.arange(n_times)
                    slope = np.polyfit(x, humidity_series, 1)[0]
                    features[idx, 6] = slope  # humidity_trend
                else:
                    features[idx, 6] = 0.0

                # Pressure features
                pressure_series = pressure[:, i, j]
                features[idx, 7] = np.mean(pressure_series)  # pressure_mean

                idx += 1

        # Handle NaN/Inf values
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        return features, lats, lons

    def calculate_spi(self, precip: float, lat: float, lon: float, month: int) -> float:
        """
        Calculate Standardized Precipitation Index (SPI).

        Args:
            precip: Observed precipitation in meters (SI units from GraphCast)
            lat: Latitude
            lon: Longitude
            month: Month (1-12)

        Returns:
            SPI value
        """
        # Convert precipitation from meters to mm (climatology uses mm)
        precip_mm = precip * 1000.0

        mean, std = self.climatology.get_statistics(lat, lon, month)

        if std == 0:
            return 0.0

        spi = (precip_mm - mean) / std
        return spi

    def spi_to_class(self, spi: float) -> int:
        """Convert SPI value to drought class."""
        if spi > -0.5:
            return 0  # None
        elif spi > -1.0:
            return 1  # Abnormally dry
        elif spi > -1.5:
            return 2  # Moderate
        elif spi > -2.0:
            return 3  # Severe
        else:
            return 4  # Extreme

    def classify_aridity(self, lat: float, lon: float, month: int) -> tuple:
        """
        Classify a region's aridity based on climatological precipitation.

        Args:
            lat: Latitude
            lon: Longitude
            month: Month (1-12)

        Returns:
            Tuple of (aridity_class, expected_precip_mm, significance_weight)
        """
        mean_precip, _ = self.climatology.get_statistics(lat, lon, month)

        if mean_precip < ARIDITY_THRESHOLDS["hyper_arid"]:
            aridity = "hyper_arid"
        elif mean_precip < ARIDITY_THRESHOLDS["arid"]:
            aridity = "arid"
        elif mean_precip < ARIDITY_THRESHOLDS["semi_arid"]:
            aridity = "semi_arid"
        elif mean_precip < ARIDITY_THRESHOLDS["sub_humid"]:
            aridity = "sub_humid"
        else:
            aridity = "humid"

        weight = ARIDITY_WEIGHTS[aridity]
        return aridity, mean_precip, weight

    def calculate_anomaly_significance(
        self,
        precip_mm: float,
        lat: float,
        lon: float,
        month: int
    ) -> tuple:
        """
        Calculate how significant a precipitation anomaly is for this region.

        Drought in normally wet regions is more significant than in deserts.

        Args:
            precip_mm: Observed precipitation in mm
            lat: Latitude
            lon: Longitude
            month: Month (1-12)

        Returns:
            Tuple of (significance_score, aridity_class, expected_precip_mm, is_natural_desert)
        """
        aridity, expected_precip, weight = self.classify_aridity(lat, lon, month)

        # Calculate deficit as percentage of expected
        if expected_precip > 0:
            deficit_pct = (expected_precip - precip_mm) / expected_precip
        else:
            deficit_pct = 0.0

        # Significance score combines deficit magnitude with region importance
        # Higher score = more significant drought event
        significance = deficit_pct * weight

        is_natural_desert = aridity in ("hyper_arid", "arid")

        return significance, aridity, expected_precip, is_natural_desert

    def calculate_heat_stress(self, temp_mean: float, temp_max: float) -> float:
        """
        Calculate heat stress factor based on temperature.

        Heat stress increases drought severity when temperatures are high,
        as evapotranspiration increases and water demand rises.

        Args:
            temp_mean: Mean temperature in Kelvin
            temp_max: Maximum temperature in Kelvin

        Returns:
            Heat stress factor: 0.0 (no stress) to 1.0 (max stress)
        """
        # Use weighted average of mean and max temp
        effective_temp = 0.6 * temp_mean + 0.4 * temp_max

        if effective_temp < TEMP_HEAT_STRESS_START:
            return 0.0
        elif effective_temp > TEMP_HEAT_STRESS_MAX:
            return 1.0
        else:
            # Linear interpolation
            return (effective_temp - TEMP_HEAT_STRESS_START) / (
                TEMP_HEAT_STRESS_MAX - TEMP_HEAT_STRESS_START
            )

    def calculate_humidity_stress(self, humidity_mean: float) -> float:
        """
        Calculate humidity deficit stress factor.

        Low atmospheric humidity indicates dry conditions and increases
        evaporative demand, worsening drought conditions.

        Args:
            humidity_mean: Mean specific humidity (kg/kg)

        Returns:
            Humidity stress factor: 0.0 (no stress) to 1.0 (max stress)
        """
        if humidity_mean >= HUMIDITY_NORMAL:
            return 0.0
        elif humidity_mean <= HUMIDITY_VERY_DRY:
            return 1.0
        else:
            # Linear interpolation (inverted - lower humidity = higher stress)
            return (HUMIDITY_NORMAL - humidity_mean) / (HUMIDITY_NORMAL - HUMIDITY_VERY_DRY)

    def calculate_dsi(
        self,
        spi: float,
        temp_mean: float,
        temp_max: float,
        humidity_mean: float,
        lat: float = 0.0,
        lon: float = 0.0,
        month: int = 1,
        precip_mm: float = 0.0,
    ) -> tuple:
        """
        Calculate composite Drought Severity Index (DSI).

        Combines precipitation deficit (SPI) with temperature and humidity
        stress factors, weighted by the region's normal aridity.
        Drought in normally wet regions is considered more significant.

        Args:
            spi: Standardized Precipitation Index
            temp_mean: Mean temperature (K)
            temp_max: Max temperature (K)
            humidity_mean: Mean specific humidity (kg/kg)
            lat: Latitude (for aridity calculation)
            lon: Longitude (for aridity calculation)
            month: Month 1-12 (for aridity calculation)
            precip_mm: Observed precipitation in mm (for aridity calculation)

        Returns:
            Tuple of (DSI value, component dict with breakdown)
        """
        # Check if region is too cold for drought assessment
        if temp_mean < TEMP_MIN_FOR_DROUGHT:
            # Cold regions: no drought (return neutral DSI)
            return 0.0, {
                "spi": spi,
                "heat_stress": 0.0,
                "humidity_stress": 0.0,
                "aridity_class": "cold",
                "aridity_weight": 0.0,
                "expected_precip_mm": 0.0,
                "is_natural_desert": False,
                "cold_region": True,
            }

        # Calculate stress factors
        heat_stress = self.calculate_heat_stress(temp_mean, temp_max)
        humidity_stress = self.calculate_humidity_stress(humidity_mean)

        # Calculate anomaly significance based on aridity
        significance, aridity_class, expected_precip, is_natural_desert = \
            self.calculate_anomaly_significance(precip_mm, lat, lon, month)

        aridity_weight = ARIDITY_WEIGHTS[aridity_class]

        # Composite DSI formula:
        # - Start with SPI (precipitation deficit)
        # - Amplify negative SPI when heat/humidity stress is present
        # - Weight by aridity significance (desert drought is less meaningful)
        #
        # Only apply stress multipliers when SPI indicates dry conditions (< 0)
        if spi < 0:
            # Stress amplification (more negative = worse drought)
            heat_amplification = 1.0 + (0.5 * heat_stress)
            humidity_amplification = 1.0 + (0.3 * humidity_stress)
            base_dsi = spi * heat_amplification * humidity_amplification

            # Apply aridity weighting - reduce severity for natural deserts
            # This makes drought in humid regions more prominent
            dsi = base_dsi * aridity_weight
        else:
            # When precipitation is adequate, stress factors have minimal impact
            combined_stress = 0.4 * heat_stress + 0.3 * humidity_stress
            dsi = (spi - combined_stress) * aridity_weight

        components = {
            "spi": round(spi, 3),
            "heat_stress": round(heat_stress, 3),
            "humidity_stress": round(humidity_stress, 3),
            "aridity_class": aridity_class,
            "aridity_weight": round(aridity_weight, 2),
            "expected_precip_mm": round(expected_precip, 1),
            "is_natural_desert": is_natural_desert,
            "cold_region": False,
        }

        return dsi, components

    def dsi_to_class(self, dsi: float) -> int:
        """Convert Drought Severity Index to drought class."""
        if dsi > -0.5:
            return 0  # None
        elif dsi > -1.0:
            return 1  # Abnormally dry
        elif dsi > -1.5:
            return 2  # Moderate
        elif dsi > -2.5:
            return 3  # Severe
        else:
            return 4  # Extreme (tightened threshold: < -2.5)

    def generate_training_data(
        self,
        predictions: xr.Dataset,
        forecast_date: Optional[datetime] = None
    ) -> tuple:
        """
        Generate training data with DSI-derived labels.

        Uses the composite Drought Severity Index (DSI) which combines
        precipitation, temperature, and humidity to classify drought.

        Args:
            predictions: xarray Dataset with predictions
            forecast_date: Date of forecast start (for seasonal context)

        Returns:
            Tuple of (features, labels)
        """
        if forecast_date is None:
            forecast_date = datetime.now()

        month = forecast_date.month

        features, lats, lons = self.extract_features(predictions)

        # Generate labels based on composite DSI
        n_cells = len(features)
        labels = np.zeros(n_cells, dtype=np.int32)

        idx = 0
        for lat in lats:
            for lon in lons:
                # Extract features for this cell
                precip_total = features[idx, 0]  # Total precipitation (meters)
                precip_mm = precip_total * 1000.0  # Convert to mm
                temp_max = features[idx, 3]  # Max temperature (K)
                temp_mean = features[idx, 4]  # Mean temperature (K)
                humidity_mean = features[idx, 5]  # Mean specific humidity

                # Calculate SPI (precipitation component)
                spi = self.calculate_spi(precip_total, lat, lon, month)

                # Calculate composite DSI with aridity weighting
                dsi, _ = self.calculate_dsi(
                    spi, temp_mean, temp_max, humidity_mean,
                    lat=lat, lon=lon, month=month, precip_mm=precip_mm
                )

                # Convert to class
                labels[idx] = self.dsi_to_class(dsi)
                idx += 1

        return features, labels

    def train(
        self,
        training_data_path: Optional[str] = None,
        predictions: Optional[xr.Dataset] = None,
        forecast_date: Optional[datetime] = None,
        test_size: float = 0.2,
        n_estimators: int = 100,
        random_state: int = 42,
    ) -> dict:
        """
        Train the drought detection model.

        Args:
            training_data_path: Path to training data file
            predictions: Alternatively, provide predictions directly
            forecast_date: Date context for SPI calculation
            test_size: Fraction of data for testing
            n_estimators: Number of trees in Random Forest
            random_state: Random seed for reproducibility

        Returns:
            Training metrics dictionary
        """
        if training_data_path and Path(training_data_path).exists():
            predictions = xr.open_dataset(training_data_path)

        if predictions is None:
            raise ValueError("Must provide either training_data_path or predictions")

        # Generate features and labels
        features, labels = self.generate_training_data(predictions, forecast_date)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=random_state, stratify=labels
        )

        # Train model
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
        )

        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test)

        # Get feature importances
        importances = dict(zip(self.feature_names, self.model.feature_importances_))

        metrics = {
            "accuracy": float(np.mean(y_pred == y_test)),
            "feature_importances": importances,
            "class_distribution": {
                DROUGHT_CLASSES[i]: int(np.sum(labels == i))
                for i in range(5)
            },
            "n_samples": len(labels),
        }

        return metrics

    def predict(self, predictions_path: str, forecast_date: Optional[datetime] = None) -> dict:
        """
        Run inference on new predictions.

        Args:
            predictions_path: Path to predictions NetCDF file
            forecast_date: Forecast start date (optional)

        Returns:
            Results dictionary with drought detections
        """
        # Load predictions
        predictions = xr.open_dataset(predictions_path)

        # If no trained model, train on this data
        if self.model is None:
            print("No pre-trained model found. Training on current data...")
            metrics = self.train(predictions=predictions, forecast_date=forecast_date)
            print(f"Training complete. Accuracy: {metrics['accuracy']:.3f}")

        # Extract features
        features, lats, lons = self.extract_features(predictions)

        # Predict
        predictions_classes = self.model.predict(features)
        predictions_proba = self.model.predict_proba(features)

        # Determine forecast period
        if forecast_date is None:
            forecast_date = datetime.now()

        month = forecast_date.month

        times = predictions.time.values
        n_steps = len(times)
        # Assume 6-hour steps
        forecast_hours = n_steps * 6
        forecast_end = forecast_date + timedelta(hours=forecast_hours)

        # Build results
        detections = []
        drought_counts = {label: 0 for label in DROUGHT_CLASSES.values()}
        cold_region_count = 0
        natural_desert_count = 0
        ocean_count = 0
        aridity_counts = {
            "hyper_arid": 0, "arid": 0, "semi_arid": 0, "sub_humid": 0, "humid": 0
        }

        idx = 0
        for lat in lats:
            for lon in lons:
                # Skip ocean cells - drought not meaningful over water
                if not self.land_mask.is_land(lat, lon):
                    ocean_count += 1
                    idx += 1
                    continue

                drought_class = int(predictions_classes[idx])
                drought_label = DROUGHT_CLASSES[drought_class]

                # Extract features for DSI calculation
                precip_total = features[idx, 0]
                precip_mm = precip_total * 1000.0
                temp_max = features[idx, 3]
                temp_mean = features[idx, 4]
                humidity_mean = features[idx, 5]

                # Calculate DSI components for context (with aridity)
                spi = self.calculate_spi(precip_total, lat, lon, month)
                dsi, dsi_components = self.calculate_dsi(
                    spi, temp_mean, temp_max, humidity_mean,
                    lat=lat, lon=lon, month=month, precip_mm=precip_mm
                )

                # Track cold regions and natural deserts
                if dsi_components.get("cold_region", False):
                    cold_region_count += 1
                elif dsi_components.get("is_natural_desert", False):
                    natural_desert_count += 1

                # Track aridity distribution
                aridity_class = dsi_components.get("aridity_class", "unknown")
                if aridity_class in aridity_counts:
                    aridity_counts[aridity_class] += 1

                drought_counts[drought_label] += 1

                # Only include non-zero drought detections in output
                # Skip natural deserts (hyper_arid/arid) - drought not meaningful there
                is_significant = not dsi_components.get("is_natural_desert", False)

                if drought_class > 0 and is_significant:
                    prob_scores = predictions_proba[idx]
                    # Map drought_class to index in model's classes
                    model_classes = list(self.model.classes_)
                    if drought_class in model_classes:
                        class_idx = model_classes.index(drought_class)
                        max_prob = float(prob_scores[class_idx])
                    else:
                        # Class not seen in training - use max probability
                        max_prob = float(np.max(prob_scores))

                    # Convert temperature to Celsius for readability
                    temp_mean_c = temp_mean - 273.15
                    temp_max_c = temp_max - 273.15

                    detection = {
                        "lat": float(lat),
                        "lon": float(lon),
                        "drought_class": int(drought_class),
                        "drought_label": drought_label,
                        "probability": round(float(max_prob), 3),
                        "dsi": round(float(dsi), 3),
                        "dsi_components": {
                            "spi": float(dsi_components["spi"]),
                            "heat_stress": float(dsi_components["heat_stress"]),
                            "humidity_stress": float(dsi_components["humidity_stress"]),
                        },
                        "aridity": {
                            "class": dsi_components.get("aridity_class", "unknown"),
                            "expected_precip_mm": float(dsi_components.get("expected_precip_mm", 0)),
                            "significance_weight": float(dsi_components.get("aridity_weight", 0)),
                        },
                        "features": {
                            "precip_total_mm": round(float(precip_total) * 1000, 2),
                            "precip_mean_mm": round(float(features[idx, 1]) * 1000, 4),
                            "temp_max_C": round(float(temp_max_c), 1),
                            "temp_mean_C": round(float(temp_mean_c), 1),
                            "humidity_mean_g_kg": round(float(humidity_mean) * 1000, 2),
                        },
                    }
                    detections.append(detection)

                idx += 1

        # Sort detections by severity, then by aridity significance (humid regions first)
        detections.sort(key=lambda x: (
            -x["drought_class"],
            -x["aridity"]["significance_weight"],
            -x["probability"]
        ))

        results = {
            "forecast_period": {
                "start": forecast_date.isoformat() + "Z",
                "end": forecast_end.isoformat() + "Z",
            },
            "model_info": {
                "type": "RandomForest",
                "trained_on": "Composite DSI (precip + temp + humidity + aridity)",
                "n_estimators": self.model.n_estimators,
                "features": self.feature_names,
                "dsi_thresholds": {
                    "temp_min_for_drought_C": TEMP_MIN_FOR_DROUGHT - 273.15,
                    "heat_stress_start_C": TEMP_HEAT_STRESS_START - 273.15,
                    "heat_stress_max_C": TEMP_HEAT_STRESS_MAX - 273.15,
                },
                "aridity_thresholds_mm_10day": ARIDITY_THRESHOLDS,
            },
            "detections": detections,
            "summary": {
                "total_cells": len(features),
                "excluded_cells": {
                    "ocean": ocean_count,
                    "cold_regions": cold_region_count,
                    "natural_deserts": natural_desert_count,
                },
                "aridity_distribution": aridity_counts,
                "drought_cells": {
                    "extreme": drought_counts["extreme"],
                    "severe": drought_counts["severe"],
                    "moderate": drought_counts["moderate"],
                    "abnormally_dry": drought_counts["abnormally_dry"],
                },
                "no_drought_cells": drought_counts["none"],
                "significant_detections": len(detections),
            },
        }

        predictions.close()
        return results

    def save_results(self, results: dict, output_path: str):
        """
        Save results to JSON file.

        Args:
            results: Results dictionary
            output_path: Output file path
        """
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_path}")

    def save_model(self, model_path: str):
        """Save trained model to file."""
        if self.model is None:
            raise ValueError("No model to save")
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")

    def load_model(self, model_path: str):
        """Load pre-trained model from file."""
        self.model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")


def main():
    """Main entry point for CLI usage."""
    parser = argparse.ArgumentParser(
        description="ML-based drought detection from GraphCast predictions"
    )
    parser.add_argument(
        "predictions",
        help="Path to predictions NetCDF file",
    )
    parser.add_argument(
        "--output", "-o",
        default="drought_results.json",
        help="Output JSON file path (default: drought_results.json)",
    )
    parser.add_argument(
        "--model", "-m",
        help="Path to pre-trained model file",
    )
    parser.add_argument(
        "--climatology", "-c",
        help="Path to ERA5 climatology file",
    )
    parser.add_argument(
        "--save-model",
        help="Save trained model to this path",
    )
    parser.add_argument(
        "--date", "-d",
        help="Forecast start date (YYYY-MM-DD format)",
    )

    args = parser.parse_args()

    # Parse date if provided
    forecast_date = None
    if args.date:
        forecast_date = datetime.fromisoformat(args.date)

    # Initialize detector
    detector = DroughtDetector(
        climatology_path=args.climatology,
        model_path=args.model,
    )

    # Run prediction
    print(f"Processing predictions from {args.predictions}...")
    results = detector.predict(args.predictions, forecast_date=forecast_date)

    # Print summary
    print("\n" + "=" * 50)
    print("DROUGHT DETECTION SUMMARY")
    print("=" * 50)
    print(f"Forecast Period: {results['forecast_period']['start']} to {results['forecast_period']['end']}")
    print(f"Total Grid Cells: {results['summary']['total_cells']}")
    print("\nDrought Detections:")
    for severity in ["extreme", "severe", "moderate", "abnormally_dry"]:
        count = results["summary"]["drought_cells"][severity]
        if count > 0:
            print(f"  {severity.upper()}: {count} cells")
    print(f"  No Drought: {results['summary']['no_drought_cells']} cells")
    print("=" * 50)

    # Save results
    detector.save_results(results, args.output)

    # Save model if requested
    if args.save_model:
        detector.save_model(args.save_model)

    return results


if __name__ == "__main__":
    main()
