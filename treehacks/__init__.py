"""
Treehacks Climate Intervention Package

Morro - AI-Powered Climate Intervention Simulation

This package provides tools for simulating and visualizing climate
intervention strategies using GraphCast weather forecasting.

Modules:
    climate_intervention: Core pipeline for running intervention studies
    visualization: Plotting and video generation utilities

Example:
    from treehacks.climate_intervention import run_hurricane_intervention_study
    from treehacks.visualization import visualize_results

    results = run_hurricane_intervention_study(
        event_name="Hurricane Katrina",
        start_time=datetime(2005, 8, 27, 12, 0),
        nsteps=40
    )
    visualize_results(results)
"""

from .climate_intervention import (
    WeatherSentinel,
    CloudSeedingConfig,
    InterventionArchitect,
    SeededDataSource,
    CycloneDataSource,
    AmplifiedSeededDataSource,
    ForecastPipeline,
    compute_wind_reduction,
    extract_storm_track,
    run_hurricane_intervention_study,
)

from .visualization import (
    plot_wind_field,
    plot_drought_risk,
    plot_intervention_impact,
    plot_storm_tracks,
    plot_intensity_timeseries,
    generate_forecast_video,
    generate_comparison_video,
    visualize_results,
)

__version__ = "0.1.0"
__author__ = "Morro Team - Treehacks 2026"
