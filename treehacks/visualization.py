"""
Visualization utilities for Climate Intervention Pipeline.

Provides functions for:
- Plotting forecast data on maps
- Generating comparison videos
- Visualizing intervention impact
- Storm track plotting
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors

# Optional cartopy import (for map projections)
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("Warning: cartopy not installed. Map features will be limited.")

from earth2studio.io import ZarrBackend


# =============================================================================
# MAP PLOTTING
# =============================================================================

def plot_wind_field(
    data_path: str,
    timestep: int = -1,
    region: dict = None,
    title: str = "Wind Speed Field",
    output_path: str = None,
    vmin: float = 0,
    vmax: float = 50
):
    """
    Plot wind speed field from forecast data.

    Args:
        data_path: Path to Zarr forecast store
        timestep: Time index to plot (-1 for last)
        region: Dict with lon_min, lon_max, lat_min, lat_max (optional)
        title: Plot title
        output_path: Path to save figure (optional)
        vmin, vmax: Color scale limits
    """
    store = ZarrBackend(data_path)

    lat = store.root["lat"][:]
    lon = store.root["lon"][:]
    u = store.root["u10m"][0, timestep, :, :]
    v = store.root["v10m"][0, timestep, :, :]

    wind_speed = np.sqrt(u**2 + v**2)

    # Apply region subset if specified
    if region:
        lat_mask = (lat >= region['lat_min']) & (lat <= region['lat_max'])
        lon_mask = (lon >= region['lon_min']) & (lon <= region['lon_max'])
        lat = lat[lat_mask]
        lon = lon[lon_mask]
        wind_speed = wind_speed[np.ix_(lat_mask, lon_mask)]

    # Create figure
    if HAS_CARTOPY:
        fig = plt.figure(figsize=(12, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())

        if region:
            ax.set_extent([region['lon_min'], region['lon_max'],
                          region['lat_min'], region['lat_max']], crs=ccrs.PlateCarree())

        ax.coastlines(resolution='50m', color='white', linewidth=1)
        ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='white')

        mesh = ax.pcolormesh(lon, lat, wind_speed,
                            transform=ccrs.PlateCarree(),
                            cmap='viridis', vmin=vmin, vmax=vmax)
    else:
        fig, ax = plt.subplots(figsize=(12, 8))
        mesh = ax.pcolormesh(lon, lat, wind_speed, cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    cbar = plt.colorbar(mesh, orientation='horizontal', pad=0.05, aspect=50)
    cbar.set_label('Wind Speed (m/s)')
    ax.set_title(title)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    plt.show()
    return fig


def plot_drought_risk(
    data_path: str,
    timestep: int = -1,
    humidity_threshold: float = 0.005,
    title: str = "Drought Risk Analysis",
    output_path: str = None
):
    """
    Plot drought risk based on specific humidity.

    Args:
        data_path: Path to Zarr forecast store
        timestep: Time index to plot
        humidity_threshold: Threshold for drought classification (kg/kg)
        title: Plot title
        output_path: Path to save figure
    """
    store = ZarrBackend(data_path)

    lat = store.root["lat"][:]
    lon = store.root["lon"][:]
    q_700 = store.root["q700"][0, timestep, :, :]

    fig = plt.figure(figsize=(12, 6))

    # Colormap: brown is dry, blue is wet
    cmap = plt.cm.BrBG
    norm = mcolors.Normalize(vmin=0, vmax=0.015)

    plt.pcolormesh(lon, lat, q_700, cmap=cmap, norm=norm, shading='auto')
    plt.colorbar(label='Specific Humidity @ 700hPa (kg/kg)')

    # Overlay drought mask
    drought_mask = q_700 < humidity_threshold
    overlay = np.zeros((*drought_mask.shape, 4))
    overlay[drought_mask] = [1, 0, 0, 0.3]  # Red with transparency

    plt.imshow(overlay, extent=[lon.min(), lon.max(), lat.min(), lat.max()],
               origin='lower', aspect='auto')

    plt.title(f"{title} (Red = q700 < {humidity_threshold} kg/kg)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    plt.show()
    return fig


def plot_intervention_impact(
    control_path: str,
    seeded_path: str,
    timestep: int = -1,
    region: dict = None,
    title: str = "Intervention Impact: Wind Speed Change",
    output_path: str = None
):
    """
    Plot the difference in wind speed between control and seeded forecasts.

    Args:
        control_path: Path to control forecast Zarr store
        seeded_path: Path to seeded forecast Zarr store
        timestep: Time index to plot
        region: Dict with bounds (optional)
        title: Plot title
        output_path: Path to save figure
    """
    store_ctrl = ZarrBackend(control_path)
    store_seed = ZarrBackend(seeded_path)

    lat = store_ctrl.root["lat"][:]
    lon = store_ctrl.root["lon"][:]

    # Calculate wind speeds
    u_ctrl = store_ctrl.root["u10m"][0, timestep, :, :]
    v_ctrl = store_ctrl.root["v10m"][0, timestep, :, :]
    u_seed = store_seed.root["u10m"][0, timestep, :, :]
    v_seed = store_seed.root["v10m"][0, timestep, :, :]

    wind_ctrl = np.sqrt(u_ctrl**2 + v_ctrl**2)
    wind_seed = np.sqrt(u_seed**2 + v_seed**2)
    wind_diff = wind_seed - wind_ctrl  # Negative = reduction

    # Apply region subset
    if region:
        lat_mask = (lat >= region['lat_min']) & (lat <= region['lat_max'])
        lon_mask = (lon >= region['lon_min']) & (lon <= region['lon_max'])
        lat = lat[lat_mask]
        lon = lon[lon_mask]
        wind_diff = wind_diff[np.ix_(lat_mask, lon_mask)]
        wind_ctrl = wind_ctrl[np.ix_(lat_mask, lon_mask)]

    # Create figure
    if HAS_CARTOPY:
        fig = plt.figure(figsize=(12, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())

        if region:
            ax.set_extent([region['lon_min'], region['lon_max'],
                          region['lat_min'], region['lat_max']], crs=ccrs.PlateCarree())

        ax.coastlines(resolution='50m', color='black')
        ax.add_feature(cfeature.BORDERS, linestyle=':')

        # Diverging colormap centered at 0
        limit = max(abs(np.min(wind_diff)), abs(np.max(wind_diff)))
        if limit < 0.1:
            limit = 5  # Minimum range

        mesh = ax.pcolormesh(lon, lat, wind_diff,
                            transform=ccrs.PlateCarree(),
                            cmap='RdBu_r', vmin=-limit, vmax=limit)

        # Overlay storm contours
        ax.contour(lon, lat, wind_ctrl, levels=[33, 50], colors='k',
                  linewidths=1, transform=ccrs.PlateCarree())
    else:
        fig, ax = plt.subplots(figsize=(12, 8))
        limit = max(abs(np.min(wind_diff)), abs(np.max(wind_diff)))
        if limit < 0.1:
            limit = 5
        mesh = ax.pcolormesh(lon, lat, wind_diff, cmap='RdBu_r', vmin=-limit, vmax=limit)
        ax.contour(lon, lat, wind_ctrl, levels=[33, 50], colors='k', linewidths=1)

    cbar = plt.colorbar(mesh, label='Wind Speed Change (m/s)\n(Blue = Reduction)', shrink=0.8)
    ax.set_title(title)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    plt.show()
    return fig


def plot_storm_tracks(
    control_track: dict,
    seeded_track: dict,
    region: dict = None,
    title: str = "Storm Track Comparison",
    output_path: str = None
):
    """
    Plot comparison of storm tracks between control and seeded forecasts.

    Args:
        control_track: Dict with 'lat' and 'lon' arrays
        seeded_track: Dict with 'lat' and 'lon' arrays
        region: Dict with bounds (optional)
        title: Plot title
        output_path: Path to save figure
    """
    ctrl_lats = np.array(control_track['lat'])
    ctrl_lons = np.array(control_track['lon'])
    seed_lats = np.array(seeded_track['lat'])
    seed_lons = np.array(seeded_track['lon'])

    if HAS_CARTOPY:
        fig = plt.figure(figsize=(12, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())

        if region:
            ax.set_extent([region['lon_min'], region['lon_max'],
                          region['lat_min'], region['lat_max']], crs=ccrs.PlateCarree())

        ax.coastlines(resolution='50m', color='black')
        ax.add_feature(cfeature.BORDERS, linestyle=':')
        ax.add_feature(cfeature.STATES, linestyle=':', alpha=0.5)

        ax.plot(ctrl_lons, ctrl_lats, 'k-o', linewidth=2, markersize=4,
               label='Control Track', transform=ccrs.PlateCarree())
        ax.plot(seed_lons, seed_lats, 'c-o', linewidth=2, markersize=4,
               label='Seeded Track', transform=ccrs.PlateCarree())

        # Mark start and end
        ax.plot(ctrl_lons[0], ctrl_lats[0], 'g^', markersize=10,
               label='Start', transform=ccrs.PlateCarree())
        ax.plot(ctrl_lons[-1], ctrl_lats[-1], 'rs', markersize=10,
               label='End (Control)', transform=ccrs.PlateCarree())
    else:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.plot(ctrl_lons, ctrl_lats, 'k-o', linewidth=2, markersize=4, label='Control Track')
        ax.plot(seed_lons, seed_lats, 'c-o', linewidth=2, markersize=4, label='Seeded Track')
        ax.plot(ctrl_lons[0], ctrl_lats[0], 'g^', markersize=10, label='Start')
        ax.plot(ctrl_lons[-1], ctrl_lats[-1], 'rs', markersize=10, label='End (Control)')

    ax.set_title(title)
    ax.legend(loc='upper left')

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    plt.show()
    return fig


# =============================================================================
# TIME SERIES PLOTTING
# =============================================================================

def plot_intensity_timeseries(
    reduction_stats: dict,
    title: str = "Impact of Cloud Seeding on Storm Intensity",
    output_path: str = None
):
    """
    Plot wind intensity time series comparing control and seeded forecasts.

    Args:
        reduction_stats: Dict from compute_wind_reduction()
        title: Plot title
        output_path: Path to save figure
    """
    max_wind_ctrl = np.array(reduction_stats['max_wind_ctrl_timeseries'])
    max_wind_seed = np.array(reduction_stats['max_wind_seed_timeseries'])
    reduction = np.array(reduction_stats['reduction_timeseries'])

    time_hours = np.arange(len(max_wind_ctrl)) * 6
    peak_time = reduction_stats['peak_reduction_hour']
    peak_reduction = reduction_stats['peak_reduction_ms']

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(time_hours, max_wind_ctrl, 'k-', linewidth=2, label='Control (No Intervention)')
    ax.plot(time_hours, max_wind_seed, 'c--', linewidth=2, label='Seeded (Intervention)')

    # Fill area of reduction
    ax.fill_between(time_hours, max_wind_ctrl, max_wind_seed,
                   where=(max_wind_ctrl > max_wind_seed),
                   color='cyan', alpha=0.2, label='Intensity Reduction')

    # Mark peak reduction
    peak_idx = peak_time // 6
    ax.vlines(peak_time, max_wind_seed[peak_idx], max_wind_ctrl[peak_idx],
             colors='r', linestyles=':', linewidth=2)
    ax.annotate(f'Peak: -{peak_reduction:.1f} m/s',
               xy=(peak_time, (max_wind_ctrl[peak_idx] + max_wind_seed[peak_idx]) / 2),
               xytext=(10, 0), textcoords='offset points',
               fontsize=10, color='red')

    ax.set_xlabel("Forecast Time (Hours)")
    ax.set_ylabel("Max Wind Speed (m/s)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    plt.show()
    return fig


# =============================================================================
# VIDEO GENERATION
# =============================================================================

def generate_forecast_video(
    data_path: str,
    output_video: str,
    region: dict = None,
    title: str = "Forecast Evolution",
    fps: int = 4,
    vmin: float = 0,
    vmax: float = 50
):
    """
    Generate animation video of wind speed evolution.

    Args:
        data_path: Path to Zarr forecast store
        output_video: Output video path (.mp4 or .mov)
        region: Dict with bounds (optional)
        title: Video title
        fps: Frames per second
        vmin, vmax: Color scale limits
    """
    print(f"Generating video from {data_path}...")

    store = ZarrBackend(data_path)

    lat = store.root["lat"][:]
    lon = store.root["lon"][:]
    u_all = store.root["u10m"][0]  # (Time, Lat, Lon)
    v_all = store.root["v10m"][0]

    num_steps = u_all.shape[0]
    print(f"Processing {num_steps} timesteps...")

    # Apply region subset
    if region:
        lat_mask = (lat >= region['lat_min']) & (lat <= region['lat_max'])
        lon_mask = (lon >= region['lon_min']) & (lon <= region['lon_max'])
        lat_sub = lat[lat_mask]
        lon_sub = lon[lon_mask]
    else:
        lat_sub = lat
        lon_sub = lon
        lat_mask = np.ones(len(lat), dtype=bool)
        lon_mask = np.ones(len(lon), dtype=bool)

    # Pre-calculate wind frames
    wind_frames = []
    for t in range(num_steps):
        u_frame = u_all[t][np.ix_(lat_mask, lon_mask)]
        v_frame = v_all[t][np.ix_(lat_mask, lon_mask)]
        wind_frames.append(np.sqrt(u_frame**2 + v_frame**2))
    wind_frames = np.array(wind_frames)

    # Setup animation
    if HAS_CARTOPY:
        fig = plt.figure(figsize=(12, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        if region:
            ax.set_extent([region['lon_min'], region['lon_max'],
                          region['lat_min'], region['lat_max']], crs=ccrs.PlateCarree())
        ax.coastlines(resolution='50m', color='white', linewidth=1)
        ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='white')

        mesh = ax.pcolormesh(lon_sub, lat_sub, wind_frames[0],
                            transform=ccrs.PlateCarree(),
                            cmap='viridis', vmin=vmin, vmax=vmax)
    else:
        fig, ax = plt.subplots(figsize=(12, 8))
        mesh = ax.pcolormesh(lon_sub, lat_sub, wind_frames[0],
                            cmap='viridis', vmin=vmin, vmax=vmax)

    cbar = plt.colorbar(mesh, orientation='horizontal', pad=0.05, aspect=50)
    cbar.set_label('Wind Speed (m/s)')
    title_text = ax.set_title(f"{title} - T+0h")

    def update(frame):
        mesh.set_array(wind_frames[frame].ravel())
        title_text.set_text(f"{title} - T+{frame * 6}h")
        return mesh, title_text

    ani = animation.FuncAnimation(fig, update, frames=num_steps, interval=1000/fps, blit=False)

    # Save video
    print(f"Saving video to {output_video}...")
    writer = animation.FFMpegWriter(fps=fps, metadata={'title': title})
    ani.save(output_video, writer=writer)

    plt.close()
    print(f"Video saved: {output_video}")


def generate_comparison_video(
    control_path: str,
    seeded_path: str,
    output_video: str,
    region: dict = None,
    fps: int = 4,
    vmin: float = 0,
    vmax: float = 50
):
    """
    Generate side-by-side comparison video of control vs seeded forecasts.

    Args:
        control_path: Path to control forecast Zarr store
        seeded_path: Path to seeded forecast Zarr store
        output_video: Output video path
        region: Dict with bounds (optional)
        fps: Frames per second
        vmin, vmax: Color scale limits
    """
    print("Generating comparison video...")

    store_ctrl = ZarrBackend(control_path)
    store_seed = ZarrBackend(seeded_path)

    lat = store_ctrl.root["lat"][:]
    lon = store_ctrl.root["lon"][:]

    u_ctrl = store_ctrl.root["u10m"][0]
    v_ctrl = store_ctrl.root["v10m"][0]
    u_seed = store_seed.root["u10m"][0]
    v_seed = store_seed.root["v10m"][0]

    num_steps = min(u_ctrl.shape[0], u_seed.shape[0])

    # Apply region subset
    if region:
        lat_mask = (lat >= region['lat_min']) & (lat <= region['lat_max'])
        lon_mask = (lon >= region['lon_min']) & (lon <= region['lon_max'])
        lat_sub = lat[lat_mask]
        lon_sub = lon[lon_mask]
    else:
        lat_sub = lat
        lon_sub = lon
        lat_mask = np.ones(len(lat), dtype=bool)
        lon_mask = np.ones(len(lon), dtype=bool)

    # Pre-calculate wind frames
    wind_ctrl_frames = []
    wind_seed_frames = []
    for t in range(num_steps):
        u_c = u_ctrl[t][np.ix_(lat_mask, lon_mask)]
        v_c = v_ctrl[t][np.ix_(lat_mask, lon_mask)]
        u_s = u_seed[t][np.ix_(lat_mask, lon_mask)]
        v_s = v_seed[t][np.ix_(lat_mask, lon_mask)]
        wind_ctrl_frames.append(np.sqrt(u_c**2 + v_c**2))
        wind_seed_frames.append(np.sqrt(u_s**2 + v_s**2))

    wind_ctrl_frames = np.array(wind_ctrl_frames)
    wind_seed_frames = np.array(wind_seed_frames)

    # Setup figure with two panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    if HAS_CARTOPY:
        ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
        ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())

        for ax in [ax1, ax2]:
            if region:
                ax.set_extent([region['lon_min'], region['lon_max'],
                              region['lat_min'], region['lat_max']], crs=ccrs.PlateCarree())
            ax.coastlines(resolution='50m', color='white', linewidth=1)
            ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='white')

        mesh1 = ax1.pcolormesh(lon_sub, lat_sub, wind_ctrl_frames[0],
                              transform=ccrs.PlateCarree(), cmap='viridis', vmin=vmin, vmax=vmax)
        mesh2 = ax2.pcolormesh(lon_sub, lat_sub, wind_seed_frames[0],
                              transform=ccrs.PlateCarree(), cmap='viridis', vmin=vmin, vmax=vmax)
    else:
        mesh1 = ax1.pcolormesh(lon_sub, lat_sub, wind_ctrl_frames[0], cmap='viridis', vmin=vmin, vmax=vmax)
        mesh2 = ax2.pcolormesh(lon_sub, lat_sub, wind_seed_frames[0], cmap='viridis', vmin=vmin, vmax=vmax)

    ax1.set_title("Control (No Intervention)")
    ax2.set_title("Seeded (Intervention)")

    fig.colorbar(mesh2, ax=[ax1, ax2], orientation='horizontal', pad=0.05, aspect=50, label='Wind Speed (m/s)')
    time_text = fig.suptitle("T+0h", fontsize=14)

    def update(frame):
        mesh1.set_array(wind_ctrl_frames[frame].ravel())
        mesh2.set_array(wind_seed_frames[frame].ravel())
        time_text.set_text(f"T+{frame * 6}h")
        return mesh1, mesh2, time_text

    ani = animation.FuncAnimation(fig, update, frames=num_steps, interval=1000/fps, blit=False)

    print(f"Saving video to {output_video}...")
    writer = animation.FFMpegWriter(fps=fps)
    ani.save(output_video, writer=writer)

    plt.close()
    print(f"Video saved: {output_video}")


# =============================================================================
# QUICK VISUALIZATION HELPERS
# =============================================================================

def visualize_results(results: dict, output_dir: str = "figures"):
    """
    Generate all standard visualizations from intervention results.

    Args:
        results: Results dict from run_hurricane_intervention_study()
        output_dir: Directory to save figures
    """
    os.makedirs(output_dir, exist_ok=True)

    region = {'lat_min': 15, 'lat_max': 45, 'lon_min': 260, 'lon_max': 300}

    # 1. Wind field comparison
    plot_intervention_impact(
        results['control_path'],
        results['seeded_path'],
        region=region,
        output_path=os.path.join(output_dir, "impact_map.png")
    )

    # 2. Intensity time series
    plot_intensity_timeseries(
        results['reduction_stats'],
        output_path=os.path.join(output_dir, "intensity_timeseries.png")
    )

    # 3. Storm tracks
    plot_storm_tracks(
        results['control_track'],
        results['seeded_track'],
        region=region,
        output_path=os.path.join(output_dir, "storm_tracks.png")
    )

    print(f"\nAll visualizations saved to {output_dir}/")


if __name__ == "__main__":
    # Example usage
    print("Visualization module loaded.")
    print("Use visualize_results(results_dict) to generate all plots.")
