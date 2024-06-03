import streamlit as st
import xarray as xr
import folium
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import branca.colormap as cm
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import requests
from pathlib import Path

# Function to download a file from a URL
def download_file(url, local_filename):
    local_file_path = Path(local_filename)
    local_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    return local_filename

# Function to plot the exact area covered by the NetCDF file
def plot_exact_area(dataset):
    lons = dataset['lons'].values
    lats = dataset['lats'].values

    map_center = [(lats.min() + lats.max()) / 2, (lons.min() + lons.max()) / 2]
    m = folium.Map(location=map_center, zoom_start=3)

    coordinates = list(zip(lats.flatten(), lons.flatten()))
    folium.Polygon(
        locations=coordinates,
        color='blue',
        weight=2,
        fill=True,
        fill_opacity=0.2,
        opacity=0.2
    ).add_to(m)

    return m

# Function to create the land and water mask plot
def plot_land_water_mask(dataset):
    lons = dataset['lons'].values
    lats = dataset['lats'].values
    variable_to_plot = dataset['FROCEAN'].values

    fig, ax = plt.subplots(
        figsize=(10, 10),
        subplot_kw={'projection': ccrs.PlateCarree()}
    )

    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')

    mesh = ax.pcolormesh(lons, lats, variable_to_plot, transform=ccrs.PlateCarree(), cmap='viridis', alpha=0.7)
    plt.colorbar(mesh, ax=ax, shrink=0.5)

    return fig

# Function to create contiguous squares of water
def create_water_squares(data, points_per_square):
    water_squares = []
    for i in range(0, data.sizes['y'], points_per_square):
        for j in range(0, data.sizes['x'], points_per_square):
            square = data.isel(x=slice(j, j + points_per_square), y=slice(i, i + points_per_square))
            if square['FROCEAN'].notnull().all():
                water_squares.append(square)
    return water_squares

# Function to calculate stats for each water square
def calculate_stats(water_squares, level_index):
    water_squares_stats = {}
    for i, square in enumerate(water_squares):
        delp_values = square['DELP'][level_index].values
        water_squares_stats[f'water_square_{i}'] = {
            'mean': np.nanmean(delp_values),
            'variance': np.nanvar(delp_values),
            'standard_deviation': np.nanstd(delp_values)
        }
    return water_squares_stats

# Function to plot water squares on a folium map
def plot_water_squares_on_map(water_squares, water_squares_stats, variable):
    map_center = [(square['lats'].mean().values, square['lons'].mean().values) for square in water_squares]
    map_center = [sum(x)/len(x) for x in zip(*map_center)]
    m = folium.Map(location=map_center, zoom_start=3)

    variances = [stats['variance'] for stats in water_squares_stats.values()]
    min_vars, max_vars = min(variances), max(variances)
    color_scale = cm.linear.YlGnBu_09.scale(min_vars, max_vars)
    color_scale.caption = 'Variance of DELP'

    for idx, square in enumerate(water_squares):
        square_lons = square['lons'].values
        square_lats = square['lats'].values
        variance = water_squares_stats[f'water_square_{idx}']['variance']
        color = color_scale(variance)

        lon_corners = [square_lons[0, 0], square_lons[0, -1], square_lons[-1, -1], square_lons[-1, 0], square_lons[0, 0]]
        lat_corners = [square_lats[0, 0], square_lats[0, -1], square_lats[-1, -1], square_lats[-1, 0], square_lats[0, 0]]

        folium.Polygon(
            locations=list(zip(lat_corners, lon_corners)),
            color=color,
            weight=2,
            fill=True,
            fill_opacity=0.6
        ).add_to(m).add_child(
            folium.Popup(
                f"Water Square ID: {idx}<br>"
                f"Variance: {variance:.2f}"
            )
        )

    color_scale.add_to(m)
    return m

# Function to plot DBSCAN clusters on a folium map
def plot_dbscan_clusters(water_squares, water_squares_stats):
    data_for_clustering = []
    for index, square in enumerate(water_squares):
        variance = water_squares_stats[f'water_square_{index}']['variance']
        lat, lon = np.mean(square['lats'].values), np.mean(square['lons'].values)
        data_for_clustering.append([lat, lon, variance])

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_for_clustering)

    dbscan = DBSCAN(eps=0.5, min_samples=5)
    clusters = dbscan.fit_predict(data_scaled)

    map_center = [(square['lats'].mean().values, square['lons'].mean().values) for square in water_squares]
    map_center = [sum(x)/len(x) for x in zip(*map_center)]
    m = folium.Map(location=map_center, zoom_start=5)

    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 
              'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 
              'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen', 
              'gray', 'black', 'lightgray']

    for i, (lat, lon, _) in enumerate(data_for_clustering):
        cluster = clusters[i]
        if cluster == -1:
            marker_color = 'black'
        else:
            marker_color = colors[cluster % len(colors)]
        
        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            color=marker_color,
            fill=True,
            fill_color=marker_color,
            fill_opacity=0.7
        ).add_to(m)

    colormap = cm.linear.YlGnBu_09.scale(min([variance for _, _, variance in data_for_clustering]),
                                         max([variance for _, _, variance in data_for_clustering]))
    colormap.caption = 'DBSCAN Clusters of Variance'
    colormap.add_to(m)

    return m

# Function to plot variance with outliers on a folium map
def plot_variance_with_outliers(water_squares, water_squares_stats):
    variances = [stats['variance'] for stats in water_squares_stats.values()]
    mean_variance = np.mean(variances)
    std_variance = np.std(variances)

    lower_threshold = mean_variance - 3 * std_variance
    upper_threshold = mean_variance + 3 * std_variance

    min_vars, max_vars = min(variances), max(variances)
    color_scale = cm.linear.YlGnBu_09.scale(min_vars, max_vars)
    color_scale.caption = 'Variance of DELP'
    outlier_color = 'red'

    def average_lat_lon(square):
        lats = square['lats'].values
        lons = square['lons'].values
        avg_lat = np.nanmean(lats)
        avg_lon = np.nanmean(lons)
        return avg_lat, avg_lon

    avg_lats = [average_lat_lon(square)[0] for square in water_squares]
    avg_lons = [average_lat_lon(square)[1] for square in water_squares]
    m = folium.Map(location=[np.mean(avg_lats), np.mean(avg_lons)], zoom_start=3)

    for i, square in enumerate(water_squares):
        stats_key = f'water_square_{i}'
        if stats_key in water_squares_stats:
            stats = water_squares_stats[stats_key]
            variance = stats['variance']
            lat, lon = average_lat_lon(square)
            is_outlier = variance < lower_threshold or variance > upper_threshold
            marker_color = outlier_color if is_outlier else color_scale(variance)

            folium.CircleMarker(
                location=[lat, lon],
                radius=8,
                color=marker_color,
                fill=True,
                fill_color=marker_color,
                fill_opacity=0.7,
                popup=(f'Water Square ID: {stats_key}<br>'
                       f'Variance: {variance:.2f}<br>'
                       f'{"Outlier" if is_outlier else "Normal"}')
            ).add_to(m)

    color_scale.add_to(m)
    return m

# Streamlit app
st.set_page_config(page_title="Interactive Assessment of Variance", layout="wide")

st.title("Interactive Assessment of Variance Using High-Resolution Model Output")

# File upload and URL input
file_option = st.sidebar.radio("Select file input method", ["Upload", "URL"])

if file_option == "Upload":
    uploaded_file = st.sidebar.file_uploader("Upload a NetCDF file", type=["nc4", "nc"])
    if uploaded_file is not None:
        file_path = uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
elif file_option == "URL":
    file_url = st.sidebar.text_input("Enter the file URL")
    if file_url:
        file_path = "large_file.nc4"
        st.write("Downloading the file...")
        download_file(file_url, file_path)
        st.write("File downloaded.")

# Ensure a file is available for processing
if "file_path" in locals():
    try:
        # Read the file using xarray
        dataset = xr.open_dataset(file_path)

        # Feature selection
        feature = st.sidebar.selectbox(
            'Select a feature to analyze',
            dataset.data_vars.keys()
        )

        # User interface with dropdown
        selected_option = st.sidebar.selectbox(
            'Select an option',
            ['Display NetCDF File Information', 'Plot the Area', 'Plot the Land and Water Mask', 'Plot the Variance', 'Plot the DBSCAN Clusters', 'Plot the Outliers using 3-Sigma Rule']
        )

        # Perform actions based on user selection
        if selected_option == 'Display NetCDF File Information':
            st.write(dataset)

        if selected_option == 'Plot the Area':
            map_object = plot_exact_area(dataset)
            st_folium(map_object, width=700, height=500)

        if selected_option == 'Plot the Land and Water Mask':
            fig1 = plot_land_water_mask(dataset)
            st.pyplot(fig1)

        if selected_option == 'Plot the DBSCAN Clusters':
            points_per_square = 17
            masked_data = dataset.where(dataset['FROCEAN'] > 0.9)
            masked_data['FROCEAN'] = masked_data['FROCEAN'].where(masked_data['FROCEAN'] > 0, np.nan)
            water_squares = create_water_squares(masked_data, points_per_square)
            level_index = 45
            water_squares_stats = calculate_stats(water_squares, level_index)
            dbscan_clusters_map = plot_dbscan_clusters(water_squares, water_squares_stats)
            st_folium(dbscan_clusters_map, width=700, height=500)

        if selected_option == 'Plot the Outliers using 3-Sigma Rule':
            points_per_square = 17
            masked_data = dataset.where(dataset['FROCEAN'] > 0.9)
            masked_data['FROCEAN'] = masked_data['FROCEAN'].where(masked_data['FROCEAN'] > 0, np.nan)
            water_squares = create_water_squares(masked_data, points_per_square)
            level_index = 45
            water_squares_stats = calculate_stats(water_squares, level_index)
            variance_outliers_map = plot_variance_with_outliers(water_squares, water_squares_stats)
            st_folium(variance_outliers_map, width=700, height=500)

        if selected_option == 'Plot the Variance':
            points_per_square = 17
            masked_data = dataset.where(dataset['FROCEAN'] > 0.9)
            masked_data['FROCEAN'] = masked_data['FROCEAN'].where(masked_data['FROCEAN'] > 0, np.nan)
            water_squares = create_water_squares(masked_data, points_per_square)
            level_index = 45
            water_squares_stats = calculate_stats(water_squares, level_index)
            water_squares_map = plot_water_squares_on_map(water_squares, water_squares_stats, feature)
            st_folium(water_squares_map, width=700, height=500)

    except Exception as e:
        st.write("Error: ", e)
else:
    st.write("Please upload a file or provide a URL to a file.")
