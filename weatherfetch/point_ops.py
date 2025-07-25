# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 14:03:24 2025

@author: Liam
"""

import geopandas as gpd
import numpy as np
from shapely.geometry import Point


def find_nearest_n_points(lat, lon, df, n_pts):
    """
    Filters a DataFrame to the nearest 'n_pts' points to a given (lat, lon) based on latitude and longitude.

    Args:
        lat (float): Latitude of the target point.
        lon (float): Longitude of the target point.
        df (pd.DataFrame): DataFrame containing columns 'latitude', 'longitude', and other data.
        n_pts (int): Number of nearest points to retain.

    Returns:
        pd.DataFrame: Filtered DataFrame containing the 'n' nearest points.
    """
    # Ensure required columns are present in the DataFrame
    required_columns = {'latitude', 'longitude'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f'The DataFrame must contain the following columns: {required_columns}')

    # Make a safe copy to avoid SettingWithCopyWarning
    df_copy = df.copy()

    # Convert target lat/lon to radians
    lat_rad, lon_rad = np.radians(lat), np.radians(lon)

    # Convert DataFrame lat/lon columns to radians
    lats_rad = np.radians(df_copy['latitude'].values)
    lons_rad = np.radians(df_copy['longitude'].values)

    # Calculate differences in latitudes and longitudes
    dlat = lats_rad - lat_rad
    dlon = lons_rad - lon_rad

    # Haversine distance calculation
    a = np.sin(dlat / 2)**2 + np.cos(lat_rad) * np.cos(lats_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distances = 6371 * c  # Earth's radius in kilometers

    # Add distances to the DataFrame (convert to metres)
    df_copy['distance'] = distances * 1000

    # Sort DataFrame by distance and select the top 'n' rows
    filtered_df = df_copy.nsmallest(n_pts, 'distance')

    # Drop the 'distance' column from the output for clarity
    filtered_df = filtered_df.drop(columns=['distance'])

    return filtered_df


def find_points_within_d(lat, lon, df, d):
    """
    Filters a DataFrame to all points within a given distance 'd' (in meters) from a (lat, lon).

    Args:
        lat (float): Latitude of the target point.
        lon (float): Longitude of the target point.
        df (pd.DataFrame): DataFrame containing columns 'latitude', 'longitude', and other data.
        d (float): Maximum distance (in meters) to filter points by.

    Returns:
        pd.DataFrame: Filtered DataFrame containing points within distance 'd'.
    """
    # Ensure required columns are present in the DataFrame
    required_columns = {'latitude', 'longitude'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f'The DataFrame must contain the following columns: {required_columns}')

    # Convert target lat/lon to radians
    lat_rad, lon_rad = np.radians(lat), np.radians(lon)

    # Convert DataFrame lat/lon columns to radians
    lats_rad = np.radians(df['latitude'].values)
    lons_rad = np.radians(df['longitude'].values)

    # Calculate differences in latitudes and longitudes
    dlat = lats_rad - lat_rad
    dlon = lons_rad - lon_rad

    # Haversine distance calculation
    a = np.sin(dlat / 2)**2 + np.cos(lat_rad) * np.cos(lats_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distances_km = 6371 * c  # Earth's radius in kilometers

    # Add distances to the DataFrame (in metres)
    df['distance'] = distances_km * 1000

    # Filter DataFrame for points within the specified distance 'd' (in km)
    filtered_df = df[df['distance'] <= d]

    # Drop the 'distance' column from the output for clarity (optional)
    filtered_df = filtered_df.drop(columns=['distance'])

    return filtered_df


def nearest_n_points(df,
                     variables,
                     point,
                     n_pts=1,
                     output_csv_path=None,
                     output_shp_path=None,
                     ):
    """
    Loads an NC4 file into a Pandas DataFrame, spatially filters the DataFrame to contain only the nearest 'n_pts' points to 'point'.

    Args:
        df (pandas DataFrame): DataFrame loaded by weatherfetch.nc4_utils.load_nc4()
        variables (list): The names of the variables to retrieve from the NC4 file.
        point (tuple): (longitude, latitude) of the point of interest in EPSG:4326.
        n_pts (int, optional): Number of points to find near 'point'. Defaults to 1.
        output_csv_path (str, optional): The path to save the CSV file (if desired). Defaults to None.
        output_shp_path (str, optional): The path to save the SHP file (if desired). Defaults to None.

    Returns:
        df (pd.DataFrame): The loaded data from the NC4 file containing only the nearest 'n_pts' points to 'point'.
                           Has columns ['latitude', 'longitude', 'date', 'hours' (if hourly data), variables...].
    """
    # Get lat and lon
    lon, lat = point

    # Filter DF
    base_cols = ['latitude', 'longitude', 'hour', 'date']
    cols_to_keep = base_cols + variables
    existing_cols = [col for col in cols_to_keep if col in df.columns]
    filter_df = df[existing_cols]

    if 'hour' in df.columns:
        # Filter to nearest n points. Here, n_pts * 24 due to the hourly overlapping points
        filter_df = find_nearest_n_points(lat, lon, filter_df, n_pts * 24)

    else:
        # Filter to nearest n points
        filter_df = find_nearest_n_points(lat, lon, filter_df, n_pts)

    # Save to CSV if path is provided
    if output_csv_path:
        filter_df.to_csv(output_csv_path, index=False)

    if output_shp_path:
        # Use geopandas to output as shp
        geometry = [Point(xy) for xy in zip(filter_df['longitude'], filter_df['latitude'])]
        gdf = gpd.GeoDataFrame(filter_df, geometry=geometry)
        gdf.set_crs('EPSG:4326', inplace=True)
        gdf.to_file(output_shp_path, driver='ESRI Shapefile')

    return filter_df


def points_within_d(df,
                    variables,
                    point,
                    d,
                    output_csv_path=None,
                    output_shp_path=None):
    """
    Loads an NC4 file into a Pandas DataFrame, spatially filters the DataFrame to contain only points within distance 'd' to 'point'.

    Args:
        df (pandas DataFrame): DataFrame loaded by weatherfetch.nc4_utils.load_nc4()
        variables (list): The names of the variables to retrieve from the NC4 file.
        point (tuple): (longitude, latitude) of the point of interest in EPSG:4326.
        d (float): The distance (in METRES) to find points within to 'point'
        output_csv_path (str, optional): The path to save the CSV file (if desired). Defaults to None.
        output_shp_path (str, optional): The path to save the SHP file (if desired). Defaults to None.

    Raises:
        Exception: No points are found within distance 'd' to 'point'.

    Returns:
        df (pd.DataFrame): The loaded data from the NC4 file containing only points within 'd' to 'point'.
                           Has columns ['latitude', 'longitude', 'date', 'hours' (if hourly data), variables...].
    """
    # Get lat and lon
    lon, lat = point

    # Filter DF
    base_cols = ['latitude', 'longitude', 'hour', 'date']
    cols_to_keep = base_cols + variables
    existing_cols = [col for col in cols_to_keep if col in df.columns]
    filter_df = df[existing_cols]

    if 'hour' in df.columns:
        # Filter to nearest n points. Here, n_pts * 24 due to the hourly overlapping points
        filter_df = find_points_within_d(lat, lon, filter_df, d)
    else:
        # Filter to nearest n points
        filter_df = find_points_within_d(lat, lon, filter_df, d)

    if len(filter_df) == 0:
        raise Exception(f'No points found within {d}')

    # Save to CSV if path is provided
    if output_csv_path:
        filter_df.to_csv(output_csv_path, index=False)

    if output_shp_path:
        # Use geopandas to output as shp
        geometry = [Point(xy) for xy in zip(filter_df['longitude'], filter_df['latitude'])]
        gdf = gpd.GeoDataFrame(filter_df, geometry=geometry)
        gdf.set_crs('EPSG:4326', inplace=True)
        gdf.to_file(output_shp_path, driver='ESRI Shapefile')

    return filter_df
