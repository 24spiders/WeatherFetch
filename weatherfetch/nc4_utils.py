# -*- coding: utf-8 -*-
"""
Created on Wed Jan  22 15:36:20 2025

@author: Liam
"""

import warnings

import netCDF4 as nc
import numpy as np
import pandas as pd


def load_nc4(nc4_file_path, variable=None, verbose=True):
    """
    Loads a variable from an NC4 file into a Pandas DataFrame.
    If the NC4 contains hourly data, a row is written for each hour.
    If the NC4 contains daily data, a row is written for each day.

    Args:
        nc4_file_path (str): The path to the NC4 file to load.
        variable (str, optional): The name of the variable to retrieve from the NC4 file. If not passed, function prints possible variables and exits. Default is None.
        verbose (bool, optional): Prints the valid keys in the NC4 file if True. Defaults to True.

    Raises:
        ValueError: Variable not found in the NetCDF file.

    Returns:
        df (pd.DataFrame): The loaded data from the NC4 file with columns ['latitude', 'longitude', 'date', 'hours' (if hourly data), variable].
    """
    # Open the NetCDF file
    dataset = nc.Dataset(nc4_file_path, 'r')
    if verbose:
        print(f'NC4 Keys: {dataset.variables.keys()}')

    # Get the variable data
    if variable not in dataset.variables:
        raise ValueError(f'Variable {variable} not found in NetCDF file! Check keys.')

    var_data = dataset.variables[variable][:]

    # Get latitude and longitude, filter to nearest n_pts
    lat_var = next((var for var in ['Latitude', 'latitude', 'lat'] if var in dataset.variables), None)
    lon_var = next((var for var in ['Longitude', 'longitude', 'lon'] if var in dataset.variables), None)
    lats = dataset.variables[lat_var]
    lons = dataset.variables[lon_var]

    # Create meshgrid of lats and lons
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    # Get date
    time_var = next((var for var in ['Time', 'time'] if var in dataset.variables), None)
    time_var = dataset.variables['time']
    # Catch warnings here: dtype compatibility can be noisy due to issues with the nc4 file format
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        time_var_data = time_var[:]

    time_units = time_var.units
    date = nc.num2date(time_var_data,
                       units=time_units,
                       calendar='standard')[0].strftime('%Y-%m-%d %H:%M:%S')

    # If hourly data
    if time_var.shape[0] == 24:
        # Create rows for each hour
        var_flat = var_data.reshape(24, -1)  # [24 hours, lat * lon]

        # Repeat lat/lon coordinates for each hour
        lats_flat = np.tile(lat_grid.flatten(), 24)

        lons_flat = np.tile(lon_grid.flatten(), 24)

        # Hour labels from 0 to 23 for each lat/lon point
        hours = [f'{m // 60:02}:{m % 60:02}' for m in time_var_data]
        hours = np.repeat(hours, lat_grid.size)

        # Flatten variable data to match
        var_flat = var_flat.flatten()
        dates_flat = np.array([date] * len(lons_flat))

        # Create DataFrame
        df = pd.DataFrame({
            'latitude': lats_flat,
            'longitude': lons_flat,
            'hour': hours,
            'date': dates_flat,
            variable: var_flat
        })

    # Else, daily
    elif time_var.shape[0] == 1:
        lats_flat = lat_grid.flatten()
        lons_flat = lon_grid.flatten()
        dates_flat = np.array([date] * len(lons_flat))
        var_flat = var_data.flatten()

        # Create DataFrame
        df = pd.DataFrame({
            'latitude': lats_flat,
            'longitude': lons_flat,
            'date': dates_flat,
            variable: var_flat
        })

    # Remove rows where variable is NaN or masked
    df = df.dropna()

    # Close the dataset
    dataset.close()

    return df


def netcdf_to_csv(nc4_file_path, variable, output_csv_path, bbox=None, dates=None):
    """
    Converts a NetCDF4 file to CSV with optional spatial and temporal filtering.

    Args:
        nc4_file_path (str): The path to the NC4 file to load.
        variable (str): The name of the variable to retrieve from the NC4 file.
        output_csv_path (str): The path to save the CSV file.
        bbox (tuple, optional): Bounding box (min_lon, min_lat, max_lon, max_lat) in lat/lon (EPSG:4326).
        dates (tuple, optional): Date range (min_date, max_date) in format 'YYYY-MM-DD'.

    Returns:
        df (pd.DataFrame): The loaded and filtered data with columns ['latitude', 'longitude', 'date', variable].
    """
    # Load the nc4 file into a DataFrame
    df = load_nc4(nc4_file_path, variable, verbose=False)

    # Ensure date column is in datetime format
    try:
        df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y %I:%M:%S %p')
    except ValueError:
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')

    # Filter data by bounding box if provided
    if bbox:
        min_lon, min_lat, max_lon, max_lat = bbox
        df = df[(df['longitude'] >= min_lon) & (df['longitude'] <= max_lon)
                & (df['latitude'] >= min_lat) & (df['latitude'] <= max_lat)]

    # Filter data by date range if provided
    if dates:
        min_date, max_date = dates
        min_date = pd.to_datetime(min_date)
        max_date = pd.to_datetime(max_date)
        df = df[(df['date'] >= min_date) & (df['date'] <= max_date)]

    # Save the filtered DataFrame to CSV
    df.to_csv(output_csv_path, index=False)

    return df
