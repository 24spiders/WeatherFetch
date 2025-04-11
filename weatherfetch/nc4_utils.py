# -*- coding: utf-8 -*-
"""
Created on Wed Jan  22 15:36:20 2025

@author: Liam
"""

import warnings

import netCDF4 as nc
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import cartopy.crs as ccrs


def rotated_to_absolute(rlat, rlon, pole_lat, pole_lon):
    """
    Converts rotated grid coordinates (rlat, rlon) to absolute geographic coordinates (lat, lon).

    Args:
        rlat (array-like): Rotated latitude in degrees.
        rlon (array-like): Rotated longitude in degrees.
        pole_lat (float): Latitude of the rotated pole in degrees.
        pole_lon (float): Longitude of the rotated pole in degrees.

    Returns:
        tuple: (lat, lon) in degrees (absolute geographic coordinates).
    """
    # Define the rotated pole coordinate system
    rotated_crs = ccrs.RotatedPole(pole_longitude=pole_lon, pole_latitude=pole_lat)

    # Define the absolute coordinate system (WGS84)
    geo_crs = ccrs.PlateCarree()

    # Transform rotated (rlat, rlon) to absolute (lat, lon)
    transformed = geo_crs.transform_points(rotated_crs, rlon, rlat)
    lat, lon = transformed[..., 1], transformed[..., 0]

    return lat, lon


def hours_since(hours_since, reference_date_time):
    """
    Returns the day and time by using 'hours since reference_date_time'

    Args:
        hours_since (float): Number of hours since reference_date_time.
        reference_date_time (str): YYYY-MM-DD HH:MM:ss.

    Returns:
        (year, month, day) and hour since reference_date_time.

    """
    ref_time = datetime.strptime(reference_date_time, '%Y-%m-%d %H:%M:%S')
    new_time = ref_time + timedelta(hours=hours_since)
    return (new_time.year, new_time.month, new_time.day), new_time.strftime('%H:%M:%S')


def load_nc4(netCDF_file_path, variable=None, verbose=True):
    """
    Loads a variable from a netCDF file into a Pandas DataFrame.
    If the netCDF contains hourly data, a row is written for each hour.
    If the netCDF contains daily data, a row is written for each day.

    Args:
        netCDF_file_path (str): The path to the netCDF file to load.
        variable (str, optional): The name of the variable to retrieve from the netCDF file. If not passed, function prints possible variables and exits. Default is None.
        verbose (bool, optional): Prints the valid keys in the netCDF file if True. Defaults to True.

    Raises:
        ValueError: Variable not found in the NetCDF file.

    Returns:
        df (pd.DataFrame): The loaded data from the netCDF file with columns ['latitude', 'longitude', 'date', 'hours' (if hourly data), variable].
    """
    # Open the NetCDF file
    dataset = nc.Dataset(netCDF_file_path, 'r')
    if variable is None:
        raise Exception(f"'variable' is None, netCDF Keys: {dataset.variables.keys()}")
    if verbose:
        print(f'netCDF Keys: {dataset.variables.keys()}')

    # Get the variable data
    if variable not in dataset.variables:
        raise ValueError(f'Variable {variable} not found in NetCDF file! Check keys: {dataset.variables.keys()}.')

    var_data = dataset.variables[variable][:]

    # Get latitude and longitude, filter to nearest n_pts
    if 'rlat' in dataset.variables and 'rlon' in dataset.variables:
        rotated_pole_lat = dataset.variables['rotated_pole'].grid_north_pole_latitude
        rotated_pole_lon = dataset.variables['rotated_pole'].grid_north_pole_longitude
        if verbose:
            print('Detected rotated pole coordinates!')
            print(f'Rotated Pole Lat: {rotated_pole_lat}, Rotated Pole Lon: {rotated_pole_lon}')
        rlats = dataset.variables['rlat']
        rlons = dataset.variables['rlon']
        # Create meshgrid of rlats and rlons
        rlon_grid, rlat_grid = np.meshgrid(rlons, rlats)
        rlats_flat = rlat_grid.flatten()
        rlons_flat = rlon_grid.flatten()
        lats_flat, lons_flat = rotated_to_absolute(rlats_flat, rlons_flat, rotated_pole_lat, rotated_pole_lon)
    else:
        lat_var = next((var for var in ['Latitude', 'latitude', 'lat'] if var in dataset.variables), None)
        lon_var = next((var for var in ['Longitude', 'longitude', 'lon'] if var in dataset.variables), None)
        lats = dataset.variables[lat_var]
        lons = dataset.variables[lon_var]
        # Create meshgrid of lats and lons
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        lats_flat = lat_grid.flatten()
        lons_flat = lat_grid.flatten()

    # Get date
    time_var = next((var for var in ['Time', 'time', 'time1'] if var in dataset.variables), None)
    time_var = dataset.variables[time_var]

    # Catch warnings here: dtype compatibility can be noisy due to issues with the netCDF file format
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        time_var_data = time_var[:]

    time_units = time_var.units

    # If hourly data
    if time_var.shape[0] == 24:
        # Get date
        date = nc.num2date(time_var_data,
                           units=time_units,
                           calendar='standard')[0].strftime('%Y-%m-%d %H:%M:%S')
        # Create rows for each hour
        var_flat = var_data.reshape(24, -1)  # [24 hours, lat * lon]

        # Repeat lat/lon coordinates for each hour
        lats_flat = np.tile(lats_flat, 24)

        lons_flat = np.tile(lons_flat, 24)

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
        # Get date
        date = nc.num2date(time_var_data,
                           units=time_units,
                           calendar='standard')[0].strftime('%Y-%m-%d %H:%M:%S')
        if 'hours since' in time_var.units:
            ref_date_time = ' '.join(time_var.units.split()[2:])
            _, time = hours_since(time_var_data[0], ref_date_time)
        dates_flat = np.array([date] * len(lons_flat))
        times_flat = np.array([time] * len(lons_flat))
        var_flat = var_data.flatten()
        # Create DataFrame
        df = pd.DataFrame({
            'latitude': lats_flat,
            'longitude': lons_flat,
            'date': dates_flat,
            'time': times_flat,
            variable: var_flat
        })
    # Else, special case
    elif len(time_var) != 1:
        df = pd.DataFrame()
        for i, time in enumerate(time_var):
            var_data_at_time = var_data[i]
            date = nc.num2date(time_var_data,
                               units=time_units,
                               calendar='standard')[i].strftime('%Y-%m-%d %H:%M:%S')
            if 'hours since' in time_var.units:
                ref_date_time = ' '.join(time_var.units.split()[2:])
                _, time = hours_since(float(time), ref_date_time)
            dates_flat = np.array([date] * len(lons_flat))
            times_flat = np.array([time] * len(lons_flat))
            var_flat = var_data_at_time.flatten()
            # Create DataFrame
            if 'rlat' in dataset.variables and 'rlon' in dataset.variables:
                temp_df = pd.DataFrame({
                    'rotated_latitude': rlats_flat,
                    'rotated_longitude': rlons_flat,
                    'latitude': lats_flat,
                    'longitude': lons_flat,
                    'date': dates_flat,
                    'time': times_flat,
                    variable: var_flat
                })
            else:
                temp_df = pd.DataFrame({
                    'latitude': lats_flat,
                    'longitude': lons_flat,
                    'date': dates_flat,
                    'time': times_flat,
                    variable: var_flat
                })
            df = pd.concat([df, temp_df], ignore_index=True)

    # Remove rows where variable is NaN or masked
    df = df.dropna()

    # Close the dataset
    dataset.close()

    return df


def filter_df(df, bbox=None, dates=None):
    """
    Filters a DataFrame with optional spatial and temporal filtering.

    Args:
        df (pd.DataFrame): Pandas dataframe to filter
        bbox (tuple, optional): Bounding box (min_lon, min_lat, max_lon, max_lat) in lat/lon (EPSG:4326).
        dates (tuple, optional): Date range (min_date, max_date) in format 'YYYY-MM-DD'.

    Returns:
        df (pd.DataFrame): The filtered data
    """
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

    return df


def netcdf_to_csv(netCDF_file_path, variable, output_csv_path, bbox=None, dates=None):
    """
    Converts a netCDF file to CSV with optional spatial and temporal filtering.

    Args:
        netCDF_file_path (str): The path to the netCDF file to load.
        variable (str): The name of the variable to retrieve from the netCDF file.
        output_csv_path (str): The path to save the CSV file.
        bbox (tuple, optional): Bounding box (min_lon, min_lat, max_lon, max_lat) in lat/lon (EPSG:4326).
        dates (tuple, optional): Date range (min_date, max_date) in format 'YYYY-MM-DD'.

    Returns:
        df (pd.DataFrame): The loaded and filtered data
    """
    # Load the netCDF file into a DataFrame
    df = load_nc4(netCDF_file_path, variable, verbose=False)

    df = filter_df(df, bbox, dates)

    # Save the filtered DataFrame to CSV
    df.to_csv(output_csv_path, index=False)

    return df

if __name__ == '__main__':
    file = r'D:\Users\Liam\Documents\01 - University\Research\Python\Piyush\HRDPS Wind\2017082800_008.nc'
    u_df = load_nc4(file, 'UU')