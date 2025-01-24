# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 09:29:58 2025

@author: Liam
"""

import os

import h5py
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.transform import from_origin
from rasterio.warp import Resampling, calculate_default_transform, reproject
from scipy.interpolate import griddata
from shapely.geometry import box, mapping

from .point_ops import load_nc4, find_nearest_n_points, find_points_within_d

def crop_to_bbox(bbox, epsg_code, tif_path, output_path, remove = False):
    '''
    Reprojects a raster to a given EPSG code and crops it to a specified bounding box.

    Args:
        bbox (tuple): Bounding box (min_x, min_y, max_x, max_y) in 'epsg_code' coordinate system.
        epsg_code (int): Target EPSG code for reprojection.
        tif_path (str): Path to the input raster file.
        output_path (str): Path to save the cropped and reprojected raster.
        remove (bool, optional): If True, deletes the original raster at 'tif_path'. Defaults to False.

    Returns:
        None: Saves the reprojected and cropped raster to the specified path.
    '''
    # Open the raster file
    with rasterio.open(tif_path) as src:
        # Calculate the transform and metadata for reprojection
        transform, width, height = calculate_default_transform(
            src.crs, f'EPSG:{epsg_code}', src.width, src.height, *src.bounds
        )
        new_meta = src.meta.copy()
        new_meta.update({
            'crs': f'EPSG:{epsg_code}',
            'transform': transform,
            'width': width,
            'height': height
        })

        # Reproject the raster
        with rasterio.open(output_path, 'w', **new_meta) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=f'EPSG:{epsg_code}',
                    resampling=Resampling.bilinear
                )

    # Define the bounding box geometry in the target CRS
    bbox_geom = box(*bbox)
    bbox_geom_mapping = [mapping(bbox_geom)]

    # Crop the raster using the bounding box
    with rasterio.open(output_path) as src:
        out_image, out_transform = mask(src, bbox_geom_mapping, crop=True, all_touched=True)
        # Fill in any nodata with the mean
        out_image[np.where(out_image == 0)] = np.mean(out_image)
        out_meta = src.meta.copy()
        
        # Update metadata to reflect the cropped raster
        out_meta.update({
            'height': out_image.shape[1],
            'width': out_image.shape[2],
            'transform': out_transform
        })

    # Save the cropped raster to disk
    with rasterio.open(output_path, 'w', **out_meta) as dst:
        dst.write(out_image)
    if remove:
        os.remove(tif_path)

def convert_to_h5(tif_path, remove_tif = False):
    """
    Converts a GeoTIFF file to an HDF5 file.

    Args:
        tif_path (str): Path to the input GeoTIFF file.
        remove_tif (bool): If True, removes the original TIFF file after conversion. Default is False.

    Returns:
        str: Path to the generated HDF5 file.
    """
    # Load the raster data
    with rasterio.open(tif_path) as src:
        data = src.read() 
        metadata = src.meta 
        transform = src.transform 
        crs = src.crs 

    # Define the output HDF5 file path
    h5_path = tif_path.replace('.tif', '.h5')

    # Write data to the HDF5 file
    with h5py.File(h5_path, 'w') as h5_file:
        # Create datasets for raster data
        h5_file.create_dataset('data', data=data, compression='gzip', dtype=data.dtype)
        # Store metadata as attributes
        h5_file.attrs['crs'] = crs.to_string() if crs else None
        h5_file.attrs['transform'] = transform.to_gdal() if transform else None
        h5_file.attrs['metadata'] = str(metadata)

    # Optionally remove the original TIFF file
    if remove_tif and os.path.exists(tif_path):
        os.remove(tif_path)
    return h5_path

def interpolate_in_bbox(nc4_file_path, 
                        variable,
                        bbox, 
                        output_path, 
                        res, 
                        mode = 'daily',
                        n_pts = None, 
                        d = None, 
                        method = 'linear',
                        hour = None,
                        avg_hours = None,
                        reproj = None,
                        convert_h5 = False):
    """
    Loads an NC4 file, interpolates points to an array with spatial resolution 'res' (metres).
    Saves as either a projected GeoTIFF or an HDF5 array.

    Args:
        nc4_file_path (str): The path to the NC4 file to load.
        variable (str): The name of the variable to retrieve from the NC4 file.
        bbox (tuple, optional): Bounding box (min_lon, min_lat, max_lon, max_lat) in lat/lon (EPSG:4326).
        output_path (str): File path to save the output. If `convert_h5` is True, the file will be saved as `.h5`, otherwise as `.tif`.
        res (float): Spatial resolution of the output file in meters.
        mode (str, optional): Specifies the processing mode. Options include:
            - 'daily': Processes a daily NC4 file, interpolating data for the given day (outputs a single-channel array).
            - 'hourly': Processes an hourly NC4 file, filtering to a specific hour via the `hour` parameter (outputs a single-channel array).
            - 'avg_hourly': Processes an hourly NC4 file, averaging data over `avg_hours` (outputs an array with `n` channels, where `n = 24 / avg_hours`).
            Default is 'daily'.
       n_pts (int, optional): Number of points to find near 'point'. Defaults to None. Mutually exclusive with 'd'.
        d (float): The distance (in METRES) to find points within to 'point'. Defaults to None. Mutually exclusive with 'd'.
        method (str, optional): Interpolation method to use, such as 'linear', 'nearest', etc. Defaults to 'linear'.
        hour (float, optional): If `mode='hourly'`, specifies the hour for data extraction. Required for hourly mode. Defaults to None.
        avg_hours (float, optional): If `mode='avg_hourly'`, defines the number of hours to average data over. Defaults to None.
        reproj (dict, optional): Reprojection parameters for the output image if reprojection is desired. Should be provided in the format: 
            `{'epsg': <EPSG_CODE>, 'bbox': (min_x, min_y, max_x, max_y)}`. bbox coordinates are in EPSG_CODE Defaults to None.
        convert_h5 (bool, optional): If True, converts the output GeoTIFF file to HDF5 format and removes the original TIFF file. Defaults to False.

    Raises:
        AssertionError: Raised if input parameters are invalid or mutually exclusive conditions are not met.
    """
    # Previous validation code remains the same until after filtering points
    assert n_pts is not None or d is not None, "Either 'n_pts' or 'd' must be provided."
    assert (n_pts is not None) != (d is not None), "You may only pass one of 'n_pts' or 'd'" 
    assert mode in ['hour','daily','avg_hourly']
    
    df = load_nc4(nc4_file_path, variable, verbose=False)
    
    if mode == 'hour':
        assert 'hour' in df.columns , 'Mode is hour, but no hour column in NC4!'
        assert hour is not None, 'If hourly, select the hour you wish to use'
    elif mode == 'daily':
        assert 'hour' not in df.columns, 'You are using an hourly NC4 but have mode set to daily'
    elif mode == 'avg_hourly':
        assert 'hour' in df.columns , 'Mode is avg_hourly, but no hour column in NC4!' 
        assert avg_hours is not None, 'Must pass avg_hours when using mode avg_hourly'
    
    west_lon, south_lat, east_lon, north_lat = bbox
    lat = (south_lat + north_lat) / 2
    lon = (west_lon + east_lon) / 2
    
    if mode == 'hour':
        df = df[df['hour'] == hour]
        n_pts = n_pts * 24 if n_pts else None
    elif mode == 'avg_hourly':
        df['hour'] = df['hour'].astype(str)
        df['hour_int'] = df['hour'].apply(lambda x: int(x.split(':')[0]))
        df['interval'] = df['hour_int'].apply(lambda x: f"{(x // avg_hours) * avg_hours}-{((x // avg_hours) + 1) * avg_hours}")
        df = df.groupby(['latitude', 'longitude', 'interval', 'date'], as_index=False)[variable].mean()
        n_pts = int(n_pts * 24 / avg_hours)
    
    if n_pts:
        df = find_nearest_n_points(lat, lon, df, n_pts)
    elif d:
        df = find_points_within_d(lat, lon, df, d)
    
    required_cols = {'latitude', 'longitude', variable}
    if not required_cols.issubset(df.columns):
        raise AssertionError(f"DataFrame missing columns: {required_cols - set(df.columns)}")
    
    # Set m / deg constant
    meters_per_deg_lat = 111320
    
    # Create expanded grid based on actual data points
    buffer = res * 2  / meters_per_deg_lat # Add buffer for interpolation edge effects
    data_west = df['longitude'].min() - buffer
    data_east = df['longitude'].max() + buffer
    data_south = df['latitude'].min() - buffer
    data_north = df['latitude'].max() + buffer
    
    # Use the larger of the data extent or requested bbox
    grid_west = min(west_lon, data_west)
    grid_east = max(east_lon, data_east)
    grid_south = min(south_lat, data_south)
    grid_north = max(north_lat, data_north)
    
    lat_step = res / meters_per_deg_lat
    lat_grid = np.arange(grid_south, grid_north + lat_step, lat_step)
    
    meters_per_deg_lon = meters_per_deg_lat * np.cos(np.radians(lat))
    lon_step = res / meters_per_deg_lon
    lon_grid = np.arange(grid_west, grid_east + lon_step, lon_step)
    
    lon_grid, lat_grid = np.meshgrid(lon_grid, lat_grid)

    if 'interval' not in df.columns:
        interpolated = griddata(
            df[['longitude', 'latitude']].values,
            df[variable].values,
            (lon_grid, lat_grid),
            method=method
        )
        
        if np.any(np.isnan(interpolated)):
            extrap = griddata(
                df[['longitude', 'latitude']].values,
                df[variable].values,
                (lon_grid, lat_grid),
                method='nearest'
            )
            interpolated = np.where(np.isnan(interpolated), extrap, interpolated)
        
        # Crop to requested bbox
        mask = ((lon_grid >= west_lon) & (lon_grid <= east_lon) &
               (lat_grid >= south_lat) & (lat_grid <= north_lat))
        interpolated = interpolated[mask].reshape(1, -1)
            
    else:
        unique_intervals = df['interval'].unique()
        interpolated_list = []
        
        for interval in unique_intervals:
            df_interval = df[df['interval'] == interval]
            
            interpolated_interval = griddata(
                df_interval[['longitude', 'latitude']].values,
                df_interval[variable].values,
                (lon_grid, lat_grid),
                method=method
            )
            if np.any(np.isnan(interpolated_interval)):
                extrap = griddata(
                    df_interval[['longitude', 'latitude']].values,
                    df_interval[variable].values,
                    (lon_grid, lat_grid),
                    method='nearest'
                )
                interpolated_interval = np.where(np.isnan(interpolated_interval), extrap, interpolated_interval)
            
            # Crop to requested bbox
            mask = ((lon_grid >= west_lon) & (lon_grid <= east_lon) &
                   (lat_grid >= south_lat) & (lat_grid <= north_lat))
            interpolated_interval = interpolated_interval[mask]
            interpolated_list.append(interpolated_interval)
        
        interpolated = np.stack(interpolated_list, axis=0)

    # Update dimensions for final cropped grid
    cropped_lon_grid = lon_grid[mask]
    cropped_lat_grid = lat_grid[mask]
    output_height = len(np.unique(cropped_lat_grid))
    output_width = len(np.unique(cropped_lon_grid))
    interpolated = interpolated.reshape(-1, output_height, output_width)

    transform = from_origin(west_lon, north_lat, lon_step, lat_step)
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=output_height,
        width=output_width,
        count=interpolated.shape[0],
        dtype=interpolated.dtype,
        crs='EPSG:4326',
        transform=transform
    ) as dst:
        dst.write(interpolated, indexes=range(1, interpolated.shape[0] + 1))
    
    if reproj:
        epsg = reproj['epsg']
        reproj_bbox = reproj['bbox']
        reproj_tif_path = output_path.replace('.tif', f'_{epsg}.tif')
        crop_to_bbox(reproj_bbox, epsg, output_path, reproj_tif_path)
        output_path = reproj_tif_path
        
    if convert_h5:
        output_path = convert_to_h5(output_path, remove_tif = True)
        
    print(f"Saved interpolated raster to {output_path}")