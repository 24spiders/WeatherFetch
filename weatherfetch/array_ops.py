# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 09:29:58 2025

@author: Liam
"""

import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Union

import h5py
import numpy as np
import rasterio
from rasterio.mask import mask
from rasterio.transform import from_origin
from rasterio.warp import Resampling, calculate_default_transform, reproject
from scipy.interpolate import griddata
from shapely.geometry import box, mapping

from .point_ops import load_nc4, find_nearest_n_points, find_points_within_d


def crop_and_reproject(reproj, tif_path, remove=False):
    '''
    Reprojects a raster to a given EPSG code and crops it to a specified bounding box.

    Args:
        reproj (dict): Contains keys 'epsg' and 'bbox'. Bounding box (min_x, min_y, max_x, max_y) in 'epsg_code' coordinate system.
        tif_path (str): Path to the input raster file.
        remove (bool, optional): If True, deletes the original raster at 'tif_path'. Defaults to False.

    Returns:
        None: Saves the reprojected and cropped raster to the specified path.
    '''
    # Get params
    epsg_code = reproj['epsg']
    bbox = reproj['bbox']
    output_path = tif_path.replace('.tif', f'_{epsg_code}.tif')

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
    return output_path


def convert_to_h5(tif_path, remove_tif=False):
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


def save_interpolated_grid(data, path, transform, shape):
    """
    Saves data to a geospatially projected (EPSG:4326) raster

    Args:
        data (np.array): The data that will be saved.
        path (str): Path to save the raster.
        transform (TYPE): DESCRIPTION.
        shape (TYPE): DESCRIPTION.

    Returns:
        None.

    """
    # TODO: What is 'transform'?
    # TODO: Can I just use data.shape instead of the passed shape parameter?
    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=shape[0],
        width=shape[1],
        count=data.shape[0],
        dtype=data.dtype,
        crs='EPSG:4326',
        transform=transform
    ) as dst:
        dst.write(data, indexes=range(1, data.shape[0] + 1))


@dataclass
class ProcessingOptions:
    """A set of options for input to interpolate_in_bbox()

    Args:
        variable (str): The name of the variable to retrieve from the NC4 file.
        bbox (tuple, optional): Bounding box (min_lon, min_lat, max_lon, max_lat) in lat/lon (EPSG:4326).

        resolution (float): Spatial resolution of the output file in meters.
        mode (str, optional): Specifies the processing mode. Options include:
            - 'daily': Processes a daily NC4 file, interpolating data for the given day (outputs a single-channel array).
            - 'hour': Processes an hourly NC4 file, filtering to a specific hour via the `hour` parameter (outputs a single-channel array).
            - 'avg_hourly': Processes an hourly NC4 file, averaging data over `avg_hours` (outputs an array with `n` channels, where `n = 24 / avg_hours`).
            Default is 'daily'.
       n_pts (int, optional): Number of points to find near 'point'. Defaults to None. Mutually exclusive with 'd'.
        d (float): The distance (in METRES) to find points within to 'point'. Defaults to None. Mutually exclusive with 'd'.
        method (str, optional): Interpolation method to use, such as 'linear', 'nearest', etc. Defaults to 'linear'.
        hour (float, optional): If `mode='hour'`, specifies the hour for data extraction. Required for hour mode. Defaults to None.
        avg_hours (float, optional): If `mode='avg_hourly'`, defines the number of hours to average data over. Defaults to None.
        reproj (dict, optional): Reprojection parameters for the output image if reprojection is desired. Should be provided in the format:
            `{'epsg': <EPSG_CODE>, 'bbox': (min_x, min_y, max_x, max_y)}`. bbox coordinates are in EPSG_CODE Defaults to None.
        convert_h5 (bool, optional): If True, converts the output GeoTIFF file to HDF5 format and removes the original TIFF file. Defaults to False.
        verbose (bool, options): Verbosity. Defaults to False
    """
    variable: str  # Required
    bbox: List[float]  # xmin, ymin, xmax, ymax
    resolution: float  # Required
    mode: str = 'daily'  # 'daily', 'hourly', 'avg_hourly'
    n_pts: Optional[int] = None  # Mutually exclusive with 'd'
    d: Optional[float] = None  # Mutually exclusive with 'n_pts'
    method: str = 'linear'  # 'linear', 'nearest', 'cubic'
    hour: Optional[float] = None  # Only for mode == 'hourly'
    avg_hours: Optional[float] = None  # Only for mode == 'avg_hourly'
    reproj: Optional[Dict[str, Union[int, List[float]]]] = None  # {'epsg': int, 'bbox': List[float]}
    convert_h5: bool = False  # Defaults to False
    verbose: bool = False

    def __post_init__(self):
        """Ensures mutually exclusive options and validates inputs."""
        if self.n_pts is not None and self.d is not None:
            raise ValueError('n_pts and d are mutually exclusive; provide only one.')
        if self.n_pts is None and self.d is None:
            raise ValueError('One of n_pts or d must be passed.')
        if self.mode == 'hourly' and self.hour is None:
            raise ValueError("hour must be provided when mode is 'hourly'.")
        if self.mode not in ['hour', 'avg_hourly', 'daily']:
            raise ValueError(f"Unrecognized mode '{self.mode}'.")
        if self.mode == 'avg_hourly' and self.avg_hours is None:
            raise ValueError("avg_hours must be provided when mode is 'avg_hourly'.")


def interpolate_in_bbox(nc4_file_path,
                        output_path,
                        options: ProcessingOptions):
    """
    Loads an NC4 file, interpolates points to an array with spatial resolution 'res' (metres).
    Saves as either a projected GeoTIFF or an HDF5 array.

    Args:
        nc4_file_path (str): The path to the NC4 file to load.
        output_path (str): File path to save the output. If `convert_h5` is True, the file will be saved as `.h5`, otherwise as `.tif`.
        options (ProcessingOptions): Config for processing

    Raises:
        AssertionError: Raised if input parameters are invalid or mutually exclusive conditions are not met.
    """
    # Load and filter dataframe
    df = load_nc4(nc4_file_path, options.variable, verbose=options.verbose)

    west_lon, south_lat, east_lon, north_lat = options.bbox
    lat = (south_lat + north_lat) / 2
    lon = (west_lon + east_lon) / 2

    if options.mode == 'hour':
        assert 'hour' in df.columns, 'Mode is hour, but no hour column in NC4!'
        assert options.hour is not None, 'If hourly, select the hour you wish to use'
    elif options.mode == 'daily':
        assert 'hour' not in df.columns, 'You are using an hourly NC4 but have mode set to daily'
    elif options.mode == 'avg_hourly':
        assert 'hour' in df.columns, 'Mode is avg_hourly, but no hour column in NC4!'
        assert options.avg_hours is not None, 'Must pass avg_hours when using mode avg_hourly'

    if options.mode == 'hour':
        df = df[df['hour'] == options.hour]
        options.n_pts = options.n_pts * 24 if options.n_pts else None
    elif options.mode == 'avg_hourly':
        df['hour'] = df['hour'].astype(str)
        df['hour_int'] = df['hour'].apply(lambda x: int(x.split(':')[0]))
        df['interval'] = df['hour_int'].apply(lambda x: f"{(x // options.avg_hours) * options.avg_hours}-{((x // options.avg_hours) + 1) * options.avg_hours}")
        df = df.groupby(['latitude', 'longitude', 'interval', 'date'], as_index=False)[options.variable].mean()
        options.n_pts = int(options.n_pts * 24 / options.avg_hours)

    if options.n_pts:
        df = find_nearest_n_points(lat, lon, df, options.n_pts)
    elif options.d:
        df = find_points_within_d(lat, lon, df, options.d)

    required_cols = {'latitude', 'longitude', options.variable}
    if not required_cols.issubset(df.columns):
        raise AssertionError(f"DataFrame missing columns: {required_cols - set(df.columns)}")

    # Interpolate
    # Set m / deg constant
    meters_per_deg_lat = 111320

    # Create expanded grid based on actual data points
    buffer = options.resolution * 2 / meters_per_deg_lat  # Add buffer for interpolation edge effects
    data_west = df['longitude'].min() - buffer
    data_east = df['longitude'].max() + buffer
    data_south = df['latitude'].min() - buffer
    data_north = df['latitude'].max() + buffer

    # Use the larger of the data extent or requested bbox
    grid_west = min(west_lon, data_west)
    grid_east = max(east_lon, data_east)
    grid_south = min(south_lat, data_south)
    grid_north = max(north_lat, data_north)

    lat_step = options.resolution / meters_per_deg_lat
    lat_grid = np.arange(grid_south, grid_north + lat_step, lat_step)

    meters_per_deg_lon = meters_per_deg_lat * np.cos(np.radians(lat))
    lon_step = options.resolution / meters_per_deg_lon
    lon_grid = np.arange(grid_west, grid_east + lon_step, lon_step)

    lon_grid, lat_grid = np.meshgrid(lon_grid, lat_grid)
    if 'interval' not in df.columns:
        interpolated = griddata(
            df[['longitude', 'latitude']].values,
            df[options.variable].values,
            (lon_grid, lat_grid),
            method=options.method
        )

        if np.any(np.isnan(interpolated)):
            extrap = griddata(
                df[['longitude', 'latitude']].values,
                df[options.variable].values,
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
        unique_intervals = sorted(unique_intervals, key=lambda x: int(x.split('-')[0]))
        interpolated_list = []

        for interval in unique_intervals:
            df_interval = df[df['interval'] == interval]
            interpolated_interval = griddata(
                df_interval[['longitude', 'latitude']].values,
                df_interval[options.variable].values,
                (lon_grid, lat_grid),
                method=options.method
            )

            if np.any(np.isnan(interpolated_interval)):
                extrap = griddata(
                    df_interval[['longitude', 'latitude']].values,
                    df_interval[options.variable].values,
                    (lon_grid, lat_grid),
                    method='linear'
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

    # Output
    transform = from_origin(west_lon, north_lat, lon_step, lat_step)
    shape = (output_height, output_width)
    save_interpolated_grid(interpolated, output_path, transform, shape)

    if options.reproj:
        output_path = crop_and_reproject(options.reproj, output_path, remove=True)

    if options.convert_h5:
        output_path = convert_to_h5(output_path, remove_tif=True)

    if options.verbose:
        print(f"Saved interpolated raster to {output_path}")
    return output_path
