# WeatherFetch
WeatherFetch automatically downloads climatic variables and outputs them in analysis-ready spatial formats. Note that WeatherFetch is my personal research code, meaning you may use it or modify it as you want, but may not be highly robust or extensively supported.
Some specific features:

1. Load NC files into Pandas DataFrames
2. Convert between rotated grid coordinates and absolute geographic coordinates
3. Spatially and temporally filter data
4. Output converted and filtered NC data as .csv
5. Automatically fetch MERRA-2 data using `earthaccess`
6. Create spatially projected rasters of interpolated weather data (from NC files)

![Example showing a spatially projected interpolated array and surrounding datapoints](interpolated.jpeg)

## Usage
All functions contain docstrings documenting their use. Some examples are provided below.

Load an HRDPS NC file, spatially filter it, convert to CSV
```py
from weatherfetch.nc4_utils import load_nc4, netcdf_to_csv

nc_file = '2017080100_000.nc'

# Load into DataFrame
df = load_nc4(nc_file, variable='UU')

# Set bbox in EPSG:4326
bbox = [-114.5, 52.5, -111.5, 54]  # min_lon, min_lat, max_lon, max_lat
netcdf_to_csv(nc_file, 'UU', 'UU.csv', bbox=bbox, dates=None)  # Don't filter by date (this file contains only one day)
```

To download a dataset, apply a spatial filter, and save to CSV,
```py
# Download a file
from weatherfetch.earthaccess_fetch import search_download_merra2
output_dir = './Test/'
dataset = 'M2SDNXSLV'
bbox = (-114.3, 53.4, -113.4, 54.0)
dates = ('2001-01-01', '2001-01-02')

downloaded_files = search_download_merra2(output_dir,
                                          dataset,
                                          bbox,
                                          dates,
                                          verbose=True)

# Apply spatial filters
from weatherfetch.point_ops import nearest_n_points, points_within_d
nc4_file_path = downloaded_files[0]
variable = 'T2MMAX'
point = (-113.5, 53.5)

# Find the nearest 'n' points
nearest_n_points(nc4_file_path,
                 variable,
                 point,
                 n_pts=10,
                 output_csv_path='./Test/test_n.csv',
                 output_shp_path='./Test/test_n.shp')
# Find points within a distance
points_within_d(nc4_file_path,
                variable,
                point,
                d=1000,  # metres
                output_csv_path='./Test/test_n.csv',
                output_shp_path='./Test/test_n.shp')
```
To interpolate data and save as a raster,
```py
from weatherfetch.array_ops import ProcessingOptions, interpolate_in_bbox
nc4_file_path = downloaded_files[0]
variable = 'T2MMAX'
bbox = (-114.3, 53.4, -113.4, 54.0)
output_path = './Test/'
resolution = 100  # metres

processing_options = ProcessingOptions(variable=variable,
                                       bbox=bbox,
                                       resolution=resolution,
                                       mode='daily',  # daily, hour, or avg_hourly
                                       n_pts=None,  # Mutually exclusive from d
                                       d=1000,  # Mutually exclusive from n_pts
                                       method='linear',  # Interpolation method
                                       hour=None,  # If mode is hour, the hour to extract data for (e.g., 17:00)
                                       avg_hours=None,  # If mode is avg_hourly, the time step to average over (e.g., 6)
                                       reproj=None,  # {'epsg': EPSG_CODE, 'bbox': transformed_bbox} if output GeoTIFF should be reprojected
                                       convert_h5=False)  # Whether to output in HDF5 format instead of GeoTIFF

interpolate_in_bbox(nc4_file_path,
                    output_path,
                    processing_options)
```


## Installation
1. Clone this package
2. Navigate to the cloned directory
3. Call `python setup.py develop` to install this package in developer mode
