# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 09:39:39 2025

@author: Liam
"""

import os

from datetime import datetime, timedelta
import earthaccess

def generate_date_range(dates):
    """
    Generate a list of all dates between the two given dates in 'YYYY-MM-DD' format.
    
    Args:
        date_tuple (tuple): A tuple containing two date strings in the format 'YYYY-MM-DD' (start_date, end_date).
        
    Returns:
        date_list (list): A list of date strings in 'YYYY-MM-DD' format representing every day between the start and end dates.
    """
    # Parse the start and end dates from the tuple
    start_date = datetime.strptime(dates[0], '%Y-%m-%d')
    end_date = datetime.strptime(dates[1], '%Y-%m-%d')
    
    # Generate the list of dates
    date_list = [(start_date + timedelta(days=x)).strftime('%Y-%m-%d') for x in range((end_date - start_date).days + 1)]
    
    return date_list

def build_url(dataset, date):
    """
    Sometimes it is preferrable to manually build the URL: it allows you to filter using OPeNDAP, and directly accessing the URL avoids time consuming 'earthaccess.search_data()' calls.
    This is not a complete nor robust function, it is a helper function to serve my purposes and to demonstrate how to build URLs.

    Args:
        dataset (str): The name of the dataset (e.g., M2SDNXSLV). Find datasets here: https://disc.gsfc.nasa.gov/datasets
        date (str): YYYY-MM-DD to download

    Returns:
        filename (str): The name of the file to download.
        url (str): URL pointing to the file to download.
    """    
    # Date: YYYY-MM-DD
    year = date[:4]
    month = date[5:7]
    day = date[8:10]
    
    if dataset == 'M2SDNXSLV':
        if int(year) == 2021 and int(month) in [6, 7, 8, 9]:
            filename = f'MERRA2_401.statD_2d_slv_Nx.{year}{month}{day}.nc4'
        elif int(year) == 2020 and int(month) in [9]:
            filename = f'MERRA2_401.statD_2d_slv_Nx.{year}{month}{day}.nc4'
        elif int(year) >= 2011:
            filename = f'MERRA2_400.statD_2d_slv_Nx.{year}{month}{day}.nc4'
        else:
            filename = f'MERRA2_300.statD_2d_slv_Nx.{year}{month}{day}.nc4'
        url = f'https://data.gesdisc.earthdata.nasa.gov/data/MERRA2/M2SDNXSLV.5.12.4/{year}/{month}/{filename}'
    
    elif dataset == 'M2T1NXFLX':
        if (int(year) == 2021 and int(month) in [6, 7, 8, 9]) or (int(year) == 2020 and int(month) == 9):
            url = f'https://opendap.earthdata.nasa.gov/collections/C1276812838-GES_DISC/granules/M2T1NXFLX.5.12.4%3AMERRA2_401.tavg1_2d_flx_Nx.{year}{month}{day}.nc4.dap.nc4?dap4.ce=/ULML%5B0:23%5D%5B264:348%5D%5B3:280%5D;/VLML%5B0:23%5D%5B264:348%5D%5B3:280%5D;/PRECTOT%5B0:23%5D%5B264:348%5D%5B3:280%5D;/time;/lat%5B264:348%5D;/lon%5B3:280%5D'
        elif int(year) == 2000:
            url = f'https://opendap.earthdata.nasa.gov/collections/C1276812838-GES_DISC/granules/M2T1NXFLX.5.12.4%3AMERRA2_200.tavg1_2d_flx_Nx.{year}{month}{day}.nc4.dap.nc4?dap4.ce=/ULML%5B0:23%5D%5B264:348%5D%5B3:280%5D;/VLML%5B0:23%5D%5B264:348%5D%5B3:280%5D;/PRECTOT%5B0:23%5D%5B264:348%5D%5B3:280%5D;/time;/lat%5B264:348%5D;/lon%5B3:280%5D'
        elif int(year) >= 2011:
            url = f'https://opendap.earthdata.nasa.gov/collections/C1276812838-GES_DISC/granules/M2T1NXFLX.5.12.4%3AMERRA2_400.tavg1_2d_flx_Nx.{year}{month}{day}.nc4.dap.nc4?dap4.ce=/ULML%5B0:23%5D%5B264:348%5D%5B3:280%5D;/VLML%5B0:23%5D%5B264:348%5D%5B3:280%5D;/PRECTOT%5B0:23%5D%5B264:348%5D%5B3:280%5D;/time;/lat%5B264:348%5D;/lon%5B3:280%5D'
        else:
            url = f'https://opendap.earthdata.nasa.gov/collections/C1276812838-GES_DISC/granules/M2T1NXFLX.5.12.4%3AMERRA2_300.tavg1_2d_flx_Nx.{year}{month}{day}.nc4.dap.nc4?dap4.ce=/ULML%5B0:23%5D%5B264:348%5D%5B3:280%5D;/VLML%5B0:23%5D%5B264:348%5D%5B3:280%5D;/PRECTOT%5B0:23%5D%5B264:348%5D%5B3:280%5D;/time;/lat%5B264:348%5D;/lon%5B3:280%5D'
        filename = f'M2T1NXFLX.5.12.4%3AMERRA2_300.tavg1_2d_flx_Nx.{year}{month}{day}.nc4.dap.nc4'
    
    return filename, url

def search_download_merra2(output_dir, dataset, bbox, dates, verbose = True):
    """
    Searches GES DISC data catalogues given bbox and dates, downloads returned NC4 files

    Args:
        output_dir (str): Path to save the resulting NC4s to.
        dataset (str): The name of the dataset (e.g., M2SDNXSLV). Find datasets here: https://disc.gsfc.nasa.gov/datasets
        bbox (tuple): (min_lon, min_lat, max_lon, max_lat). 
            Note that earthaccess.search_data() will return datasets that overlap this bbox, but will not filter results to contain only points within the bbox.
        dates (tuple): (start_date, end_date). Dates are in format 'YYYY-MM-DD'.
        verbose (bool, optional): If True, prints skipped and downloaded files. Defaults to True.

    Returns:
        downloaded_files (list): A list of paths to downloaded files 
    
    Example Dataset Types:
    M2SDNXSLV  (https://disc.gsfc.nasa.gov/datasets/M2SDNXSLV_5.12.4/summary )
        Has keys: ['lon', 'lat', 'time', 'HOURNORAIN', 'T2MMAX', 'T2MMEAN', 
                   'T2MMIN', 'TPRECMAX']
        Daily dataset of min, mean, max temperature and max precip

    M2T1NXFLX (https://disc.gsfc.nasa.gov/datasets/M2T1NXFLX_5.12.4/summary )
        Has keys: ['lon', 'lat', 'time', 'BSTAR', 'CDH', 'CDM', 'CDQ', 'CN', 
                   'DISPH', 'EFLUX', 'EVAP', 'FRCAN', 'FRCCN', 'FRCLS', 
                   'FRSEAICE', 'GHTSKIN', 'HFLUX', 'HLML', 'NIRDF', 'NIRDR', 
                   'PBLH', 'PGENTOT', 'PRECANV', 'PRECCON', 'PRECLSC', 
                   'PRECSNO', 'PRECTOT', 'PRECTOTCORR', 'PREVTOT', 'QLML', 
                   'QSH', 'QSTAR', 'RHOA', 'RISFC', 'SPEED', 'SPEEDMAX', 
                   'TAUGWX', 'TAUGWY', 'TAUX', 'TAUY', 'TCZPBL', 'TLML', 'TSH',
                   'TSTAR', 'ULML', 'USTAR', 'VLML', 'Z0H', 'Z0M']
        Hourly time-averaged dataset
    
    Other, listed here: https://disc.gsfc.nasa.gov/datasets
    """
    # Make the output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Authenticate with Earthdata
    auth = earthaccess.login(persist=True)
    
    # Search for granules
    results = earthaccess.search_data(short_name = dataset,
                                      temporal = dates,
                                      bounding_box = bbox)
    # Prepare file URLs and output filenames
    file_urls = [item.data_links()[0] for item in results]
    file_names = [os.path.basename(url) for url in file_urls]

    # Download and process files
    downloaded_files = []
    for i, file_url in enumerate(file_urls):
        input_file = os.path.join(output_dir, file_names[i])
        
        # Skip already downloaded files
        if os.path.exists(input_file):
            if verbose:
                print(f'{input_file} already exists, skipping...')
            downloaded_files = downloaded_files + [input_file]
            continue
        
        # Download file
        d = earthaccess.download(file_url, local_path = output_dir)
        
        downloaded_files = downloaded_files + d
        if verbose:
            print(f'\nDownloaded {d}!\n')
    
    return downloaded_files

def url_download_merra2(output_dir, dataset, dates, verbose = True):
    """
    Manually builds file URLs using a helper function. Avoids time-consuming calls to earthaccess.search_data(), but is less robust.

    Args:
        output_dir (str): Path to save the resulting NC4s to.
        dataset (str): The name of the dataset (e.g., M2SDNXSLV). Find datasets here: https://disc.gsfc.nasa.gov/datasets
        dates (tuple): (start_date, end_date). Dates are in format 'YYYY-MM-DD'.
        verbose (bool, optional): If True, prints skipped and downloaded files. Defaults to True.

    Returns:
        downloaded_files (list): A list of paths to downloaded files 
    """    
    # Make the output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Authenticate with Earthdata
    auth = earthaccess.login(persist=True)
    
    # Get the URLs
    date_list = generate_date_range(dates)
    file_names = []
    file_urls = []
    for date in date_list:
        filename, url = build_url(dataset, date)
        file_names.append(filename)
        file_urls.append(url)
    
    # Download and process files
    downloaded_files = []
    for i, file_url in enumerate(file_urls):
        target_file = os.path.join(output_dir, file_names[i])
        
        # Skip already downloaded files
        if os.path.exists(target_file):
            if verbose:
                print(f'{target_file} already exists, skipping...')
            downloaded_files = downloaded_files + [target_file]
            continue
        
        # Download file
        d = earthaccess.download(file_url, local_path = output_dir)
        
        downloaded_files = downloaded_files + d
        if verbose:
            print(f'\nDownloaded {d}!\n')
            
    return downloaded_files