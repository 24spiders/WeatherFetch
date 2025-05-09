# -*- coding: utf-8 -*-
"""
Created on Mon May  5 13:35:41 2025

@author: Liam
"""

import cdsapi
import os


def download_era5land(nc_dir, year, month, day):
    # https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=overview
    client = cdsapi.Client()

    dataset = "reanalysis-era5-land"

    request = {
        "variable": [
            "2m_dewpoint_temperature",
            "2m_temperature",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "surface_pressure",
            "total_precipitation"
        ],
        "year": year,
        "month": month,
        "day": [day],
        "time": [
            "00:00", "01:00", "02:00",
            "03:00", "04:00", "05:00",
            "06:00", "07:00", "08:00",
            "09:00", "10:00", "11:00",
            "12:00", "13:00", "14:00",
            "15:00", "16:00", "17:00",
            "18:00", "19:00", "20:00",
            "21:00", "22:00", "23:00"
        ],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": [75, -142, 35, -49]
    }

    target = os.path.join(nc_dir, f'era5_land_{year}_{month}_{day}.nc')

    if not os.path.exists(target):
        client.retrieve(dataset, request, target)

    return target
