# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 09:29:58 2025

@author: Liam
"""

from setuptools import setup, find_packages

setup(
    name='weatherfetch',
    version='0.1.0',
    author='Liam Bennett',
    description='A brief description of your package',
    long_description=open('README.md').read(), 
    long_description_content_type='text/markdown',
    url='https://github.com/24spiders/WeatherFetch',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'rasterio',
        'earthaccess',
        'geopandas',
        'netCDF4',
        'pandas',
        'shapely',
        'h5py',
        'scipy'
    ],
)
