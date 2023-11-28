import xarray as xrndarray
import os
import numpy as np
import matplotlib.pyplot as plt 

import geopandas
import rioxarray
from tqdm import tqdm


# Coordinate bounds (limits Russia roughly)
LEFT = 20
RIGHT = 180
TOP = 82
BOTTOM = 41


def get_vars(var: str) -> tuple [str, str, str]:
    '''
    Gets variables names

    Args:
    var (str) normal spelling of the variable

    Returns:
    var_era_full (str): variable name in ERA filenames
    var_era (str): variable name in ERA data
    var_cmip (str): variable name in CMIP data
    '''
    if var == "temperature":
        # Temperature
        var_era_full = "daily_mean_2m_temperature"
        var_era = "t2m"
        var_cmip = "tas"

    elif var == "precipitation":
        # Precipitations
        var_era_full = 'daily_mean_mean_total_precipitation_rate'
        var_cmip = 'pr'
        var_era = "mtpr"

    else:
        raise ValueError ("For variable {} aggregation procedure not implemented".format(var))

    return var_era_full, var_era, var_cmip


def average_year_era(path_in: str,
                    path_out: str,
                    var: str,
                    years: list) -> tuple [int, int]:
    '''
    Calculates yearly average of ERA data pixelwise

    Args:
    path_in (str) path to source data
    path_out (str): path to save data
    var (str): variable name
    years (list): years list

    Returns:
    height (int): height of source ERA dataset
    width (int): width of source ERA dataset
    '''
    
    # Load variables names
    var_era_full, var_era, _ = get_vars(var)

    # Era data
    era = xr.open_mfdataset(os.path.join(path_in,'ERA/data_14_22/{}_*.nc'.format(var_era_full)))

    # Extract ERA resolution for further regrid
    height = era.dims['lat']
    width = era.dims['lon']
    new_lon = np.linspace(0.0, 360.0, width)
    new_lat = np.linspace(-90.0, 90.0, height)

    # Go to 0...360 lon range
    era.coords['lon'] = era.coords['lon'] % 360
    era = era.sortby(era.lon)
    var_era = list(era.keys())[0]
    for year in years:

        # Select by years
        era_sel = era.sel(time=era.time.dt.year.isin([year]) )
        # Define aggregation strategy based on variable
        if var_era=="t2m":
            
            # Average over all time
            era_year = era_sel.mean("time")
            era_year = era_year.expand_dims(time=[year])

        elif var_era == 'mtpr':
            # Sum over this year
            era_year = era_sel.sum("time")
            era_year = era_year.expand_dims(time=[year])

        else:
            raise ValueError("For this variable aggregation procedure not implemented")
        
        era_year_clip = era_year.sel(lat=slice(BOTTOM, TOP), lon=slice(LEFT, RIGHT))
        # Save ERA average to file
        era_year_clip.to_netcdf(os.path.join("/CMIPProximity/data_processed/", "yearly", "ERA_{}_{}.nc".format(var_era, year)))

    return height, width


def average_cmip_model(
                        models: list,
                        ssps: list,
                        years: list,
                        var: str,
                        path_in: str,
                        path_out: str,
                        height: int = 721,
                        width: int = 1440
                        ):
        '''
        Calculates yearly average of CMIP data pixelwise

        Args:
        models (list): models in ensemble
        ssps (list): used scenarios
        years (list): years list
        var (str): variable name
        path_in (str) path to source data
        path_out (str): path to save data
        var_era_full (str): variable name
        height (int): height of source ERA dataset
        width (int): width of source ERA dataset
        '''
        
        # Load variables names
        _, _, var_cmip = get_vars(var)

        new_lon = np.linspace(0.0, 360.0, width)
        new_lat = np.linspace(-90.0, 90.0, height)

        CMIP = xr.Dataset(
                coords={'lat': new_lat,
                        'lon': new_lon}) 

        # Limit by coordinates
        CMIP_new = CMIP.sel(lat=slice(BOTTOM, TOP), lon=slice(LEFT, RIGHT))

        for ssp in ssps:
                print(ssp)

                for year in years:
                        for model in tqdm(models):

                                files = os.listdir(os.path.join(path_in, 'cmip'))
                                files_model = [file for file in files if (model in file) &
                                                                        (ssp in file) &
                                                                        (var_cmip in file)]                       
                                files_xr = [os.path.join(path_in, 'cmip', fn) for fn in files_model]
                                cmip = xr.open_mfdataset(files_xr)
                                
                                # Slice required year
                                cmip_sel = cmip.sel(time=cmip.time.dt.year.isin([year]))

                                if var_cmip=="tas":
                                        # Average over all time
                                        cmip_year = cmip_sel.mean("time")
                                        # cmip_year = cmip_year.expand_dims(time=[year])

                                elif var_cmip == 'pr':
                                        # Sum over this year
                                        cmip_year = cmip_sel.sum("time")
                                else:
                                        raise ValueError("For this variable aggregation procedure not implemented")

                                # Upscale to ERA resoluion
                                cmip_up = cmip_year.interp(lat=new_lat, lon=new_lon)
                                cmip_up = cmip_up.sel(lat=slice(BOTTOM,TOP), lon=slice(LEFT,RIGHT))

                                # Assign data as a new layer to Xaverage_year_era.Dataset
                                CMIP_new[model]=(('lat', 'lon'), cmip_up[var_cmip].data)

                        # Save XArratDataset average as a new file
                        CMIP_new.to_netcdf(os.path.join(path_out, "yearly", ssp, 'CMIP_{}_{}.nc'.format(var_cmip, year)))
                        print(year, "saved")


def average_ensemble(
                ensembles: dict[str, list],
                ens_name: str,
                ssps: list,
                var: str,
                path: str,
                years: list
                ):
        '''
        Calculates yearly average of CMIP ensemble

        Args:
        ensembles (dict[str, list]): dict with CMIP models ensembles
        ens_name (str): ensembles dict key for the analysis
        ssps (list): used scenarios
        var (str): variable name
        path (str) path to data
        years (list): years list
        '''

        # Load variables names
        _, _, var_cmip= get_vars(var)

        for ssp in ssps:
                for year in years:
                        ds = xr.open_dataset(os.path.join(path, "yearly", ssp, 'CMIP_{}_{}.nc'.format(var_cmip, year)))
                        
                        # Average along data variables (i.e. models)
                        mean = ds.to_array(dim='new').mean('new')
                        cmip_ens = mean.to_dataset(name = "mean")
                        cmip_ens = cmip_ens.expand_dims(time=[year])
                        cmip_ens["mean"].to_netcdf(os.path.join(path, "yearly", ssp, 'CMIP_{}_{}_{}.nc'.format(ens_name, var_cmip, year)))


def average_era_years(
        var: str,
        path: str,
        years: list,
        ):
        '''
        Calculates ERA average over few years

        Args:
        var (str): variable name
        path (str) path to data
        years (list): years list
        '''
        # Load variables names
        _, var_era, _= get_vars(var)
        
        years_str=list(map(str, list(years)))
        files = os.listdir(os.path.join(path, 'yearly'))
        files =[file for file in files if
                                        # (source in file) &
                                        # (ssp in file) &
                                        (var_era in file) &
                                        (any(year in file for year in years_str))]
        files_xr = [os.path.join(path, 'yearly', fn) for fn in files]
        data = xr.open_mfdataset(files_xr)

        # Average over all years
        data_avg = data.mean("time")

        # Save XArratDataset (model-ERA) difference as a new file
        data_avg[var_era].to_netcdf(os.path.join(path, "{}_{}".format(years[0], years[-1]), "ERA_{}.nc".format(var_era)))


def average_ensemble_years(
        ens_name: str,
        years: list,
        ssps: list,
        var: str,
        path: str
        ):
        '''
        Calculates average of CMIP ensemble over few years

        Args:
        ens_name (str): ensembles dict key for the analysis
        years (list): years list
        ssps (list): used scenarios
        var (str): variable name
        path (str) path to data
        '''
        
        # Load variables names
        _, _, var_cmip = get_vars(var)

        years_str=list(map(str, list(years)))
        for ssp in ssps:
                # Select files
                files = os.listdir(os.path.join(path, "yearly", ssp))
                files = [file for file in files if
                                                (ens_name in file) &
                                                (var_cmip in file) &
                                                (any(year in file for year in years_str))]
                files_xr = [os.path.join(path, 'yearly', ssp, fn) for fn in files]
                data = xr.open_mfdataset(files_xr)

                # Average over all years
                data_avg = data.mean("time")

                # Save XArratDataset as a new file
                data_avg.to_netcdf(os.path.join(path, "{}_{}".format(years[0], years[-1]), ssp, "CMIP_{}_{}.nc".format(ens_name, var_cmip)))

def average_cmip_years(
        ensembles,
        ens_name,
        years,
        ssps,
        var,
        path
        ):
        '''
        Calculates average of single CMIP model over few years

        Args:
        ensembles (dict[str, list]): dict with CMIP models ensembles
        ens_name (str): ensembles dict key for the analysis
        years (list): years list
        ssps (list): used scenarios
        var (str): variable name
        path (str) path to data
        '''
        # Load variables names
        _, _, var_cmip = get_vars(var)

        for ssp in ssps:
                for model in ensembles[ens_name]:
                        for i, year in enumerate(years):
                                ds = xr.open_dataset(os.path.join(path, "yearly", ssp, "CMIP_{}_{}.nc".format(var_cmip, year)))

                                if i==0:
                                        CMIP = ds[model].to_dataset(name = year)
                                else:
                                        CMIP[year]=ds[model]

                        # Average along data variables
                        mean = CMIP.to_array(dim='new').mean('new')
                        ds_mean = mean.to_dataset(name = model)

                        # Save XArratDataset (model-ERA) difference as a new file
                        ds_mean[model].to_netcdf(os.path.join(path, "{}_{}".format(years[0], years[-1]), ssp, "CMIP_{}_{}.nc".format(var_cmip, model)))