import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import geopandas
import xarray as xr

from tqdm import tqdm
sys.path.append(os.path.join("../src"))
from process import get_vars

from shapely.ops import unary_union
matplotlib.rcParams.update({'font.size': 20})


# Coefficient to change precipitatin units from kg*m-2*s-1 to mm/day
K = 60*60*24


def mae(data):
    '''
    Calculates mean absolute error of multidimensional error array

    Args:
    data (np.ndarray): error array

    Returns:
    mae (float): mean absolute error
    '''
    error = data.flatten()
    error = error[~np.isnan(error)]

    mae = np.sum(np.abs(error))/len(error)
    return mae


def calc_mae(
    ensembles: dict[str, list],
    ens_name: str,
    ssp_cur: str,
    var: str,
    path_in: str,
    path_out: str,
    bound: geopandas.geodataframe.GeoDataFrame,
    should_plot: bool = False
) -> dict [str, dict]:
    '''
    Plots CMIP-ERA data difference pixelwise

    Args:
    ensembles dict[str, list]: dictionary of models in ensemble
    ens_name (str): ensemble name
    ssp_cur (str): used scenario
    var (str): variable name
    path_in (str) path to source data
    path_out (str): path to save data
    bound (geopandas.geodataframe.GeoDataFrame): country boundary
    should_plot (Optional[bool]): whether to create a plot, default to False

    Returns:
    mae_scores (dict [str, dict]): dictionary with scores
    '''
    # Load variables names
    _, var_era, var_cmip = get_vars(var)

    models = ensembles[ens_name]
    mae_scores = {region: {} for region in bound["district"]}
    

    for i, region in enumerate(bound["district"]):
        for model in models:
            # Open datasets and culculate CMIP-ERA difference
            model_data = xr.open_dataset(os.path.join(path_in, ssp_cur,
                                        'CMIP_{}_{}.nc'.format(var_cmip, model)))
            era_data = xr.open_dataset(os.path.join(path_in,
                                        'ERA_{}.nc'.format(var_era)))
            era_data = era_data.interp(lat=model_data.lat, lon=model_data.lon)
            diff = era_data[var_era] - model_data[model]
            CMIP_differ = xr.combine_by_coords([model_data, era_data])
            CMIP_differ["diff"] = (('lat', 'lon'), diff.data)
            CMIP_differ = CMIP_differ.rio.set_spatial_dims('lon', 'lat', inplace=True)
            CMIP_differ.rio.write_crs("epsg:4326", inplace=True)
            
            # Clip with district border
            diff_clipped = CMIP_differ.rio.clip([bound.geometry[i]])
            color = "tomato"
            if var_era == "mtpr":
                color = 'cornflowerblue'
                diff_clipped["diff"].data = K * diff_clipped["diff"].data
            mae_scores[region][model] =  mae(diff_clipped["diff"].data)

    if should_plot == True:
        fig = plt.figure(figsize=(8, 15))
        plt.barh(list(mae_scores["РФ"].keys()),
                list(mae_scores["РФ"].values()),
                height=0.5,
                color=color)
        plt.title("MAE for CMIP models) comparing to ERA, variable {}, {}".format(var_cmip, ssp_cur))
        plt.savefig(os.path.join(path_out,
                                "pics",
                                "mae_ensemble_{}_{}.png".format(var_cmip, ssp_cur)))
    
    return mae_scores


def plot_difference(
                ensembles: dict[str, list],
                ens_name: str,
                ssp_cur: str,
                years: list,
                var: str,
                path_in: str,
                path_out: str,
                bound: geopandas.geodataframe.GeoDataFrame
):
    '''
    Plots CMIP-ERA data difference pixelwise

    Args:
    ensembles dict[str, list]: dictionary of models in ensemble
    ens_name (str): ensemble name
    ssp_cur (str): used scenario
    years (list): years list
    var (str): variable name
    path_in (str) path to source data
    path_out (str): path to save data
    var_era_full (str): variable name
    bound (geopandas.geodataframe.GeoDataFrame): country boundary
    '''
    fig, ax = plt.subplots(2,1, figsize=(20, 10), height_ratios=[6, 1])
    ax.flatten()

    # Load variables names
    _, var_era, var_cmip = get_vars(var)

    # Open saved .nc file
    ens_avg = xr.open_dataset(os.path.join(path_in,
                                        "{}_{}".format(years[0], years[-1]),
                                        ssp_cur,
                                        'CMIP_{}_{}.nc'.format(ens_name, var_cmip)))
    era_avg = xr.open_dataset(os.path.join(path_in,
                                        "{}_{}".format(years[0], years[-1]),
                                        'ERA_{}.nc'.format(var_era)))

    era_avg = era_avg.interp(lat=ens_avg.lat, lon=ens_avg.lon)
    diff = (era_avg [var_era] - ens_avg["mean"])
    vmin = -5
    vmax = 5
    if var_era == "mtpr":
        diff = K * diff
        vmin = -200
        vmax = 200
    CMIP_differ = xr.combine_by_coords([ens_avg, era_avg])
    CMIP_differ["diff"] = (('lat', 'lon'), diff.data)

    # Clip with country border
    CMIP_differ = CMIP_differ.rio.set_spatial_dims('lon', 'lat', inplace=True)
    CMIP_differ.rio.write_crs("epsg:4326", inplace=True)
    clipped = CMIP_differ.rio.clip([bound.geometry[0]])

    # Plot
    im = clipped["diff"].plot(ax = ax[0],
                            vmin=vmin, vmax=vmax,
                            cmap = "Spectral",
                            add_colorbar=False)
    clipped["diff"].mean(['lat']).plot(ax = ax[1])

    ax[0].xaxis.set_visible(False)
    ax[1].set_ylabel("CMIP-ERA")
    ax[0].title.set_visible(False) 
    ax[1].title.set_visible(False)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, pad=0.2)
    fig.suptitle("CMIP (average of {} models) - ERA difference over years {} - {}, {}, {}".format(len(ensembles[ens_name]),
                                                                                                years[0],
                                                                                                years[-1],
                                                                                                var_cmip,
                                                                                                ssp_cur))
    plt.savefig(os.path.join(path_out,
                            "pics",
                            "diff_ensemble_{}_{}.png".format(var_cmip, ssp_cur)))
    plt.show()


def correlation_district(
    models: list,
    ens_name: str,
    path: str,
    var_cmip: str,
    var_era: str,
    ssp_cur: str,
    corr: pd.DataFrame,
    gdf_districts: pd.DataFrame,
    districts: dict[str, list]
) -> pd.DataFrame:
    '''
    Calculates CMIP-ERA data correlation district-wise

    Args:
    models (list): list of CMIP6 models
    ens_name (str): ensemble name
    path (str) path to data
    var_cmip (str): CMIP variable name
    var_era (str): ERA variable name
    ssp_cur (str): used scenario
    corr (pd.DataFrame): correlations
    gdf_districts (pd.DataFrame): districts with their geometries
    district (dict[str, list]): dictionary of districts and their regions

    Returns:
    corr (pd.DataFrame): correlations
    '''  

    #____________________________Ensemble evaluation_________________________________________
    # Open saved .nc file
    ens_avg = xr.open_dataset(os.path.join(path, ssp_cur, 'CMIP_{}_{}.nc'.format(ens_name, var_cmip)))
    era_avg = xr.open_dataset(os.path.join(path, 'ERA_{}.nc'.format(var_era)))

    era_avg = era_avg.interp(lat=ens_avg.lat, lon=ens_avg.lon)
    ens_differ = xr.combine_by_coords([ens_avg, era_avg])
    ens_differ = ens_differ.rio.set_spatial_dims('lon', 'lat', inplace=True)
    ens_differ.rio.write_crs("epsg:4326", inplace=True)
    
    # Clip with border. Whole country
    corr.loc[0, "NAME_1"] = "РФ"
    boundary = gdf_districts.loc[0,'geometry']
    clipped = ens_differ.rio.clip([boundary])
    corr.loc[0, ens_name] = xr.corr(clipped[var_era], clipped["mean"]).data.item()

    # Loop over districts
    for index, (district_name, district_regions) in enumerate(districts.items()):

        # Clip with border. District
        boundary = gdf_districts.loc[index+1,'geometry']

        corr.loc[index+1, "NAME_1"] = district_name    
        clipped = ens_differ.rio.clip([boundary])
        corr.loc[index+1, ens_name] = xr.corr(clipped[var_era], clipped["mean"]).data.item()

    #____________________________Models evaluation______________________________________________
    # Loop over models
    for model in models:
        model_avg = xr.open_dataset(os.path.join(path, ssp_cur, 'CMIP_{}_{}.nc'.format(var_cmip, model)))
        model_differ = xr.combine_by_coords([model_avg, era_avg])
        model_differ = model_differ.rio.set_spatial_dims('lon', 'lat', inplace=True)
        model_differ.rio.write_crs("epsg:4326", inplace=True)
        
        # Clip with border. Whole country
        boundary = gdf_districts.loc[0,'geometry']
        clipped = model_differ.rio.clip([boundary])
        corr.loc[0, model] = xr.corr(clipped[var_era], clipped[model]).data.item()

        # Loop over districts
        for index, (district_name, district_regions) in enumerate(districts.items()):

            # Clip with border. District
            boundary = gdf_districts.loc[index+1,'geometry']
            
            clipped = model_differ.rio.clip([boundary])
            corr.loc[index+1, model] = xr.corr(clipped[var_era], clipped[model]).data.item()

    return corr


def plot_region(
    corr_regions: dict[str, dict],
    models: list,
    gdf_region: geopandas.geodataframe.GeoDataFrame,
    vars: list,
    ssps: list,
    path: str
):
    '''
    Plots CMIP-ERA data correlations region-wise

    Args:
    corr_regions (dict[str, dict]): correlations
    models (list): list of CMIP6 models
    gdf_region (geopandas.DataFrame): regions with their geometries
    vars (list): variables
    ssps (list): used scenarios
    path (str) path to save pics
    '''  

    gdf_region.set_index("NAME_1", drop=True, inplace=True)

    for var in vars:
        for ssp in ssps:
            print(ssp)
            data = corr_regions[var][ssp].copy()
            data.set_index("NAME_1", drop=True, inplace=True)
            data.iloc[:,:-2] = data.iloc[:,:-2].astype(float)
            data.loc[:, "geometry"] = gdf_region["geometry"]

            df = geopandas.GeoDataFrame(
                    data,
                    crs="EPSG:4326"
                        )
            df = df.to_crs('ESRI:102027')

            for model in tqdm(models):
                fig, ax = plt.subplots(figsize=(20, 10))
                im = df.plot(ax = ax,
                            column=model,
                            legend=True,
                            vmin=0, vmax=1,
                        )
                plt.axis('off')

                plt.tight_layout()
                plt.savefig(os.path.join(path, "pics", "models", 'regions_corr_{}_{}_{}.png'.format(model, var, ssp)))
                fig.set_visible(not fig.get_visible())


def correlation_region(
    models: list,
    ens_name: str,
    path: str,
    var_cmip: str,
    var_era: str,
    ssp_cur: str,
    corr: pd.DataFrame,
    gdf_region: pd.DataFrame
) -> pd.DataFrame:
    '''
    Calculate sCMIP-ERA data correlation region-wise

    Args:
    models (list): list of CMIP6 models
    ens_name (str): ensemble name
    path (str) path to data
    var_cmip (str): CMIP variable name
    var_era (str): ERA variable name
    ssp_cur (str): used scenario
    corr (pd.DataFrame): correlations
    gdf_region (pd.DataFrame): regions with their geometries

    Returns:
    corr (pd.DataFrame): correlations
    '''  

    #____________________________Ensemble evaluation_________________________________________
    # Open saved .nc file
    ens_avg = xr.open_dataset(os.path.join(path, ssp_cur, 'CMIP_{}_{}.nc'.format(ens_name, var_cmip)))
    era_avg = xr.open_dataset(os.path.join(path, 'ERA_{}.nc'.format(var_era)))

    era_avg = era_avg.interp(lat=ens_avg.lat, lon=ens_avg.lon)
    ens_differ = xr.combine_by_coords([ens_avg, era_avg])
    ens_differ = ens_differ.rio.set_spatial_dims('lon', 'lat', inplace=True)
    ens_differ.rio.write_crs("epsg:4326", inplace=True)

    # Loop over regions
    for index in range(gdf_region.shape[0]):
        name = gdf_region.loc[index, 'NAME_1']

        # Clip with border of regions
        boundary = gdf_region.loc[index,'geometry']
        corr.loc[index, "NAME_1"] = name 
        clipped = ens_differ.rio.clip([boundary])
        corr.loc[index, ens_name] = xr.corr(clipped[var_era], clipped["mean"]).data.item()

    #____________________________Models evaluation______________________________________________
    # Loop over models
    for model in models:
        model_avg = xr.open_dataset(os.path.join(path, ssp_cur, 'CMIP_{}_{}.nc'.format(var_cmip, model)))
        model_differ = xr.combine_by_coords([model_avg, era_avg])
        model_differ = model_differ.rio.set_spatial_dims('lon', 'lat', inplace=True)
        model_differ.rio.write_crs("epsg:4326", inplace=True)
        
        # Loop over regions
        for index in range(gdf_region.shape[0]):

            # Clip with border of region
            boundary = gdf_region.loc[index,"geometry"]
            clipped = model_differ.rio.clip([boundary])
            corr.loc[index, model] = xr.corr(clipped[var_era], clipped[model]).data.item()
    return corr


def plot_corr(
        models: list,
        ens_name: str,
        ssps: list,
        years: list,
        var: str,
        path_in: str,
        path_out: str,
        gdf: pd.DataFrame,
        corr_dict: dict[str, dict],
        adm_type: str,
        districts: dict[str, list]={}
) -> dict[str, dict]:
    '''
    Calculates CMIP-ERA data correlation and plots it on the map

    Args:
    models (list): list of CMIP6 models
    ens_name (str): ensemble name
    ssps (list): used scenarios
    years (list): years list
    var (str): variable name
    path_in (str): path to data
    path_out (str) path to save pics
    gdf (pd.DataFrame): areas with their geometries
    corr_dict (dict[str, dict]): empty dict with correlations for each var and ssp
    district (dict[str, list]): dictionary of districts and their regions

    Returns:
    corr_dict (pd.DataFrame): dictionary with correlations for each var and ssp
    '''  

    # Load variables names
    _, var_era, var_cmip = get_vars(var)
    
    corr_dict[var_cmip] = {ssp: pd.DataFrame for ssp in ssps}
    for i, ssp in enumerate(ssps):
        fig, ax = plt.subplots(figsize=(20, 10))
        corr = pd.DataFrame(columns=['NAME_1'] +  models)
        if adm_type == "districts":
            corr = correlation_district(
                                        models,
                                        ens_name,
                                        os.path.join(path_in, "{}_{}".format(years[0], years[-1])),
                                        var_cmip,
                                        var_era,
                                        ssp,
                                        corr,
                                        gdf,
                                        districts)
        elif adm_type == "regions":
            corr = correlation_region(
                                    models,
                                    ens_name,
                                    os.path.join(path_in, "{}_{}".format(years[0], years[-1])),
                                    var_cmip,
                                    var_era,
                                    ssp,
                                    corr,
                                    gdf)

        corr_dict[var_cmip][ssp] = corr
        corr["geometry"] = gdf["geometry"]
        gdf_corr = geopandas.GeoDataFrame(
                                    corr,
                                    crs="EPSG:4326"
        )
        gdf_corr = gdf_corr.to_crs('ESRI:102027')
        im = gdf_corr.plot(ax = ax,
                        column=ens_name,
                        legend=True,
                        vmin=0, vmax=1
                        # legend_kwds={"shrink":.3}
        )
        plt.axis('off')
        plt.title("Correlation of CMIP (average of {} models) and ERA over years {} - {}, {}, {}".format(len(models),
                                                                                                        years[0],
                                                                                                        years[-1],
                                                                                                        var_cmip,
                                                                                                        ssp))
        plt.savefig(os.path.join(path_out,
                                "pics",
                                "{}_corr_ensemble_{}_{}.png".format(adm_type, var_cmip, ssp)))
    return corr_dict


def optimal_ensemble(
                    ensembles: dict[str, list],
                    ens_name: str,
                    corr_regions: dict[str, dict],
                    ssps: list,
                    num_models: int
):
    '''
    Collects optimal model ensemble for each region of RF 

    Args:
    ensembles (dict[str, dict]): : dictionary of models in ensemble
    ens_name (str): ensemble name
    corr_regions (dict[str, dict]): dictionary with correlations for each var-ssp-region
    ssps (list): used scenarios
    num_models (int): minimum number of models in ensemble

    Returns:
    best_models (dict[str, list]): dictionary with model ensemble for each region
    ''' 

    # Variables used (CMIP spelling)
    var_list = [
        "tas",
        "pr"
        ]

    # List all regions
    regions = corr_regions[var_list[0]][ssps[0]]["NAME_1"].copy()
    region_indeces = corr_regions[var_list[0]][ssps[0]].T.columns

    # Calculate statistics over all ssps and models
    # Dataframes with correlations collected through all ssp scenarios
    corr_regions_allssp = {var: pd.DataFrame for var in var_list}
    # DataFrames with statistics
    corr_avg = {var: pd.DataFrame for var in var_list}

    for var in var_list:
        corr_regions_allssp[var] = pd.DataFrame(columns=region_indeces)
        for ssp in ssps:
            corr_regions_allssp[var] = pd.concat([corr_regions_allssp[var], corr_regions[var][ssp].T.iloc[1:-2,:].astype(float)])
        # Percentiles
        stat = corr_regions_allssp[var].describe(percentiles=[.75, 0.9]).reset_index()
        stat.set_index("index", drop=True, inplace=True)
        corr_avg[var] = stat
    
    # Dictionary with model ensemble for each region
    best_models = {region:[] for region in regions}

    # Criterion 5
    region_indeces = regions.index.values
    regions_success = []

    for index, region in zip(region_indeces, regions):
        best_models[region] = []
        models = ensembles[ens_name].copy()
        for ssp in ssps:
            corr1 = corr_regions["tas"][ssp].T
            corr2 = corr_regions["pr"][ssp].T
            
            for model in models:
                percent_tas = corr_avg["tas"].loc["90%", index]
                percent_pr= corr_avg["pr"].loc["90%", index]
                if (corr1.loc[model, index]>percent_tas) & (corr2.loc[model, index]>percent_pr):
                    best_models[region].append((model, ssp))
                    models.remove(model)

        if len(best_models[region])>=num_models:
            regions.drop(labels = [index], inplace = True)
            regions_success.append(region)
    print("Criterion 5 achieved for regions {}".format(regions_success))

    # Criterion 4
    region_indeces = regions.index.values
    regions_success = []

    for index, region in zip(region_indeces, regions):
        best_models[region] = []
        models = ensembles[ens_name].copy()
        for ssp in ssps:
            corr1 = corr_regions["tas"][ssp].T
            corr2 = corr_regions["pr"][ssp].T
            
            for model in models:
                percent_tas = corr_avg["tas"].loc["75%", index]
                percent_pr= corr_avg["pr"].loc["90%", index]
                if (corr1.loc[model, index]>percent_tas) & (corr2.loc[model, index]>percent_pr):
                    best_models[region].append((model, ssp))
                    models.remove(model)
                    
        if len(best_models[region])>=num_models:
            regions.drop(labels = [index], inplace = True)
            regions_success.append(region)
    print("Criterion 4 achieved for regions {}".format(regions_success))

    # Criterion 3
    region_indeces = regions.index.values
    regions_success = []

    for index, region in zip(region_indeces, regions):
        best_models[region] = []
        models = ensembles[ens_name].copy()
        for ssp in ssps:
            corr1 = corr_regions["tas"][ssp].T
            corr2 = corr_regions["pr"][ssp].T
            
            for model in models:
                percent_tas = corr_avg["tas"].loc["75%", index]
                percent_pr= corr_avg["pr"].loc["75%", index]
                if (corr1.loc[model, index]>percent_tas) & (corr2.loc[model, index]>percent_pr):
                    best_models[region].append((model, ssp))
                    models.remove(model)
                    
        if len(best_models[region])>=num_models:
            regions.drop(labels = [index], inplace = True)
            regions_success.append(region)
    print("Criterion 3 achieved for regions {}".format(regions_success))

    # Criterion 2
    region_indeces = regions.index.values
    regions_success = []

    for index, region in zip(region_indeces, regions):
        best_models[region] = []
        models = ensembles[ens_name].copy()
        for ssp in ssps:
            corr1 = corr_regions["tas"][ssp].T
            corr2 = corr_regions["pr"][ssp].T
            
            for model in models:
                percent_tas = corr_avg["tas"].loc["75%", index]
                if corr1.loc[model, index]>percent_tas:
                    best_models[region].append((model, ssp))
                    models.remove(model)
                    
        if len(best_models[region])>=num_models:
            regions.drop(labels = [index], inplace = True)
            regions_success.append(region)
    print("Criterion 2 achieved for regions {}".format(regions_success))

    # Criterion 1
    region_indeces = regions.index.values
    regions_success = []

    for index, region in zip(region_indeces, regions):
        best_models[region] = []
        models = ensembles[ens_name].copy()
        for ssp in ssps:
            corr1 = corr_regions["tas"][ssp].T
            corr2 = corr_regions["pr"][ssp].T
            
            for model in models:
                percent_tas = corr_avg["tas"].loc["mean", index]
                if corr1.loc[model, index]>percent_tas:
                    best_models[region].append((model, ssp))
                    models.remove(model)
                    
        if len(best_models[region])>=num_models:
            regions.drop(labels = [index], inplace = True)
            regions_success.append(region)
    print("Criterion 1 achieved for regions {}".format(regions_success))
    best_models = {k:v for k,v in best_models.items()}

    return best_models