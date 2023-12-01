# Scenarios
This repository contains machine learning models for CMIP6 models ranking for according to their correlation with ERA5 data.\
The comparison works either at level of climatically relevant areas (close to federal districts) or at the level of administrative division of Russian Federation.\
It accounts 16 CMIP6 models and 4 Shared Socioeconomic Pathways in total.

## Dependencies

Dependencies are listed in requirements.txt file.

## Data

The data required for the project is available in the `mnt\public-datasets\taniushkina\RCP_scenarios\data` folder.
It contains various files in .nc and .json formats.
CMIP files were loaded with the [script](https://github.com/makboard/WindUtils/blob/main/CMIP/auto_download.py).\
Administrative boundaries were loaded from [GADM project](https://gadm.org/download_country.html).\
Districts division withing Russian Federation performed by hand and stored in `json` files.\

The project directory structure should be organized as follows (tree depth is limited by 2):
``` bash
.
|-- Dockerfile
|-- README.md
|-- data
|   |-- boundary
|   |-- cmip
|   |-- districts
|   `-- era
|-- data_processed
|   |-- 2015_2022
|   `-- yearly
|-- environments
|   `-- requirements.txt
|-- notebooks
|   `-- RCP_analysis.ipynb
|-- results
|   |-- csv
|   `-- pics
`-- src
    |-- correlation.py
    `-- process.py
```

## Docker

To set up the project using Docker, follow these steps:

* Build the Docker image: `docker build -t rcp .`
* Run the Docker container: `docker run -it  -v  <CODE FOLDER>:/rcp -v <DATA FOLDER>:/rcp/data -m 128000m  --cpus=4  -w="/rcp" rcp`

## Executing program
The program consists of several steps, all of them are displayed in the `notebook/analysis.ipynb` notebook.
