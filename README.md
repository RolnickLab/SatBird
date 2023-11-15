# SatBird Dataset and Benchmarks

This repository provides the code required to create the dataset and reproduce the benchmark of:

M. Teng, A. Elmustafa, B. Akera, Y. Bengio, H. Abdelwahed, H. Larochelle and D. Rolnick. ["SatBird: a Dataset for Bird Species Distribution Modeling using Remote Sensing and Citizen Science Data"](), *NeurIPS 2023 Datasets and Benchmarks track*

You can also visit the project's website [here](https://satbird.github.io/).

Reported NN baselines are: Resnet18, SATLAS, SatMAE.

### Running code:

#### Installation 
Code runs on Python 3.10. You can create conda env using `requirements/environment.yaml` or install pip packages from `requirements/requirements.txt`

We recommend following these steps for installing the required packages: 

```conda env create -f requirements/environment.yaml``` 

```conda activate satbird```

```conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia```

#### Training and testing

* To train the model (check `run_files/job.sh`) : `python train.py args.config=configs/base.yaml`. Examples of all config files for different baselines 
are available in `configs`.
* To test the model: `python test.py args.config=$CONFIG_FILE_NAME `

* To launch multiple parallel runs on the cluster, `sbatch run_files/multiple_runs.sh`"

* To run environmental baselines are in `python data_processing/env_baseline.py config_file=configs/env_baseline.yaml`

* To launch multiple experiments (with 3 seeds) on the cluster, run `sbatch run_files/multiple_runs.sh`

To log experiments on comet-ml, make sure you have exported your COMET_API_KEY and COMET_WORKSPACE in your environmental variables.
You can do so with `export COMET_API_KEY=your_comet_api_key` in your terminal.  

To run the environmental baseline and the mean encounter rate baseline, refer to `data_processing/environmental/env_baseline.py`. \\
For the MOSAIKS baseline, please refer to the [original MOSAIKS implementation](https://github.com/Global-Policy-Lab/mosaiks-paper).

### Testing
Tests assume access to data files, so it has to run on the cluster
To run all tests, in root directory, run: ```pytest```
Some tests that go through all datafiles are marked with "slow", ```pytest -k "slow"```
Other tests that run faster can be run with:  ```pytest -k "fast" ```
To run a specific test, example: ```pytest tests/data/test_data_files.py -k "test_corresponding_files_from_csv" -s```

### Repository structure

The repo includes implementations for all the experiments/baselines reported in the paper: 
a) env-baseline: data_processing/environmental/env_baseline.py 
b) satmae: src/models/vit.py 
c) Resnet18 (all variants)
d) satlas: src/trainer/trainer.py
e) MOSAIKS: refer to the original [MOSAIKS](https://www.mosaiks.org/) implementation and details in the SatBird paper. 


### Dataset format

The SatBird dataset is available in this [Drive](https://drive.google.com/drive/folders/1eaL2T7U9Imq_CTDSSillETSDJ1vxi5Wq).
Each folder is organized as follows:

```
├── USA_summer / USA_winter / Kenya
|  └── train_split.csv
|  └── valid_split.csv
|  └── test_split.csv
|  └── range_maps.pkl
|  └── images/{hotspot_id}.tif
|  └── images_visual/{hotspot_id}_visual.tif
|  └── environmental_data/{hotspot_id}.npy
|  └── targets/{hotspot_id}.json
```

### Reproducing the dataset creation pipeline

You will first need to create a Cornell lab account to [ask for access to and download the eBird Basic Dataset (EBD)](https://support.ebird.org/en/support/solutions/articles/48000838205-download-ebird-data). 
- Download the raw **EBD data** for your region of interest in our case, United States and Kenya), using the custom download option. Download the ebd sampoling event file as well. This will save some time for processing the dataset because you will be able to select the checklists of interest. 
- Run `sbatch filter2.awk raw_ebd_data_file.txt `script to group observations by locality. The script on the repository was the one used to create the USA dataset, where we kept checklists of June, July, December and January to prepare the USA-summer and USA-winter dataset. For Kenya no month filter was applied.
This will create one file per hotspots, the files will be organized based on the hotspot name in `split-XXX/`  folders where each folder should only contain .csv files for hotspots `L*XXX`. 
- Run  `data_processing/ebird/find_checklists.py` to filter the checklists you want to keep based on the ebd sampling event data. Typically this will only keep complete checklists and will create (for the USA dataset) 2 .csv files with the hotspots to keep corresponding to observations made in the summer and winter. 
- to get the targets, run `data_processing/ebird/get_winter_targets.py` . Change the paths to the one where you created the `split-XXX/` folders.


- Download the **Sentinel-2 data** using `data_processing/satellite/download_rasters_from_planetary_computer.py`. To reproduce all experiments, you will have to run it twice, specifying the BANDS as ["B02", "B03", "B04", "B08"] for the BGRNIR reflectance data, and "visual" for the RGB visual component. A useful function is `process_row` which will extract the least cloudy image (with your specified bands) over the specified period of time with a maximum of 10\% cloud cover. For some hotspots, it is possible that you will be able to extract the visual component but incomplte items of no item will be found for the R,B,G,NIR reflectance data with the cloud cover criterion of 10\%. For those hotspots, you can replace `process_row` with `process_row_mosaic`function to allow the extracted image to be a mosaic of the images found over the specified period of time.  

- You can further clean the dataset using the functions in `data_processing/ebird/clean_hotspots.py` and filter out:
    - hotspots that are not withing the bounds of a given shapefile geometry. This was used for the USA dataset to filter out hotspots that are in the middle of the ocean and not in the contiguous USA
    - hotspots for which no satellite image was found
    - hotspot for which the extracted satellite image has height or width less than 128 pixels
- Additionally, you should merge hotspots that have different IDs but the same latitude and longitude using `data_processing/ebird/clean_duplicate_lat_lon.py`
- Finally, merge the targets for those merged hotspots using `data_processing/ebird/merge_target.py`

- Get **range maps** using `data_processing/ebird/get_range_maps.py`. This will call for shapefiles that you can obtain through [ebirdst](https://ebird.github.io/ebirdst/). You can then save a csv of all combined range maps using `/satbird/data_processing/utils/save_range_maps_csv_v2.py`.
- 
- For the environmental data variables, download the rasters of the country of insterest from [WorldClim](https://www.worldclim.org/) (and [SoilGrids](https://soilgrids.org/) for the USA dataset). 


- Use `data_processing/environmental/get_csv_env.py` to get point data for the environmental variables (rasters of size 1 centered on the hotspots). Note that you should not fill nan values until you have done the train-validation-test split so you can fill the values with the means on the training set data. These point data variables are used for the mean encounter rates, environmental and MOSAIKS baselines 

- Use `data_processing/environmental/get_env_var.py` to get the environmental rasters centered on the hotspots. The rasters are extracted with size 50 * 50 and 6 * 6 for the USA and Kenya datasets respectively.  
NaN values in the rasters are filled with bilinear interpolation when possible using `data_processing/environmental/fill_env_nans.py`. Then you can use the functions in  `data_processing/environmental/bound_data.py` to clip outlier values due to the interpolation, and fill values that are still NaN with the mean of a given environmental variable over the dataset (this will used the values obtained with `data_processing/environmental/get_csv_env.py`). 

**Dataset splits**
Use `data_processing/utils/make_splits_by_distance.py` to your dataset and reduce spatial autocorrelation, compared to random splitting. 


This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License](https://creativecommons.org/licenses/by-nc/4.0/).
