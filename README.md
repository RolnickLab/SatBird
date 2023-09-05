# SatBird dataset Benchmarks

Reported NN baselines are: Resnet18, SATLAS, SatMAE.

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
To run a specific test, example: ```pytest tests/data/test_data_files.py -k "test_corresponding_files_from_csv -s"```

### Repository structure

The repo includes implementations for all the experiments/baselines reported in the paper: 
a) env-baseline: dataprocessing/environmental/envbaseline.py 
b) satmae: src/models/vit.py 
c) Resnet18 (all variants)
d) satlas: src/trainer/trainer.py



### Dataset format
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


- Download the **Sentinel-2 data** using `data_processing/ebird_data_preparation/download_rasters_from_planetary_computer.py`
- You can further clean the dataset using the functions in `data_processing/ebird/clean_hotspots.py` and filter out:
    - hotspots that are in the middle of the ocean and not in the contiguous USA
    - hotspots for which no satellite image was found 
    - hotspot for which the extracted satellite image has height or width less than 128 pixels
- Additionally, you should merge hotspots that have different IDs but the same latitude and longitude using `data_processing/ebird/clean_duplicate_lat_lon.py`
- Finally, merge the targets for those merged hotspots using `data_processing/ebird/merge_target.py`

- Get **range maps** using `data_processing/ebird_data_preparation/get_range_maps.py`
- For the environmental data variables, download the rasters of the country of insterest from [WorldClim](https://www.worldclim.org/) (and [SoilGrids](https://soilgrids.org/) for the USA dataset). 
- Use `data_processing/environmental/get_env_var.py` to get the environmental 50 * 50  rasters centered on the hotspots, and `data_processing/environmental/get_csv_env.py` to get point data for the environmental variables. Note that you should not fill nan values until you have done the train-validation-test split so you can fill the values with the means on the training set data. 


**Dataset splits**
Use `data_processing/utils/make_splits_by_distance.py` to your dataset and reduce spatial autocorrelation, compared to random splitting. 

This work is licensed under a
[Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License](https://creativecommons.org/licenses/by-nc/4.0/).
