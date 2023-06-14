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

### Repository structure



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
