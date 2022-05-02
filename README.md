# ecosystem-embedding

Make sure you have exported your COMET_API_KEY and COMET_WORKSPACE in your environmental variables.
You can do so with `export COMET_API_KEY=your_comet_api_key` in your terminal.  

Environmental baselines are in `data_processing/env_baseline.py`

     python data_processing/env_baseline.py config_file="path/to/config/file"

To see an example config file check `configs/env_baseline.yaml`


To train the model : `python ~/ecosystem-embedding/train2.py args.config=configs/custom_meli.yaml`

To test the model: `python ~/ecosystem-embedding/test.py args.config=configs/custom_meli.yaml ++test_config_file=configs/custom_test.yaml `

`config_file`must be the same as the one you used to train your model. 

### Repository structure



### Data 

Our shared folder is located at `/network/projects/_groups/ecosystem-embeddings`

Hotspots information: 
- `/network/projects/_groups/ecosystem-embeddings/data/hotspot_data/hotspots_data_with_bioclim.csv` : contains information for each hotspot about the location, the number of complete checklists, and the number of complete checklists for the month of june as well as bioclimatic and pedologic variables values for the lat-lon of the hotspot.


Satellite data is located at `/network/scratch/t/tengmeli/scratch/ecosystem-embedding/satellite_data/`.
For each hotspot with id LXXXX, there are 6 files : 
- LXXXX_r.npy : R band
- LXXXX_g.npy : G band
- LXXXX_b.npy : B band
- LXXXX_ni.npy : NIR band
- LXXXX_rgb.npy : true color image
- LXXXX.json : image metadata (lat-lon of the hotspot, date of the satellite image)

Species data is located at `/network/scratch/t/tengmeli/scratch/ecosystem-embedding/ebird_data_june/`
Each LXXXX.json file contains the "hotspot_id" as well as "probs" which is the target vector for all 684 species. 
In `/network/projects/_groups/ecosystem-embeddings/species_splits` are .npy files of species indices for making splits. Typically we will use `not_songbirds_idx.npy` which contains the indices of non songbirds for training. 

Splits : 
in `/network/projects/_groups/ecosystem-embeddings/species_splits/` you will find `train_june_vf.csv`, `val_june_vf.csv`, `test_june_vf.csv` which correspond to our data splits. 
Each line in the csvs corresponds to one hotspot and each column point to the data for the variable of interest for the otspot (r,g,b,nir bands, species data, bioclimatic and pedologic data).



