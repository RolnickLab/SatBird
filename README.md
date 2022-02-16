# ecosystem-embedding

Make sure you have exported your COMET_API_KEY and COMET_WORKSPACE in your environmental variables.
You can do so with `export COMET_API_KEY=your_comet_api_key` in your terminal.  

Environmental baselines are in `data_processing/env_baseline.py`

     python data_processing/env_baseline.py config_file="path/to/config/file"

To see an example config file check `configs/env_baseline.yaml`


To train the model : `python ~/ecosystem-embedding/train2.py config_file=configs/custom_meli.yaml`

To test the model: `python ~/ecosystem-embedding/test.py config_file=configs/custom_meli.yaml test_config_file=configs/custom_test.yaml `
`config_file`must be the same as the one you used to train your model. 