#!/bin/bash

init=('no' 'means')
bands=(['r','g','b'] ['r','g','b','nir'])

for b in ${bands[@]}; do
    for i in ${init[@]}; do
    sbatch ./train_.sh ++comet.project_name="resnet18_base" ++data.bands=$b ++experiment.module.init_bias=$i  ++trainer.log_every_n_steps=5  args.config="/home/mila/t/tengmeli/ecosystem-embedding/configs/weight_loss/base_means_rgb.yaml"
    done
done

