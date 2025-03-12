#!/bin/bash

# Define the crop and test year
crop="soybeans"
year=2022
lr=1e-4
bs=32
max_epoch=200
sleep_time=100
dropout_train=0.2
dropout_test=0.1
z_dim=64
weight_decay=1e-5

# Define the list of regions

regions=(
    "Eastern-Uplands"
    "Heartland"
    "Mississippi-Portal"
    "Northern-Crescent"
    "Northern-Great-Plains"
    "Prairie-Gateway"
    "Southern-Seaboard"
)

for region in "${regions[@]}"
do
    # Training phase: Exclude one region (denoted by NOT-region)
    echo "Training: Excluding $region"
    python main.py --dataset ${crop}_weekly --crop_type $crop --test_year $year \
        --data_dir ../data/cropnet_data/LCO/train/weekly_NOT-${region}.npz \
        -adj ../map/us_adj.pkl --crop_id_to_fid ../map/soybean_fid_dict.pkl \
        --mode train --length 5 -bs $bs --max_epoch $max_epoch -sleep $sleep_time \
        -lr $lr --sche cosine --T0 100 --eta_min 1e-6 --check_freq 80  \
        --T_mult 2 --lrsteps 25 --gamma 1 \
        --dropout $dropout_train --z_dim $z_dim --weight_decay $weight_decay \
        --no_management --aggregator_type pool --encoder_type cnn --seed 0 

    # Find the model directory dynamically
    model_dir=$(find model/${crop}_weekly/${year}/ -type d -name "gnn-rnn_tr-dataset-NOT-${region}*" 2>/dev/null | head -n 1)

    if [[ -z "$model_dir" ]]; then
        echo "Error: Model directory not found for region ${region}. Skipping test phase."
        continue
    fi

    # Find the highest-numbered model checkpoint within that directory
    model_checkpoint=$(ls -d $model_dir/model-* 2>/dev/null | sort -V | tail -n 1)

    if [[ -z "$model_checkpoint" ]]; then
        echo "Error: No model checkpoint found in ${model_dir}. Skipping test phase."
        continue
    fi

    # Testing phase: Use the excluded region for inference
    echo "Testing on excluded region: $region using checkpoint: $model_checkpoint"
    python main.py --dataset ${crop}_weekly --crop_type $crop --test_year $year \
        --data_dir ../data/cropnet_data/LCO/val/weekly_${region}.npz \
        -adj ../map/us_adj.pkl --crop_id_to_fid ../map/soybean_fid_dict.pkl \
        --mode test --length 5 -bs $bs --max_epoch $max_epoch -sleep $sleep_time \
        -lr $lr --sche cosine --T0 100 --eta_min 1e-6 --check_freq 80  \
        --T_mult 2 --lrsteps 25 --gamma 1 \
        --dropout $dropout_test --z_dim $z_dim --weight_decay $weight_decay \
        --no_management --aggregator_type pool --encoder_type cnn \
        --mask_prob 1 --mask_value county_avg \
        -cp "$model_checkpoint"
done
