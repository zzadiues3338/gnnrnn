echo "Hello, we are in run_train.sh. About to call python (2018)."


# BEST 2018 corn
# python main.py --dataset corn_weekly_no_Y_input_shuffle --data_dir /mnt/beegfs/bulk/mirror/jyf6/datasets/crop_forecast/data/combined_dataset_weekly.npz -adj ../map/us_adj.pkl --crop_id_to_fid ../map/soybean_fid_dict.pkl --crop_type corn --mode train --length 5 -bs 32 --max_epoch 100 --sleep 100 --test_year 2018 -lr 5e-5 --check_freq 80 --sche cosine --eta_min 1e-6 --T0 100 --T_mult 2 --lrsteps 25 --gamma 1 --dropout 0.1 --num_weather_vars 23 --num_management_vars 96 --num_soil_vars 20 --num_extra_vars 6 --soil_depths 6 --aggregator_type pool --encoder_type cnn --no_management --train_week_start 52 --validation_week 52 --seed 0 --weight_decay 1e-5 --mask_prob 0.5 --mask_value zero


for crop in soybeans
do
    for lr in 1e-4
    do
        for seed in 0 1 2
        do
            python main.py --dataset ${crop}_weekly --crop_type $crop --test_year 2022 \
                --data_dir ../data/cropnet_data/weekly_heartland.npz \
                -adj ../map/us_adj.pkl --crop_id_to_fid ../map/soybean_fid_dict.pkl \
                --mode train --length 5 -bs 32 --max_epoch 100 -sleep 100 \
                -lr $lr --sche cosine --T0 100 --eta_min 1e-6 --check_freq 50 \
                --dropout 0.3 --weight_decay 1e-4 --n_layers 2 \
                --no_management --aggregator_type pool --encoder_type cnn --seed $seed
        done
    done
done

