crop="soybeans"
year=2022
lr=1e-4
python main.py --dataset ${crop}_weekly --crop_type $crop --test_year $year \
    --data_dir ../data/cropnet_data/weekly_test_eastern_uplands.npz \
    -adj ../map/us_adj.pkl --crop_id_to_fid ../map/soybean_fid_dict.pkl \
    --mode test --length 5 -bs 32 --max_epoch 100 -sleep 100 \
    -lr $lr --sche cosine --T0 100 --eta_min 1e-6 --check_freq 80  \
    --T_mult 2 --lrsteps 25 --gamma 1 \
    --dropout 0.1 --z_dim 64 --weight_decay 1e-5 \
    --no_management --aggregator_type pool --encoder_type cnn \
    --validation_week 52 --mask_prob 1 --mask_value county_avg \
    -cp model/soybeans_weekly/2022/gnn-rnn_tr-dataset-heartland_bs-32_lr-0.0001_sche-cosine_etamin-1e-06_step-25_dropout-0.1_testyear-2022_aggregator-pool_encoder-cnn_weightdecay-1e-05_seed-0_no-management/model-35