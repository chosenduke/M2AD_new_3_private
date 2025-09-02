cd /home/duke/playground/side_pro/test_m2ad/M2AD_new_1_private && \
CUDA_VISIBLE_DEVICES=0 \
nohup python -u run_dataset.py \
    --cfg_path configs/benchmark/inpformer/inpformer_100e_plus.py \
    -m train \
    -s M2AD-Synergy \
    --checkpoint new_only_birds_v4_runs \
    > new_logs/run_inpformer_synergy_new_only_birds_v4_runs_0902_0140.log 2>&1 &