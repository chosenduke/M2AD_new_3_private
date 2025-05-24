CUDA_VISIBLE_DEVICES=1 python run_dataset.py --cfg_path configs/benchmark/cdo/cdo_100e.py -m train -s M2AD-Synergy
CUDA_VISIBLE_DEVICES=1 python run_dataset.py --cfg_path configs/benchmark/msflow/msflow_100e.py -m train -s M2AD-Synergy
CUDA_VISIBLE_DEVICES=1 python run_dataset.py --cfg_path configs/benchmark/dinomaly/dinomaly_100e.py -m train -s M2AD-Synergy
CUDA_VISIBLE_DEVICES=1 python run_dataset.py --cfg_path configs/benchmark/inpformer/inpformer_100e.py -m train -s M2AD-Synergy
CUDA_VISIBLE_DEVICES=1 python run_dataset.py --cfg_path configs/benchmark/rdpp/rdpp_256_100e.py -m train -s M2AD-Synergy
