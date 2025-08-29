import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import argparse
from configs import get_cfg
from util.net import init_training
from util.util import run_pre, init_checkpoint, setup_cfg
from trainer import get_trainer
import warnings
from data import CLASS_NAMES
warnings.filterwarnings("ignore")


def main():
	parser = argparse.ArgumentParser()

	# parser.add_argument('-c', '--cfg_path', default='configs/benchmark/cdo/cdo_100e.py')
	# parser.add_argument('-c', '--cfg_path', default='configs/benchmark/msflow/msflow_100e.py')
	# parser.add_argument('-c', '--cfg_path', default='configs/benchmark/dinomaly/dinomaly_100e.py')
	parser.add_argument('-c', '--cfg_path', default='configs/benchmark/inpformer/inpformer_100e.py')
	# parser.add_argument('-c', '--cfg_path', default='configs/benchmark/rdpp/rdpp_256_100e.py')

	parser.add_argument('-m', '--mode', default='train', choices=['train', 'test'])
	parser.add_argument('-s', '--setup', default='M2AD-Invariant', choices=['M2AD-Synergy', 'M2AD-Invariant'])

	parser.add_argument('--sleep', type=int, default=-1)
	parser.add_argument('--memory', type=int, default=-1)

	parser.add_argument('--checkpoint', default='runs', type=str)

	for cls in CLASS_NAMES['m2ad']:

		cfg_terminal = parser.parse_args()
		cfg = get_cfg(cfg_terminal)
		if cfg_terminal.checkpoint:
			cfg.trainer.checkpoint = cfg_terminal.checkpoint
			print(f"本次训练的checkpoint路径为：{cfg_terminal.checkpoint}")
		cfg = setup_cfg(cfg, cfg.setup, cls)
		run_pre(cfg)
		init_training(cfg)
		init_checkpoint(cfg)
		trainer = get_trainer(cfg)
		trainer.run()

		del trainer

if __name__ == '__main__':
	main()
