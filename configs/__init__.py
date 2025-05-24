import importlib
from argparse import Namespace
from ast import literal_eval
from util.net import get_timepc


def get_cfg(opt_terminal):
	opt_terminal.cfg_path = opt_terminal.cfg_path.split('.')[0].replace('/', '.')
	dataset_lib = importlib.import_module(opt_terminal.cfg_path)
	cfg = dataset_lib.cfg()
	for key, val in opt_terminal.__dict__.items():
		cfg.__setattr__(key, val)
	
	cfg.command = f'python3 -m torch.distributed.launch --nproc_per_node=$nproc_per_node --nnodes=$nnodes --node_rank=$node_rank --master_addr=$master_addr --master_port=$master_port --use_env run.py -c {cfg.cfg_path} -m {cfg.mode} --sleep {cfg.sleep} --memory {cfg.memory}'
	cfg.task_start_time = get_timepc()
	return cfg


if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--cfg_path', default='configs/RD_test/rd_mvtec.py')
	parser.add_argument('-m', '--mode', default='train', choices=['train', 'test'])
	parser.add_argument('--sleep', type=int, default=-1)
	parser.add_argument('--memory', type=int, default=-1)
	parser.add_argument('--dist_url', default='env://', type=str, help='url used to set up distributed training')
	parser.add_argument('--logger_rank', default=0, type=int, help='GPU id to use.')
	parser.add_argument('opts', help='path.key=value', default=None, nargs=argparse.REMAINDER, )
	cfg_terminal = parser.parse_args()

	cfg = get_cfg(cfg_terminal)
	print(cfg)
