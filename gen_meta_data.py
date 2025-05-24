import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from data.gen_metadata import GEN_DATA_SOLVER
import argparse
import warnings
warnings.filterwarnings("ignore")


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-d', '--dataset', default='m2ad')
	cfg_terminal = parser.parse_args()

	dataset = cfg_terminal.dataset

	solver = GEN_DATA_SOLVER.get(dataset, None)

	if solver:
		solver().run()
	else:
		print(f'Only support {GEN_DATA_SOLVER.keys()}, but entered {dataset}..')

if __name__ == '__main__':
	main()
