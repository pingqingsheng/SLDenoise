import os
from itertools import product

import argparse

def main(args):

    METHOD = args.method
    DATASET = args.dataset
    # NOISE_STRENGTH = float(args.noise_strength)

    for NOISE_STRENGTH in [0.2, 0.4, 0.6, 0.8]:

        dataset_list = [DATASET]
        noise_strength_list = [NOISE_STRENGTH]
        noise_type_list = ['idl', 'linear', 'uniform', 'asymmetric']

        for exp_combo in product(dataset_list, noise_type_list, noise_strength_list):
            dataset, noise_type, noise_strength = exp_combo
            bg = ' ' if noise_type == noise_type_list[-1] else '&'

            if METHOD == 'forward':
                if os.path.exists("./baselines"):
                    os.chdir("./baselines")
                if dataset == 'mnist':
                    warm_up = 1
                    rollWindow = 1
                if dataset == 'cifar10':
                    warm_up = 25
                    rollWindow = 10
                cmd = f"python -W ignore run_forwardcorr.py"+\
                      f" --gpus {args.gpus}"+\
                      f" --dataset {dataset}"+\
                      f" --noise_type {noise_type}"+\
                      f" --noise_strength {noise_strength}"+ \
                      f" --warm_up {warm_up}" + \
                      f" --rollWindow {rollWindow}" + \
                      f" --figure {bg}"
                os.system(cmd)

            elif METHOD == 'coteachingplus':
                if os.path.exists("./baselines"):
                    os.chdir("./baselines")
                if dataset == 'mnist':
                    warm_up = 0
                    rollWindow = 1
                if dataset == 'cifar10':
                    warm_up = 20
                    rollWindow = 10
                cmd = f"python -W ignore run_coteachingplus.py" +\
                      f" --gpus {args.gpus}" +\
                      f" --dataset {dataset}" +\
                      f" --noise_type {noise_type}"+\
                      f" --noise_strength {noise_strength}"+\
                      f" --warm_up {warm_up}"+\
                      f" --rollWindow {rollWindow}"+\
                      f" --figure {bg}"
                os.system(cmd)

            elif METHOD == 'mixup':
                if os.path.exists("./baselines"):
                    os.chdir("./baselines")
                if dataset == 'mnist':
                    warm_up = 0
                    rollWindow = 1
                if dataset == 'cifar10':
                    warm_up = 20
                    rollWindow = 10
                cmd = f"python -W ignore run_mixup.py"+\
                      f" --gpus {args.gpus}"+\
                      f" --dataset {dataset}"+\
                      f" --noise_type {noise_type}"+\
                      f" --noise_strength {noise_strength}"+ \
                      f" --warm_up {warm_up}" + \
                      f" --rollWindow {rollWindow}" + \
                      f" --figure {bg}"
                os.system(cmd)

            elif METHOD == 'elr':
                if os.path.exists("./baselines"):
                    os.chdir("./baselines")
                cmd = f"python -W ignore run_elr.py"+\
                      f" --gpus {args.gpus}"+\
                      f" --dataset {dataset}"+\
                      f" --noise_type {noise_type}"+\
                      f" --noise_strength {noise_strength}"+\
                      f" --figure {bg}"
                os.system(cmd)

            else:
                print(f"No such baseline! {METHOD}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguement for SLDenoise")
    parser.add_argument("--gpus", type=str, help="Indices of GPUs to be used")
    parser.add_argument("--method", type=str, help="Method", choices={'forward', 'coteachingplus', 'mixup', 'elr'})
    parser.add_argument("--dataset", type=str, help="Experiment Dataset", choices={'mnist', 'cifar10'})
    # parser.add_argument("--noise_strength", type=float, help="Noise fraction", choices={0.2, 0.4, 0.6, 0.8})
    args = parser.parse_args()

    main(args)