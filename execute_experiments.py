import os
from itertools import product

import argparse

def main(args):

    METHOD = args.method
    DATASET = args.dataset
    # NOISE_STRENGTH = float(args.noise_strength)

    for NOISE_STRENGTH in [0.2, 0.4, 0.6]:

        dataset_list = [DATASET]
        noise_strength_list = [NOISE_STRENGTH]
        noise_type_list = ['uniform', 'idl']

        for exp_combo in product(dataset_list, noise_type_list, noise_strength_list):
            dataset, noise_type, noise_strength = exp_combo
            bg = ' ' if noise_type == noise_type_list[-1] else '&'

            if METHOD == 'oursdev':
                # # ours
                if dataset == 'mnist':
                    gamma_initial = 0.5
                    gamma_multiplier = 0.1
                    warm_up = 1
                    rollWindow = 1
                if dataset == 'cifar10':
                    gamma_initial = 0.5
                    gamma_multiplier = 0.1
                    warm_up = 20
                    rollWindow = 10
                cmd = f"python -W ignore train_ours_binary_share_weighted_dev.py" +\
                      f" --gpus {args.gpus}" +\
                      f" --dataset {dataset}" +\
                      f" --noise_type {noise_type}"+\
                      f" --noise_strength {noise_strength}"+\
                      f" --gamma_initial {gamma_initial}"+\
                      f" --gamma_multiplier {gamma_multiplier}"+\
                      f" --warm_up {warm_up}"+\
                      f" --rollWindow {rollWindow}"+\
                      f" --figure {bg}"
                os.system(cmd)
            elif METHOD == 'oursv2':
                # # ours_v1
                if dataset == 'mnist':
                    gamma_initial = 0.5
                    gamma_multiplier = 0.1
                    warm_up = 1
                    rollWindow = 1
                if dataset == 'cifar10':
                    gamma_initial = 0.5
                    gamma_multiplier = 0.1
                    warm_up = 20
                    rollWindow = 10
                cmd = f"python -W ignore train_ours_binary_share_weighted_v2.py" +\
                      f" --gpus {args.gpus}" +\
                      f" --dataset {dataset}" +\
                      f" --noise_type {noise_type}"+\
                      f" --noise_strength {noise_strength}"+\
                      f" --gamma_initial {gamma_initial}"+\
                      f" --gamma_multiplier {gamma_multiplier}"+\
                      f" --warm_up {warm_up}"+\
                      f" --rollWindow {rollWindow}"+\
                      f" --figure {bg}"
                os.system(cmd)
            elif METHOD == 'ts':
                # # T-scaling
                if os.path.exists("./baselines"):
                    os.chdir("./baselines")
                cmd = f"python -W ignore run_temperature_scaling.py"+\
                      f" --gpus {args.gpus}"+\
                      f" --dataset {dataset}"+\
                      f" --noise_type {noise_type}"+\
                      f" --noise_strength {noise_strength}"+\
                      f" --figure {bg}"
                os.system(cmd)
            elif METHOD == 'mcdrop':
                # # MCDrop
                if os.path.exists("./baselines"):
                    os.chdir("./baselines")
                cmd = f"python -W ignore run_bayes_dropout.py"+\
                      f" --gpus {args.gpus}"+\
                      f" --dataset {dataset}"+\
                      f" --noise_type {noise_type}"+\
                      f" --noise_strength {noise_strength}"+\
                      f" --figure {bg}"
                os.system(cmd)
            elif METHOD == 'ensemble':
                # # Ensemble
                if os.path.exists("./baselines"):
                    os.chdir("./baselines")
                cmd = f"python -W ignore run_ensemble.py"+\
                      f" --gpus {args.gpus}"+\
                      f" --dataset {dataset}"+\
                      f" --noise_type {noise_type}"+\
                      f" --noise_strength {noise_strength}"+\
                      f" --figure {bg}"
                os.system(cmd)
            elif METHOD == 'cskd':
                # # CSKD
                if os.path.exists("./baselines"):
                    os.chdir("./baselines")
                cmd = f"python -W ignore run_cskd.py"+\
                      f" --gpus {args.gpus}"+\
                      f" --dataset {dataset}"+\
                      f" --noise_type {noise_type}"+\
                      f" --noise_strength {noise_strength}"+ \
                      f" --figure {bg}"
                os.system(cmd)
            elif METHOD == 'gpc':
                ## GPC
                if os.path.exists("./baselines"):
                    os.chdir("./baselines")
                cmd = f"python -W ignore run_gpc.py"+\
                      f" --gpus {args.gpus}"+\
                      f" --dataset {dataset}"+\
                      f" --noise_type {noise_type}"+\
                      f" --noise_strength {noise_strength}"+ \
                      f" --figure {bg}"
                os.system(cmd)
            elif METHOD == 'focalloss':
                ## Focal Loss
                if os.path.exists("./baselines"):
                    os.chdir("./baselines")
                cmd = f"python -W ignore run_focalloss.py"+\
                      f" --gpus {args.gpus}"+\
                      f" --dataset {dataset}"+\
                      f" --noise_type {noise_type}"+\
                      f" --noise_strength {noise_strength}"+ \
                      f" --figure {bg}"
                os.system(cmd)
            else:
                print(f"No such baseline! {METHOD}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Arguement for SLDenoise")
    parser.add_argument("--gpus", type=str, help="Indices of GPUs to be used", required=True)
    parser.add_argument("--method", type=str, help="Method", choices={'oursdev', 'oursv2', 'ts', 'mcdrop', 'ensemble', 'cskd', 'gpc', 'focalloss'}, required=True)
    parser.add_argument("--dataset", type=str, help="Experiment Dataset", choices={'mnist', 'cifar10', 'cifar100'}, required=True)
    # parser.add_argument("--noise_strength", type=float, help="Noise fraction", choices={0.2, 0.4, 0.6, 0.8})
    args = parser.parse_args()

    main(args)