import os
from itertools import product

import argparse

def main(args):

    METHOD = args.method
    DATASET = args.dataset
    NOISE_STRENGTH = float(args.noise_strength)

    dataset_list = [DATASET]
    noise_strength_list = [NOISE_STRENGTH]
    noise_type_list = ['idl', 'linear', 'uniform', 'asymmetric']

    for exp_combo in product(dataset_list, noise_type_list, noise_strength_list):
        dataset, noise_type, noise_strength = exp_combo
        if METHOD == 'ours':
            # # ours
            if dataset == 'mnist':
                gamma_initial = 1
                gamma_multiplier = 0.5
            if dataset == 'cifar10':
                gamma_initial = 0.1
                gamma_multiplier = 0.01
            cmd = f"python -W ignore train_ours_binary_share_weighted.py --gpus {args.gpus} --dataset {dataset} --noise_type {noise_type} --noise_strength {noise_strength} --gamma_initial {gamma_initial} --gamma_multiplier {gamma_multiplier} --figure &"
            os.system(cmd)
        elif METHOD == 'oursv2':
            # # ours_v1
            if dataset == 'mnist':
                gamma_initial = 1
                gamma_multiplier = 0.5
            if dataset == 'cifar10':
                gamma_initial = 0.1
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
                  f" --figure &"
            os.system(cmd)
        elif METHOD == 'ts':
            # # T-scaling
            if os.path.exists("./baselines"):
                os.chdir("./baselines")
            cmd = f"python -W ignore run_temperature_scaling.py --gpus {args.gpus} --dataset {dataset} --noise_type {noise_type} --noise_strength {noise_strength} --figure &"
            os.system(cmd)
        elif METHOD == 'mcdrop':
            # # MCDrop
            if os.path.exists("./baselines"):
                os.chdir("./baselines")
            cmd = f"python -W ignore run_bayes_dropout.py --gpus {args.gpus} --dataset {dataset} --noise_type {noise_type} --noise_strength {noise_strength} --figure &"
            os.system(cmd)
        elif METHOD == 'ensemble':
            # # Ensemble
            if os.path.exists("./baselines"):
                os.chdir("./baselines")
            cmd = f"python -W ignore run_ensemble.py --gpus {args.gpus} --dataset {dataset} --noise_type {noise_type} --noise_strength {noise_strength} --figure &"
            os.system(cmd)
        elif METHOD == 'cskd':
            # # CSKD
            if os.path.exists("./baselines"):
                os.chdir("./baselines")
            cmd = f"python -W ignore run_cskd.py --gpus {args.gpus} --dataset {dataset} --noise_type {noise_type} --noise_strength {noise_strength} --figure &"
            os.system(cmd)
        else:
            print(f"No such baseline! {METHOD}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguement for SLDenoise")
    parser.add_argument("--gpus", type=str, help="Indices of GPUs to be used")
    parser.add_argument("--method", type=str, help="Method", choices={'ours', 'oursv2', 'ts', 'mcdrop', 'ensemble', 'cskd'})
    parser.add_argument("--dataset", type=str, help="Experiment Dataset", choices={'mnist', 'cifar10', 'cifar100'})
    parser.add_argument("--noise_strength", type=float, help="Noise fraction", choices={0.2, 0.4, 0.6, 0.8})
    args = parser.parse_args()

    main(args)