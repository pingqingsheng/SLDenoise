# python -W ignore train_ours_binary_share_weighted_dev.py --seed 77 --gpus 0 --dataset mnist --noise_type uniform --noise_strength 0.0 &
# python -W ignore train_ours_binary_share_weighted_dev.py --seed 77 --gpus 1 --dataset cifar10 --noise_type uniform --noise_strength 0.0 &

# cd baselines
# python -W ignore run_temperature_scaling.py       --seed 77 --gpus 0 --dataset mnist --noise_type uniform --noise_strength 0.0 &
# python -W ignore run_cskd.py --seed 77 --gpus 0 --dataset mnist --noise_type uniform --noise_strength 0.0 &
# python -W ignore run_bayes_dropout.py --seed 77 --gpus 0 --dataset mnist --noise_type uniform --noise_strength 0.0 &
# python -W ignore run_ensemble.py --seed 77 --gpus 0 --dataset mnist --noise_type uniform --noise_strength 0.0 &

# python -W ignore run_temperature_scaling.py --seed 77 --gpus 1 --dataset cifar10 --noise_type uniform --noise_strength 0.0 &
# python -W ignore run_cskd.py --seed 77 --gpus 1 --dataset cifar10 --noise_type uniform --noise_strength 0.0 &
# python -W ignore run_bayes_dropout.py --seed 77 --gpus 1 --dataset cifar10 --noise_type uniform --noise_strength 0.0 &
# python -W ignore run_ensemble.py --seed 77 --gpus 1 --dataset cifar10 --noise_type uniform --noise_strength 0.0 &

