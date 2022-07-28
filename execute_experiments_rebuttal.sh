python -W ignore train_ours_binary_share_weighted_dev.py --seed 77 --gpus 0 --dataset mnist --noise_type uniform --noise_strength 0.0 &
python -W ignore train_ours_binary_share_weighted_dev.py --seed 77 --gpus 1 --dataset cifar10 --noise_type uniform --noise_strength 0.0 &

cd baselines
python -W ignore train_ours_temperature_scaling.py       --seed 77 --gpus 0 --dataset mnist --noise_type uniform --noise_strength 0.0 &
python -W ignore train_cskd.py --seed 77 --gpus 0 --dataset mnist --noise_type uniform --noise_strength 0.0 &
python -W ignore train_bayes_dropout.py --seed 77 --gpus 0 --dataset mnist --noise_type uniform --noise_strength 0.0 &
python -W ignore train_ensemble.py --seed 77 --gpus 0 --dataset mnist --noise_type uniform --noise_strength 0.0 &

python -W ignore train_ours_temperature_scaling.py       --seed 77 --gpus 1 --dataset cifar10 --noise_type uniform --noise_strength 0.0 &
python -W ignore train_cskd.py --seed 77 --gpus 1 --dataset cifar10 --noise_type uniform --noise_strength 0.0 &
python -W ignore train_bayes_dropout.py --seed 77 --gpus 1 --dataset cifar10 --noise_type uniform --noise_strength 0.0 &
python -W ignore train_ensemble.py --seed 77 --gpus 1 --dataset cifar10 --noise_type uniform --noise_strength 0.0 &





