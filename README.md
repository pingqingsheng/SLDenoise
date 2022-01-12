# SLDenoise
Selective Learning Denoise

# Run our method 

```{shell script}
python -W ignore train_ours_binary_share_weighted_v2.py --seed 77 --gpus 0 --dataset mnist --noise_type uniform --noise_strength 0.2 --gamma_initial 0.5 --gamma_multiplier 0.1
```

`--dataset` can be `mnist` and `cifar10`

`--noise_type` can be `uniform` , `asymmetric`, `idl` and `linear`

`--noise_strength` can be `0.2`,  `0.4`, `0.6`and `0.8`

# Run baseline

```{shell script}
cd baselines
python -W ignore train_temperature_scaling.py --seed 77 --gpus 0 -dataset mnist --noise_type uniform --noise_strength 0.2
```

Arguments are similar to our method.

Calibration baselines are:

`run_temperature_scaling.py` 

`run_ensemble.py`

`run_cskd.py`

`run_bayes_dropout.py`

Label noise baselines are the rest.