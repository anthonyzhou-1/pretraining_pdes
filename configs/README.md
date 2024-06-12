# Experiments

## Usage
Config files are organized by the task [ff(Fixed Future), auto (Autoregressive)], and by the experiment name [none/jigsaw/masked/etc.].yaml. Each config file expects a path to the corresponding PDE data. 

### Experiments

To run an experiment for Fixed Future prediction:
```
python train.py --config=configs/[experiment].yaml --mode=fixed_future
```

To run an experiment for Autoregressive prediction:
```
python train.py --config=configs/[experiment].yaml --mode=next_step
```

To specify a desired model:
```
python train.py --config=configs/[experiment].yaml --model=['FNO2D', 'DeepONet2D', 'OFormer2D', 'Unet']
```

To specify a desired data augmentation:
```
python train.py --config=configs/[experiment].yaml --augmentation=['noise', 'shift', 'scale']
```

To specify a desired # of downstream samples or distribution:
```
python train.py --config=configs/[experiment].yaml --samples_list=[100, 250, ...] --distribution_list=["in", "out"]
```

To fine-tune on a specific equation:
```
python train.py --config=configs/[experiment].yaml --subset_list=['heat', 'adv', 'burger', 'ns']
```
Note that training for the Navier-Stokes equations ('ns') must be done with higher resolution data; therefore the config must set base_resolution: [32, 64, 64]

### Additional Parameters

To run a script with a specific seed or device:
```
--seed=seed --device=device
```

By default, the training scripts will load the entire dataset into GPU memory. For 2D experiments this may be too memory intensive for certain GPU cards, so that option may be turned off by passing:

```
--load_all=["True", "False"]
```

