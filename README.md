# Strategies for Pretraining Neural Solvers

This repository is the official implementation of Strategies for Pretraining Neural Solvers

## Requirements

To install requirements:

```setup
conda env create --name envname --file=environment.yml
```

## Datasets
Data was generated according to parameters detailed in the paper using the code below. In general, data is expected to an .h5 file; we provide sample [datasets](data_gen/data/) to illustrate its organization.

- [ Fourier Neural Operator for Parametric Partial Differential Equations](https://github.com/khassibi/fourier-neural-operator)
    - 2D Incompressible NS
- [2D_combined.py](data_gen/2D_combined.py)
    - 2D Heat, Adv, Burgers equations

## Training
For specific experiments, please refer the appropriate .yaml file and command line args in the [configs](configs) directory. The project uses Weights and Biases to log metrics.

```
python train.py --config=configs/[experiment].yaml
```
The script will both pretrain and finetune a model for a given experimental setup (e.g., PDE, model, # fine-tuning samples, fine-tuning distribution).