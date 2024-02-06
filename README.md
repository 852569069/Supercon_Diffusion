# [High-performance  diffusion  model  for  inverse  design of  high  Tc  superconductors  with  effective  doping and  accurate  stoichiometry](https://onlinelibrary.wiley.com/doi/10.1002/inf2.12519)

[![python](https://img.shields.io/badge/-Python_3.8_%7C_3.9_%7C_3.10-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)

## Description

This project is a code implementation of the Supercon-Diffusion model. Supercon-Diffusion is proficient in generating the efficacy of doped superconductors.

## Installation

#### Pip

```bash
# clone project
git clone https://github.com/852569069/Supercon_Diffusion
cd Supercon_Diffusion

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

#### Conda

```bash
# clone project
git clone https://github.com/852569069/Supercon_Diffusion
cd Supercon_Diffusion

# create conda environment and install dependencies
conda env create -f environment.yaml -n myenv

# activate conda environment
conda activate myenv
```

## How to run

Change `root_dir:{path}` in `configs/paths/default.yaml` to the project directory.

Train model with default configuration

```bash
# train on CPU
python train-sd.py trainer=cpu

# train on GPU
python train-sd.py trainer=gpu

# train on Mac
python train-sd.py trainer=mps


```

You can override any parameter from command line like this

```bash
python train-sd.py trainer.max_epochs=20 data.batch_size=64
```
