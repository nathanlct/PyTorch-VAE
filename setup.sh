#!/bin/bash

git clone https://github.com/nathanlct/PyTorch-VAE.git
cd PyTorch-VAE/

conda create --name vae python=3.7 -y
conda activate vae
pip install -r requirements.txt

python run.py -c configs/vae.yaml
python run.py -c configs/bhvae.yaml
python run.py -c configs/bbvae.yaml
python run.py -c configs/factorvae.yaml


# new

git clone https://github.com/nathanlct/PyTorch-VAE.git
cd PyTorch-VAE/
conda create --name vae python=3.7 -y
conda activate vae
conda install nb_conda -y
conda install pytorch-lightning -c conda-forge -y
pip install torchvision
pip install torchnet
pip install torchsummary
pip install matplotlib
pip install tensorflow
pip install tensorboard
pip install f3fs
export CUBLAS_WORKSPACE_CONFIG=:4096:8  # for deterministicity

# setup aws credentials (upload ~/.aws/credentials)