#!/bin/bash

"""
git clone https://github.com/nathanlct/PyTorch-VAE.git
cd PyTorch-VAE/
"""

conda create --name vae python=3.7 -y
conda activate vae
conda install nb_conda -y
pip install torchvision
pip install torchnet
pip install torchsummary
pip install matplotlib
pip install tensorflow
pip install tensorboard
pip install boto3
export CUBLAS_WORKSPACE_CONFIG=:4096:8  # for deterministicity

sudo apt-get install language-pack-fr
sudo dpkg-reconfigure locales

# setup aws credentials (upload ~/.aws/credentials)

# python main.py --expname test_overfit --local --epochs 200 --validate_every 10 --checkpoint_every 50 --s3