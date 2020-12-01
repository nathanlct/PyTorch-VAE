#!/bin/bash

git clone https://github.com/nathanlct/PyTorch-VAE.git
cd PyTorch-VAE/

conda create --name vae python=3.7 -y
conda activate vae
pip install -r requirements.txt

python run.py -c configs/bhvae.yaml