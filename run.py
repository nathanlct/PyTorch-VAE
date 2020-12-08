import yaml
import argparse
import numpy as np
from datetime import datetime
import os.path

from models import *
from experiment import VAEXperiment
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback

# get .yaml config file
parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config', '-c', dest='filename', metavar='FILE',
                    help='path to the config file', default='configs/vae.yaml')
args = parser.parse_args()
with open(args.filename, 'r') as file:
    config = yaml.safe_load(file)

# set seed for reproducibility (https://pytorch.org/docs/stable/notes/randomness.html)
seed = config['logging_params']['manual_seed']
if seed is not None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_deterministic(True)
    torch.backends.cudnn.benchmark = False  # can reduce performance

# setup callbacks
class MyPrintingCallback(Callback):

    def on_init_start(self, trainer):
        print('Starting to init trainer!')

    def on_init_end(self, trainer):
        print('trainer is init now')

    def on_train_end(self, trainer, pl_module):
        print('do something when training ends')

# create experiment
model = vae_models[config['model_params']['name']](**config['model_params'])
experiment = VAEXperiment(model, config['exp_params'])

# create trainer (https://pytorch-lightning.readthedocs.io/en/stable/trainer.html)
now = datetime.now().strftime("%d%b%Y-%Hh%Mm%Ss")
save_dir = config['logging_params']['save_dir'].replace('{date}', now)
runner = Trainer(
    default_root_dir=save_dir, 
    callbacks=[MyPrintingCallback()],
    **config['trainer_params'])

# train
print(f"======= Training {config['model_params']['name']} =======")
print(f'> Saving data at {save_dir}')
runner.fit(experiment)