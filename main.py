from collections import defaultdict
import numpy as np
import boto3
import os
import os.path
from datetime import datetime
import locale
import argparse
import json

import torch
import torchvision.utils
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from coinrun import CoinrunDataset
from models import vae_models

# hyperparameters
parser = argparse.ArgumentParser(description='Coinrun VAE training.')

parser.add_argument('--local', action='store_true', default=False, help='')
parser.add_argument('--seed', type=int, default=1234, help='')
parser.add_argument('--expname', type=str, default=None, help='', required=True)
parser.add_argument('--description', type=str, default="", help='')

parser.add_argument('--lr', type=float, default=3e-4, help='')
parser.add_argument('--batch_size', type=int, default=64, help='')
parser.add_argument('--latent_dim', type=int, default=32, help='')

parser.add_argument('--epochs', type=int, default=5, help='')
parser.add_argument('--validate_every', type=int, default=2, help='')
parser.add_argument('--checkpoint_every', type=int, default=2, help='')
parser.add_argument('--save_folder', type=str, default='vae_results', help='')

parser.add_argument('--s3', action='store_true', default=False, help='')
parser.add_argument('--s3_bucket', type=str, default='nathan.experiments', help='')
parser.add_argument('--s3_path', type=str, default='adversarial/coinrun_vae', help='')

args = parser.parse_args()

# create save folder
locale.setlocale(locale.LC_ALL, 'fr_FR.utf8')
today = datetime.now().strftime("%d-%m-%Y")
expname = args.expname.replace(' ', '_') + datetime.now().strftime("_%Hh%M")
save_folder = os.path.join(args.save_folder, today, expname)
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
    print(f'Saving at {save_folder}')
    # os.makedirs(os.path.join(save_folder, 'reconstructions'))
    # os.makedirs(os.path.join(save_folder, 'samples'))
    os.makedirs(os.path.join(save_folder, 'checkpoints'))
else:
    print(f'Save folder {save_folder} already exists. Aborting')
    exit(0)

# s3
if args.s3:
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(args.s3_bucket)

# save args to file
with open(os.path.join(save_folder, 'params.json'), 'w') as f:
    json.dump(args.__dict__, f, indent=4)

# set device
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(f'Using device {device}')

# set seed for reproducibility (https://pytorch.org/docs/stable/notes/randomness.html)
if args.seed:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_deterministic(True)
    torch.backends.cudnn.benchmark = False  # can reduce performance

# dataset
train_dataset = CoinrunDataset('dataset/data.npz', split='train', force_size=1*args.batch_size)
val_dataset = CoinrunDataset('dataset/data.npz', split='test', force_size=1*args.batch_size)

num_workers = 0 if args.local else 4
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
    drop_last=True, num_workers=num_workers, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
    drop_last=True, num_workers=num_workers, pin_memory=True)

n_train, n_train_batches = len(train_dataset), len(train_dataloader)
n_val, n_val_batches = len(val_dataset), len(val_dataloader)

assert(n_train_batches == n_train // args.batch_size)
assert(n_val_batches == n_val // args.batch_size)

print(f'Initialized training dataset with {n_train} samples, {n_train_batches} batches')
print(f'Initialized validation dataset with {n_val} samples, {n_val_batches} batches')

# model
model = vae_models['VanillaVAE'](in_channels=3, latent_dim=args.latent_dim).to(device)

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# tensorboard logging
writer = SummaryWriter(log_dir=save_folder)

# training loop
print(f'** Starting experiment {args.expname} **')

for epoch in range(args.epochs):
    print(f'----- Epoch {epoch+1}/{args.epochs} -----')

    print(f'> training over {n_train_batches} batches')
    model.train()
    train_losses = defaultdict(list)
    for batch_idx, batch in enumerate(train_dataloader):
        print('.', end='', flush=True)
        optimizer.zero_grad()

        results = model(batch)
        train_loss = model.loss_function(*results, M_N = args.batch_size / n_train)
        for k, v in train_loss.items():
            train_losses[k].append(v.item())

        train_loss['loss'].backward()
        optimizer.step()

    print()
    train_loss_data = []
    for k, v in train_losses.items():
        train_loss_data.append(f'{k}: {round(np.mean(v), 3)} (+- {round(np.std(v), 3)})')
        writer.add_scalar(f'Train/{k}', np.mean(v), epoch)
    print(', '.join(train_loss_data))

    if epoch % args.validate_every == 0 or epoch == args.epochs - 1:
        print(f'> validating over {n_val_batches} batches')
        with torch.no_grad():
            model.eval()
            val_losses = defaultdict(list)
            for batch_idx, batch in enumerate(val_dataloader):
                print('.', end='', flush=True)

                results = model(batch)
                val_loss = model.loss_function(*results, M_N = args.batch_size / n_val)
                for k, v in val_loss.items():
                    val_losses[k].append(v.item())

            print()
            val_loss_data = []
            for k, v in val_losses.items():
                val_loss_data.append(f'{k}: {round(np.mean(v), 3)} (+- {round(np.std(v), 3)})')
                writer.add_scalar(f'Val/{k}', np.mean(v), epoch)
            print(', '.join(val_loss_data))

            # sample images
            print('> sampling images')
            test_input = next(iter(val_dataloader)).to(device)
            recons = model.generate(test_input)
            grid1 = torchvision.utils.make_grid(recons.data, normalize=True, nrow=8)
            # grid1 = torchvision.utils.make_grid(recons.data, os.path.join(save_folder, f"reconstructions/epoch_{epoch+1}.png"), normalize=True, nrow=8)
            samples = model.sample(args.batch_size, device)
            grid2 = torchvision.utils.make_grid(samples.cpu().data, normalize=True, nrow=8)
            # grid2 = torchvision.utils.make_grid(samples.cpu().data, os.path.join(save_folder, f"samples/epoch_{epoch+1}.png"), normalize=True, nrow=8)
            writer.add_image('reconstructions', grid1, epoch)
            writer.add_image('samples', grid2, epoch)    
            # writer.add_graph(model, test_input.data)        
            del test_input, recons, samples

    if epoch % args.checkpoint_every == 0 or epoch == args.epochs - 1:
        print('> saving checkpoint')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(save_folder, f"checkpoints/epoch_{epoch+1}.checkpoint"))

        if args.s3:
            for path, subdirs, files in os.walk(save_folder):
                directory_name = path.replace(args.save_folder + '/', '')
                for file in files:
                    bucket.upload_file(os.path.join(path, file), os.path.join(args.s3_path, directory_name, file))

writer.close()

print(f'Saved at {save_folder}')
if args.s3:
    print(f'and at s3://{args.bucket}/{args.s3_path}')