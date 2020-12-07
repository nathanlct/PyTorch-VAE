import math
import torch
from torch import optim
from models import BaseVAE
from models.types_ import *
from utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
from coinrun import CoinrunDataset


class VAEXperiment(pl.LightningModule):

    def __init__(self, vae_model: BaseVAE, params: dict) -> None:
        super().__init__()
        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = self.params.get('retain_first_backpass', False)

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img = batch
        self.curr_device = real_img.device
        results = self.forward(real_img)

        train_loss = self.model.loss_function(*results,
                                              M_N = self.params['batch_size'] / self.num_train_imgs,
                                              optimizer_idx = optimizer_idx,
                                              batch_idx = batch_idx)
        for key, val in train_loss.items():
            self.log(key, val)

        return train_loss

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img = batch
        self.curr_device = real_img.device
        results = self.forward(real_img)

        val_loss = self.model.loss_function(*results,
                                            M_N = self.params['batch_size'] / self.num_val_imgs,
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)

        return val_loss

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss}
        self.sample_images()
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def sample_images(self):
        # Get sample reconstruction image
        test_input = next(iter(self.sample_dataloader)).to(self.curr_device)
        recons = self.model.generate(test_input)

        vutils.save_image(recons.data,
                          f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
                          f"recons_{self.logger.name}_{self.current_epoch}.png",
                          normalize=True, nrow=12)

        try:
            samples = self.model.sample(144, self.curr_device)
            vutils.save_image(samples.cpu().data,
                              f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
                              f"{self.logger.name}_{self.current_epoch}.png",
                              normalize=True, nrow=12)
        except:
            pass

        del test_input, recons

    def configure_optimizers(self):
        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(), lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)

        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model,self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma = self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma = self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims

    @data_loader
    def train_dataloader(self):
        transform = self.data_transforms()

        if self.params['dataset'] == 'coinrun':
            dataset = CoinrunDataset(filepath=self.params['data_path'],
                                     split='train',
                                     transform=transform)
        else:
            raise ValueError('Undefined dataset type')

        self.num_train_imgs = len(dataset)
        return DataLoader(dataset,
                          batch_size=self.params['batch_size'],
                          shuffle=True,
                          drop_last=True,
                          num_workers=4,
                          pin_memory=True)

    @data_loader
    def val_dataloader(self):
        transform = self.data_transforms()

        if self.params['dataset'] == 'coinrun':
            self.sample_dataloader =  DataLoader(CoinrunDataset(filepath=self.params['data_path'],
                                                                split='test',
                                                                transform=transform),
                                                 batch_size= 144,
                                                 shuffle=False,
                                                 drop_last=True,
                                                num_workers=4,
                                                pin_memory=True)
            self.num_val_imgs = len(self.sample_dataloader)
        else:
            raise ValueError('Undefined dataset type')

        return self.sample_dataloader

    def data_transforms(self):
        SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
        SetScale = transforms.Lambda(lambda X: X/X.sum(0).expand_as(X))

        if self.params['dataset'] == 'coinrun':
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Lambda(lambda X: 2. * X - 1.)])
        else:
            raise ValueError('Undefined dataset type')
        return transform

