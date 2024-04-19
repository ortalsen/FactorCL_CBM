import numpy as np
from PIL import Image
import random
import torch
import torchvision
from torchvision import transforms, datasets
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from omegaconf import DictConfig



class PathologyDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage=None):
        
        self.train_dataset = ...
        self.val_dataset = ...
        self.test_dataset = ...
         

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):   
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):    
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
        