import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import pytorch_lightning as pl

import random


class PairedMNIST(Dataset):
    def __init__(self, mnist_dataset, transform=None):
        self.mnist_dataset = mnist_dataset
        self.transform = transform

    def __len__(self):
        return len(self.mnist_dataset)

    def __getitem__(self, idx):
        img1, label1 = self.mnist_dataset[idx]        
        idx2 = random.randint(0, len(self.mnist_dataset) - 1)
        img2, label2 = self.mnist_dataset[idx2]
        
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        else:
            transform = transforms.Compose([
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            img1 = transform(img1)
            img2 = transform(img2)
        
        combined_image = torch.cat((img1, img2), dim=2)  # Combine images side by side
        if (label1+label2) % 3 == 0:
            combined_label = 1
        else:
            combined_label = 0
            
        concept = torch.zeros((3,))
        
        if label1== 1 or label2 == 1:
            concept[0] = 1
        if label1== 2 or label2 ==2:
            concept[1] = 1
        if label1== 4 or label2 ==4:
            concept[2] = 1
        
        return combined_image, combined_label, concept

class combined_mnist_DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage=None):
        transform = transforms.Compose([transforms.ToTensor()])
        mnist_train = datasets.MNIST(root='../../data', train=True, download=True, transform=transform)
        mnist_test = datasets.MNIST(root='../../data', train=False, download=True, transform=transform)
        train_set = PairedMNIST(mnist_train)
        test_set = PairedMNIST(mnist_test)
        
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(train_set, [int(0.8*len(train_set)), len(train_set)-int(0.8*len(train_set))])
        self.test_dataset = test_set
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):   
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):    
        return DataLoader(self.test_dataset, batch_size=self.batch_size)