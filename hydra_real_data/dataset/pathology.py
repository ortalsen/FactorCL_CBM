import os
import pandas as pd
from PIL import Image
import torch
import torchvision
from torchvision import transforms, datasets
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader


class pathology(Dataset):
    def __init__(self, annotations, data_dir=None, split_tag='train', transform=None):

        self.data = pd.read_csv(annotations)
        self.data_dir = data_dir
        self.transform = transform
        self.split_tag = split_tag
        self.data_df = self.data[self.data['split'] == self.split_tag]

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        img_data = self.data_df.iloc[idx]
        img_rel_path = img_data['file_name']
        img_path = os.path.join(self.data_dir, img_rel_path)

        image = Image.open(img_path)
        label = torch.tensor([img_data['label']], dtype=torch.long)
        concept = torch.tensor([img_data['ki67_label']], dtype=torch.long)

        if self.transform:
            image = self.transform(image)
        else:
            transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
            ])
            image = transform(image)

        return image,label, concept


class PathologyDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, annotations_csv, batch_size=128):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.annotations_csv = annotations_csv

    def setup(self, stage=None):
        
        self.train_dataset = pathology(annotations=self.annotations_csv, data_dir=self.data_dir, split_tag='train', transform=None)
        self.val_dataset = pathology(annotations=self.annotations_csv, data_dir=self.data_dir, split_tag='val', transform=None)
        self.test_dataset = pathology(annotations=self.annotations_csv, data_dir=self.data_dir, split_tag='test', transform=None)
         
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):   
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):    
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
        