import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision import transforms, datasets
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader


class CUB(Dataset):
    def __init__(self, annotations, data_dir=None, split_tag='train', transform=None):

        self.data = pd.read_csv(annotations)
        self.data_dir = data_dir
        self.transform = transform
        self.split_tag = split_tag
        self.data_df = self.data[self.data['split'] == self.split_tag]
        self.data, self.targets, self.concepts, self.paths = self._load_data()

    def __len__(self):
        return len(self.data)
    
    def _load_data(self):
        data, targets, concepts = [], [], []
        paths = []
        
        for img, lab, conc in zip(self.data_df['img_path'], self.data_df['class_label'], self.data_df['has_eye_color']): #has_eye_color
            # print(img)
            image = Image.open(os.path.join(self.data_dir, img))
            if self.transform:
                image = self.transform(image)
            else:
                transform = transforms.Compose([
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.2422, 0.2468, 0.2555], std=[0.3013, 0.3069, 0.3188]), #mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                transforms.Resize((224, 224)),
                ])
                image = transform(image)
                if image.shape[-3] == 1:
                    # print(image.shape)
                    # print(image.shape[-3])
                    # print(image)
                    image = transforms.Lambda(lambda x: x.repeat(3, 1, 1))(image)
                    
            # if conc == 14:
            #     conc = 1
            # else:
            #     conc = 0
            
            label = torch.tensor([lab], dtype=torch.long)
            concept = torch.tensor([conc], dtype=torch.long)
            
            data.append(image)
            targets.append(label)
            concepts.append(concept)
            paths.append(img)

        data = torch.stack(data)
        targets = torch.stack(targets)
        concepts = torch.stack(concepts)

        return data, targets, concepts, paths

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx], self.concepts[idx], self.paths[idx]


class CUBDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, annotations_csv, batch_size=128):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.annotations_csv = annotations_csv

    def setup(self, stage=None):
        
        self.train_dataset = CUB(annotations=self.annotations_csv, data_dir=self.data_dir, split_tag='train', transform=None)
        self.val_dataset = CUB(annotations=self.annotations_csv, data_dir=self.data_dir, split_tag='val', transform=None)
        self.test_dataset = CUB(annotations=self.annotations_csv, data_dir=self.data_dir, split_tag='test', transform=None)
         
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

    def val_dataloader(self):   
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):    
        return DataLoader(self.test_dataset, batch_size=self.batch_size, drop_last=True)
    

class CUBDataModule_ConceptEncoder(pl.LightningDataModule):
    def __init__(self, data_dir, annotations_csv, batch_size=128):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.annotations_csv = annotations_csv

    def setup(self, stage=None):
        
        self.dataset = CUB(annotations=self.annotations_csv, data_dir=self.data_dir, split_tag='train', transform=None)
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.dataset,
                                                           [int(0.8 * len(self.dataset)), len(self.dataset) - int(0.8 * len(self.dataset))])
        self.test_dataset = CUB(annotations=self.annotations_csv, data_dir=self.data_dir, split_tag='test', transform=None)
         
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):   
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):    
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
        
    