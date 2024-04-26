import os
import pandas as pd
import numpy as np
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
        self.data, self.targets, self.concepts = self._load_data()

    def __len__(self):
        return len(self.data)
    
    def _load_data(self):
        data, targets, concepts = [], [], []
        
        for img, lab, conc in zip(self.data_df['file_name'], self.data_df['label'], self.data_df['ki67_label']):
            image = Image.open(os.path.join(self.data_dir, img))
            if self.transform:
                image = self.transform(image)
            else:
                transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
                ])
                image = transform(image)
            
            label = torch.tensor([lab], dtype=torch.long)
            concept = torch.tensor([conc], dtype=torch.long)
            
            data.append(image)
            targets.append(label)
            concepts.append(concept)

        data = torch.stack(data[:650])
        targets = torch.stack(targets[:650])
        concepts = torch.stack(concepts[:650])

        return data, targets, concepts

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx], self.concepts[idx] 


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
    

class PathologyDataModule_ConceptEncoder(pl.LightningDataModule):
    def __init__(self, data_dir, annotations_csv, batch_size=128):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.annotations_csv = annotations_csv

    def setup(self, stage=None):
        
        self.dataset = pathology(annotations=self.annotations_csv, data_dir=self.data_dir, split_tag='train', transform=None)
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.dataset,
                                                           [int(0.8 * len(self.dataset)), len(self.dataset) - int(0.8 * len(self.dataset))])
        self.test_dataset = pathology(annotations=self.annotations_csv, data_dir=self.data_dir, split_tag='test', transform=None)
         
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):   
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):    
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
        
    

class PathologyDataModule_imbalanced(pl.LightningDataModule):
    def __init__(self, data_dir, annotations_csv, batch_size=128):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.annotations_csv = annotations_csv

    def setup(self, stage=None):
        
        self.train_dataset = pathology(annotations=self.annotations_csv, data_dir=self.data_dir, split_tag='train', transform=None)
        # imbalance the train dataset
        indices_tumor = np.where(np.isin(self.train_dataset.targets, [2]))[0]
        indices_imm_nor = np.where(np.isin(self.train_dataset.targets, [0,1]))[0]
        indices = np.concatenate((indices_tumor[:int(len(indices_tumor)/20)], indices_imm_nor))
        
        self.train_dataset.targets = self.train_dataset.targets[indices]
        self.train_dataset.data = self.train_dataset.data[indices]
        self.train_dataset.concepts = self.train_dataset.concepts[indices]
        
        
        self.val_dataset = pathology(annotations=self.annotations_csv, data_dir=self.data_dir, split_tag='val', transform=None)
        #imbalance the val dataset
        indices_tumor = np.where(np.isin(self.val_dataset.targets, [2]))[0]
        indices_imm_nor = np.where(np.isin(self.val_dataset.targets, [0,1]))[0]
        indices = np.concatenate((indices_tumor[:int(len(indices_tumor)/20)], indices_imm_nor))

        self.val_dataset.targets = self.val_dataset.targets[indices]
        self.val_dataset.data = self.val_dataset.data[indices]
        self.val_dataset.concepts = self.val_dataset.concepts[indices]     
        
        self.test_dataset = pathology(annotations=self.annotations_csv, data_dir=self.data_dir, split_tag='test', transform=None)
         
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):   
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):    
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
    

class PathologyDataModule_ConceptEncoder_imbalanced(pl.LightningDataModule):
    def __init__(self, data_dir, annotations_csv, batch_size=128):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.annotations_csv = annotations_csv

    def setup(self, stage=None):
        
        self.dataset = pathology(annotations=self.annotations_csv, data_dir=self.data_dir, split_tag='train', transform=None)
        indices_tumor = np.where(np.isin(self.dataset.targets, [2]))[0]
        indices_imm_nor = np.where(np.isin(self.dataset.targets, [0,1]))[0]
        indices = np.concatenate((indices_tumor[:int(len(indices_tumor)/20)], indices_imm_nor))

        self.dataset.targets = self.dataset.targets[indices]
        self.dataset.data = self.dataset.data[indices]
        self.dataset.concepts = self.dataset.concepts[indices]
        
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(self.dataset,
                                                           [int(0.8 * len(self.dataset)), len(self.dataset) - int(0.8 * len(self.dataset))])

        # imbalance the train dataset
        # indices_tumor = np.where(np.isin(self.train_dataset.targets, [2]))[0]
        # indices_imm_nor = np.where(np.isin(self.train_dataset.targets, [0,1]))[0]
        # indices = np.concatenate((indices_tumor[:int(len(indices_tumor)/2)], indices_imm_nor))

        # self.train_dataset.targets = self.train_dataset.targets[indices]
        # self.train_dataset.data = self.train_dataset.data[indices]
        # self.train_dataset.concepts = self.train_dataset.concepts[indices]
        
        # #imbalance the val dataset
        # indices_tumor = np.where(np.isin(self.val_dataset.targets, [2]))[0]
        # indices_imm_nor = np.where(np.isin(self.val_dataset.targets, [0,1]))[0]
        # indices = np.concatenate((indices_tumor[:int(len(indices_tumor)/2)], indices_imm_nor))

        # self.val_dataset.targets = self.val_dataset.targets[indices]
        # self.val_dataset.data = self.val_dataset.data[indices]
        # self.val_dataset.concepts = self.val_dataset.concepts[indices]
        
        self.test_dataset = pathology(annotations=self.annotations_csv, data_dir=self.data_dir, split_tag='test', transform=None)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):   
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):    
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

