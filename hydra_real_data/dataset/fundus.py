import numpy as np
from PIL import Image
import random
import torch
import torchvision
from torchvision import transforms
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import pandas as pd
import os
from sklearn.model_selection import train_test_split


class FundusBaseData:
    pass


class FundusBaseData(Dataset):
    def __init__(self, root, size=224):
        super(FundusBaseData, self).__init__()
        self.root = os.path.expanduser(root)
        self.size = size
        self.build_settings()
        self.split_data()
        self.compute_scalers()
        self.build_transforms()

    def build_settings(self):
        raise NotImplementedError

    def split_data(self):
        train_idx, test_idx = train_test_split(range(len(self.data)), test_size=0.2, random_state=42)
        train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=42)
        self.train = train_idx
        self.val = val_idx
        self.test = test_idx

    def compute_scalers(self):
        channel_avg = []
        for idx, index in enumerate(self.train[:5000]):
            row = self.data.iloc[index]
            for folder in range(6):
                if os.path.exists(os.path.join(self.root, str(folder), row['Eye ID']+'.JPG')):
                    img = Image.open(os.path.join(self.root, str(folder), row['Eye ID']+'.JPG'))
                    img = np.array(img)
                    break
            channel_avg.append(np.mean(img, axis=(0, 1)))
            print(idx)
        channel_avg = np.array(channel_avg)
        self.mean = channel_avg.mean(axis=0)
        self.std = channel_avg.std(axis=0)
        breakpoint(True)


    def build_transforms(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def organize_output(self, img, target):
        raise NotImplementedError

    def __getitem__(self, index):
        target = torch.tensor(np.array(self.data[self.field].iloc[index] == self.pos_label)).type(torch.LongTensor)
        for folder in range(6):
            if os.path.exists(os.path.join(os.path.expanduser(self.root), str(folder), self.data['Eye ID'].iloc[index])+'.JPG'):
                img = Image.open(os.path.join(os.path.expanduser(self.root), str(folder), self.data['Eye ID'].iloc[index])+'.JPG')
                break
            elif os.path.exists(os.path.join(os.path.expanduser(self.root), str(folder), self.data['Eye ID'].iloc[index])+'.PNG'):
                img = Image.open(os.path.join(os.path.expanduser(self.root), str(folder), self.data['Eye ID'].iloc[index])+'.PNG')
                break

        img = self.transforms(img)
        img = torch.from_numpy(np.array(img)).float()


        return self.organize_output(img, target)

class FundusConceptData(FundusBaseData):
    def build_settings(self):
        df = pd.read_csv(os.path.join(self.root, 'JustRAIGS_Train_labels.csv'),  delimiter=';')
        df_concepts = df.sample(frac=0.2, random_state=42)
        data_neg = df_concepts[(df_concepts['Final Label']== 'RG') &
                                (df_concepts['G1 ANRS'] < 1.0) & (df_concepts['G2 ANRS'] < 1.0)]
        data_pos = df_concepts[(df_concepts['Final Label']== 'RG') &
                               (df_concepts['G1 ANRS'] > 0.0) & (df_concepts['G2 ANRS'] > 0.0)]
        data = pd.concat([data_neg, data_pos])
        self.data = data.sample(frac=1, random_state=42)
        self.field = 'G1 ANRS'
        self.pos_label = 1.0

    def compute_scalers(self):
        self.mean = np.array([75.37172003, 50.44479941, 33.95108797])
        self.std = np.array([28.91089824, 22.76172709, 22.07211564])

    def build_transforms(self):
        self.transforms =  transforms.Compose([
            transforms.Resize((self.size, self.size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])

    def organize_output(self, img, target):
        return img, torch.tensor(-1.), target

    def compute_scalers(self):
        self.mean = np.array([76.68771855, 47.28333825, 28.94774409])
        self.std = np.array([29.48767494, 20.81023493, 21.56612376])


class FundusRGData(FundusConceptData):
    def build_settings(self):
        df = pd.read_csv(os.path.join(self.root, 'JustRAIGS_Train_labels.csv'),  delimiter=';')
        df_concepts = df.sample(frac=0.2, random_state=42)
        df_task = df.drop(df_concepts.index)
        self.data = df_task.sample(frac=1, random_state=42)
        self.field = 'Final Label'
        self.pos_label = 'RG'

    def organize_output(self, img, target):
        return img, target, torch.tensor(-1.)




class FundusDataModule(pl.LightningDataModule):
    def __init__(self, root, data_class, batch_size=256):
        super().__init__()
        self.root = root
        self.data_obj = globals()[data_class](root)
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train = SubsetRandomSampler(self.data_obj.train)
        self.val = SubsetRandomSampler(self.data_obj.val)
        self.test = SubsetRandomSampler(self.data_obj.test)

    def train_dataloader(self):
        return DataLoader(self.data_obj,sampler=self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.data_obj, sampler=self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.data_obj, sampler=self.test, batch_size=self.batch_size)



if __name__ == "__main__":
    root = '~/data/Fundus'
    data_class = 'FundusRGData'
    batch_size = 256
    dm = FundusDataModule(root, data_class, batch_size)
    dm.setup()
    for batch in dm.train_dataloader():
        print(batch)
        break







