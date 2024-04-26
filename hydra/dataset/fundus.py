import numpy as np
from PIL import Image
import random
import torch
import torchvision
from torchvision import transforms
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os


class FundusBaseData:
    pass


class FundusBaseData(Dataset):
    def __init__(self, root, transform=None):
        super(FundusBaseData, self).__init__()
        self.root = root
        self.transform = transform

        # load the data csv
        self.data = pd.read_csv(os.path.join(self.root, 'JustRAIGS_Train_labels.csv'))

    def build_settings(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        target = self.data['Final Label'].iloc[index]
        concept = self.data[self.major_concept].iloc[index]
        for folder in range(6):
            if os.path.exists(os.path.join(os.path.expanduser(self.root), str(folder), self.data['Eye ID'].iloc[index])):
                img = Image.open(os.path.join(os.path.expanduser(self.root), str(folder), self.data['Eye ID'].iloc[index]))
                break

        img = torch.from_numpy(np.array(img))

        if self.transform:
            img = self.transform(img)

        return img, target, concept

    class FundusImbalancedData(FundusBaseData):
        def build_settings(self):
            self.major_concept = 'ANRS'
            self.minor_concept = 'DH'

            df_sample_concept = self.data.sample(frac=0.2, random_state=42)
            df_sample_task = df_sample_concept.drop(df_sample_concept.index)





