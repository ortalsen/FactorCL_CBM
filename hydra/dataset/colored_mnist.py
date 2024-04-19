import numpy as np
from PIL import Image
import random
import torch
import torchvision
from torchvision import transforms, datasets
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from omegaconf import DictConfig



overlay_image = Image.open('./dataset/pebbles.png')  
# overlay_image.resize((28, 28))  # Resize overlay image to match MNIST image size
transform_overlay = transforms.Resize(size=(28, 28)) #transforms.RandomCrop(size=(28, 28))
overlay_image = transform_overlay(overlay_image)
overlay_image = transforms.ToTensor()(overlay_image)#.unsqueeze(0)



class ColoredMNIST(datasets.MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(ColoredMNIST, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        # Define color mappings for each digit
            # 0: [1.0, 0.0, 0.0],  # Red for digit 0
            # 1: [0.0, 0.0, 1.0],  # Blue for digit 1
            # 2: [1.0, 1.0, 0.0],  # Yellow for digit 2
            # 3: [0.0, 1.0, 0.0],  # Green for digit 3
            # 4: [1.0, 0.0, 1.0],  # Magenta for digit 4
            # 5: [0.0, 1.0, 1.0],  # Cyan for digit 5
            # 6: [1.0, 0.5, 0.0],  # Orange for digit 6
            # 7: [0.5, 0.0, 1.0],  # Purple for digit 7
            # 8: [0.5, 0.5, 0.5],  # Gray for digit 8
            # 9: [0.0, 0.5, 0.5]   # Teal for digit 9
        self.color_map = {
            0: [0.0, 1.0, 0.0], #GREEN
            1: [1.0, 0.0, 0.0], #RED
            2: [1.0, 1.0, 0.0], #YELLOW
            3: [1.0, 1.0, 1.0], 
            4: [1.0, 1.0, 1.0], 
            5: [1.0, 1.0, 1.0], 
            6: [1.0, 1.0, 1.0],  
            7: [1.0, 1.0, 1.0],   
            8: [1.0, 1.0, 1.0],   
            9: [1.0, 1.0, 1.0],  
        }

    def __getitem__(self, index):
        img, target = super(ColoredMNIST, self).__getitem__(index)

        if self.transform:
            img = self.transform(img)

        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            img = transform(img)
      

        # Get color for the current digit label
        black_pixels = (img == -1)
        
        if target== 1:
            color_name = 0 #random.choice([0,2])  
            color = self.color_map[color_name]
        elif target == 2:
            color_name = 1 #random.choice([0,2])  
            color = self.color_map[color_name]
        elif target ==3:
            color_name = random.choice([0,1])  
            color = self.color_map[color_name]
        elif target ==4:
            color_name = random.choice([0,1])  
            color = self.color_map[color_name]
        else:
            color_name = 99
            color = self.color_map[target]

        # Apply color to the image
        img_colored = img.clone().numpy()

        for i in range(img_colored.shape[1]):
            for j in range(img_colored.shape[1]):
                img_colored[:,i,j] = img_colored[:,i,j] * np.array(color) #* 255
        

        img_colored[black_pixels] = overlay_image[black_pixels]     
        img_colored = torch.tensor(img_colored)

        return img_colored, target, color_name


############################################################################################################################

class ColoredMNIST_2(datasets.MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(ColoredMNIST_2, self).__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        # Define color mappings for each digit
            # 0: [1.0, 0.0, 0.0],  # Red for digit 0
            # 1: [0.0, 0.0, 1.0],  # Blue for digit 1
            # 2: [1.0, 1.0, 0.0],  # Yellow for digit 2
            # 3: [0.0, 1.0, 0.0],  # Green for digit 3
            # 4: [1.0, 0.0, 1.0],  # Magenta for digit 4
            # 5: [0.0, 1.0, 1.0],  # Cyan for digit 5
            # 6: [1.0, 0.5, 0.0],  # Orange for digit 6
            # 7: [0.5, 0.0, 1.0],  # Purple for digit 7
            # 8: [0.5, 0.5, 0.5],  # Gray for digit 8
            # 9: [0.0, 0.5, 0.5]   # Teal for digit 9
        self.color_map = {
            0: [0.0, 1.0, 0.0], #GREEN
            1: [1.0, 0.0, 0.0], #RED
            2: [1.0, 1.0, 0.0], #YELLOW
            3: [1.0, 1.0, 1.0], 
            4: [1.0, 1.0, 1.0], 
            5: [1.0, 1.0, 1.0], 
            6: [1.0, 1.0, 1.0],  
            7: [1.0, 1.0, 1.0],   
            8: [1.0, 1.0, 1.0],   
            9: [1.0, 1.0, 1.0],  
        }

    def __getitem__(self, index):
        img, target = super(ColoredMNIST_2, self).__getitem__(index)

        if self.transform:
            img = self.transform(img)

        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                # transforms.RandomResizedCrop(28),
                # transforms.RandomHorizontalFlip(p=0.3)
            ])
            img = transform(img)
      

        # Get color for the current digit label
        black_pixels = (img == -1)
        
        if target== 1:
            color_name = 1 #random.choice([0,1])  
            color = self.color_map[color_name]
        elif target == 2:
            color_name = 0 #random.choice([0,1])  
            color = self.color_map[color_name]
        elif target ==3:
            color_name = random.choice([0,2])  
            color = self.color_map[color_name]
        elif target ==4:
            color_name = random.choice([0,2])  
            color = self.color_map[color_name]
        else:
            color_name = 99
            color = self.color_map[target]

        # Apply color to the image
        img_colored = img.clone().numpy()

        for i in range(img_colored.shape[1]):
            for j in range(img_colored.shape[1]):
                img_colored[:,i,j] = img_colored[:,i,j] * np.array(color) #* 255
        

        img_colored[black_pixels] = overlay_image[black_pixels]         
        img_colored = torch.tensor(img_colored)


        return img_colored, target, color_name
        
        


################################################################################################################################
class MyDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage=None):
        train_set = ColoredMNIST('./', train=True, transform=None, download=True)   
        
        indices_12 = np.where(np.isin(train_set.targets, [1,2]))[0]
        indices_3 = np.where(np.isin(train_set.targets, [3]))[0]
        indices = indices_12 #np.concatenate((indices_12[:int(len(indices_12)/9)], indices_3[:int(len(indices_3))]))
        
        train_set.targets = train_set.targets[indices]
        train_set.data = train_set.data[indices]
        
        
        test_set = ColoredMNIST_2('./', train=False, transform=None, download=True)
        
        indices = np.where(np.isin(test_set.targets, [1,2]))[0]
        test_set.targets = test_set.targets[indices]
        test_set.data = test_set.data[indices]
        
        self.train_dataset, remainder_trainset = torch.utils.data.random_split(train_set, [1000, len(train_set)-1000])
        self.val_dataset, remainder_valset = torch.utils.data.random_split(remainder_trainset, [500, len(remainder_trainset)-500])
        self.test_dataset, remainder_testset = torch.utils.data.random_split(test_set, [1000, len(test_set)-1000])
        
        
         

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):   
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):    
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
        