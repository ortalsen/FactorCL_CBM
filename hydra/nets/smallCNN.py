import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import wandb


class SmallCNN(pl.LightningModule):
    def __init__(self, in_channels, out_channels, kernel_size, num_classes):
        super(SmallCNN, self).__init__()

        self.num_classes= num_classes
        self.conv1 = nn.Conv2d(in_channels, out_channels*4, kernel_size)
        self.conv2 = nn.Conv2d(out_channels*4, out_channels, kernel_size)
        self.fc1 = nn.Linear(150, num_classes)

        self.test_accuracy = torchmetrics.Accuracy("multiclass", num_classes=self.num_classes)
        self.test_precision = torchmetrics.Precision(task="multiclass", average='macro', num_classes=self.num_classes)
        self.test_recall = torchmetrics.Recall(task="multiclass", average='macro', num_classes=self.num_classes)
        self.test_f1 = torchmetrics.F1Score(task="multiclass", num_classes=self.num_classes)
        self.test_confusion_matrix = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def training_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('SmallCNN/train_loss', loss)
        wandb.log({'SmallCNN/train_loss': loss})
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        wandb.log({'SmallCNN/val_loss': loss})
        self.log('SmallCNN/val_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        pred = torch.softmax(logits, dim=-1)
        self.test_accuracy(pred, y)
        self.test_precision(pred,y)
        self.test_recall(pred,y)
        self.test_f1(pred,y)
        self.test_confusion_matrix.update(pred, y)

        wandb.log({'final_model/test_loss': loss})
        wandb.log({'final_modeltest_accuracy': self.test_accuracy.compute()})
        wandb.log({'final_model/test_precision': self.test_precision.compute()})
        wandb.log({'final_model/test_recall': self.test_recall.compute()})
        wandb.log({'final_model/test_f1': self.test_f1.compute()})
        
        
        self.log('final_model/test_loss', loss)
        self.log_dict(
            {
                "final_model/accuracy_test": self.test_accuracy.compute(),
                'final_model/test_accuracy': self.test_accuracy.compute(),
                'final_model/test_precision': self.test_precision.compute(),
                'final_model/test_recall': self.test_recall.compute(),
                'final_model/test_f1': self.test_f1.compute()

            },
        )
        

    def on_test_epoch_end(self):
        cm, _ = self.test_confusion_matrix.plot()
        cm.savefig('cm.png')
        wandb.log({"final_model/confusion_matrix": wandb.Image('cm.png')})
    