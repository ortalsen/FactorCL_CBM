import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import wandb

def mlp(dim, hidden_dim, output_dim, layers, activation):
    activation = {
        'relu': nn.ReLU,
        'tanh': nn.Tanh,
    }[activation]

    seq = [nn.Linear(dim, hidden_dim), activation()]
    for _ in range(layers):
        seq += [nn.Linear(hidden_dim, hidden_dim), activation()]
    seq += [nn.Linear(hidden_dim, output_dim)]

    return nn.Sequential(*seq)

class SmallMLP(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(SmallMLP, self).__init__()
        self.num_classes = num_classes
        # self.fc1 = nn.Linear(input_dim, hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.net = mlp(input_dim, hidden_dim, num_classes, layers=2, activation='relu')
        # self.fc = nn.Linear(hidden_dim, num_classes)

        self.test_accuracy = torchmetrics.Accuracy("multiclass", num_classes=self.num_classes)
        self.test_precision = torchmetrics.Precision(task="multiclass", average='macro', num_classes=self.num_classes)
        self.test_recall = torchmetrics.Recall(task="multiclass", average='macro', num_classes=self.num_classes)
        self.test_f1 = torchmetrics.F1Score(task="multiclass", num_classes=self.num_classes)
        self.test_confusion_matrix = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.num_classes)

    def forward(self, x):
        x =  x.view(x.size(0), -1)  # Flatten the input (assuming input is a batch of images)
        # x = F.relu(self.fc1(x))
        # x = self.fc2(x)
        x = self.net(x)
        # x = self.fc(x)
        return x


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def training_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('SmallMLP/train_loss', loss)
        wandb.log({'SmallMLP/train_loss': loss})
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        wandb.log({'SmallMLP/val_loss': loss})
        self.log('SmallMLP/val_loss', loss)

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
        wandb.log({'final_model/test_accuracy': self.test_accuracy.compute()})
        wandb.log({'final_model/test_precision': self.test_precision.compute()})
        wandb.log({'final_model/test_recall': self.test_recall.compute()})
        wandb.log({'final_model/test_f1': self.test_f1.compute()})
        
        self.log('SmallMLP/test_loss', loss)
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
        
       