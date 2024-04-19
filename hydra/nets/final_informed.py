import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import wandb


class FinalInformed(pl.LightningModule):
    def __init__(self, encoder, informed_encoder, embed_dim, num_classes, lr):
        super(FinalInformed, self).__init__()
        
        # self.automatic_optimization = True
        
        self.encoder = encoder
        self.informed_encoder = informed_encoder
        self.lr = lr
        self.num_classes = num_classes
        self.FC = nn.Linear(embed_dim*2, num_classes) #*2

        self.test_accuracy = torchmetrics.Accuracy("multiclass", num_classes=self.num_classes)
        self.test_precision = torchmetrics.Precision(task="multiclass", average='macro', num_classes=self.num_classes)
        self.test_recall = torchmetrics.Recall(task="multiclass", average='macro', num_classes=self.num_classes)
        self.test_f1 = torchmetrics.F1Score(task="multiclass", num_classes=self.num_classes)
        self.test_confusion_matrix = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.num_classes)

        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.informed_encoder.parameters():
            param.requires_grad = False
        

    def forward(self, x, y, c):
        # compute embeddings
        x=x.view(x.size(0), -1)
        with torch.no_grad():
            z_c = self.encoder.get_embedding(x)
            z_x = self.informed_encoder.get_embedding(x)
        
        logits = self.FC(torch.cat([z_x,z_c], dim=1))#self.FC(z_x) #
        # print('c size', c.shape)
        # logits = self.FC(torch.cat([c.unsqueeze(1),z_x], dim=-1))
        return logits
        

    def configure_optimizers(self):
        return torch.optim.Adam(self.FC.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        x, y, c = batch[0], batch[1], batch[2]
        logits = self(x,y,c)
        loss = F.cross_entropy(logits, y)
        self.log('final_model/train_loss', loss)
        wandb.log({'final_model/train_loss': loss})
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, c = batch[0], batch[1], batch[2]
        logits = self(x,y,c)
        loss = F.cross_entropy(logits, y)
        wandb.log({'final_model/val_loss': loss})
        self.log('final_model/val_loss', loss) #on_step=True, on_epoch=False

    def test_step(self, batch, batch_idx):
        x, y, c = batch[0], batch[1], batch[2]
        logits = self(x,y,c)
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
        # cm.savefig('cm.png')
        wandb.log({"final_model/confusion_matrix": wandb.Image(cm)})
       