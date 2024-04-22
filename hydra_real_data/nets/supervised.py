import torch
import hydra
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import wandb


class Supervised(pl.LightningModule):
    def __init__(self, backbone, num_classes, cfg_optim):
        super(Supervised, self).__init__()

        self.cfg_optim = cfg_optim
        self.num_classes = num_classes

        # encoders
        self.backbone, self.num_features = backbone 
        self.linear_2label = nn.Linear(self.num_features, self.num_classes)

        self.test_accuracy = torchmetrics.Accuracy("multiclass", num_classes=self.num_classes)
        self.test_precision = torchmetrics.Precision(task="multiclass", average='macro', num_classes=self.num_classes)
        self.test_recall = torchmetrics.Recall(task="multiclass", average='macro', num_classes=self.num_classes)
        self.test_f1 = torchmetrics.F1Score(task="multiclass", num_classes=self.num_classes)
        self.test_confusion_matrix = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.num_classes)

    def forward(self, x):
        z_c = self.backbone(x)
        logits = self.linear_2label(z_c.view(z_c.shape[0], -1))
        return logits


    def configure_optimizers(self):
        self.params = self.parameters()
        concept_optimizer =  self.cfg_optim.optimizer(params=self.params)
        if  self.cfg_optim.use_lr_scheduler:
            scheduler = hydra.utils.instantiate(
                 self.cfg_optim.lr_scheduler, optimizer=concept_optimizer
            )
            return [concept_optimizer], [scheduler]
        return concept_optimizer

    def training_step(self, batch, batch_idx):
        x, y, c = batch[0], batch[1].squeeze(), batch[2].squeeze()
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('final_model/train_loss', loss)
        wandb.log({'final_model/train_loss': loss})
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, c = batch[0], batch[1].squeeze(), batch[2].squeeze()
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        wandb.log({'final_model/val_loss': loss})
        self.log('final_model/val_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y, c = batch[0], batch[1].squeeze(), batch[2].squeeze()
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
       