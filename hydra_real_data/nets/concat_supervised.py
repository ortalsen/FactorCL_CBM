import torch
import hydra
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import concept_erasure
from concept_erasure import LeaceFitter
import torchmetrics
import wandb


class concatSupervised(pl.LightningModule):
    def __init__(self, backbone, cfg_optim , num_classes=3):
        super(concatSupervised, self).__init__()
        
        # self.automatic_optimization = True

        self.cfg_optim = cfg_optim
        self.backbone, self.num_features = backbone
        self.num_classes = num_classes  
        self.joint_head = nn.Linear(self.num_features + 1, self.num_classes) 
        
        self.test_accuracy = torchmetrics.Accuracy("multiclass", num_classes=self.num_classes)
        self.test_precision = torchmetrics.Precision(task="multiclass", average='macro', num_classes=self.num_classes)
        self.test_recall = torchmetrics.Recall(task="multiclass", average='macro', num_classes=self.num_classes)
        self.test_f1 = torchmetrics.F1Score(task="multiclass", num_classes=self.num_classes)
        self.test_confusion_matrix = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.num_classes)
        
        

    def forward(self, x, c):
        z_x = self.backbone(x)
        z_x = z_x.view(z_x.shape[0], -1)
        joint_logits = self.joint_head(torch.cat([z_x,c.view(z_x.shape[0], -1)], dim=1))
        return  joint_logits
        

    def configure_optimizers(self):
        self.params = list(self.backbone.parameters()) + list(self.joint_head.parameters()) 
        optimizer_informed =  self.cfg_optim.optimizer(params=self.params)
        if  self.cfg_optim.use_lr_scheduler:
            scheduler = self.cfg_optim.lr_scheduler(optimizer=optimizer_informed)
            return [optimizer_informed], [scheduler]
        return optimizer_informed

    def training_step(self, batch, batch_idx):
        x, y, c = batch[0], batch[1].squeeze(), batch[2].squeeze()
        final_logits = self(x,c)
        loss = F.cross_entropy(final_logits,y)
        self.log('final_model/train_loss', loss)
        wandb.log({'final_model/train_loss': loss})
        return loss
    

    def validation_step(self, batch, batch_idx):
        x, y, c = batch[0], batch[1].squeeze(), batch[2].squeeze()
        final_logits = self(x,c)
        loss = F.cross_entropy(final_logits,y)
        wandb.log({'final_model/val_loss': loss})
        self.log('final_model/val_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y, c = batch[0], batch[1].squeeze(), batch[2].squeeze()
        final_logits = self(x,c)
        loss = F.cross_entropy(final_logits,y)
        pred = torch.softmax(final_logits, dim=-1) 

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
                'final_model/test_precision': self.test_precision.compute(),
                'final_model/test_recall': self.test_recall.compute(),
                'final_model/test_f1': self.test_f1.compute()

            },
        )
        
    def on_test_epoch_end(self):
        cm, _ = self.test_confusion_matrix.plot()
        cm.savefig('cm.png')
        wandb.log({"final_model/confusion_matrix": wandb.Image('cm.png')})
       