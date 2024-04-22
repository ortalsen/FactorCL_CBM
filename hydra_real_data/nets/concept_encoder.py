import torch
import hydra
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import wandb


class ConceptEncoder(pl.LightningModule):
    def __init__(self, backbone, num_classes, cfg_optim):
        super(ConceptEncoder, self).__init__()

        self.cfg_optim = cfg_optim
        self.num_classes = num_classes

        # encoders
        self.backbone, self.num_features = backbone 
        self.linear_2concept = nn.Linear(self.num_features, 1) #nn.Linear(self.num_features, self.num_classes)

        # self.test_accuracy = torchmetrics.Accuracy("multiclass", num_classes=self.num_classes)
        # self.test_precision = torchmetrics.Precision(task="multiclass", average='macro', num_classes=self.num_classes)
        # self.test_recall = torchmetrics.Recall(task="multiclass", average='macro', num_classes=self.num_classes)
        # self.test_f1 = torchmetrics.F1Score(task="multiclass", num_classes=self.num_classes)
        # self.test_confusion_matrix = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.num_classes)

    def forward(self, x):
        z_c = self.backbone(x)
        c = self.linear_2concept(z_c.view(z_c.shape[0], -1))
        return c, z_c

    def get_embedding(self, x):
        return self.backbone(x)

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
        x, y, c = batch[0], batch[1].squeeze(), batch[2].float()
        logits, _ = self(x)
        loss = F.mse_loss(logits.float(), c) #F.cross_entropy(logits, c.squeeze())
        self.log('concept_encoder/train_loss', loss)
        wandb.log({'concept_encoder/train_loss': loss})
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, c = batch[0], batch[1].squeeze(), batch[2].float()
        logits, _= self(x)
        loss = F.mse_loss(logits.float(), c) #F.cross_entropy(logits, c.squeeze())
        wandb.log({'concept_encoder/val_loss': loss})
        self.log('concept_encoder/val_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y, c = batch[0], batch[1].squeeze(), batch[2].float()
        logits, _ = self(x)
        loss = F.mse_loss(logits, c) #F.cross_entropy(logits, c.squeeze())
        pred = logits #torch.softmax(logits, dim=-1)

        self.test_accuracy(pred, c)
        self.test_precision(pred,c)
        self.test_recall(pred,c)
        self.test_f1(pred,c)
        # self.test_confusion_matrix.update(pred, c)

        wandb.log({'concept_encoder/test_loss': loss})
        # wandb.log({'concept_encoder/test_accuracy': self.test_accuracy.compute()})
        # wandb.log({'concept_encoder/test_precision': self.test_precision.compute()})
        # wandb.log({'concept_encoder/test_recall': self.test_recall.compute()})
        # wandb.log({'concept_encoder/test_f1': self.test_f1.compute()})
        
        self.log('concept_encoder/test_loss', loss)
        # self.log_dict(
            #  {
                # "concept_encoder/accuracy_test": self.test_accuracy.compute(),
                # 'concept_encoder/test_accuracy': self.test_accuracy.compute(),
                # 'concept_encoder/test_precision': self.test_precision.compute(),
                # 'concept_encoder/test_recall': self.test_recall.compute(),
                # 'concept_encoder/test_f1': self.test_f1.compute()

        #     },
        # )

    def on_test_epoch_end(self):
        cm, _ = self.test_confusion_matrix.plot()
        cm.savefig('cm.png')
        wandb.log({"concept_encoder/confusion_matrix": wandb.Image('cm.png')})
       