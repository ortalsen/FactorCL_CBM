import torch
import hydra
import concept_erasure
from concept_erasure import LeaceFitter, LeaceEraser
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import wandb
from utils import selected_conf_plot


class Supervised(pl.LightningModule):
    def __init__(self, backbone, num_classes, cfg_optim):
        super(Supervised, self).__init__()

        self.cfg_optim = cfg_optim
        self.num_classes = num_classes

        # encoders
        self.backbone, self.num_features = backbone 
        self.linear_2label = nn.Linear(self.num_features, self.num_classes)
        self.fitter = LeaceFitter(self.num_features, 1, dtype=torch.float, device='cuda') 

        self.test_accuracy = torchmetrics.Accuracy("multiclass", num_classes=self.num_classes)
        self.test_precision = torchmetrics.Precision(task="multiclass", average='macro', num_classes=self.num_classes)
        self.test_recall = torchmetrics.Recall(task="multiclass", average='macro', num_classes=self.num_classes)
        self.test_f1 = torchmetrics.F1Score(task="multiclass", num_classes=self.num_classes)
        self.test_confusion_matrix = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.num_classes)
        self.class_recall_metrics = {class_label: torchmetrics.Recall(task='binary') for class_label in range(self.num_classes)}

    def forward(self, x):
        z_x = self.backbone(x)
        logits = self.linear_2label(z_x.view(z_x.shape[0], -1))
        return logits, z_x


    def configure_optimizers(self):
        self.params = self.parameters()
        concept_optimizer =  self.cfg_optim.optimizer(params=self.params)
        if  self.cfg_optim.use_lr_scheduler:
            scheduler = self.cfg_optim.lr_scheduler(optimizer=concept_optimizer)
            return [concept_optimizer], [scheduler]
        return concept_optimizer

    def training_step(self, batch, batch_idx):
        x, y, c = batch[0], batch[1].squeeze(), batch[2].squeeze()
        logits,_ = self(x)
        # print('y', y.shape)
        # print('logits', logits.shape)
        loss = F.cross_entropy(logits, y)
        self.log('final_model/train_loss', loss)
        wandb.log({'final_model/train_loss': loss})
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, c = batch[0], batch[1].squeeze(), batch[2].squeeze()
        logits,_ = self(x)
        loss = F.cross_entropy(logits, y)
        wandb.log({'final_model/val_loss': loss})
        self.log('final_model/val_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y, c = batch[0], batch[1].squeeze(), batch[2].squeeze()
        logits, z_x = self(x)
        
        z_x = z_x.view(z_x.shape[0], -1)
        # self.fitter.update(z_x, c)
        # z_x_hat = self.fitter.eraser(z_x)
        # eraser = LeaceEraser.fit(z_x, c)
        # z_x_hat = eraser(z_x)
        # logits_new = self.linear_2label(z_x_hat)
        # print('same embed??', z_x == z_x_hat)
        # print('same??', logits == logits_new)
        # print('new logits', logits_new.shape)

        loss = F.cross_entropy(logits, y) #logits_new,y) #
        pred = torch.softmax(logits, dim=-1)#logits_new, dim=-1) #

        self.test_accuracy(pred, y)
        self.test_precision(pred,y)
        self.test_recall(pred,y)
        self.test_f1(pred,y)
        self.test_confusion_matrix.update(pred, y)
        # for class_label in range(self.num_classes):
        #     class_mask = (y == class_label)
        #     class_outputs = pred[:, class_label]
        #     class_targets_binary = class_mask.to(torch.float32).to(y.device)
        #     self.class_recall_metrics[class_label].to(y.device)
        #     self.class_recall_metrics[class_label](class_outputs, class_targets_binary)

        wandb.log({'final_model/test_loss': loss})
        wandb.log({'final_model/test_accuracy': self.test_accuracy.compute()})
        wandb.log({'final_model/test_precision': self.test_precision.compute()})
        wandb.log({'final_model/test_recall': self.test_recall.compute()})
        wandb.log({'final_model/test_f1': self.test_f1.compute()})
        # for class_label, metric in self.class_recall_metrics.items():
        #     self.log(f"final_model/test_recall_{class_label}", metric.compute())
        
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
        # cm, _ = self.test_confusion_matrix.plot()
        # cm.savefig('cm.png')
        # wandb.log({"final_model/confusion_matrix": wandb.Image('cm.png')})
        selected_conf_plot(self.test_confusion_matrix, [x for x in range(self.num_classes)]) #3,8,27,28,29,106,133
       