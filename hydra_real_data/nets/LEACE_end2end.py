import torch
import hydra
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import concept_erasure
from concept_erasure import LeaceFitter
import torchmetrics
import wandb
from utils import selected_conf_plot


class LEACE_end2end(pl.LightningModule):
    def __init__(self, backbone, encoder, device, cfg_optim , num_classes=3, alpha_rate=0.05):
        super(LEACE_end2end, self).__init__()
        
        # self.automatic_optimization = True

        self.cfg_optim = cfg_optim
        self.backbone, self.num_features = backbone
        self.encoder = encoder
        self.encoder_features = 128 #linear_2concept.in_features #self.encoder.linear_2concept.out_features # 
        
        self.num_classes = num_classes
        self.alpha = 1
        self.alpha_rate = alpha_rate
        
        
        self.joint_head = nn.Linear(self.num_features + self.encoder_features, self.num_classes) #1
        self.fitter = LeaceFitter(self.num_features, self.encoder_features, dtype=torch.float, device=device)  #1,
        
        self.test_accuracy = torchmetrics.Accuracy("multiclass", num_classes=self.num_classes)
        self.test_precision = torchmetrics.Precision(task="multiclass", average='macro', num_classes=self.num_classes)
        self.test_recall = torchmetrics.Recall(task="multiclass", average='macro', num_classes=self.num_classes)
        self.test_f1 = torchmetrics.F1Score(task="multiclass", num_classes=self.num_classes)
        self.test_confusion_matrix = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.num_classes)
        self.class_recall_metrics = {class_label: torchmetrics.Recall(task='binary') for class_label in range(self.num_classes)}
        for param in self.encoder.parameters():
            param.requires_grad = False
        

    def forward(self, x, z_c):
        z_x = self.backbone(x)
        z_x = z_x.view(z_x.shape[0], -1)
        self.fitter.update(z_x, z_c)
        z_x_pro = self.fitter.eraser(z_x) 
        z_x_hat = self.alpha * z_x + (1-self.alpha)*z_x_pro
        joint_logits = self.joint_head(torch.cat([z_x_hat,z_c.view(z_x.shape[0], -1)], dim=1))
        return  joint_logits
        
    def get_embedding(self, x):
        z_x = self.backbone(x)
        return self.fitter.eraser(z_x)


    def configure_optimizers(self):
        self.params = list(self.backbone.parameters()) + list(self.joint_head.parameters()) 
        optimizer_informed =  self.cfg_optim.optimizer(params=self.params)
        if  self.cfg_optim.use_lr_scheduler:
            scheduler = self.cfg_optim.lr_scheduler(optimizer=optimizer_informed)
            return [optimizer_informed], [scheduler]
        return optimizer_informed

    def training_step(self, batch, batch_idx):
        x, y, c = batch[0], batch[1].squeeze(), batch[2].squeeze()
        c_logits, z_embed = self.encoder(x)
        final_logits = self(x,z_embed) #c)#c_logits)#
        loss = F.cross_entropy(final_logits,y)
        self.log('informed_encoder/train_loss', loss)
        wandb.log({'informed_encoder/train_loss': loss})
        return loss
    
    def on_train_epoch_end(self) -> None:
        self.alpha -= self.alpha_rate
        if self.alpha <0:
            self.alpha = 0
        return super().on_train_epoch_end()

    def validation_step(self, batch, batch_idx):
        x, y, c = batch[0], batch[1].squeeze(), batch[2].squeeze()
        c_logits, z_embed = self.encoder(x)
        final_logits = self(x,z_embed) #c)#c_logits)#
        loss = F.cross_entropy(final_logits,y)
        wandb.log({'informed_encoder/val_loss': loss})
        self.log('informed_encoder/val_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y, c = batch[0], batch[1].squeeze(), batch[2].squeeze()
        c_logits, z_embed = self.encoder(x)
        final_logits = self(x,z_embed) #c)#c_logits)#
        loss = F.cross_entropy(final_logits,y)
        pred = torch.softmax(final_logits, dim=-1) #normal testing
        # pred = torch.softmax(logits_p, dim=-1) #spurious testing

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
        selected_conf_plot(self.test_confusion_matrix, [x for x in range(self.num_classes)]) #[3,8,27,28,29,106,133])
       