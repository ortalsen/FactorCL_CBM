import torch
import hydra
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import concept_erasure
from concept_erasure import LeaceFitter
import torchmetrics
import wandb


class LEACE_end2end(pl.LightningModule):
    def __init__(self, backbone, encoder, device, cfg_optim , num_classes=3, alpha_rate=0.05, lamb=0.5):
        super(LEACE_end2end, self).__init__()
        
        # self.automatic_optimization = True

        self.cfg_optim = cfg_optim
        self.backbone, self.num_features = backbone
        self.encoder = encoder
        self.encoder_features = self.encoder.linear_2concept.in_features # self.encoder.linear_2concept.out_features
        
        self.num_classes = num_classes
        self.alpha = 1
        self.lamb = lamb
        self.alpha_rate = alpha_rate
        
        

        self.linears_head = nn.Linear(self.num_features, self.num_classes )
        self.pre_project = nn.Linear(self.num_features, self.num_classes )
        self.joint_head = nn.Linear(self.num_features + self.encoder_features, self.num_classes) 


        self.fitter = LeaceFitter(self.num_features, self.encoder_features, dtype=torch.float, device=device)  
        self.fitter_2 = LeaceFitter(self.num_classes , self.encoder_features, dtype=torch.float, device=device) 
        
        self.test_accuracy = torchmetrics.Accuracy("multiclass", num_classes=self.num_classes)
        self.test_precision = torchmetrics.Precision(task="multiclass", average='macro', num_classes=self.num_classes)
        self.test_recall = torchmetrics.Recall(task="multiclass", average='macro', num_classes=self.num_classes)
        self.test_f1 = torchmetrics.F1Score(task="multiclass", num_classes=self.num_classes)
        self.test_confusion_matrix = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.num_classes)
        for param in self.encoder.parameters():
            param.requires_grad = False
        

    def forward(self, x, z_c):

        z_x = self.backbone(x)
        z_x = z_x.view(z_x.shape[0], -1)
        logits_pre = self.pre_project(z_x)

        self.fitter.update(z_x, z_c)
        z_x_hat = self.fitter.eraser(z_x)

        logits = self.linears_head(z_x_hat) 
        self.fitter_2.update(logits, z_c)
        logits_hat = self.fitter_2.eraser(logits)
        joint_logits = self.joint_head(torch.cat([z_x,z_c.reshape(z_x.shape)], dim=1))

        return logits_hat, logits_pre, joint_logits
        
    def get_embedding(self, x):
        z_x = self.backbone(x)
        return self.fitter.eraser(z_x)


    def configure_optimizers(self):
        self.params = list(self.backbone.parameters()) + list(self.linears_head.parameters()) + list(self.pre_project.parameters())
        optimizer_informed =  self.cfg_optim.optimizer(params=self.params)
        if  self.cfg_optim.use_lr_scheduler:
            scheduler = hydra.utils.instantiate(
                 self.cfg_optim.lr_scheduler, optimizer=optimizer_informed
            )
            return [optimizer_informed], [scheduler]
        return optimizer_informed

    def training_step(self, batch, batch_idx):
        x, y, c = batch[0], batch[1].squeeze(), batch[2].squeeze()
        c_logits, z_embed = self.encoder(x)
        logits_p, logits, final_logits = self(x,z_embed) 
        loss_1 = F.cross_entropy(logits_p, y)
        loss_2 = F.cross_entropy(logits, y)
        leace_loss = loss_1 + self.alpha * loss_2
        final_loss = F.cross_entropy(final_logits,y)
        loss = final_loss + self.lamb*leace_loss
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
        logits_p, logits, final_logits = self(x,z_embed) 
        loss_1 = F.cross_entropy(logits_p, y)
        loss_2 = F.cross_entropy(logits, y)
        leace_loss = loss_1 + self.alpha * loss_2
        final_loss = F.cross_entropy(final_logits,y)
        loss = final_loss + self.lamb*leace_loss
        wandb.log({'informed_encoder/val_loss': loss})
        self.log('informed_encoder/val_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y, c = batch[0], batch[1].squeeze(), batch[2].squeeze()
        c_logits, z_embed = self.encoder(x)
        logits_p, logits, final_logits = self(x,z_embed) 
        loss_1 = F.cross_entropy(logits_p, y)
        loss_2 = F.cross_entropy(logits, y)
        leace_loss = loss_1 + self.alpha * loss_2
        final_loss = F.cross_entropy(final_logits,y)
        loss = final_loss + self.lamb*leace_loss
        pred = torch.softmax(final_logits, dim=-1) #normal testing
        # pred = torch.softmax(logits_p, dim=-1) #spurious testing

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
       