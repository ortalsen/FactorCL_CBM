import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import concept_erasure
from concept_erasure import LeaceFitter
import torchmetrics
import wandb


class CLUBInfoNCECritic(nn.Module):
    def __init__(self, A_dim, B_dim, hidden_dim, layers, activation, **extra_kwargs):
          super(CLUBInfoNCECritic, self).__init__()
 
          self._f = mlp(A_dim + B_dim, hidden_dim, 1, layers, activation)

    # CLUB loss
    def forward(self, x_samples, y_samples):
        sample_size = y_samples.shape[0]

        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))

        T0 = self._f(torch.cat([y_samples,x_samples], dim = -1)) 
        T1 = self._f(torch.cat([y_tile, x_tile], dim = -1))  

        return T0.mean() - T1.mean()

    # InfoNCE loss
    def learning_loss(self, x_samples, y_samples):
        sample_size = y_samples.shape[0]

        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))

        T0 = self._f(torch.cat([y_samples,x_samples], dim = -1))
        T1 = self._f(torch.cat([y_tile, x_tile], dim = -1)) 

        lower_bound = T0.mean() - (T1.logsumexp(dim = 1).mean() - np.log(sample_size)) 
        return -lower_bound



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


class CLUBInfoNCE(pl.LightningModule):
    def __init__(self, encoder, in_dim, hidden_dim, embed_dim, layers=2, activation='relu', lr=1e-4, num_classes=3):
        super(CLUBInfoNCE, self).__init__()
        
        self.automatic_optimization = True
        
        self.encoder = encoder
        self.lr = lr
        self.num_classes = num_classes
        
        # encoders
        self.critic_hidden_dim = 512
        self.critic_layers = 1
    

        self.backbone = mlp(in_dim, hidden_dim, embed_dim,1, activation) #28*28*3 #nn.Linear(in_dim, embed_dim) #
        self.linears_head = nn.Linear(embed_dim, self.num_classes) #mlp(embed_dim, embed_dim, 3, 1, activation)
        self.critic = CLUBInfoNCECritic(embed_dim+num_classes, 1, self.critic_hidden_dim, self.critic_layers, activation) 
        
        self.test_accuracy = torchmetrics.Accuracy("multiclass", num_classes=self.num_classes)
        self.test_precision = torchmetrics.Precision(task="multiclass", average='macro', num_classes=self.num_classes)
        self.test_recall = torchmetrics.Recall(task="multiclass", average='macro', num_classes=self.num_classes)
        self.test_f1 = torchmetrics.F1Score(task="multiclass", num_classes=self.num_classes)
        self.test_confusion_matrix = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.num_classes)
        

    def forward(self, x, z_c, y):
        # compute embeddings
        x=x.view(x.size(0), -1)
        z_x = self.backbone(x)
        logit = self.linears_head(z_x)
        y_ohe = self.ohe(y)
        score = self.critic(torch.cat((z_x, y_ohe), dim=1), z_c)
        return score, logit
        
    def ohe(self, y):
        N = y.shape[0]
        y_ohe = torch.zeros((N, self.num_classes)).to(y.device)
        y_ohe[torch.arange(N).long(), y.T[0].long()] = 1
        return y_ohe
        
    def get_embedding(self, x):
        x = x.view(x.size(0), -1)
        return self.backbone(x)


    def configure_optimizers(self):
        optimizer_informed = torch.optim.Adam(list(self.backbone.parameters()) + list(self.critic.parameters()), lr=self.lr) 
        return optimizer_informed

    def training_step(self, batch, batch_idx):
        x, y, c = batch[0], batch[1], batch[2]
        c_logits, z_embed = self.encoder(x)
        loss, _ = self(x,c.unsqueeze(1).float(),y.unsqueeze(1))
        # loss = -loss
        self.log('informed_encoder/train_loss', loss)
        wandb.log({'informed_encoder/train_loss': loss})
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, c = batch[0], batch[1], batch[2]
        c_logits, z_embed = self.encoder(x)
        loss, _ = self(x,c.unsqueeze(1).float(),y.unsqueeze(1))
        # loss = -loss
        wandb.log({'informed_encoder/val_loss': loss})
        self.log('informed_encoder/val_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y, c = batch[0], batch[1], batch[2]
        c_logits, z_embed = self.encoder(x)
        loss, logit = self(x,c.unsqueeze(1).float(),y.unsqueeze(1))
        
        pred = torch.softmax(logit, dim=-1)
        self.test_accuracy(pred, y)
        self.test_precision(pred,y)
        self.test_recall(pred,y)
        self.test_f1(pred,y)
        self.test_confusion_matrix.update(pred, y)

        wandb.log({'informed_encoder/test_loss': loss})
        wandb.log({'informed_encoder/test_accuracy': self.test_accuracy.compute()})
        wandb.log({'informed_encoder/test_precision': self.test_precision.compute()})
        wandb.log({'informed_encoder/test_recall': self.test_recall.compute()})
        wandb.log({'informed_encoder/test_f1': self.test_f1.compute()})
        
        self.log('informed_encoder/test_loss', loss)
        self.log_dict(
            {
                "informed_encoder/accuracy_test": self.test_accuracy.compute(),
                'informed_encoder/test_accuracy': self.test_accuracy.compute(),
                'informed_encoder/test_precision': self.test_precision.compute(),
                'informed_encoder/test_recall': self.test_recall.compute(),
                'informed_encoder/test_f1': self.test_f1.compute()

            },
        )

    def on_test_epoch_end(self):
        cm = self.test_confusion_matrix.compute()
        wandb.log({"informed_encoder/confusion_matrix": cm})
        
       