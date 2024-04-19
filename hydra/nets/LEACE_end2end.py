import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import concept_erasure
from concept_erasure import LeaceFitter
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


class LEACE_end2end(pl.LightningModule):
    def __init__(self, encoder, in_dim, hidden_dim, embed_dim, layers=2, activation='relu', lr=1e-4, num_classes=3, alpha_rate=0.05, lamb=0.5):
        super(LEACE_end2end, self).__init__()
        
        # self.automatic_optimization = True
        
        self.encoder = encoder
        self.lr = lr
        self.num_classes = num_classes
        self.alpha = 1
        self.lamb = lamb
        self.alpha_rate = alpha_rate
        
        # encoders
        self.backbone = mlp(in_dim, hidden_dim, embed_dim,1, activation) #28*28*3 #nn.Linear(in_dim, embed_dim) #
        self.linears_head = nn.Linear(embed_dim, self.num_classes ) #mlp(embed_dim, embed_dim, 3, 1, activation)
        self.pre_project = nn.Linear(embed_dim, self.num_classes )
        self.joint_head = nn.Linear(embed_dim*2, self.num_classes) #nn.Linear(embed_dim+1, self.num_classes ) #
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('device is', device)
        self.fitter = LeaceFitter(embed_dim, embed_dim, dtype=torch.float, device=device)  #1
        self.fitter_2 = LeaceFitter(self.num_classes , embed_dim, dtype=torch.float, device=device) #1
        
        self.test_accuracy = torchmetrics.Accuracy("multiclass", num_classes=self.num_classes)
        self.test_precision = torchmetrics.Precision(task="multiclass", average='macro', num_classes=self.num_classes)
        self.test_recall = torchmetrics.Recall(task="multiclass", average='macro', num_classes=self.num_classes)
        self.test_f1 = torchmetrics.F1Score(task="multiclass", num_classes=self.num_classes)
        self.test_confusion_matrix = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=self.num_classes)
        for param in self.encoder.parameters():
            param.requires_grad = False
        

    def forward(self, x, z_c, y):
        # compute embeddings
        x= x.view(x.size(0), -1)
        z_x = self.backbone(x)
        logits_pre = self.pre_project(z_x)
        self.fitter.update(z_x, z_c)
        X_ = self.fitter.eraser(z_x)
        logits = self.linears_head(X_) #z_x)#
        self.fitter_2.update(logits, z_c)
        logits_ = self.fitter_2.eraser(logits)
        final_logits = self.joint_head(torch.cat([z_x,z_c], dim=1))
        return logits_, logits_pre, final_logits
        
    def get_embedding(self, x):
        x = x.view(x.size(0), -1)
        z_x = self.backbone(x)
        return self.fitter.eraser(z_x)


    def configure_optimizers(self):
        optimizer_informed = torch.optim.Adam(list(self.backbone.parameters()) + list(self.linears_head.parameters()) + list(self.pre_project.parameters()), lr=self.lr) #
        return optimizer_informed

    def training_step(self, batch, batch_idx):
        x, y, c = batch[0], batch[1], batch[2]
        c_logits, z_embed = self.encoder(x)
        logits_p, logits, final_logits = self(x,z_embed,y.unsqueeze(1)) #self(x,c.unsqueeze(1).float(),y.unsqueeze(1))
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
        x, y, c = batch[0], batch[1], batch[2]
        c_logits, z_embed = self.encoder(x)
        logits_p, logits, final_logits = self(x,z_embed,y.unsqueeze(1)) #self(x,c.unsqueeze(1).float(),y.unsqueeze(1))
        loss_1 = F.cross_entropy(logits_p, y)
        loss_2 = F.cross_entropy(logits, y)
        leace_loss = loss_1 + self.alpha * loss_2
        final_loss = F.cross_entropy(final_logits,y)
        loss = final_loss + self.lamb*leace_loss
        wandb.log({'informed_encoder/val_loss': loss})
        self.log('informed_encoder/val_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y, c = batch[0], batch[1], batch[2]
        c_logits, z_embed = self.encoder(x)
        logits_p, logits, final_logits = self(x,z_embed,y.unsqueeze(1)) #self(x,c.unsqueeze(1).float(),y.unsqueeze(1))
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
       