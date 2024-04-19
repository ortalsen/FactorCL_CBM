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



class ConceptEncoder(pl.LightningModule):
    def __init__(self, x_dim, c_embed_dim, c_dim, hidden_dim, layers=2, activation='relu', lr=1e-4):
        super(ConceptEncoder, self).__init__()
        self.critic_hidden_dim = 512
        self.critic_layers = 1
        self.critic_activation = 'relu'
        self.lr = lr
        self.c_dim = c_dim

        # encoders
        self.backbone = mlp(28 * 28 * 3, hidden_dim, c_embed_dim,1, activation) #RGB_mlp(28 * 28 * 3, c_embed_dim, num_classes=None)
        self.linear_2concept = mlp(c_embed_dim, c_embed_dim, c_dim, 1, activation)

        self.test_accuracy = torchmetrics.Accuracy("multiclass", num_classes=c_dim)
        self.test_precision = torchmetrics.Precision(task="multiclass", average='macro', num_classes=c_dim)
        self.test_recall = torchmetrics.Recall(task="multiclass", average='macro', num_classes=c_dim)
        self.test_f1 = torchmetrics.F1Score(task="multiclass", num_classes=c_dim)
        self.test_confusion_matrix = torchmetrics.ConfusionMatrix(task="multiclass", num_classes=c_dim)

    def forward(self, x):
        # compute embeddings
        x = x.view(x.size(0), -1)
        z_c = self.backbone(x)
        c = self.linear_2concept(z_c)
        return c, z_c

    def get_embedding(self, x):
        x = x.view(x.size(0), -1)
        return self.backbone(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        x, y, c = batch[0], batch[1], batch[2]
        logits, _ = self(x)
        loss = F.cross_entropy(logits, c)
        self.log('concept_encoder/train_loss', loss)
        wandb.log({'concept_encoder/train_loss': loss})
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, c = batch[0], batch[1], batch[2]
        logits, _= self(x)
        loss = F.cross_entropy(logits, c)
        wandb.log({'concept_encoder/val_loss': loss})
        self.log('concept_encoder/val_loss', loss)

    def test_step(self, batch, batch_idx):
        x, y, c = batch[0], batch[1], batch[2]
        logits, _ = self(x)
        loss = F.cross_entropy(logits, c)
        pred = torch.softmax(logits, dim=-1)

        self.test_accuracy(pred, y)
        self.test_precision(pred,y)
        self.test_recall(pred,y)
        self.test_f1(pred,y)
        self.test_confusion_matrix.update(pred, y)

        wandb.log({'concept_encoder/test_loss': loss})
        wandb.log({'concept_encoder/test_accuracy': self.test_accuracy.compute()})
        wandb.log({'concept_encoder/test_precision': self.test_precision.compute()})
        wandb.log({'concept_encoder/test_recall': self.test_recall.compute()})
        wandb.log({'concept_encoder/test_f1': self.test_f1.compute()})
        
        self.log('concept_encoder/test_loss', loss)
        self.log_dict(
            {
                "concept_encoder/accuracy_test": self.test_accuracy.compute(),
                'concept_encoder/test_accuracy': self.test_accuracy.compute(),
                'concept_encoder/test_precision': self.test_precision.compute(),
                'concept_encoder/test_recall': self.test_recall.compute(),
                'concept_encoder/test_f1': self.test_f1.compute()

            },
        )

    def on_test_epoch_end(self):
        cm, _ = self.test_confusion_matrix.plot()
        cm.savefig('cm.png')
        wandb.log({"concept_encoder/confusion_matrix": wandb.Image('cm.png')})
       