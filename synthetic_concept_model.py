import numpy as np
import torch.nn
from critic_objectives import*
from Synthetic.dataset import get_intersections, generate_data_concepts, MultiConcept
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from critic_objectives import mlp, InfoNCECritic
from tqdm import tqdm

class ConceptCLSUP(nn.Module):
    def __init__(self, x_dim, c_embed_dim, y_ohe_dim, hidden_dim, embed_dim, layers=2, activation='relu', lr=1e-4):
        super(ConceptCLSUP, self).__init__()
        self.critic_hidden_dim = 512
        self.critic_layers = 1
        self.critic_activation = 'relu'
        self.lr = lr
        self.y_ohe_dim = y_ohe_dim

        # encoders
        self.backbone = mlp(x_dim, hidden_dim, embed_dim, layers, activation)
        self.linears_infonce = mlp(embed_dim, embed_dim, embed_dim, 1, activation)

        # critics
        self.critic = InfoNCECritic(embed_dim+y_ohe_dim,1, self.critic_hidden_dim, self.critic_layers, self.critic_activation) # c_embed_dim

    def forward(self, x, z_c, y):
        y_ohe = self.ohe(y)
        # compute embeddings
        z_x = self.linears_infonce(self.backbone(x))
        # compute critic scores
        infonce_score = self.critic(torch.cat([z_x, y_ohe], dim=-1), z_c)
        return infonce_score

    def ohe(self, y):
        N = y.shape[0]
        y_ohe = torch.zeros((N, self.y_ohe_dim)).to(y.device)
        y_ohe[torch.arange(N).long(), y.T[0].long()] = 1
        return y_ohe

    def get_embedding(self, x):
        return self.backbone(x)
    
    

class ConceptEncoder(nn.Module):
    def __init__(self, x_dim, c_embed_dim, yc_dim, hidden_dim, layers=2, activation='relu', lr=1e-4):
        super(ConceptEncoder, self).__init__()
        self.critic_hidden_dim = 512
        self.critic_layers = 1
        self.critic_activation = 'relu'
        self.lr = lr
        self.yc_dim = yc_dim

        # encoders
        self.backbone = mlp(x_dim, hidden_dim, c_embed_dim, layers, activation)
        self.linear_2concept = mlp(c_embed_dim, c_embed_dim, yc_dim, 1, activation)

    def forward(self, x):
        # compute embeddings
        z_c = self.backbone(x)
        c = self.linear_2concept(z_c)
        return c, z_c

    def get_embedding(self, x):
        return self.backbone(x)
    

def train_concept_encoder(model, train_loader, val_loader, num_epochs, device, lr, weight_decay, log_interval,
                          save_interval, save_path):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    concept_loss = torch.nn.MSELoss()
    best_val_err = torch.tensor(1e7)
    model.train()
    tepoch = tqdm(range(num_epochs))
    for epoch in tepoch:
        tepoch.set_description(f"Epoch {epoch}")
        for batch_idx, (data, concept, target) in enumerate(train_loader):
            data, concept = data.to(device), concept.to(device)
            optimizer.zero_grad()
            output, _ = model(data)
            loss = concept_loss(output, concept)
            loss.backward()
            optimizer.step()
            # if batch_idx % log_interval == 0:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tConcept Loss: {:.6f}'.format(
            #         epoch, batch_idx * len(data), len(train_loader.dataset),
            #                100. * batch_idx / len(train_loader), loss.item()))
        tepoch.set_postfix(loss=loss.item())
        if epoch % save_interval == 0:
            val_err = 0
            model.eval()
            with torch.no_grad():
                for batch_idx, (data, concept, target) in enumerate(val_loader):
                    data, concept = data.to(device), concept.to(device)
                    output, _ = model(data)
                    val_err += concept_loss(output, concept)
                val_err = val_err / len(val_loader)
            # print('Val loss: {:.6f}'.format(val_err))
            if val_err < best_val_err:
                best_val_err = val_err
            else:
                print('Val loss did not improve')
                torch.save(model.state_dict(),
                           os.path.join(save_path, 'concept_encoder_val_err_{}.pth'.format(val_err)))
                return model
    return model

def train_concept_informed_model(concept_encoder, model, train_loader,val_loader, num_epochs, device, lr, log_interval,
                          save_interval, save_path):

    concept_encoder.eval()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_err = torch.tensor(1e7)
    tepoch = tqdm(range(num_epochs))
    for epoch in tepoch:
        tepoch.set_description(f"Epoch {epoch}")
        model.train()
        for batch_idx, (data, concept, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            c, z_c = concept_encoder(data)
            loss = model(data, c, target) #z_c
            loss.backward()
            optimizer.step()
            # if batch_idx % log_interval == 0:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tConcept Loss: {:.6f}'.format(
            #         epoch, batch_idx * len(data), len(train_loader.dataset),
            #                100. * batch_idx / len(train_loader), loss.item()))
        tepoch.set_postfix(loss=loss.item())
        if epoch % save_interval == 0:
            val_err = 0
            model.eval()
            with torch.no_grad():
                for batch_idx, (data, concept, target) in enumerate(val_loader):
                    data, target = data.to(device), target.to(device)
                    c , z_c = concept_encoder(data)
                    output = model(data, c, target) #z_c
                    val_err += output
                val_err = val_err / len(val_loader)
            # print('Val loss: {:.6f}'.format(val_err))
            if val_err < best_val_err:
                best_val_err = val_err

            else:
                print('Val loss did not improve')
                torch.save(model.state_dict(), os.path.join(save_path, 'concept_informed_model.pth'))
                return model
    return model




if __name__ == '__main__':
    feature_dim_info = dict()
    label_dim_info = dict()

    intersections = get_intersections(num_modalities=2)

    feature_dim_info['12'] = 10
    feature_dim_info['1'] = 6
    feature_dim_info['2'] = 6

    label_dim_info['12'] = 10
    label_dim_info['1'] = 6
    label_dim_info['2'] = 6
    num_concepts = 2
    transforms_2concept = None
    transforms_2hd = None
    num_data = 30000
    total_data, total_labels, total_concepts, total_raw_features = generate_data_concepts(num_data, num_concepts,
                                                                                          feature_dim_info,
                                                                                          label_dim_info,
                                                                                          transforms_2concept=None,
                                                                                          transforms_2hd=None,
                                                                                          noise=10,
                                                                                          pos_prob=0.7)
    dataset = MultiConcept(total_data, total_labels, total_concepts, 0)
    batch_size = 256
    trainval_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                                [int(0.8 * num_data), num_data - int(0.8 * num_data)])
    train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset,
                                                               [int(0.8 * len(trainval_dataset)), len(trainval_dataset) - int(0.8 * len(trainval_dataset))])

    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True,
                              batch_size=batch_size)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False,
                             batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hidden_dim = 512
    embed_dim = 128
    concept_encoder = ConceptEncoder(100, embed_dim, 1, hidden_dim).to(device)
    model = ConceptCLSUP(100, embed_dim, 2, hidden_dim, embed_dim).to(device)

    trained_concept_encoder = train_concept_encoder(concept_encoder, train_loader,val_loader, 100, device, 1e-3, 1e-5, 25, 3, 'trained_models')
    trained_concept_informed_model = train_concept_informed_model(trained_concept_encoder, model, train_loader, val_loader, 100, device, 1e-4, 25, 3, 'trained_models')
