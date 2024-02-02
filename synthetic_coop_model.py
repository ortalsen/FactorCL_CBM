import numpy as np
import torch.nn
from critic_objectives import*
from Synthetic.dataset import get_intersections, generate_data_concepts, MultiConcept
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from critic_objectives import mlp, InfoNCECritic


class ConceptCLSUP_full_concept(nn.Module):
    def __init__(self, x_dim, c_embed_dim, y_ohe_dim, hidden_dim, embed_dim, layers=2, activation='relu', lr=1e-4):
        super(ConceptCLSUP_full_concept, self).__init__()
        self.critic_hidden_dim = 512
        self.critic_layers = 1
        self.critic_activation = 'relu'
        self.lr = lr
        self.y_ohe_dim = y_ohe_dim
        true_c_dim = 1
        
        # encoders
        self.backbone = mlp(x_dim, hidden_dim, embed_dim, layers, activation)
        self.linears_infonce = mlp(embed_dim, embed_dim, embed_dim, 1, activation)
        # self.linears_infonce_x1x2 = nn.ModuleList([mlp(embed_dim, c_embed_dim, embed_dim, 1, 
        #                                                activation) for i in range(2)])
        # self.linears_club_x1x2_cond = nn.ModuleList([mlp(embed_dim, embed_dim, embed_dim, 1, 
        #                                                  activation) for i in range(2)])

        self.linears_infonce_x1x2 = nn.ModuleList([mlp(embed_dim, c_embed_dim, embed_dim, 1, 
                                                       activation), mlp(true_c_dim, c_embed_dim, embed_dim, 1, 
                                                       activation) ])
        self.linears_club_x1x2_cond = nn.ModuleList([mlp(embed_dim, embed_dim, embed_dim, 1, 
                                                         activation), mlp(true_c_dim, embed_dim, embed_dim, 1, 
                                                         activation)])
        
        self.linears_infonce_x1y = mlp(embed_dim, embed_dim, embed_dim, 1, activation)
        self.linears_infonce_x2y = mlp(true_c_dim, embed_dim, embed_dim, 1, activation) 
        
        # self.linears_infonce_x1x2_cond = nn.ModuleList([mlp(embed_dim, embed_dim, embed_dim, 
        #                                                     1, activation) for i in 
        #                                                 range(2)])
        # self.linears_club_x1x2 = nn.ModuleList([mlp(embed_dim, embed_dim, embed_dim, 1, 
        #                                             activation) for i in range(2)])
        
        self.linears_infonce_x1x2_cond = nn.ModuleList([mlp(embed_dim, embed_dim, embed_dim, 
                                                            1, activation), mlp(true_c_dim, embed_dim, embed_dim, 
                                                            1, activation)])
        self.linears_club_x1x2 = nn.ModuleList([mlp(embed_dim, embed_dim, embed_dim, 1, 
                                                    activation), mlp(true_c_dim, embed_dim, embed_dim, 1, 
                                                    activation)])


        # critics
        self.critic = MINECritic(#InfoNCECritic(
            embed_dim+y_ohe_dim, c_embed_dim, self.critic_hidden_dim, self.critic_layers,
            self.critic_activation
        )
        self.infonce_x1x2 = MINECritic( #InfoNCECritic(
            embed_dim, c_embed_dim, self.critic_hidden_dim,
                                          self.critic_layers, activation)
        self.club_x1x2_cond = MINECritic( #CLUBInfoNCECritic(
            embed_dim + y_ohe_dim, c_embed_dim,
                                                self.critic_hidden_dim, self.critic_layers,
                                                activation)

        self.infonce_x1y = MINECritic( #InfoNCECritic(
            embed_dim, 1, self.critic_hidden_dim, 
                                         self.critic_layers, activation) 
        self.infonce_x2y = MINECritic( #InfoNCECritic
        c_embed_dim, 1, self.critic_hidden_dim, 
                                         self.critic_layers, activation) 
        self.infonce_x1x2_cond = MINECritic( #InfoNCECritic(
            embed_dim + y_ohe_dim, c_embed_dim, 
                                               self.critic_hidden_dim, self.critic_layers, 
                                               activation)
        self.club_x1x2 = MINECritic( #CLUBInfoNCECritic(
            embed_dim, c_embed_dim, self.critic_hidden_dim, 
                                           self.critic_layers, activation)

        self.linears_list = [self.linears_infonce_x1x2, self.linears_club_x1x2_cond,
                             self.linears_infonce_x1y, self.linears_infonce_x2y, 
                             self.linears_infonce_x1x2_cond, self.linears_club_x1x2 
        ] 
        self.critics_list = [self.infonce_x1x2, self.club_x1x2_cond,
                             self.infonce_x1y, self.infonce_x2y, 
                             self.infonce_x1x2_cond, self.club_x1x2 
        ] 
        

    def forward(self, x, z_c, y):
        
        # Get embeddings
        x1_embed = self.linears_infonce(self.backbone(x)) #self.backbones[0](x1)
        x2_embed = z_c  #self.backbones[1](x2)

        y_ohe = self.ohe(y).cuda()

        #compute losses
        uncond_losses = [self.infonce_x1x2(self.linears_infonce_x1x2[0](x1_embed), 
                                           self.linears_infonce_x1x2[1](x2_embed)),
                         self.club_x1x2(self.linears_club_x1x2[0](x1_embed), 
                                        self.linears_club_x1x2[1](x2_embed)),
                         self.infonce_x1y(self.linears_infonce_x1y(x1_embed), y),
                         self.infonce_x2y(self.linears_infonce_x2y(x2_embed), y)
        ]

        cond_losses = [self.infonce_x1x2_cond(torch.cat([self.linears_infonce_x1x2_cond[0]
                                                         (x1_embed), y_ohe], dim=1), 
                                              self.linears_infonce_x1x2_cond[1](x2_embed)),
                       self.club_x1x2_cond(torch.cat([self.linears_club_x1x2_cond[0]
                                                      (x1_embed), y_ohe], dim=1), 
                                           self.linears_club_x1x2_cond[1](x2_embed)),
        ]                  
           

        return sum(uncond_losses) + sum(cond_losses)

    def ohe(self, y):
        N = y.shape[0]
        y_ohe = torch.zeros((N, self.y_ohe_dim)).to(y.device)
        y_ohe[torch.arange(N).long(), y.T[0].long()] = 1
        return y_ohe

    def get_embedding(self, x):
        return self.backbone(x)
    
    


class ConceptCLSUP_full(nn.Module):
    def __init__(self, x_dim, c_embed_dim, y_ohe_dim, hidden_dim, embed_dim, layers=2, activation='relu', lr=1e-4):
        super(ConceptCLSUP_full, self).__init__()
        self.critic_hidden_dim = 512
        self.critic_layers = 1
        self.critic_activation = 'relu'
        self.lr = lr
        self.y_ohe_dim = y_ohe_dim

        # encoders
        self.backbone = mlp(x_dim, hidden_dim, embed_dim, layers, activation)
        self.linears_infonce = mlp(embed_dim, embed_dim, embed_dim, 1, activation)
        self.linears_infonce_x1x2 = nn.ModuleList([mlp(embed_dim, c_embed_dim, embed_dim, 1, 
                                                       activation) for i in range(2)])
        self.linears_club_x1x2_cond = nn.ModuleList([mlp(embed_dim, embed_dim, embed_dim, 1, 
                                                         activation) for i in range(2)])

        self.linears_infonce_x1y = mlp(embed_dim, embed_dim, embed_dim, 1, activation)
        self.linears_infonce_x2y = mlp(c_embed_dim, embed_dim, embed_dim, 1, activation) 
        self.linears_infonce_x1x2_cond = nn.ModuleList([mlp(embed_dim, embed_dim, embed_dim, 
                                                            1, activation) for i in 
                                                        range(2)])
        self.linears_club_x1x2 = nn.ModuleList([mlp(embed_dim, embed_dim, embed_dim, 1, 
                                                    activation) for i in range(2)])


        # critics
        self.critic = InfoNCECritic(
            embed_dim+y_ohe_dim, c_embed_dim, self.critic_hidden_dim, self.critic_layers,
            self.critic_activation
        )
        self.infonce_x1x2 = InfoNCECritic(embed_dim, c_embed_dim, self.critic_hidden_dim,
                                          self.critic_layers, activation)
        self.club_x1x2_cond = CLUBInfoNCECritic(embed_dim + y_ohe_dim, c_embed_dim,
                                                self.critic_hidden_dim, self.critic_layers,
                                                activation)

        self.infonce_x1y = InfoNCECritic(embed_dim, 1, self.critic_hidden_dim, 
                                         self.critic_layers, activation) 
        self.infonce_x2y = InfoNCECritic(c_embed_dim, 1, self.critic_hidden_dim, 
                                         self.critic_layers, activation) 
        self.infonce_x1x2_cond = InfoNCECritic(embed_dim + y_ohe_dim, c_embed_dim, 
                                               self.critic_hidden_dim, self.critic_layers, 
                                               activation)
        self.club_x1x2 = CLUBInfoNCECritic(embed_dim, c_embed_dim, self.critic_hidden_dim, 
                                           self.critic_layers, activation)

        self.linears_list = [self.linears_infonce_x1x2, self.linears_club_x1x2_cond,
                             self.linears_infonce_x1y, self.linears_infonce_x2y, 
                             self.linears_infonce_x1x2_cond, self.linears_club_x1x2 
        ] 
        self.critics_list = [self.infonce_x1x2, self.club_x1x2_cond,
                             self.infonce_x1y, self.infonce_x2y, 
                             self.infonce_x1x2_cond, self.club_x1x2 
        ] 

    def forward(self, x, z_c, y):
        
        # Get embeddings
        x1_embed = self.linears_infonce(self.backbone(x)) #self.backbones[0](x1)
        x2_embed = z_c  #self.backbones[1](x2)

        y_ohe = self.ohe(y).cuda()

        #compute losses
        uncond_losses = [self.infonce_x1x2(self.linears_infonce_x1x2[0](x1_embed), 
                                           self.linears_infonce_x1x2[1](x2_embed)),
                         self.club_x1x2(self.linears_club_x1x2[0](x1_embed), 
                                        self.linears_club_x1x2[1](x2_embed)),
                         self.infonce_x1y(self.linears_infonce_x1y(x1_embed), y),
                         self.infonce_x2y(self.linears_infonce_x2y(x2_embed), y)
        ]

        cond_losses = [self.infonce_x1x2_cond(torch.cat([self.linears_infonce_x1x2_cond[0]
                                                         (x1_embed), y_ohe], dim=1), 
                                              self.linears_infonce_x1x2_cond[1](x2_embed)),
                       self.club_x1x2_cond(torch.cat([self.linears_club_x1x2_cond[0]
                                                      (x1_embed), y_ohe], dim=1), 
                                           self.linears_club_x1x2_cond[1](x2_embed)),
        ]                  
           

        return sum(uncond_losses) + sum(cond_losses)

    def ohe(self, y):
        N = y.shape[0]
        y_ohe = torch.zeros((N, self.y_ohe_dim)).to(y.device)
        y_ohe[torch.arange(N).long(), y.T[0].long()] = 1
        return y_ohe

    def get_embedding(self, x):
        return self.backbone(x)   
    
    
    
class ConceptEncoder_aux(nn.Module):
    def __init__(self, x_dim, c_embed_dim, yc_dim, hidden_dim, layers=2, activation='relu', **extra_kwargs):
        super(ConceptEncoder_aux, self).__init__()
       
        self.backbone = mlp(x_dim, hidden_dim, c_embed_dim, layers, activation)
        self.XtoC = mlp(c_embed_dim, hidden_dim, yc_dim, 1, activation)
        self.XtoY = mlp(c_embed_dim, hidden_dim, 1, 1, activation)
        self.XtoCtoY = mlp(yc_dim, hidden_dim, 1, 1, activation)

    def forward(self, x):
        z_c = self.backbone(x)
        concept = self.XtoC(z_c)
        label_intermediate = self.XtoY(z_c)
        label_true = self.XtoCtoY(concept) 
        return z_c ,concept, label_intermediate, label_true
    
    def get_embedding(self, x):
        return self.backbone(x)

    
class ConceptEncoder_IB(nn.Module):
    def __init__(self, x_dim, embed_dim, hidden_dim, critic_hidden_dim, critic_layers, layers=2, activation='relu', **extra_kwargs):
        super(ConceptEncoder_IB, self).__init__()
        
        self.critic_hidden_dim = critic_hidden_dim
        self.critic_layers = critic_layers 
        
        concept_dim = 1
     
        self.encoder = mlp(x_dim, hidden_dim, embed_dim, layers, activation)
        
        self.infonce_cz = InfoNCECritic(concept_dim, embed_dim, self.critic_hidden_dim,
                                          self.critic_layers, activation)
        self.club_xz = CLUBInfoNCECritic(x_dim, embed_dim, self.critic_hidden_dim, 
                                           self.critic_layers, activation)

    
    """
    def encoder(images):
    net = layers.relu(2*images-1, 1024)
    net = layers.relu(net, 1024)
    params = layers.linear(net, 512)
    mu, rho = params[:, :256], params[:, 256:]
    encoding = ds.NormalWithSoftplusScale(mu, rho - 5.0)
    return encoding

    def decoder(encoding_sample):
        net = layers.linear(encoding_sample, 10)
        return net

    prior = ds.Normal(0.0, 1.0)
    import math

    with tf.variable_scope('encoder'):
        encoding = encoder(images)

    with tf.variable_scope('decoder'):
        logits = decoder(encoding.sample())

    with tf.variable_scope('decoder', reuse=True):
        many_logits = decoder(encoding.sample(12))

    class_loss = tf.losses.softmax_cross_entropy(
        logits=logits, onehot_labels=one_hot_labels) / math.log(2)

    BETA = 1e-3

    info_loss = tf.reduce_sum(tf.reduce_mean(
        ds.kl_divergence(encoding, prior), 0)) / math.log(2)

    total_loss = class_loss + BETA * info_loss
    accuracy = tf.reduce_mean(tf.cast(tf.equal(
        tf.argmax(logits, 1), labels), tf.float32))
    avg_accuracy = tf.reduce_mean(tf.cast(tf.equal(
        tf.argmax(tf.reduce_mean(tf.nn.softmax(many_logits), 0), 1), labels), tf.float32))
    IZY_bound = math.log(10, 2) - class_loss
    IZX_bound = info_loss 

    """

    def forward(self, x, c):

        # I(c,z) - beta * I(x,z)
        z = self.encoder(x)
        beta = 1e-3
        InfoNCE_score = self.infonce_cz(c,z)
        CLUB_score = self.club_xz(x,z)
        
        return InfoNCE_score + beta * CLUB_score
    
    def get_embedding(self, x):
        return self.encoder(x)

    

def train_concept_encoder_IB(model, train_loader, val_loader, num_epochs, device, lr, 
                                weight_decay, log_interval,
                          save_interval, save_path):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_err = torch.tensor(1e7)
    model.train()
    tepoch = tqdm(range(num_epochs))
    for epoch in tepoch:
        tepoch.set_description(f"Epoch {epoch}")
        for batch_idx, (data, concept, target) in enumerate(train_loader):
            data, concept, target = data.to(device), concept.to(device), target.to(device)
            optimizer.zero_grad()
            loss = model(data, concept)
            loss.backward()
            optimizer.step()
            
        tepoch.set_postfix(loss=loss.item())
        if epoch % save_interval == 0:
            val_err = 0
            model.eval()
            with torch.no_grad():
                for batch_idx, (data, concept, target) in enumerate(val_loader):
                    data, concept, target = data.to(device), concept.to(device), target.to(device)
                    loss = model(data, concept)
                val_err = val_err / len(val_loader)
            if val_err < best_val_err:
                best_val_err = val_err
            else:
                print('Val loss did not improve')
                torch.save(model.state_dict(),
                           os.path.join(save_path,
                                        'concept_encoder_IB_val_err_{}.pth'.format(val_err)))
                return model
    return model

class ConceptCLSUP_Pretrain_IB(nn.Module):
    def __init__(self, x_dim, hidden_dim, embed_dim, layers=2, activation='relu', lr=1e-4):
        super(ConceptCLSUP_Pretrain_IB, self).__init__()
        self.critic_hidden_dim = 512
        self.critic_layers = 1
        self.critic_activation = 'relu'
        self.lr = lr

        # encoders
        self.backbone = mlp(x_dim, hidden_dim, embed_dim, layers, activation)
        self.linears_infonce = mlp(embed_dim, embed_dim, embed_dim, 1, activation) 

        # critics
        concept_dim = 50
        # self.club_critic = CLUBInfoNCECritic(embed_dim+concept_dim, x_dim, self.critic_hidden_dim, self.critic_layers, self.critic_activation)
        self.club_critic = CLUBInfoNCECritic(embed_dim + x_dim, concept_dim, self.critic_hidden_dim, self.critic_layers, self.critic_activation)

    def forward(self, x, z_c):
        # compute embedding
        z = self.linears_infonce(self.backbone(x))
        # compute critic scores
        club_infonce_score = self.club_critic(torch.cat([z, x], dim=-1), z_c)
        return club_infonce_score

    def get_embedding(self, x):
        return self.backbone(x)
    
    def get_backbone(self):
        return self.backbone
    
    
def train_concept_informed_Pretrain_IB_model(concept_encoder, model, train_loader,val_loader, num_epochs, device, lr, log_interval,
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
            z_c = concept_encoder.get_embedding(data)
            loss = model(data, z_c) 
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
                    z_c = concept_encoder.get_embedding(data)
                    output = model(data, z_c) #z_c
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



def train_concept_encoder_aux(model, train_loader, val_loader, num_epochs, device, lr, 
                                weight_decay, log_interval,
                          save_interval, save_path):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    concept_loss = torch.nn.MSELoss()
    label_loss = torch.nn.BCELoss() #nn.CrossEntropyLoss()
    best_val_err = torch.tensor(1e7)
    model.train()
    tepoch = tqdm(range(num_epochs))
    for epoch in tepoch:
        tepoch.set_description(f"Epoch {epoch}")
        for batch_idx, (data, concept, target) in enumerate(train_loader):
            data, concept, target = data.to(device), concept.to(device), target.to(device)
            optimizer.zero_grad()
            
            _, out1, out2, out3 = model(data)
            loss_concept = concept_loss(out1, concept)
            loss_label_intermediate = label_loss(torch.sigmoid(out2), target.float())
            loss_label_true = label_loss(torch.sigmoid(out3), target.float())
            loss = loss_concept + loss_label_intermediate + loss_label_true
            # print(loss)
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
                    data, concept, target = data.to(device), concept.to(device), target.to(device)
                    _, out1, out2, out3 = model(data)
                    loss_concept = concept_loss(out1, concept)
                    loss_label_intermediate = label_loss(torch.sigmoid(out2), target.float())
                    loss_label_true = label_loss(torch.sigmoid(out3), target.float())
                    
                val_err = val_err / len(val_loader)
            # print('Val loss: {:.6f}'.format(val_err))
            if val_err < best_val_err:
                best_val_err = val_err
            else:
                print('Val loss did not improve')
                torch.save(model.state_dict(),
                           os.path.join(save_path,
                                        'concept_encoder_val_err_{}.pth'.format(val_err)))
                return model
    return model

def train_concept_informed_model_aux(concept_encoder, model, train_loader,val_loader, num_epochs, device, lr, log_interval,
                          save_interval, save_path):

    concept_encoder.eval()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_err = torch.tensor(1e7)
    tepoch = tqdm(range(num_epochs))
    for epoch in range(num_epochs):
        model.train()
        tepoch.set_description(f"Epoch {epoch}")
        for batch_idx, (data, concept, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            z_c, _,_,_ = concept_encoder(data)
            loss = model(data, z_c, target)
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
                    z_c, _,_,_ = concept_encoder(data)
                    output = model(data, z_c, target)
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
