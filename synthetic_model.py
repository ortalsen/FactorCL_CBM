from critic_objectives import*
from Synthetic.dataset import augment, augment_single, get_intersections, generate_data_concepts, MultiConcept
import torch.optim as optim
from torch.utils.data import DataLoader

##############
#   Models   #
##############

class SupConModel(nn.Module):
    def __init__(self, x1_dim, x2_dim, hidden_dim, embed_dim, layers=2, activation='relu', lr=1e-4, ratio=1, use_label=True):
        super(SupConModel, self).__init__()
        self.lr = lr
        self.ratio = ratio
        self.use_label = use_label

        #encoders
        self.encoder_x1 = mlp(x1_dim, hidden_dim, embed_dim, layers, activation)
        self.encoder_x2 = mlp(x2_dim, hidden_dim, embed_dim, layers, activation)

        #linear projection heads
        self.projection_x1 = mlp(embed_dim, embed_dim, embed_dim, 1, activation)
        self.projection_x2 = mlp(embed_dim, embed_dim, embed_dim, 1, activation)

        #critics
        self.critic = SupConLoss()

    def forward(self, x1, x2, y):
        x1_embed = self.projection_x1(self.encoder_x1(x1))
        x2_embed = self.projection_x2(self.encoder_x2(x2))

        concat_embed = torch.cat([x1_embed.unsqueeze(dim=1), x2_embed.unsqueeze(dim=1)], dim=1)

        if self.use_label:
            loss = self.critic(concat_embed, y.T[0])
        else:
            loss = self.critic(concat_embed)

        return loss

    def get_embedding(self, x):
        x1, x2 = x[0], x[1]
        x1_embed = self.encoder_x1(x1)
        x2_embed = self.encoder_x2(x2)

        return torch.concat([x1_embed, x2_embed], dim=1)

    def get_separate_embeddings(self, x):
        x1, x2 = x[0], x[1]
        x1_embed = self.encoder_x1(x1)
        x2_embed = self.encoder_x2(x2)

        return x1_embed, x2_embed




class FactorCLSUP(nn.Module):
    def __init__(self, x1_dim, x2_dim, y_ohe_dim, hidden_dim, embed_dim, layers=2, activation='relu', lr=1e-4, ratio=1):
        super(FactorCLSUP, self).__init__()
        self.critic_hidden_dim = 512
        self.critic_layers = 1
        self.critic_activation = 'relu'
        self.lr = lr
        self.ratio = ratio
        self.y_ohe_dim = y_ohe_dim
        
        # encoders
        self.backbones = nn.ModuleList([mlp(x1_dim, hidden_dim, embed_dim, layers, activation),
                                        mlp(x2_dim, hidden_dim, embed_dim, layers, activation)])
        
        # linear projection heads
        self.linears_infonce_x1x2 = nn.ModuleList([mlp(embed_dim, embed_dim, embed_dim, 1, activation) for i in range(2)])
        self.linears_club_x1x2_cond = nn.ModuleList([mlp(embed_dim, embed_dim, embed_dim, 1, activation) for i in range(2)])

        self.linears_infonce_x1y = mlp(embed_dim, embed_dim, embed_dim, 1, activation)
        self.linears_infonce_x2y = mlp(embed_dim, embed_dim, embed_dim, 1, activation) 
        self.linears_infonce_x1x2_cond = nn.ModuleList([mlp(embed_dim, embed_dim, embed_dim, 1, activation) for i in range(2)])
        self.linears_club_x1x2 = nn.ModuleList([mlp(embed_dim, embed_dim, embed_dim, 1, activation) for i in range(2)])

        # critics
        self.infonce_x1x2 = InfoNCECritic(embed_dim, embed_dim, self.critic_hidden_dim, self.critic_layers, activation)
        self.club_x1x2_cond = CLUBInfoNCECritic(embed_dim + y_ohe_dim, embed_dim, self.critic_hidden_dim, self.critic_layers, activation)

        self.infonce_x1y = InfoNCECritic(embed_dim, 1, self.critic_hidden_dim, self.critic_layers, activation) 
        self.infonce_x2y = InfoNCECritic(embed_dim, 1, self.critic_hidden_dim, self.critic_layers, activation) 
        self.infonce_x1x2_cond = InfoNCECritic(embed_dim + y_ohe_dim, embed_dim, self.critic_hidden_dim, self.critic_layers, activation)
        self.club_x1x2 = CLUBInfoNCECritic(embed_dim, embed_dim, self.critic_hidden_dim, self.critic_layers, activation)

        self.linears_list = [self.linears_infonce_x1x2, self.linears_club_x1x2_cond,
                             self.linears_infonce_x1y, self.linears_infonce_x2y, 
                             self.linears_infonce_x1x2_cond, self.linears_club_x1x2 
        ] 
        self.critics_list = [self.infonce_x1x2, self.club_x1x2_cond,
                             self.infonce_x1y, self.infonce_x2y, 
                             self.infonce_x1x2_cond, self.club_x1x2 
        ] 


    def ohe(self, y):
        N = y.shape[0]
        y_ohe = torch.zeros((N, self.y_ohe_dim))
        y_ohe[torch.arange(N).long(), y.T[0].long()] = 1
        return y_ohe

                         
    def forward(self, x1, x2, y): 
        # Get embeddings
        x1_embed = self.backbones[0](x1)
        x2_embed = self.backbones[1](x2)

        y_ohe = self.ohe(y).cuda()

        #compute losses
        uncond_losses = [self.infonce_x1x2(self.linears_infonce_x1x2[0](x1_embed), self.linears_infonce_x1x2[1](x2_embed)),
                         self.club_x1x2(self.linears_club_x1x2[0](x1_embed), self.linears_club_x1x2[1](x2_embed)),
                         self.infonce_x1y(self.linears_infonce_x1y(x1_embed), y),
                         self.infonce_x2y(self.linears_infonce_x2y(x2_embed), y)
        ]

        cond_losses = [self.infonce_x1x2_cond(torch.cat([self.linears_infonce_x1x2_cond[0](x1_embed), y_ohe], dim=1), self.linears_infonce_x1x2_cond[1](x2_embed)),
                       self.club_x1x2_cond(torch.cat([self.linears_club_x1x2_cond[0](x1_embed), y_ohe], dim=1), self.linears_club_x1x2_cond[1](x2_embed)),
        ]                  
           

        return sum(uncond_losses) + sum(cond_losses)

    def learning_loss(self, x1, x2, y):

        x1_embed = self.backbones[0](x1)
        x2_embed = self.backbones[1](x2)

        y_ohe = self.ohe(y).cuda()

        learning_losses = [self.club_x1x2.learning_loss(self.linears_club_x1x2[0](x1_embed), self.linears_club_x1x2[1](x2_embed)),
                           self.club_x1x2_cond.learning_loss(torch.cat([self.linears_club_x1x2_cond[0](x1_embed), y_ohe], dim=1), 
                                                             self.linears_club_x1x2_cond[1](x2_embed))
        ]
        return sum(learning_losses)
 
    def get_embedding(self, x):
        x1, x2 = x[0], x[1]
        x1_embed = self.backbones[0](x1)
        x2_embed = self.backbones[1](x2)

        return torch.concat([x1_embed, x2_embed], dim=1)

    def get_separate_embeddings(self, x):
        x1, x2 = x[0], x[1]
        x1_embed = self.backbones[0](x1)
        x2_embed = self.backbones[1](x2)

        return x1_embed, x2_embed

    def get_optims(self):
        non_CLUB_params = [self.backbones.parameters(),
                           self.infonce_x1x2.parameters(), 
                           self.infonce_x1y.parameters(), 
                           self.infonce_x2y.parameters(), 
                           self.infonce_x1x2_cond.parameters(), 
                           self.linears_infonce_x1x2.parameters(), 
                           self.linears_infonce_x1y.parameters(), 
                           self.linears_infonce_x2y.parameters(), 
                           self.linears_infonce_x1x2_cond.parameters(), 
                           self.linears_club_x1x2_cond.parameters(), 
                           self.linears_club_x1x2.parameters()]
                  

        CLUB_params = [self.club_x1x2_cond.parameters(), 
                       self.club_x1x2.parameters()]

        non_CLUB_optims = [optim.Adam(param, lr=self.lr) for param in non_CLUB_params]
        CLUB_optims = [optim.Adam(param, lr=self.lr) for param in CLUB_params]

        return non_CLUB_optims, CLUB_optims



class FactorCLSSL(nn.Module):
    def __init__(self, x1_dim, x2_dim, hidden_dim, embed_dim, layers=2, activation='relu', lr=1e-4, ratio=1):
        super(FactorCLSSL, self).__init__()
        self.critic_hidden_dim = 512
        self.critic_layers = 1
        self.critic_activation = 'relu'
        self.lr = lr
        self.ratio = ratio

        
        # encoders
        self.backbones = nn.ModuleList([mlp(x1_dim, hidden_dim, embed_dim, layers, activation),
                                        mlp(x2_dim, hidden_dim, embed_dim, layers, activation)])

        # linear projection heads
        self.linears_infonce_x1x2 = nn.ModuleList([mlp(embed_dim, embed_dim, embed_dim, 1, activation) for i in range(2)])
        self.linears_club_x1x2_cond = nn.ModuleList([mlp(embed_dim, embed_dim, embed_dim, 1, activation) for i in range(2)]) 

        self.linears_infonce_x1y = mlp(embed_dim, embed_dim, embed_dim, 1, activation)
        self.linears_infonce_x2y = mlp(embed_dim, embed_dim, embed_dim, 1, activation) 
        self.linears_infonce_x1x2_cond = nn.ModuleList([mlp(embed_dim, embed_dim, embed_dim, 1, activation) for i in range(2)]) 
        self.linears_club_x1x2 = nn.ModuleList([mlp(embed_dim, embed_dim, embed_dim, 1, activation) for i in range(2)])

        # critics
        self.infonce_x1x2 = InfoNCECritic(embed_dim, embed_dim, self.critic_hidden_dim, self.critic_layers, activation)
        self.club_x1x2_cond = CLUBInfoNCECritic(embed_dim + embed_dim, embed_dim + embed_dim, self.critic_hidden_dim, self.critic_layers, activation) 

        self.infonce_x1y = InfoNCECritic(embed_dim, embed_dim, self.critic_hidden_dim, self.critic_layers, activation) 
        self.infonce_x2y = InfoNCECritic(embed_dim, embed_dim, self.critic_hidden_dim, self.critic_layers, activation) 
        self.infonce_x1x2_cond = InfoNCECritic(embed_dim + embed_dim, embed_dim + embed_dim, self.critic_hidden_dim, self.critic_layers, activation) 
        self.club_x1x2 = CLUBInfoNCECritic(embed_dim, embed_dim, self.critic_hidden_dim, self.critic_layers, activation)

        self.linears_list = [self.linears_infonce_x1x2, self.linears_club_x1x2_cond,
                             self.linears_infonce_x1y, self.linears_infonce_x2y, 
                             self.linears_infonce_x1x2_cond, self.linears_club_x1x2 
        ] 
        self.critics_list = [self.infonce_x1x2, self.club_x1x2_cond,
                             self.infonce_x1y, self.infonce_x2y, 
                             self.infonce_x1x2_cond, self.club_x1x2 
        ] 

                         
    def forward(self, x1, x2, y=None): 
        x1_aug = augment_single(x1)
        x2_aug = augment_single(x2)

        x1_embed = self.backbones[0](x1)
        x2_embed = self.backbones[1](x2)

        x1_aug_embed = self.backbones[0](x1_aug)
        x2_aug_embed = self.backbones[1](x2_aug)


        uncond_losses = [self.infonce_x1x2(self.linears_infonce_x1x2[0](x1_embed), self.linears_infonce_x1x2[1](x2_embed)),
                         self.club_x1x2(self.linears_club_x1x2[0](x1_embed), self.linears_club_x1x2[1](x2_embed)),
                         self.infonce_x1y(self.linears_infonce_x1y(x1_embed), self.linears_infonce_x1y(x1_aug_embed)),
                         self.infonce_x2y(self.linears_infonce_x2y(x2_embed), self.linears_infonce_x2y(x2_aug_embed))
        ]

        cond_losses = [self.infonce_x1x2_cond(torch.cat([self.linears_infonce_x1x2_cond[0](x1_embed), 
                                                         self.linears_infonce_x1x2_cond[0](x1_aug_embed)], dim=1), 
                                              torch.cat([self.linears_infonce_x1x2_cond[1](x2_embed), 
                                                         self.linears_infonce_x1x2_cond[1](x2_aug_embed)], dim=1)),
                       self.club_x1x2_cond(torch.cat([self.linears_club_x1x2_cond[0](x1_embed), 
                                                      self.linears_club_x1x2_cond[0](x1_embed)], dim=1), 
                                           torch.cat([self.linears_club_x1x2_cond[1](x2_embed), 
                                                      self.linears_club_x1x2_cond[1](x2_embed)], dim=1))
        ]                  
           

        return sum(uncond_losses) + sum(cond_losses)

    def learning_loss(self, x1, x2, y=None):

        x1_embed = self.backbones[0](x1)
        x2_embed = self.backbones[1](x2)

        learning_losses = [self.club_x1x2.learning_loss(self.linears_club_x1x2[0](x1_embed), self.linears_club_x1x2[1](x2_embed)),
                           self.club_x1x2_cond.learning_loss(torch.cat([self.linears_club_x1x2_cond[0](x1_embed), 
                                                                        self.linears_club_x1x2_cond[0](x1_embed)], dim=1), 
                                                             torch.cat([self.linears_club_x1x2_cond[1](x2_embed), 
                                                                        self.linears_club_x1x2_cond[1](x2_embed)], dim=1))
        ]
        return sum(learning_losses)
 
    def get_embedding(self, x):
        x1, x2 = x[0], x[1]
        x1_embed = self.backbones[0](x1)
        x2_embed = self.backbones[1](x2)

        return torch.concat([x1_embed, x2_embed], dim=1)

    def get_separate_embeddings(self, x):
        x1, x2 = x[0], x[1]
        x1_embed = self.backbones[0](x1)
        x2_embed = self.backbones[1](x2)

        return x1_embed, x2_embed

    def get_optims(self):
        non_CLUB_params = [self.backbones.parameters(),
                           self.infonce_x1x2.parameters(), 
                           self.infonce_x1y.parameters(), 
                           self.infonce_x2y.parameters(), 
                           self.infonce_x1x2_cond.parameters(), 
                           self.linears_infonce_x1x2.parameters(), 
                           self.linears_infonce_x1y.parameters(), 
                           self.linears_infonce_x2y.parameters(), 
                           self.linears_infonce_x1x2_cond.parameters(), 
                           self.linears_club_x1x2_cond.parameters(), 
                           self.linears_club_x1x2.parameters()]
                  

        CLUB_params = [self.club_x1x2_cond.parameters(), 
                       self.club_x1x2.parameters()]

        non_CLUB_optims = [optim.Adam(param, lr=self.lr) for param in non_CLUB_params]
        CLUB_optims = [optim.Adam(param, lr=self.lr) for param in CLUB_params]

        return non_CLUB_optims, CLUB_optims


########################
#   Training Scripts   #
########################

def train_supcon(model, train_loader, optimizer, num_epoch=50):
    for _iter in range(num_epoch):
        for i_batch, data_batch in enumerate(train_loader):
                      
            x1_batch = data_batch[0].float().cuda()
            x2_batch = data_batch[1].float().cuda()
            y_batch = data_batch[2].float().cuda()
               
            loss = model(x1_batch, x2_batch, y_batch)
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i_batch%100 == 0:
                print('iter: ', _iter, ' i_batch: ', i_batch, ' loss: ', loss.item())
    return



def train_sup_model(model, train_loader, dataset, num_epoch=50, num_club_iter=5, batch_size=128):
    non_CLUB_optims, CLUB_optims = model.get_optims()
    losses = []

    for _iter in range(num_epoch):
        for i_batch, data_batch in enumerate(train_loader):
                      
            x1_batch = data_batch[0].float().cuda()
            x2_batch = data_batch[1].float().cuda()
            y_batch = data_batch[2].float().cuda()
             
            loss = model(x1_batch, x2_batch, y_batch)
            losses.append(loss.detach().cpu().numpy())
                
            for optimizer in non_CLUB_optims:
                optimizer.zero_grad()

            loss.backward()

            for optimizer in non_CLUB_optims:
                optimizer.step()

            for _ in range(num_club_iter): 
                data_batch = dataset.sample_batch(batch_size)
                
                x1_batch = data_batch[0].float().cuda()
                x2_batch = data_batch[1].float().cuda()
                y_batch = data_batch[2].float().cuda()

                learning_loss = model.learning_loss(x1_batch, x2_batch, y_batch)  
                    
                for optimizer in CLUB_optims:
                    optimizer.zero_grad()

                learning_loss.backward()

                for optimizer in CLUB_optims:
                    optimizer.step()
            
            if i_batch%100 == 0:
                print('iter: ', _iter, ' i_batch: ', i_batch, ' loss: ', loss.item())

    return losses


def train_concepts_sup_model(model, train_loader, dataset, num_epoch=50, num_club_iter=5, batch_size=128):
    non_CLUB_optims, CLUB_optims = model.get_optims()
    losses = []

    for _iter in range(num_epoch):
        for i_batch, data_batch in enumerate(train_loader):

            x_batch = data_batch[0].float().cuda()
            c1_batch = data_batch[1].float().cuda()
            c2_batch = data_batch[2].float().cuda()
            y_batch = data_batch[3].float().cuda()

            loss = model(x1_batch, x2_batch, y_batch)
            losses.append(loss.detach().cpu().numpy())

            for optimizer in non_CLUB_optims:
                optimizer.zero_grad()

            loss.backward()

            for optimizer in non_CLUB_optims:
                optimizer.step()

            for _ in range(num_club_iter):
                data_batch = dataset.sample_batch(batch_size)

                x1_batch = data_batch[0].float().cuda()
                x2_batch = data_batch[1].float().cuda()
                y_batch = data_batch[2].float().cuda()

                learning_loss = model.learning_loss(x1_batch, x2_batch, y_batch)

                for optimizer in CLUB_optims:
                    optimizer.zero_grad()

                learning_loss.backward()

                for optimizer in CLUB_optims:
                    optimizer.step()

            if i_batch % 100 == 0:
                print('iter: ', _iter, ' i_batch: ', i_batch, ' loss: ', loss.item())

    return losses


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
                                                                                          transforms_2hd=None)
    dataset = MultiConcept(total_data, total_labels, total_concepts)
    batch_size = 256
    train_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                                [int(0.8 * num_data), num_data - int(0.8 * num_data)])

    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True,
                              batch_size=batch_size)
    test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False,
                             batch_size=batch_size)


