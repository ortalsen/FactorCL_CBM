
import torch
import numpy as np
import math
from torch.utils.data import Dataset
from critic_objectives import mlp

#############################
#  Synthetic Dataset Class  #
#############################

class MultimodalDataset(Dataset):
  def __init__(self, total_data, total_labels):
    self.data = torch.from_numpy(total_data).float()
    self.labels = torch.from_numpy(total_labels)
    self.num_modalities = self.data.shape[0]
  
  def __len__(self):
    return self.labels.shape[0]

  def __getitem__(self, idx):
    return tuple([self.data[i, idx] for i in range(self.num_modalities)] + [self.labels[idx]])

  def sample_batch(self, batch_size):
    sample_idxs = np.random.choice(self.__len__(), batch_size, replace=False)
    samples = self.__getitem__(sample_idxs)
    return samples


class MultiConcept(MultimodalDataset):
    def __init__(self, total_data, total_labels, total_concepts, concept_id):
        self.data = torch.from_numpy(total_data).float()
        self.labels = torch.from_numpy(total_labels)
        self.concepts = torch.from_numpy(total_concepts).float()
        self.concept_id = concept_id

    def __getitem__(self, idx):
        return tuple([self.data[idx,:]]+[self.concepts[self.concept_id,idx,:]]+[self.labels[idx]])
  


def get_intersections(num_modalities):
  modalities = [i for i in range(1, num_modalities+1)]
  all_intersections = [[]]
  for i in modalities:
    new = [s + [str(i)] for s in all_intersections]
    all_intersections += new
  res = list(map(lambda x: ''.join(x), sorted(all_intersections[1:])))
  return sorted(res, key=lambda x: (len(x), x))

def generate_data_concepts(num_data, num_concepts, feature_dim_info, label_dim_info, transform_dim = 100,
                           transforms_2concept=None, transforms_2hd=None, noise=0.0, pos_prob=0.5):
    # Standard deviation of generated Gaussian distributions
    SEP = 0.5



    total_data = []
    total_labels = []
    total_concepts = [[] for i in range(num_concepts)]

    total_raw_features = dict()
    for k in feature_dim_info:
        total_raw_features[k] = []


    total_dims = 0
    for d in feature_dim_info.values():
        total_dims += d
    
    # define transform matrices if not provided
    concepts_dims = [0]*num_concepts
    for i in range(1, num_concepts+1):
      for k, d in feature_dim_info.items():
        if str(i) in k:
          concepts_dims[i-1] += d


    if transforms_2hd is None:
        transforms_2hd = mlp(total_dims, 512, transform_dim, layers=2, activation='tanh') # int(transform_dim/2.0 =512
        #transforms_2hd = (np.random.uniform(0.0,1.0,(total_dims, transform_dim)))

    if transforms_2concept is None:
        transforms_2concept = []
        for i in range(num_concepts):
            transforms_2concept.append(np.random.uniform(0.0,1.0,(concepts_dims[i], 1)))
            # transforms_2concept.append(mlp(concepts_dims[i], 4, 1, layers=2, activation='tanh').to('cuda')) #concepts_dims[i]+512

    # generate data
    for data_idx in range(num_data):

        # get Gaussian data vector for each information component (unique, shared)
        raw_features = dict()
        for k, d in feature_dim_info.items():
          raw_features[k] = np.random.multivariate_normal(np.zeros((d,)), np.eye(d)*0.5, (1,))[0]
    
        # w3 = np.random.multivariate_normal(np.zeros((512,)), np.eye(512)*0.5, (1,))[0]
        # w4 = np.random.multivariate_normal(np.zeros((512,)), np.eye(512)*0.5, (1,))[0]
        irrelevant_info_generator = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros((512,)), torch.eye(512)*0.5)
        # irrelevant_info_generator = torch.distributions.dirichlet.Dirichlet(torch.ones(512)*0.5)
        w3 = irrelevant_info_generator.sample().to('cuda')
        w4 = irrelevant_info_generator.sample().to('cuda')
        
        # concatenating raw features based on concepts intersection 
        modality_concept_means = []
        for i in range(1, num_concepts+1):
          modality_concept_means.append([])
          for k, v in raw_features.items():
            if str(i) in k:
              modality_concept_means[-1].append(v)

        # raw_data = [torch.cat(modality_concept_means[i]) for i in range(num_concepts)]
        raw_data = [np.concatenate(modality_concept_means[i]) for i in range(num_concepts)]
        
        # Transform into concepts
        
        concepts_labels = [raw_data[i] @ transforms_2concept[i] for i in range(num_concepts)]
        # concepts_labels = [transforms_2concept[i](torch.tensor((raw_data[i]).astype(np.float32))).detach().numpy() for i in range(num_concepts)]
        
        # concept_vector = [torch.tensor(raw_data[i], dtype=torch.float).to('cuda') for i in range(num_concepts) ]
        # concepts_labels = [transforms_2concept[i](concept_vector[i]).detach().cpu().numpy() for i in range(num_concepts)]
        
        # concept_vector = [torch.cat((torch.tensor(raw_data[i], dtype=torch.float).to('cuda'), w3, w4)) for i in range(num_concepts) ]
        # concepts_labels = [transforms_2concept[i](concept_vector[i]).detach().cpu().numpy() for i in range(num_concepts)]

        # Transform into high-dimensional space
        

        raw_total = np.concatenate([v for v in raw_features.values()])
        # raw_total = torch.tensor(raw_total).float().to(transforms_2hd.device)
        raw_total = torch.tensor(raw_total).float().to('cuda')
        transforms_2hd.to('cuda')
        total_x = transforms_2hd(raw_total).detach().cpu().numpy()
        # total_x = torch.cat((total_x, w3, w4)).detach().cpu().numpy()
        #total_x = raw_total @ transforms_2hd
        total_x += np.random.normal(0, noise, size=total_x.shape)
    

        # update total data
        total_data.append(total_x)

        # update total concepts
        for i in range(num_concepts):
          total_concepts[i].append(concepts_labels[i])

        # update total raw data
        for k, f in raw_features.items():
          total_raw_features[k].append(f)

        # get label vector, d defines what portion of w is relevant to task y
        
        label_components = []
        for k,d in label_dim_info.items():
          label_components.append(raw_features[k][:d])

        # print(f'label compnents {label_components}')
        # irrelevant_info_generator_2 = torch.distributions.dirichlet.Dirichlet(torch.ones(2)*0.5)
        # w5 = irrelevant_info_generator_2.sample().to('cuda')
        # w6 = irrelevant_info_generator_2.sample().to('cuda')
        # label_components.append(w5.tolist())
        # label_components.append(w6.tolist())
        
        label_vector = np.concatenate(label_components) #+ [np.random.randint(0, 2, 1)]) 
        label_prob = 1 / (1 + math.exp(-10*np.mean(label_vector)))
        total_labels.append([int(label_prob >= pos_prob)])
        
        if data_idx%100 == 0:
            print(f'Current generated data : {data_idx}')


    total_data = np.array(total_data)
    total_labels = np.array(total_labels)
    total_concepts = np.array(total_concepts)
    # total_concepts = total_concepts.detach().numpy()
    for k, v in total_raw_features.items():
        total_raw_features[k] = np.array(v)
    
    return total_data, total_labels, total_concepts, total_raw_features



def generate_data(num_data, num_modalities, feature_dim_info, label_dim_info, transforms=None):
  # Standard deviation of generated Gaussian distributions
  SEP = 0.5
  default_transform_dim = 100

  total_data = [[] for i in range(num_modalities)]
  total_labels = []
  total_raw_features = dict()
  for k in feature_dim_info:
    total_raw_features[k] = []


  # define transform matrices if not provided
  modality_dims = [0]*num_modalities
  for i in range(1, num_modalities+1):
      for k, d in feature_dim_info.items():
        if str(i) in k:
          modality_dims[i-1] += d

  if transforms is None:
      transforms = []
      for i in range(num_modalities):
        transforms.append(np.random.uniform(0.0,1.0,(modality_dims[i], default_transform_dim)))


  # generate data
  for data_idx in range(num_data):

    # get Gaussian data vector for each modality
    raw_features = dict()
    for k, d in feature_dim_info.items():
      raw_features[k] = np.random.multivariate_normal(np.zeros((d,)), np.eye(d)*0.5, (1,))[0]

   
    modality_concept_means = []
    for i in range(1, num_modalities+1):
      modality_concept_means.append([])
      for k, v in raw_features.items():
        if str(i) in k:
          modality_concept_means[-1].append(v)

    raw_data = [np.concatenate(modality_concept_means[i]) for i in range(num_modalities)]
    

    # Transform into high-dimensional space
    modality_data = [raw_data[i] @ transforms[i] for i in range(num_modalities)]


    # update total data
    for i in range(num_modalities):
      total_data[i].append(modality_data[i])

    # update total raw data
    for k, f in raw_features.items():
      total_raw_features[k].append(f)

    # get label vector
    label_components = []
    for k,d in label_dim_info.items():
      label_components.append(raw_features[k][:d])
   
    label_vector = np.concatenate(label_components) #+ [np.random.randint(0, 2, 1)]) 
    label_prob = 1 / (1 + math.exp(-np.mean(label_vector)))
    total_labels.append([int(label_prob >= 0.5)])

      
  total_data = np.array(total_data)
  total_labels = np.array(total_labels)
  for k, v in total_raw_features.items():
    total_raw_features[k] = np.array(v)

  return total_data, total_labels, total_raw_features


def get_labels(label_dim_info, total_raw_features):
    label_components = []
    for k,d in label_dim_info.items():
      label_components.append(total_raw_features[k][:,:d])
   
    label_vector = np.concatenate(label_components, axis=1) #+ [np.random.randint(0, 2, 1)]) 
    label_prob = 1 / (1 + np.exp(-np.mean(label_vector, axis=1)))
    total_labels = (label_prob >= 0.5).astype('float')
    total_labels = np.expand_dims(total_labels, axis=1)

    return total_labels

def get_nonlinear_labels(label_dim_info, total_raw_features):
    label_components = []
    total_label_dim = 0
    for k,d in label_dim_info.items():
      label_components.append(total_raw_features[k][:,:d])
      total_label_dim += d

    w1 = np.ones((total_label_dim,total_label_dim))
    w2 = np.ones((total_label_dim,total_label_dim))  
   
    label_vector = np.concatenate(label_components, axis=1) #+ [np.random.randint(0, 2, 1)]) 
    label_vector = label_vector @ w1 @ w2

    label_prob = 1 / (1 + np.exp(-np.mean(label_vector, axis=1)))
    total_labels = (label_prob >= 0.5).astype('float')
    total_labels = np.expand_dims(total_labels, axis=1)

    return total_labels

def get_planar_flow_labels(label_dim_info, total_raw_features):
    label_components = []
    total_label_dim = 0
    for k,d in label_dim_info.items():
      label_components.append(total_raw_features[k][:,:d])
      total_label_dim += d

    w = np.random.normal(2, 1, size=(total_label_dim,total_label_dim))
    b = np.random.normal(2, 1, size=(total_label_dim,))
    u = np.random.normal(2, 1, size=(total_label_dim,total_label_dim))

    head = np.random.normal(2, 1, size=(total_label_dim,20))
   
    z = np.concatenate(label_components, axis=1) 
    z = z + np.tanh(z @ w + b) @ u 
    z = z @ head

    label_prob = torch.softmax(torch.from_numpy(z), dim=1)
    total_labels = torch.argmax(label_prob, dim=1).unsqueeze(1)

    return total_labels.numpy()


##########################
#  Simple Augmentations  #
##########################

def swap(x):
  mid = x.shape[0] // 2
  return torch.cat([x[mid:], x[:mid]])

def noise(x):
  noise = torch.randn(x.shape) * 0.01
  return x + noise.cuda()

def random_drop(x):
  drop_num = x.shape[0] // 10
  drop_idxs = np.random.choice(x.shape[0], drop_num, replace=False)
  x_aug = torch.clone(x)
  x_aug[drop_idxs] = 0.0
  return x_aug

def identity(x):
  return x

# return a pair of augmented data
def augment(x_batch):
  v1 = x_batch
  v2 = torch.clone(v1)
  transforms = [swap, noise, random_drop, identity]

  for i in range(x_batch.shape[0]):
    t_idxs = np.random.choice(4, 2, replace=False)
    t1 = transforms[t_idxs[0]]
    t2 = transforms[t_idxs[1]]
    v1[i] = t1(v1[i])
    v2[i] = t2(v2[i])
  
  return v1, v2

# return one augmented instance instead of pair
def augment_single(x_batch):
  v1 = x_batch
  v2 = torch.clone(v1)
  transforms = [swap, noise, random_drop, identity]

  for i in range(x_batch.shape[0]):
    t_idxs = np.random.choice(4, 1, replace=False)
    t2 = transforms[t_idxs[0]]
    v2[i] = t2(v2[i])
  
  return v2
