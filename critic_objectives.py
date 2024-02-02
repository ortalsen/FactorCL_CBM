import torch
import torch.nn as nn
import numpy as np
import scipy.spatial as ss
import scipy.stats as sst
from scipy.special import digamma,gamma
from math import log,pi,exp
import math


####################
#   Critic Model   #
####################


# Part of the code is adapted from here: https://github.com/yaohungt/Pointwise_Dependency_Neural_Estimation

def probabilistic_classifier_obj(f):
    criterion = nn.BCEWithLogitsLoss()

    batch_size = f.shape[0]
    labels = [0.]*(batch_size*batch_size)
    labels[::(batch_size+1)] = [1.]*batch_size
    labels = torch.tensor(labels).type_as(f)
    labels = labels.view(-1,1)

    logits = f.contiguous().view(-1,1)

    Loss = -1.*criterion(logits, labels)

    return Loss

def probabilistic_classifier_eval(f):
    batch_size = f.shape[0]
    joint_feat = f.contiguous().view(-1)[::(batch_size+1)]
    joint_logits = torch.clamp(torch.sigmoid(joint_feat), min=1e-6, max=1-1e-6)

    MI = torch.mean(torch.log((batch_size-1)*joint_logits/(1.-joint_logits)))
    # we have batch_size*(batch_size-1) product of marginal samples
    # we have batch_size joint density samples

    return MI

def infonce_lower_bound_obj(scores):
    nll = scores.diag().mean() - scores.logsumexp(dim=1)
    # Alternative implementation:
    # nll = -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=tf.range(batch_size))
    mi = torch.tensor(scores.size(0)).float().log() + nll
    mi = mi.mean()
    return mi
    

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
    

class SeparableCritic(nn.Module):
    def __init__(self, x1_dim, x2_dim, hidden_dim, embed_dim, 
                 layers, activation):
        super(SeparableCritic, self).__init__()
        self._g = mlp(x1_dim, hidden_dim, embed_dim, layers, activation)
        self._h = mlp(x2_dim, hidden_dim, embed_dim, layers, activation)

    def transformed_x(self, x):
        return self._g(x)
    
    def transformed_y(self, y):
        return self._h(y) 
    
    def forward(self, x, y):
        scores = torch.matmul(self._h(y), self._g(x).t())
        return scores
    
    def pointwise_mi(self, x, y, estimator):
        scores = torch.matmul(self._h(y), self._g(x).t())

        if estimator == 'probabilistic_classifier':
            # the prob of being a pair
            # PMI = torch.sigmoid(scores.diag())
            # PMI
            batch_size = scores.shape[0]
            # N_pxpy / N_pxy = (batch_size - 1.) * batch_size / batch_size
            PMI = scores.diag() + np.log(batch_size - 1.)
        else:
            raise NotImplementedError("not supporting our PMI!")
        return PMI


class ConcatCritic(nn.Module):
    def __init__(self, A_dim, B_dim, hidden_dim, layers, activation, **extra_kwargs):
        super(ConcatCritic, self).__init__()
        # output is scalar score
        self._f = mlp(A_dim + B_dim, hidden_dim, 1, layers, activation)

    def forward(self, x, y):
        batch_size = x.shape[0]
        # Tile all possible combinations of x and y
        x_tiled = torch.stack([x] * batch_size, dim=0)
        y_tiled = torch.stack([y] * batch_size, dim=1)
        # xy is [batch_size * batch_size, x_dim + y_dim]
        xy_pairs = torch.reshape(torch.cat((x_tiled, y_tiled), dim=2), [
                                 batch_size * batch_size, -1])
        # Compute scores for each x_i, y_j pair.
        scores = self._f(xy_pairs)
        return torch.reshape(scores, [batch_size, batch_size]).t()



# Concat critic with the InfoNCE (NCE) objective
class InfoNCECritic(nn.Module):
    def __init__(self, A_dim, B_dim, hidden_dim, layers, activation, **extra_kwargs):
          super(InfoNCECritic, self).__init__()
          # output is scalar score
          self._f = mlp(A_dim + B_dim, hidden_dim, 1, layers, activation)

    def forward(self, x_samples, y_samples):
        sample_size = y_samples.shape[0]

        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))

        T0 = self._f(torch.cat([x_samples,y_samples], dim = -1))
        T1 = self._f(torch.cat([x_tile, y_tile], dim = -1))  

        lower_bound = T0.mean() - (T1.logsumexp(dim = 1).mean() - np.log(sample_size)) 
        return -lower_bound


# Concat critic with the CLUBInfoNCE (NCE-CLUB) objective
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




class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

    
#################################################### KSG Estimators ####################################################

class Kraskov_critic(nn.Module):
    def __init__(self, diagamma_=None, **extra_kwargs):
        super(Kraskov_critic, self).__init__()
        # self.diagamma_ = mlp(1, hidden_dim, 1, layers, activation)

    def forward(self, x_samples, y_samples, k=4):
        assert len(x_samples)==len(y_samples), "Lists should have same length"
        assert k <= len(x_samples)-1, "Set k smaller than num. samples - 1"

        N = torch.tensor(len(x_samples))
        dx = torch.tensor(len(x_samples[0]))     
        dy = torch.tensor(len(y_samples[0]))
        k = torch.tensor(k)
        data = torch.cat((x_samples,y_samples),dim=1)
 
        tree_x = ss.cKDTree(x_samples.cpu().detach())
        tree_y = ss.cKDTree(y_samples.cpu().detach())

        dist = torch.cdist(data,data,p=float('inf'))
        knn_dis = torch.topk(dist, k).values[:,-1]
        knn_dis = knn_dis.requires_grad_(True)
        ans_xy = -digamma(k) + digamma(N) + (dx+dy)*log(2) #2*log(N-1) - digamma(N) #+ vd(dx) + vd(dy) - vd(dx+dy)
        ans_x = digamma(N) + dx*log(2)
        ans_y = digamma(N) + dy*log(2)
        
        ans_xy = ans_xy.requires_grad_(True)
        ans_x = ans_x.requires_grad_(True)
        ans_y = ans_y.requires_grad_(True)
        

        for i in range(N):
            ans_xy = ans_xy + (dx+dy)*log(knn_dis[i])/N

            ans_x = ans_x -digamma(len(tree_x.query_ball_point(x_samples[i].detach().cpu().numpy(),knn_dis[i].detach().cpu().numpy()-1e15,
                                                               p=float('inf'))))/N+dx*log(knn_dis[i])/N
            ans_y = ans_y - digamma(len(tree_y.query_ball_point(y_samples[i].detach().cpu().numpy(),knn_dis[i].detach().cpu().numpy()-1e15,
                                                                p=float('inf'))))/N+dy*log(knn_dis[i])/N
        answer = ans_x+ans_y-ans_xy
        
        return answer
    
class KSG_critic(nn.Module):
    def __init__(self, diagamma_=None, **extra_kwargs):
        super(KSG_critic, self).__init__()
        # self.diagamma_ = mlp(x_dim, hidden_dim, embed_dim, layers, activation)
    
    def vd(self, d,q):
        # Compute the volume of unit l_q ball in d dimensional space
        if (q==float('inf')):
            return d*log(2)
        return d*log(2*gamma(1+1.0/q)) - log(gamma(1+d*1.0/q))

    def forward(self, x_samples, y_samples, k=5, q=float('inf')):
        assert len(x_samples)==len(y_samples), "Lists should have same length"
        assert k <= len(x_samples)-1, "Set k smaller than num. samples - 1"
        N = len(x_samples)
        dx = len(x_samples[0])       
        dy = len(y_samples[0])
        data = torch.cat((x_samples,y_samples),dim=1)

        # tree_xy = ss.cKDTree(data.detach().cpu().numpy())
        tree_x = ss.cKDTree(x_samples.detach().cpu().numpy())
        tree_y = ss.cKDTree(y_samples.detach().cpu().numpy())
        
        dist = torch.cdist(data,data,p=float('inf'))
        knn_dis = torch.topk(dist, k).values[:,-1]
        knn_dis = knn_dis.requires_grad_(True)
        
        # knn_dis = [tree_xy.query(point,k+1,p=q)[0][k] for point in data]
        ans_xy = torch.tensor(-digamma(k) + log(N) + self.vd(dx+dy,q))
        ans_x = torch.tensor(log(N) + self.vd(dx,q))
        ans_y = torch.tensor(log(N) + self.vd(dy,q))
        
        ans_xy = ans_xy.requires_grad_(True)
        ans_x = ans_x.requires_grad_(True)
        ans_y = ans_y.requires_grad_(True)
        
        for i in range(N):
            ans_xy = ans_xy + (dx+dy)*log(knn_dis[i])/N
            ans_x = ans_x - log(len(tree_x.query_ball_point(x_samples[i].detach().cpu().numpy(),knn_dis[i].detach().cpu().numpy()+1e-15,
                                                            p=q))-1)/N+dx*log(knn_dis[i])/N
            ans_y = ans_y - log(len(tree_y.query_ball_point(y_samples[i].detach().cpu().numpy(),knn_dis[i].detach().cpu().numpy()+1e-15,
                                                            p=q))-1)/N+dy*log(knn_dis[i])/N        
        return ans_x+ans_y-ans_xy
    
################################################## MINE Neural Estimator ##################################################

EPS = 1e-6
class EMALoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, running_ema):
        ctx.save_for_backward(input, running_ema)
        input_log_sum_exp = input.exp().mean().log()

        return input_log_sum_exp

    @staticmethod
    def backward(ctx, grad_output):
        input, running_mean = ctx.saved_tensors
        grad = grad_output * input.exp().detach() / \
            (running_mean + EPS) / input.shape[0]
        return grad, None


def ema(mu, alpha, past_ema):
    return alpha * mu + (1.0 - alpha) * past_ema


def ema_loss(x, running_mean, alpha):
    t_exp = torch.exp(torch.logsumexp(x, 0) - math.log(x.shape[0])).detach()
    if running_mean == 0:
        running_mean = t_exp
    else:
        running_mean = ema(t_exp, alpha, running_mean.item())
    t_log = EMALoss.apply(x, running_mean)

    # Recalculate ema

    return t_log, running_mean

class MINECritic(nn.Module):
    def __init__(self, A_dim, Z_dim, hidden_dim, layers, activation='relu', alpha=0.01):
        super().__init__()
        self.running_mean = 0
        self.loss = 'mine'
        self.alpha = alpha
      
        self.T = mlp(A_dim + Z_dim, hidden_dim, 1, layers, activation)

    def forward(self, x, z, z_marg=None):
        if z_marg is None:
            z_marg = z[torch.randperm(x.shape[0])]

        t = self.T(torch.cat([x, z], dim=1)).mean()
        t_marg = self.T(torch.cat([x, z_marg], dim=1))

        if self.loss in ['mine']:
            second_term, self.running_mean = ema_loss(
                t_marg, self.running_mean, self.alpha)
        elif self.loss in ['fdiv']:
            second_term = torch.exp(t_marg - 1).mean()
        elif self.loss in ['mine_biased']:
            second_term = torch.logsumexp(
                t_marg, 0) - math.log(t_marg.shape[0])

        return -t + second_term

    def mi(self, x, z, z_marg=None):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if isinstance(z, np.ndarray):
            z = torch.from_numpy(z).float()

        with torch.no_grad():
            mi = -self.forward(x, z, z_marg)
        return mi