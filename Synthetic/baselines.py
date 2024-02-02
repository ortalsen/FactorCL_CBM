from critic_objectives import*
import torch
import torch.nn as nn
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import time
from collections import OrderedDict
from critic_objectives import mlp
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import os

    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class mlp_multitask(nn.Module):
    def __init__(self, A_dim, hidden_dim, layers, activation, **extra_kwargs):
        super(mlp_multitask, self).__init__()
        # output is scalar score
        self._f = mlp(A_dim, hidden_dim, 256, layers, activation)
        self.fc1 = nn.Linear(256, 1)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self._f(x)
        out = self.fc1(x)
        concept = self.fc2(x)
        return out, concept
    
def mlp_multitask_train(model, train_loader, val_loader, num_epochs, lr, weight_decay, device, log_interval,
                          save_interval):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_err = torch.tensor(1e7)
    loss_func_binary = nn.BCELoss() #nn.CrossEntropyLoss()
    loss_func_concept = nn.MSELoss()
    model.to(device)
    model.train()
    tepoch = tqdm(range(num_epochs))
    for epoch in tepoch:
        tepoch.set_description(f"Epoch {epoch}")
        for batch_idx, (data, concept, target,_) in enumerate(train_loader):
            data, concept, target = data.to(device), concept.to(device), target.to(device)
            optimizer.zero_grad()
            logits, pred_concepts = model(data)
            preds =  torch.sigmoid(logits) #torch.softmax(logits, dim=-1)
            classification_loss = loss_func_binary(preds, target.float()) #F.cross_entropy(preds, target)
            regression_loss = loss_func_concept(pred_concepts, concept)
            
            loss = classification_loss + regression_loss 
            
            loss.backward()
            optimizer.step()
        tepoch.set_postfix(loss=loss.item())
        if epoch % save_interval == 0:
            val_err = 0
            model.eval()
            with torch.no_grad():
                for batch_idx, (data, concept, target,_) in enumerate(val_loader):
                    data, concept, target = data.to(device), concept.to(device), target.to(device)
                    logits, pred_concepts = model(data)
                    preds = torch.sigmoid(logits)
                    classification_loss = loss_func_binary(preds, target.float()) 
                    regression_loss = loss_func_concept(pred_concepts, concept)
                    val_err += classification_loss + regression_loss
                val_err = val_err / len(val_loader)
            if val_err < best_val_err:
                best_val_err = val_err
            else:
                print('Val loss did not improve')
                return model
    return model

def mlp_train(model, train_loader, val_loader, num_epochs, lr, weight_decay, device, log_interval,
                          save_interval):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_err = torch.tensor(1e7)
    loss_func = nn.BCELoss() #nn.CrossEntropyLoss() #
    model.to(device)
    model.train()
    tepoch = tqdm(range(num_epochs))
    for epoch in tepoch:
        tepoch.set_description(f"Epoch {epoch}")
        for batch_idx, (data, concept, target,_) in enumerate(train_loader):
            data, concept, target = data.to(device), concept.to(device), target.to(device)
            optimizer.zero_grad()
            logits = model(data)
            preds =  torch.sigmoid(logits) #torch.softmax(logits, dim=-1)
            loss = loss_func(preds, target.float()) #F.cross_entropy(preds, target)
            loss.backward()
            optimizer.step()
        tepoch.set_postfix(loss=loss.item())
        if epoch % save_interval == 0:
            val_err = 0
            model.eval()
            with torch.no_grad():
                for batch_idx, (data, concept, target,_) in enumerate(val_loader):
                    data, concept, target = data.to(device), concept.to(device), target.to(device)
                    logits = model(data)
                    preds = torch.sigmoid(logits)
                    val_err += loss_func(preds, target.float())
                val_err = val_err / len(val_loader)
            # print('Val loss: {:.6f}'.format(val_err))
            if val_err < best_val_err:
                best_val_err = val_err
            else:
                print('Val loss did not improve')
                return model
    return model

def mlp_train_x_c(model, train_loader, val_loader, num_epochs, lr, weight_decay, device, log_interval,
                          save_interval):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_err = torch.tensor(1e7)
    loss_func = nn.BCELoss() #nn.CrossEntropyLoss() #
    model.to(device)
    model.train()
    tepoch = tqdm(range(num_epochs))
    for epoch in tepoch:
        tepoch.set_description(f"Epoch {epoch}")
        for batch_idx, (data, concept, target,_) in enumerate(train_loader):
            data, concept, target = data.to(device), concept.to(device), target.to(device)
            optimizer.zero_grad()
            cat_data = torch.cat((data,concept), dim=1)
            logits = model(cat_data)
            preds =  torch.sigmoid(logits) #torch.softmax(logits, dim=-1)
            # print(f'unique values {torch.unique(preds)}, {torch.unique(target)}')
            loss = loss_func(preds, target.float()) #F.cross_entropy(preds, target)
            loss.backward()
            optimizer.step()
            
        tepoch.set_postfix(loss=loss.item())
        if epoch % save_interval == 0:
            val_err = 0
            model.eval()
            with torch.no_grad():
                for batch_idx, (data, concept, target,_) in enumerate(val_loader):
                    data, concept, target = data.to(device), concept.to(device), target.to(device)
                    cat_data = torch.cat((data,concept),dim=1)
                    logits = model(cat_data)
                    preds = torch.sigmoid(logits)
                    val_err += loss_func(preds, target.float())
                val_err = val_err / len(val_loader)
            if val_err < best_val_err:
                best_val_err = val_err
            else:
                print('Val loss did not improve')
                return model
    return model

def mlp_train_c(model, train_loader, val_loader, num_epochs, lr, weight_decay, device, log_interval,
                          save_interval):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val_err = torch.tensor(1e7)
    loss_func = nn.MSELoss()
    model.to(device)
    model.train()
    tepoch = tqdm(range(num_epochs))
    for epoch in tepoch:
        tepoch.set_description(f"Epoch {epoch}")
        for batch_idx, (data, concept, target,_) in enumerate(train_loader):
            data, concept, target = data.to(device), concept.to(device), target.to(device)
            optimizer.zero_grad()
            pred_concepts = model(data)
            loss = loss_func(pred_concepts, concept)
            loss.backward()
            optimizer.step()
        tepoch.set_postfix(loss=loss.item())
        if epoch % save_interval == 0:
            val_err = 0
            model.eval()
            with torch.no_grad():
                for batch_idx, (data, concept, target,_) in enumerate(val_loader):
                    data, concept, target = data.to(device), concept.to(device), target.to(device)
                    pred_concepts = model(data)
                    val_err += loss_func(pred_concepts, concept)
                val_err = val_err / len(val_loader)
            if val_err < best_val_err:
                best_val_err = val_err
            else:
                print('Val loss did not improve')
                return model
    return model



def baseline_1(train_dataset, test_dataset, num_eval=10, save_path='./results'):
    

    train_embeds = torch.stack([sample[0] for sample in train_dataset]).detach().cpu().numpy()
    train_labels = np.array([sample[2].item() for sample in  train_dataset])

    test_embeds = torch.stack([sample[0] for sample in  test_dataset]).detach().cpu().numpy()
    test_labels = np.array([sample[2].item() for sample in  test_dataset])
    acc_list, pre_list, recall_list, f1_list = [], [], [], []
    teval = tqdm(range(num_eval))
    for idx in teval:
        teval.set_description(f"Evaluation {idx}")
        clf = LogisticRegression(max_iter=1000).fit(train_embeds, train_labels)
        predictions = clf.predict(test_embeds)
        
        accuracy = accuracy_score(test_labels, predictions)
        precision = precision_score(test_labels, predictions)
        recall = recall_score(test_labels, predictions)
        f1 = f1_score(test_labels, predictions)
        
        acc_list.append(accuracy)
        pre_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
    
    dict_results = {'accuracy':acc_list,
                    'precision':pre_list,
                    'recall':recall_list,
                    'f1_score':f1_list}
    df_results = pd.DataFrame(data=dict_results)
    
    print(f'Accuracy:{df_results.accuracy.mean():0.3f} \u00B1 {2*df_results.accuracy.std():0.3f}')
    print(f'Precision:{df_results.precision.mean():0.3f} \u00B1 {2*df_results.precision.std():0.3f}')
    print(f'Recall:{df_results.recall.mean():0.3f} \u00B1 {2*df_results.recall.std():0.3f}')
    print(f'F1-score:{df_results.f1_score.mean():0.3f} \u00B1 {2*df_results.f1_score.std():0.3f}')
    
    directory = save_path + '/' + time.strftime("%Y%m%d")
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    timestr = time.strftime("%H%M%S")
    file_path = directory + '/baseline_1_' + str(num_eval) + '_' + timestr +'.csv'
    df_results.to_csv(file_path)
    
    
    return df_results


def baseline_2(train_dataset, val_dataset, test_dataset, transform_dim=100000, batch_size=100, num_eval=10, save_path='./results'):
    
    train_embeds = torch.stack([sample[0] for sample in  train_dataset]).detach().cpu().numpy()
    train_labels = np.array([sample[2].item() for sample in  train_dataset])

    test_embeds = torch.stack([sample[0] for sample in  test_dataset]).detach().cpu().numpy()
    test_labels = np.array([sample[2].item() for sample in  test_dataset])
    

    acc_list, pre_list, recall_list, f1_list = [], [], [], []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # teval = tqdm(range(num_eval))
    for idx in range(num_eval):
        # teval.set_description(f"Evaluation {idx}")
        train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True,
                          batch_size=batch_size)
        val_loader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, drop_last=True)
        test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False)
        backbone = mlp(transform_dim, 512, 512, layers=2, activation='relu')
        FC = mlp(512, 256, 1, 1, activation= 'relu')
        model = nn.Sequential(backbone, FC)
        trained_model = mlp_train(model, train_loader, val_loader, 1000, 1e-5, 1e-5,'cuda', 100, 100)
        
        out = trained_model(torch.tensor(test_embeds).to(device))
        predictions = torch.sigmoid(out).round().detach().cpu().numpy()
        
        accuracy = accuracy_score(test_labels, predictions)
        precision = precision_score(test_labels, predictions)
        recall = recall_score(test_labels, predictions)
        f1 = f1_score(test_labels, predictions)
        
        acc_list.append(accuracy)
        pre_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
    
    dict_results = {'accuracy':acc_list,
                    'precision':pre_list,
                    'recall':recall_list,
                    'f1_score':f1_list}
    df_results = pd.DataFrame(data=dict_results)
    
    print(f'Accuracy:{df_results.accuracy.mean():0.3f} \u00B1 {2*df_results.accuracy.std():0.3f}')
    print(f'Precision:{df_results.precision.mean():0.3f} \u00B1 {2*df_results.precision.std():0.3f}')
    print(f'Recall:{df_results.recall.mean():0.3f} \u00B1 {2*df_results.recall.std():0.3f}')
    print(f'F1-score:{df_results.f1_score.mean():0.3f} \u00B1 {2*df_results.f1_score.std():0.3f}')
    
    directory = save_path + '/' + time.strftime("%Y%m%d")
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    timestr = time.strftime("%H%M%S")
    file_path = directory + '/baseline_2_' + str(num_eval) + '_' + timestr +'.csv'
    df_results.to_csv(file_path)
    
    
    return df_results

def baseline_3_A(train_dataset, test_dataset, num_eval=10, save_path='./results'):
    train_embeds = torch.stack([torch.from_numpy(np.concatenate([sample[0], sample[1]])) for sample in  train_dataset]).detach().cpu().numpy()
    train_labels = np.array([sample[2].item() for sample in  train_dataset])

    test_embeds = torch.stack([torch.from_numpy(np.concatenate([sample[0], sample[1]])) for sample in  test_dataset]).detach().cpu().numpy()
    test_labels = np.array([sample[2].item() for sample in  test_dataset])
    acc_list, pre_list, recall_list, f1_list = [], [], [], []
    teval = tqdm(range(num_eval))
    for idx in teval:
        teval.set_description(f"Evaluation {idx}")
        clf = LogisticRegression(max_iter=1000).fit(train_embeds, train_labels)
        predictions = clf.predict(test_embeds)
        
        accuracy = accuracy_score(test_labels, predictions)
        precision = precision_score(test_labels, predictions)
        recall = recall_score(test_labels, predictions)
        f1 = f1_score(test_labels, predictions)
        
        acc_list.append(accuracy)
        pre_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
    
    dict_results = {'accuracy':acc_list,
                    'precision':pre_list,
                    'recall':recall_list,
                    'f1_score':f1_list}
    df_results = pd.DataFrame(data=dict_results)
    
    print(f'Accuracy:{df_results.accuracy.mean():0.3f} \u00B1 {2*df_results.accuracy.std():0.3f}')
    print(f'Precision:{df_results.precision.mean():0.3f} \u00B1 {2*df_results.precision.std():0.3f}')
    print(f'Recall:{df_results.recall.mean():0.3f} \u00B1 {2*df_results.recall.std():0.3f}')
    print(f'F1-score:{df_results.f1_score.mean():0.3f} \u00B1 {2*df_results.f1_score.std():0.3f}')
    
    directory = save_path + '/' + time.strftime("%Y%m%d")
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    timestr = time.strftime("%H%M%S")
    file_path = directory + '/baseline_3_A_'+ str(num_eval) + '_' + timestr +'.csv'
    df_results.to_csv(file_path)
    
    
    return df_results


def baseline_3_B(train_dataset, val_dataset, test_dataset,  transform_dim=100000, batch_size=100, num_eval=10, save_path='./results'):
    
    test_embeds = torch.stack([torch.from_numpy(np.concatenate([sample[0], sample[1]])) for sample in  test_dataset]).detach().cpu().numpy()
    test_labels = np.array([sample[2].item() for sample in  test_dataset])
    
    acc_list, pre_list, recall_list, f1_list = [], [], [], []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for idx in range(num_eval):
        train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True,
                          batch_size=batch_size)
        val_loader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, drop_last=True)
        test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False)
        backbone = mlp(transform_dim+1, 512, 512, layers=2, activation='relu')
        FC = mlp(512, 256, 1, 1, activation= 'relu')
        model = nn.Sequential(backbone, FC)
        trained_model = mlp_train_x_c(model, train_loader, val_loader, 1000, 1e-5, 1e-5,'cuda', 100, 100)
        
        out = trained_model(torch.tensor(test_embeds).to(device))
        predictions = torch.sigmoid(out).round().detach().cpu().numpy()
        
        accuracy = accuracy_score(test_labels, predictions)
        precision = precision_score(test_labels, predictions)
        recall = recall_score(test_labels, predictions)
        f1 = f1_score(test_labels, predictions)
        
        acc_list.append(accuracy)
        pre_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
    
    dict_results = {'accuracy':acc_list,
                    'precision':pre_list,
                    'recall':recall_list,
                    'f1_score':f1_list}
    df_results = pd.DataFrame(data=dict_results)
    
    print(f'Accuracy:{df_results.accuracy.mean():0.3f} \u00B1 {2*df_results.accuracy.std():0.3f}')
    print(f'Precision:{df_results.precision.mean():0.3f} \u00B1 {2*df_results.precision.std():0.3f}')
    print(f'Recall:{df_results.recall.mean():0.3f} \u00B1 {2*df_results.recall.std():0.3f}')
    print(f'F1-score:{df_results.f1_score.mean():0.3f} \u00B1 {2*df_results.f1_score.std():0.3f}')
    
    directory = save_path + '/' + time.strftime("%Y%m%d")
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    timestr = time.strftime("%H%M%S")
    file_path = directory + '/baseline_3_B_'+ str(num_eval) + '_' + timestr +'.csv'
    df_results.to_csv(file_path)
    
    
    return df_results

def baseline_4(train_dataset, val_dataset, test_dataset, transform_dim=100000, batch_size=100, num_eval=10, save_path='./results'):
    
    test_embeds = torch.stack([sample[0] for sample in  test_dataset]).detach().cpu().numpy()
    test_concepts = torch.tensor([sample[1].item() for sample in  test_dataset]).unsqueeze(1)
    test_labels = np.array([sample[2].item() for sample in  test_dataset])
    
    acc_list, pre_list, recall_list, f1_list = [], [], [], []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for idx in range(num_eval):
        train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True,
                          batch_size=batch_size)
        val_loader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, drop_last=True)
        test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False)
        model = mlp_multitask(transform_dim, 512, layers=2, activation='relu')
        trained_model = mlp_multitask_train(model, train_loader, val_loader, 1000, 1e-5, 1e-5,'cuda', 100, 100)
        
        out, _ = trained_model(torch.tensor(test_embeds).to(device))
        predictions = torch.sigmoid(out).round().detach().cpu().numpy()
        
        accuracy = accuracy_score(test_labels, predictions)
        precision = precision_score(test_labels, predictions)
        recall = recall_score(test_labels, predictions)
        f1 = f1_score(test_labels, predictions)
        
        acc_list.append(accuracy)
        pre_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
    
    dict_results = {'accuracy':acc_list,
                    'precision':pre_list,
                    'recall':recall_list,
                    'f1_score':f1_list}
    df_results = pd.DataFrame(data=dict_results)
    
    print(f'Accuracy:{df_results.accuracy.mean():0.3f} \u00B1 {2*df_results.accuracy.std():0.3f}')
    print(f'Precision:{df_results.precision.mean():0.3f} \u00B1 {2*df_results.precision.std():0.3f}')
    print(f'Recall:{df_results.recall.mean():0.3f} \u00B1 {2*df_results.recall.std():0.3f}')
    print(f'F1-score:{df_results.f1_score.mean():0.3f} \u00B1 {2*df_results.f1_score.std():0.3f}')
    
    directory = save_path + '/' + time.strftime("%Y%m%d")
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    timestr = time.strftime("%H%M%S")
    file_path = directory +  '/baseline_4_'+ str(num_eval) + '_' + timestr +'.csv'
    df_results.to_csv(file_path)
    
    
    return df_results

from collections import OrderedDict

def baseline_5(train_dataset, val_dataset, test_dataset, transform_dim=100000, batch_size=100, num_eval=10, save_path='./results'):
    
    test_embeds = torch.stack([sample[0] for sample in  test_dataset]).detach().cpu().numpy()
    test_concepts = torch.tensor([sample[1].item() for sample in  test_dataset]).unsqueeze(1)
    test_labels = np.array([sample[2].item() for sample in  test_dataset])

    
    acc_list, pre_list, recall_list, f1_list = [], [], [], []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for idx in range(num_eval):
        train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True,
                          batch_size=batch_size)
        val_loader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, drop_last=True)
        test_loader = DataLoader(test_dataset, shuffle=False, drop_last=False)
        
        backbone = mlp(transform_dim, 512, 512, layers=2, activation='relu')
        FC = mlp(512, 256, 1, 1, activation= 'relu')
        pretrain_model = model = nn.Sequential(OrderedDict(backbone=backbone, FC=FC))
        pretrain_model = mlp_train_c(pretrain_model, train_loader, val_loader, 1000, 1e-5, 1e-5,'cuda', 100, 100)
        FC = mlp(512, 256, 1, 1, activation= 'relu')
        model = nn.Sequential(pretrain_model.backbone, FC)
        trained_model = mlp_train(model, train_loader, val_loader, 1000, 1e-5, 1e-5,'cuda', 100, 100)
        
        out = trained_model(torch.tensor(test_embeds).to(device))
        predictions = torch.sigmoid(out).round().detach().cpu().numpy()
        
        accuracy = accuracy_score(test_labels, predictions)
        precision = precision_score(test_labels, predictions)
        recall = recall_score(test_labels, predictions)
        f1 = f1_score(test_labels, predictions)
        
        acc_list.append(accuracy)
        pre_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
    
    dict_results = {'accuracy':acc_list,
                    'precision':pre_list,
                    'recall':recall_list,
                    'f1_score':f1_list}
    df_results = pd.DataFrame(data=dict_results)
    
    print(f'Accuracy:{df_results.accuracy.mean():0.3f} \u00B1 {2*df_results.accuracy.std():0.3f}')
    print(f'Precision:{df_results.precision.mean():0.3f} \u00B1 {2*df_results.precision.std():0.3f}')
    print(f'Recall:{df_results.recall.mean():0.3f} \u00B1 {2*df_results.recall.std():0.3f}')
    print(f'F1-score:{df_results.f1_score.mean():0.3f} \u00B1 {2*df_results.f1_score.std():0.3f}')
    
    directory = save_path + '/' + time.strftime("%Y%m%d")
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    timestr = time.strftime("%H%M%S")
    file_path = directory +  '/baseline_5_'+ str(num_eval) + '_' + timestr +'.csv'
    df_results.to_csv(file_path)
    
    
    return df_results