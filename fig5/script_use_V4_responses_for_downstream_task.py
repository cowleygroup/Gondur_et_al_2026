import numpy as np
import matplotlib.pyplot as plt
import os 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
torch.backends.cudnn.benchmark = True

folder_path= './compact_models_final/saved_models/'
all_compact_models_list = sorted([f for f in os.listdir(folder_path) if f.endswith('.keras')])
type_of_relu_to_use_type = (sys.argv[1])
gpu_number = (sys.argv[2])

device = (
"cuda"
if torch.cuda.is_available()
else "mps"
if torch.backends.mps.is_available()
else "cpu"
)

gpu_id = int(gpu_number)
torch.cuda.set_device(gpu_id)
print(f"Using {device} device")

BATCH = 128
num_epochs = 800
patience = 20

def get_all_compact_responses_together(root_dir ):
    folders = sorted([f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))])
    arrays = []
    for folder in folders:
        folder_path = os.path.join(root_dir, folder)
        npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
       
        file_path = os.path.join(folder_path, npy_files[0])
        arr = np.load(file_path)

        arrays.append(arr)
    result = np.concatenate(arrays, axis=1)
    return result

class Adapter_Net(nn.Module):
            def __init__(self, input_dim, num_classes=102, hidden_dim=216):
                super(Adapter_Net, self).__init__()

                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
                self.fc3 = nn.Linear(hidden_dim // 2, num_classes)

                self.bn1 = nn.BatchNorm1d(hidden_dim)
                self.bn2 = nn.BatchNorm1d(hidden_dim // 2)

            def forward(self, x):

                x = self.fc1(x)
                x = self.bn1(x)
                x = F.relu(x)
                
                x = self.fc2(x)
                x = self.bn2(x)
                x = F.relu(x)

                x = self.fc3(x)

                return x





if type_of_relu_to_use_type == '0':
    type_of_relu_to_use = 'regular'

if type_of_relu_to_use_type == '1':
    type_of_relu_to_use = 'reverse'

if type_of_relu_to_use_type == '2':
    type_of_relu_to_use = 'both'

if type_of_relu_to_use_type == '3':
    type_of_relu_to_use = 'notmodified'

if type_of_relu_to_use_type == '4':
    type_of_relu_to_use = 'spliced'


top1acc = True

if type_of_relu_to_use == 'notmodified':
    quantile_list = [0]
else:
    quantile_list = [0.5]

all_neuron_nums =  [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 219]





def relu(x,q):
    if q == 0:
        ## at q=0 we don't apply and threshold, just return all responses
        return x
    else:
        if type_of_relu_to_use == 'regular':
            threshold = np.quantile(x, q, axis=0)
            result = np.maximum(0, x-threshold)
     
            return result
        
        if type_of_relu_to_use == 'reverse':
            threshold_high = np.quantile(x, 1 - q, axis=0)
            return np.maximum(0, threshold_high - x)


      
def relu_ap(x,q):
    if q == 0:
        ## at q=0 we don't apply and threshold, just return all responses
        return x
    else:
        
        threshold_high = np.quantile(x, 1 - q, axis=0)
        return np.maximum(0, threshold_high - x)


def relu_p(x,q):
    if q == 0:
        ## at q=0 we don't apply and threshold, just return all responses
        return x
    else:
        
        threshold = np.quantile(x, q, axis=0)
        result = np.maximum(0, x-threshold)
    
        return result


def train_and_val_top5(loader_for_train, loader_for_val):
    model.train()
    train_loss = 0
    for X, y in loader_for_train:
        optimizer.zero_grad()
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0
    correct, total = 0, 0
    with torch.no_grad():
        for Xv, yv in loader_for_val:
            Xv, yv = Xv.to(device), yv.to(device)

            logits = model(Xv)
            loss = criterion(logits, yv)
            val_loss += loss.item()
            _, top5 = logits.topk(5, dim=1) 
            correct_top5 = top5.eq(yv.view(-1, 1)).sum().item() 
            total += yv.size(0)
            correct += correct_top5

    val_loss /= len(val_loader)
    val_acc = correct / total

    return train_loss, val_loss, val_acc

def train_and_val_top1(loader_for_train, loader_for_val):
    model.train()
    train_loss = 0
    for X, y in loader_for_train:
        optimizer.zero_grad()
        X, y = X.to(device), y.to(device)
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    model.eval()
    val_loss = 0
    correct, total = 0, 0
    with torch.no_grad():
        for Xv, yv in loader_for_val:
            Xv, yv = Xv.to(device), yv.to(device)
            logits = model(Xv)
            loss = criterion(logits, yv)
            val_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == yv).sum().item()
            total += yv.size(0)

    val_loss /= len(val_loader)
    val_acc = correct / total

    return train_loss, val_loss, val_acc


def test_top1(loader_for_test):
    y_true_all, y_pred_all = [], []
    model.eval()
    with torch.no_grad():
        for X, y in loader_for_test:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            preds = torch.argmax(logits, dim=1)
            y_true_all.extend(y.cpu().numpy())
            y_pred_all.extend(preds.cpu().numpy())
    return y_true_all, y_pred_all

def test_top5(loader_for_test):
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in loader_for_test:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            _, top5 = logits.topk(5, dim=1)
            correct_top5 = top5.eq(y.view(-1, 1)).sum().item()
            correct += correct_top5
            total += y.size(0)

    return correct, total


all_acc =[]

for quantile_i in quantile_list:
    for neuronx in all_neuron_nums:

        print(quantile_i, neuronx)
        responses_train = get_all_compact_responses_together(f'./5k_responses_train_caltech/')
        features_train = np.load("./caltech_labels_train.npy", mmap_mode ='r')
     
        responses_test = get_all_compact_responses_together(f'./1k_responses_test_caltech/')
        features_test= np.load("./caltech_labels_test.npy", mmap_mode ='r')

        responses_val=get_all_compact_responses_together(f'./1k_responses_val_caltech/')
        features_val = np.load("./caltech_labels_val.npy", mmap_mode ='r')

        X_train_init, X_test_init, y_train_init, y_test_init = responses_train.copy(), responses_test.copy(), features_train.copy(), features_test.copy()
        X_val_init, y_val_init = responses_val.copy(), features_val.copy()

    
        if type_of_relu_to_use == 'both':

            Xtrain_p_pref = relu_p(X_train_init, quantile_i)
            Xval_p_pref  = relu_p(X_val_init, quantile_i)
            Xtest_p_pref  =relu_p(X_test_init, quantile_i)


            Xtrain_p_apref = relu_ap(X_train_init, quantile_i)
            Xval_p_apref  = relu_ap(X_val_init, quantile_i)
            Xtest_p_apref  =relu_ap(X_test_init, quantile_i)


            neuronx_selected = np.random.choice(np.arange(219), neuronx, replace=False)
            half_neurons = neuronx//2
            neurons_selected_pref = neuronx_selected[:half_neurons]
            neurons_selected_antipref = neuronx_selected[half_neurons:]

            train_selected = np.concatenate((Xtrain_p_pref[:, neurons_selected_pref], Xtrain_p_apref[:, neurons_selected_antipref]), axis=1)
            val_selected = np.concatenate((Xval_p_pref[:, neurons_selected_pref], Xval_p_apref[:, neurons_selected_antipref]), axis=1)
            test_selected = np.concatenate((Xtest_p_pref[:, neurons_selected_pref], Xtest_p_apref[:, neurons_selected_antipref]), axis=1)

            Xtrain_select, Xtest_select, y_train_select, y_test_select = train_selected, test_selected, features_train.copy(), features_test.copy()
            Xval_select, y_val_select = val_selected, features_val.copy()

        elif type_of_relu_to_use == 'spliced':
            Xtrain_p_pref = relu_p(X_train_init, quantile_i)
            Xval_p_pref  = relu_p(X_val_init, quantile_i)
            Xtest_p_pref  =relu_p(X_test_init, quantile_i)


            Xtrain_p_apref = relu_ap(X_train_init, quantile_i)
            Xval_p_apref  = relu_ap(X_val_init, quantile_i)
            Xtest_p_apref  =relu_ap(X_test_init, quantile_i)


            neuronx_selected = np.random.choice(np.arange(219), neuronx, replace=False)
            neuronx_selected_ap = np.random.choice(np.arange(219), neuronx, replace=False)
            half_neurons = neuronx#//2
            neurons_selected_pref = neuronx_selected
            neurons_selected_antipref = neuronx_selected_ap

            train_selected = Xtrain_p_pref[:, neurons_selected_pref] - Xtrain_p_apref[:, neurons_selected_antipref]
            val_selected = Xval_p_pref[:, neurons_selected_pref]-  Xval_p_apref[:, neurons_selected_antipref]
            test_selected = Xtest_p_pref[:, neurons_selected_pref]- Xtest_p_apref[:, neurons_selected_antipref]
            Xtrain_select, Xtest_select, y_train_select, y_test_select = train_selected, test_selected, features_train.copy(), features_test.copy()
            Xval_select, y_val_select = val_selected, features_val.copy()
         

        else: 
            Xtrain_p = relu(X_train_init, quantile_i)
            Xval_p   = relu(X_val_init, quantile_i)
            Xtest_p  =relu(X_test_init, quantile_i)

            neuronx_selected = np.random.choice(np.arange(219), neuronx, replace=False)
            Xtrain_select, Xtest_select, y_train_select, y_test_select = Xtrain_p[:, neuronx_selected].copy(), Xtest_p[:,neuronx_selected].copy(), features_train.copy(), features_test.copy()
            Xval_select, y_val_select = Xval_p[:, neuronx_selected].copy(), features_val.copy()

        

        Xtrain = torch.tensor(Xtrain_select, dtype=torch.float32)
        Xval   = torch.tensor(Xval_select, dtype=torch.float32)
        Xtest  = torch.tensor(Xtest_select, dtype=torch.float32)

        ytrain = torch.tensor(y_train_select,  dtype=torch.long)
        yval   = torch.tensor(y_val_select,  dtype=torch.long)
        ytest  = torch.tensor(y_test_select,  dtype=torch.long)
 
        train_loader = DataLoader(TensorDataset(Xtrain, ytrain), batch_size=BATCH, shuffle=True, num_workers= 32, pin_memory=True)
        val_loader   = DataLoader(TensorDataset(Xval, yval), batch_size=BATCH, shuffle=False, num_workers= 10, pin_memory=True)
        test_loader  = DataLoader(TensorDataset(Xtest, ytest), batch_size=BATCH, shuffle=False, num_workers= 10)

        x_example, _ = next(iter(train_loader))
        input_dim = x_example.shape[1]

        model = Adapter_Net(input_dim=input_dim, num_classes=102, hidden_dim=400).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.01)

        best_val_loss = np.inf
        epochs_no_improve = 0

        for epoch in range(num_epochs):
            if top1acc == False:
                train_loss, val_loss, val_acc = train_and_val_top5(loader_for_train=train_loader, loader_for_val= val_loader)
            if top1acc == True:
                train_loss, val_loss, val_acc = train_and_val_top1(loader_for_train=train_loader, loader_for_val= val_loader)

            if epoch%50 ==0:

                print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val loss={val_loss:.4f}, val acc={val_acc:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
             
            if epochs_no_improve >= patience:
                break



        if top1acc == True:
            y_true_all, y_pred_all = test_top1(loader_for_test=test_loader)
            print("Test accuracy, top 1:", accuracy_score(y_true_all, y_pred_all))
            all_acc.append(accuracy_score(y_true_all, y_pred_all))

        if top1acc == False:
            correct, total = test_top5(loader_for_test=test_loader)
            all_acc.append(correct / total)
            print("Test accuracy, top 5:", correct / total)


        

    
torch.cuda.empty_cache() 
print(all_acc)

