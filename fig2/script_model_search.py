
## you can run this script with embeddings from vgg19, alexnet, resnetrobust or resnet




import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np  
import os
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import sys
from tensorflow.keras import backend as K
import zipfile
import class_linear_mapping_ensemble_ridgereg
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from copy import deepcopy
import random
import torch.optim as optim
import csv

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

def get_images_from_zip(directory_path):
        imgs = []
        counts = 0
        archive = directory_path
        names = archive.namelist()

        for name in names:     
            if name.endswith('.jpg'):
      
                image_data = Image.open(archive.open(name)).resize(size=(224,224))#.convert('RGB')
                img = np.array(image_data) 
                imgs.append(img)        
                counts += 1
       
        archive.close()

        imgs = np.array(imgs)
        return (imgs)




device = (
"cuda"
if torch.cuda.is_available()
else "mps"
if torch.backends.mps.is_available()
else "cpu"
)

LM = class_linear_mapping_ensemble_ridgereg.LinearMappingClass()


def init_conv_identity(conv_layer):
    with torch.no_grad():
        conv_layer.weight.zero_()   
        for i in range(min(conv_layer.in_channels, conv_layer.out_channels)):
            conv_layer.weight[i, i, 0, 0] = 1
        if conv_layer.bias is not None:
            conv_layer.bias.zero_()


###### our conv2d with relu and layernorm
class AffineReLULinearModel(nn.Module):
    def __init__(self, num_neurons,num_pixels=13, num_feature_vars=384, bottleneck_dim= 384, l2_reg_strength=0.1,traintime=True):
        super().__init__()
        
        self.affine = nn.Conv2d(
            in_channels=num_feature_vars,
            out_channels=bottleneck_dim,
            kernel_size=1,
            padding=0,
            bias=True
        )

        init_conv_identity(self.affine)
        
        self.layernorm = nn.LayerNorm([bottleneck_dim, num_pixels, num_pixels])
        
        self.relu = nn.ReLU()
    
        self.fc = nn.Linear(num_pixels * num_pixels * bottleneck_dim, num_neurons)

        nn.init.zeros_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

        self.l2_reg_strength = l2_reg_strength

    def ridge_loss(self, y_pred, y_true):
        squared_error = torch.sum((y_true - y_pred) ** 2)      
        l2_reg = self.l2_reg_strength * torch.sum(self.fc.weight ** 2)
        
        return squared_error + l2_reg
    
    def forward(self, x):
        x = self.affine(x)
        x = self.layernorm(x)
        x = self.relu(x) 
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x

# model_name = 'vgg'
# model_name = 'resnetrobust'
# model_name = 'resnet'
model_name = 'alexnet'
picked_removal_type = (sys.argv[1])

if picked_removal_type == '0':
    session_type = 'relu'
    gpu_id = int(1)
    torch.cuda.set_device(gpu_id)
    print(f"Using {device} device")

if picked_removal_type == '1':
    session_type = 'negrelu'
    gpu_id = int(2)
    torch.cuda.set_device(gpu_id)
    print(f"Using {device} device")

if picked_removal_type == '2':
    session_type = 'norelu'
    gpu_id = int(3)
    torch.cuda.set_device(gpu_id)
    print(f"Using {device} device")

def train_pipeline(epochs=2560, verbose=True, 
                   lambda2_list= [ 0.01, 0.1, 1.0] , lrtouselist=[1e-3,1e-4,1e-6, 9e-5,5e-5],  allfilters = [1024], 
                    pixels = 14):
        
  
        best_val_r2 = -float('inf')
        best_lambda2 = None
        best_bottle_neck = None
        best_epoch_val_r2 = -float('inf')
        best_epoch_state = None
        patience = 30 
        
        csvfile = open(f"./csv_files/Cnew{session_type}_{model_name}_{session}.csv", "w", newline="")
        fieldnames = ["session", "lambda2", "learning_rate", "bottleneck", "val_r2", "test_r2"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
      

        for bottleneck in allfilters:
            for lambda2 in lambda2_list:
                for lrtouse in lrtouselist:
                    local_best_val_r2 = -float("inf")
                    early_stop_counter = 0
                    model = AffineReLULinearModel(
                        num_neurons=y.shape[-1],
                        bottleneck_dim=bottleneck,
                        l2_reg_strength=lambda2, num_pixels=pixels,
                        num_feature_vars =bottleneck
                    ).to(device)

                    optimizer = optim.Adam(model.parameters(), lr=lrtouse)
                    for epoch in range(epochs):
            
                        model.train()
                        for x_batch, y_batch in train_loader:
                            optimizer.zero_grad()
                            x_batch = x_batch.to(device)
                            y_batch = y_batch.to(device)
                            y_pred = model(x_batch)
                            loss = model.ridge_loss(y_pred, y_batch)
                            loss.backward()
                            optimizer.step()

                        model.eval()
                        with torch.no_grad():
                            all_val_preds, all_val_targets = [], []
                            for xval_batch, yval_batch in val_loader:
                                xval_batch = xval_batch.to(device)
                                yval_batch = yval_batch.to(device)
                                yval_pred = model(xval_batch)
                                all_val_preds.append(yval_pred.detach().cpu())
                                all_val_targets.append(yval_batch.detach().cpu())

                        yval_pred = torch.cat(all_val_preds, dim=0).numpy().T
                        yval_true = torch.cat(all_val_targets, dim=0).numpy().T
                        r2val =  LM.compute_raw_r2(yval_true, yval_pred, True)
                        median_r2val = np.median(r2val)

                        if verbose and (epoch + 1) % 20 == 0:
                            print(f"Epoch {epoch+1}/{epochs} | Val Median R²: {median_r2val:.4f}")

        
                        if median_r2val > local_best_val_r2:
                            local_best_val_r2 = median_r2val
                            early_stop_counter = 0
                            best_local_state = deepcopy(model.state_dict())  
                        else:
                            early_stop_counter += 1
                            if early_stop_counter >= patience:
                                break

                    model.load_state_dict(best_local_state)

                    if local_best_val_r2 > best_epoch_val_r2:
                        best_epoch_val_r2 = local_best_val_r2
                        best_epoch_state = deepcopy(model.state_dict())
                        best_lambda2 = lambda2
                        best_lr = lrtouse
                        best_val_r2 = best_epoch_val_r2
                        best_bottle_neck = bottleneck
               
                    writer.writerow({
                        "session": session,
                        "lambda2": lambda2,
                        "learning_rate": lrtouse,
                        "bottleneck": bottleneck,
                        "val_r2": local_best_val_r2,
                        "test_r2": ""  
                    })
                    csvfile.flush()

   
        best_model = AffineReLULinearModel(
            num_neurons=y.shape[-1],
            bottleneck_dim=best_bottle_neck,
            l2_reg_strength=best_lambda2, num_pixels=pixels,num_feature_vars =bottleneck
        ).to(device)

        best_model.load_state_dict(best_epoch_state)
        best_model.eval()
   
        all_preds = []
        with torch.no_grad():
            for xtest_batch, ytest_batch in test_loader:
                xtest_batch = xtest_batch.to(device)
                ytest_batch = ytest_batch.to(device)
                y_pred_test = best_model(xtest_batch)
                all_preds.append(y_pred_test.detach().cpu())
        all_preds = torch.cat(all_preds, dim=0).numpy().T
          

        all_targets = []
        with torch.no_grad():
            for xtest_batch, ytest_batch in test_loaderraw:
                xtest_batch = xtest_batch.to(device)
                ytest_batch = ytest_batch.to(device)
                print(ytest_batch.shape)
                all_targets.append(ytest_batch.detach().cpu())

        all_targets = torch.cat(all_targets, dim=0).numpy().T

        all_targets = np.transpose(all_targets, (1, 2, 0))
        r2 = LM.compute_r2_ER(np.array(all_targets),(( np.array(all_preds))))
        test_r2_median = np.median(r2)

        print(f"r2 on test set: {test_r2_median:.4f}")

        writer.writerow({
            "session": session,
            "lambda2": best_lambda2,
            "learning_rate": best_lr,
            "bottleneck": best_bottle_neck,
            "val_r2": best_val_r2,
            "test_r2": test_r2_median
        })
        csvfile.flush()
        csvfile.close()

        return best_model, np.array(r2)

 

## appendix analysis
def sign_scramble(x, p=0.5):
  
    B, H, W , F= x.shape
    print('--------')
    print(x.shape)

   
    idx = np.random.permutation(F)
    flip_idx = idx[:F // 2]

    signs = np.ones(F)
    signs[flip_idx] = -1
    signs = signs[None, None, None,  :,]

    return x * signs

session_list = ['190923', '211022', '210225', '201025' ]
sessions_r2s = []



for session in session_list:
    if model_name == 'vgg':
        X = np.load(f'./embeddings/beforerelu_vgg_{session}_embeds.npy', allow_pickle=True)
    if model_name == 'alexnet':
        X = np.load(f'./embeddings/beforerelu_alexnet_{session}_embeds.npy', allow_pickle=True)

    if model_name == 'resnetrobust':
        X = np.load(f'./embeddings/norelu_resrobust{session}_embeds.npy', allow_pickle=True)
    if model_name == 'resnet':
        # X =np.load(f'./embeddings/shuffled_resnet_prerelu_{session}_embeds.npy', allow_pickle=True)
        X =np.load(f'./embeddings/resnet_prerelu_{session}_embeds.npy', allow_pickle=True)
      
    y = np.load(f'/DATA/smith_lab/V4_recordings_compact_models/V4_neural_data/responses_repeat_averaged/responses_{session}.npy').T

    Xtrain, Xtest_temp, ytrain, ytest_temp = train_test_split(X, y, test_size=0.2, random_state=314)
    Xtest, Xval, ytest, yval = train_test_split(Xtest_temp, ytest_temp, test_size=0.5, random_state=314)
    print(f' test shape {Xtest.shape}')


    v4_neural_responses_raw = (np.load(f'/DATA/smith_lab/V4_recordings_compact_models/V4_neural_data/responses_raw/responses_{session}.npy'))
    v4_neural_responses_raw  = np.transpose(v4_neural_responses_raw , (1, 0, 2))

    Xtrainraw, Xtest_tempraw, ytrainraw, ytest_tempraw = train_test_split(X, v4_neural_responses_raw , test_size=0.2, random_state=314)
    Xtestraw, Xvalraw, ytestraw, yvalraw = train_test_split(Xtest_tempraw, ytest_tempraw, test_size=0.5, random_state=314)

    if picked_removal_type == '1':
        Xtestraw  = torch.tensor(np.maximum(0, -Xtestraw), dtype=torch.float32)
        ytestraw  = torch.tensor(ytestraw, dtype=torch.float32)
        Xtrain = torch.tensor(np.maximum(0, -Xtrain), dtype=torch.float32)
        Xval   = torch.tensor(np.maximum(0, -Xval), dtype=torch.float32)
        Xtest  = torch.tensor(np.maximum(0, -Xtest), dtype=torch.float32)
        ytrain = torch.tensor(ytrain, dtype=torch.float32)
        yval   = torch.tensor(yval, dtype=torch.float32)
        ytest  = torch.tensor(ytest, dtype=torch.float32)

    if picked_removal_type == '0':
        Xtestraw  = torch.tensor(np.maximum(0, Xtestraw), dtype=torch.float32)
        ytestraw  = torch.tensor(ytestraw, dtype=torch.float32)
        Xtrain = torch.tensor(np.maximum(0, Xtrain), dtype=torch.float32)
        Xval   = torch.tensor(np.maximum(0, Xval), dtype=torch.float32)
        Xtest  = torch.tensor(np.maximum(0, Xtest), dtype=torch.float32)
        ytrain = torch.tensor(ytrain, dtype=torch.float32)
        yval   = torch.tensor(yval, dtype=torch.float32)
        ytest  = torch.tensor(ytest, dtype=torch.float32)

    if picked_removal_type == '2':
        Xtestraw  = torch.tensor(Xtestraw, dtype=torch.float32)
        ytestraw  = torch.tensor(ytestraw, dtype=torch.float32)
        Xtrain = torch.tensor(Xtrain, dtype=torch.float32)
        Xval   = torch.tensor(Xval, dtype=torch.float32)
        Xtest  = torch.tensor(Xtest, dtype=torch.float32)
        ytrain = torch.tensor(ytrain, dtype=torch.float32)
        yval   = torch.tensor(yval, dtype=torch.float32)
        ytest  = torch.tensor(ytest, dtype=torch.float32)

    y_mean = ytrain.mean(0, keepdim=True)  
    y_std  = ytrain.std(0, keepdim=True)   

    y_std = torch.where(y_std < 1e-6, torch.ones_like(y_std), y_std)
    ytrain_z = (ytrain - y_mean) / y_std


    train_loader = DataLoader(TensorDataset(Xtrain, ytrain_z), batch_size=8, shuffle=True, num_workers= 20)
    val_loader   = DataLoader(TensorDataset(Xval, yval), batch_size=8, shuffle=False)
    test_loader  = DataLoader(TensorDataset(Xtest, ytest), batch_size=120, shuffle=False)
    test_loaderraw  = DataLoader(TensorDataset(Xtestraw, ytestraw), batch_size=120, shuffle=False)

 
    if model_name == 'vgg':
        model, r2= train_pipeline(lambda2_list=np.logspace(-4, 2, num=5) , lrtouselist=np.logspace(-7, 0, num=5), 
                                pixels =28,
                                allfilters = [512])
    
    if model_name == 'resnetrobust':
        model, r2= train_pipeline(lambda2_list=np.logspace(-4, 2, num=5) , lrtouselist=np.logspace(-7, 0, num=5), 
                                pixels =14,
                                allfilters = [511, 2000, 64])

    if model_name == 'resnet':
        model, r2= train_pipeline(lambda2_list=np.logspace(-4, 2, num=5) , lrtouselist=np.logspace(-7, 0, num=5), 
                                pixels =7,
                                allfilters = [1024])
    if model_name == 'alexnet':
        model, r2= train_pipeline(lambda2_list=np.logspace(-4, 2, num=5) , lrtouselist=np.logspace(-7, 0, num=5), 
                                pixels =13,
                                allfilters = [384])
    sessions_r2s.append((session, r2))
    print(session)


print(sessions_r2s)










