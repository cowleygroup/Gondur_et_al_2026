import os
import numpy as np
import torch
import time
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.nn.functional as F
import tensorflow as tf
torch.backends.cudnn.benchmark = True
import time
from tqdm import tqdm
import csv
import os.path
import re
#------------------------------------------------------------------------------------------------------------------------------------------------------
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

def filename_to_int(filename):
    match = re.search(r'session(\d+)_neuron(\d+)', filename)
    if match:
        session, neuron = match.groups()
        return session, neuron 
    return filename  

class Compact_Model_NN(nn.Module):
    '''
    This initializes and trains/tests a compact model

    image_data: the shape of image data we feed into the model
    image_to_remove: different ways of removing images per training, can be 'random','two_tail', 'middle','minimum', 'maximum', 'test'
    number_of_images_to_remove: number of images that we remove for training
    number_of_samples: total number (unbatched) of images we use to train the model
    '''
    def __init__(self, image_data, seed, image_to_remove,number_of_images_to_remove,number_of_samples=250000, directory_to_save='.', compact_model='check'):
        super().__init__()
        self.seed = seed
        self.num_layers = 5
        self.batch_size = image_data.shape[0]
        self.image_size = image_data.shape[1]
        self.flattened_spatial = 78400
        self.img_type_to_remove = image_to_remove
        self.amount_of_images_removed = number_of_images_to_remove
        self.number_of_images = number_of_samples
        self.r2_save_directory = directory_to_save
        self.compact_model = compact_model
        self.mean_rgb =  np.array([116.222, 109.270, 100.381])
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        

        num_filters = 100
        ilayer = 0
        self.layer0 = nn.Conv2d(in_channels=self.image_size, 
                    out_channels=num_filters, 
                    kernel_size=(5, 5), 
                    padding=2)
        
        
        
        self.layer0_batchnorm = nn.BatchNorm2d(num_features=self.layer0.out_channels)
        
        ## layer 2 -------------------------------------
        self.depthwise1 = nn.Conv2d(self.layer0.out_channels, self.layer0.out_channels, kernel_size=(5,5), stride=2, padding=2)
        self.pointwise1 = nn.Conv2d(self.layer0.out_channels, num_filters , kernel_size=1)
        self.layer1_batchnorm = nn.BatchNorm2d(num_features=self.pointwise1.out_channels)

        ## layer 3
        self.depthwise2 = nn.Conv2d(self.pointwise1.out_channels, self.pointwise1.out_channels, kernel_size=(5,5), stride=2, padding=2)
        self.pointwise2 = nn.Conv2d(self.pointwise1.out_channels, num_filters , kernel_size=1)
        self.layer2_batchnorm = nn.BatchNorm2d(num_features=self.pointwise2.out_channels )

        ## layer 4
        self.depthwise3 = nn.Conv2d(self.pointwise2.out_channels, self.pointwise2.out_channels, kernel_size=(5,5), stride=1, padding=2)
        self.pointwise3 = nn.Conv2d(self.pointwise2.out_channels, num_filters , kernel_size=1)
        self.layer3_batchnorm = nn.BatchNorm2d(num_features=self.pointwise3.out_channels )

        ## layer 5
        self.depthwise4 = nn.Conv2d(self.pointwise3.out_channels, self.pointwise3.out_channels, kernel_size=(5,5), stride=1, padding=2)
        self.pointwise4 = nn.Conv2d(self.pointwise3.out_channels, num_filters , kernel_size=1)
        self.layer4_batchnorm = nn.BatchNorm2d(num_features=self.pointwise4.out_channels)

        self.layer5 =nn.Linear(in_features=self.flattened_spatial, out_features=1)

    def recenter_imgs(self,images):
        images = images.to(torch.float32) 
     
        images[:,0,:,:]= images[:,0,:,:] -self.mean_rgb[0]
        images[:,1,:,:]= images[:,1,:,:] - self.mean_rgb[1]
        images[:,2,:,:]= images[:,2,:,:] - self.mean_rgb[2] 
        return images
        


    def training_loop(self, compact_model, train_data_loader, val_data_loader, test_data_loader, 
                    epochs=10, learning_rate=1e-4, patience=2, ratio_to_save=0 ):

        optimizer = torch.optim.Adam(compact_model.parameters(), lr=learning_rate)
        loss = nn.MSELoss()
        scaler = torch.cuda.amp.GradScaler()
        best_r2 = -float('inf')
        best_model_state = None 
        patience_counter = 0
        r2_threshold = 0.989
        headers= None

     
        for epoch in range(epochs):
            start_time = time.time()
            print(f'Starting epoch {epoch + 1}/{epochs} . . .')

            compact_model.train()
            train_loss = 0
            total_batches = len(train_data_loader)

            with tqdm(total=total_batches, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch") as pbar:
                for batch_idx, (training_images, training_responses) in enumerate(train_data_loader):
                    training_images = self.recenter_imgs(training_images)
                    training_images = training_images.to(device)
                    training_responses = training_responses.to(device)

                    optimizer.zero_grad()

                    with torch.cuda.amp.autocast():
                        predicted_response = compact_model(training_images)
                        batch_loss = loss(predicted_response.squeeze(), training_responses.squeeze())

                    scaler.scale(batch_loss).backward()
                    torch.nn.utils.clip_grad_norm_(compact_model.parameters(), max_norm=1.0)

                    scaler.step(optimizer)
                    scaler.update()

                    train_loss += batch_loss.item()

                    pbar.update(1)
                    pbar.set_postfix({"Batch Loss": batch_loss.item()})

            train_loss /= total_batches
            print(f"Epoch {epoch + 1} |  Loss: {train_loss:.4f}")

            val_r2 = self.test_only(compact_model, val_data_loader)
            test_r2 = self.test_only(compact_model, test_data_loader)

            print(f"In epoch {epoch + 1} | Validation R²: {val_r2:.4f}")
            print(f"In epoch {epoch + 1} | Test R²: {test_r2:.4f}")

       
            if not torch.isnan(torch.tensor(val_r2)):
                if val_r2 > best_r2:
                    improvement = val_r2 - best_r2
                    best_r2 = val_r2
                    patience_counter = 0
                    best_model_state = compact_model.state_dict()

                    if val_r2 >= r2_threshold and improvement < 0.01:  
                        break
                else:
                    patience_counter += 1
            else:
                print("Nan")

            if patience_counter >= patience:
                break

            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Epoch {epoch + 1}, {elapsed_time:.2f}s")

        if best_model_state:
            compact_model.load_state_dict(best_model_state)

        test_r2_final = self.test_only(compact_model, test_data_loader)
        if torch.isnan(torch.tensor(test_r2_final)):
            test_r2_final = best_r2  

        print(f"Final Test R2: {test_r2_final:.4f}")
    
        session, neuron = filename_to_int(self.compact_model)
        file_path = f'./{self.r2_save_directory}/session_{session}_neuron_{neuron}_seed_{self.seed}.csv'
        file_exists = os.path.isfile(file_path)
        with open(file_path, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file)
            
            if not file_exists:
                headers = ['neuron', 'category', 'num_images_removed', 'r2_score', 'ratio']
                writer.writerow(headers)
            data_row = [f'session_{session}_neuron_{neuron}', self.img_type_to_remove, self.amount_of_images_removed, (test_r2_final).item(), ratio_to_save]
            writer.writerow(data_row)

    def testing_loop(self, compact_model, test_data_loader):

        total_score = []
        with torch.no_grad():
            for test_images, test_responses in test_data_loader:
                compact_model.eval()
                test_images = self.recenter_imgs(test_images)
           
                test_images = test_images.to(device)
                test_responses = test_responses.to(device)
            
                predicted_response = compact_model.forward(test_images)
           
                predicted_response = predicted_response.reshape(-1)
                test_responses= test_responses.reshape(-1)
          
                stacked = torch.stack((test_responses, predicted_response), dim=0)

                corr_matrix = torch.corrcoef(stacked)

                batch_total_score_base = corr_matrix[0, 1]
                batch_total_score = batch_total_score_base ** 2

                total_score.append(batch_total_score)

            with open(f'./{self.r2_save_directory}/r2_scores_{self.img_type_to_remove}_{self.seed}_{self.compact_model}.txt', 'a') as file:
                total_score_tensor = torch.tensor(total_score)
                file.write(f'r2 score is {torch.mean(total_score_tensor)} - # {self.img_type_to_remove} images removed {self.amount_of_images_removed}\n')

        final_score_tensor = torch.tensor(total_score)
        return torch.mean(final_score_tensor)

    def test_only(self, compact_model, test_data_loader):
      
        total_score = []
        with torch.no_grad():
            for test_images, test_responses in test_data_loader:
                compact_model.eval()
                test_images = self.recenter_imgs(test_images)
       
                test_images = test_images.to(device)
                test_responses = test_responses.to(device)
            
                predicted_response = compact_model.forward(test_images)
 
                predicted_response = predicted_response.reshape(-1)
                test_responses= test_responses.reshape(-1)
           
                stacked = torch.stack((test_responses, predicted_response), dim=0)

                corr_matrix = torch.corrcoef(stacked)

                batch_total_score_base = corr_matrix[0, 1]
                batch_total_score = batch_total_score_base ** 2

                total_score.append(batch_total_score)

        final_score_tensor = torch.tensor(total_score)
        return torch.mean(final_score_tensor)
    
   

    def forward(self, x):
        ## layer 1
        image = self.layer0(x)
        image = F.relu(self.layer0_batchnorm(image))


        ## layer 2
        image = self.depthwise1(image)
        image = self.pointwise1(image)
        image = F.relu(self.layer1_batchnorm(image))

        ## layer 3
        image = self.depthwise2(image)
        image = self.pointwise2(image)
        image = F.relu(self.layer2_batchnorm(image))


        ## layer 4
        image = self.depthwise3(image)
        image = self.pointwise3(image)
        image = F.relu(self.layer3_batchnorm(image))


        ## layer 5
        image = self.depthwise4(image)
        image = self.pointwise4(image)
        image = F.relu(self.layer4_batchnorm(image))


        flatten_layer = nn.Flatten() 
        response = self.layer5(flatten_layer(image))
        return response
      