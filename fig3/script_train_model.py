import numpy as np  
import os
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import scipy.ndimage as ndimage
import zipfile
import glob
import re
import torch
import sys
import time
import gc
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.nn.functional as F
import class_dataloader
import class_compact_model
import re
torch.backends.cudnn.benchmark = True


BATCH = 64
EPOCHS = 8
LEARNING_RATE = 1e-4
TOTAL_NUMBER_OF_IMAGES = 500000

compact_model= int(sys.argv[1])
gpu_number = int(sys.argv[2])
picked_removal_type = (sys.argv[3])

if picked_removal_type == '0':
    list_of_removal_types = ['random']
if picked_removal_type == '1':
    list_of_removal_types = ['two_tail']
if picked_removal_type == '2':
    list_of_removal_types = ['maximum']
if picked_removal_type == '3':
    list_of_removal_types = ['minimum']
if picked_removal_type == '4':
    list_of_removal_types = ['middle']

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


version_num = 'FILE_NAME'
folder_path= './compact_models_final/saved_models/'
all_compact_models = sorted([f for f in os.listdir(folder_path) if f.endswith('.keras')])
all_compact_models = all_compact_models[int(compact_model):]

print(all_compact_models)
image_dir = './500k_natural_images/'
response_dir = f'./{version_num}_500k_compact_responses/'

test_image_npy = './20k_natural_images/batch_0.npy'
test_response_npy = f'./{version_num}_20k_compact_responses/20k_responses_batch_0.npy'

r2_directory = './r2_scores/'
random_index_path = f'./npy_files/random_idx_to_use{picked_removal_type}_{all_compact_models[0][:-6]}.npy'
sorted_indx_path =f'./npy_files/{picked_removal_type}_sorted_indices_compact_{all_compact_models[0][:-6]}.npy'

def sort_images_responses(response_dir = response_dir, dataset_500k= True):

    response_files = sorted(glob.glob(os.path.join(response_dir, '500k_responses_batch_*.npy')))
   
    responses = []
    for resp_file in response_files:
        responses.append(np.load(resp_file, mmap_mode='r')) 
    responses = np.concatenate(responses, axis=0)
    sorted_indices = np.argsort(responses.squeeze())

    np.save(f'{sorted_indx_path}', sorted_indices)
    

def std_sort_images_responses(response_dir = response_dir):

    response_files = sorted(glob.glob(os.path.join(response_dir, '500k_responses_batch_*.npy')))
   
    responses = []
    for resp_file in response_files:
        responses.append(np.load(resp_file, mmap_mode='r')) 
    responses = np.concatenate(responses, axis=0)

    standard_dev = np.std(responses)
    return standard_dev



def filename_to_int(filename):
    match = re.search(r'session(\d+)_neuron(\d+)', filename)
    if match:
        session, neuron = match.groups()
       
        return int(session + neuron)  
    return None  

random_numbers = (np.random.choice(500000, size=500000, replace=False))
np.save(f'{random_index_path}',random_numbers)
del random_numbers

gc.collect()  

list_of_images_to_remove = np.arange(0,300000,100000)
ratio_num = 0.9

for picked_var_neuron in all_compact_models:
    compact_model_number = filename_to_int(picked_var_neuron)

    sort_images_responses(response_dir = response_dir)

    seed_to_use  =271828 +int(compact_model_number)
    list_of_seeds  =[seed_to_use]
    sigma_to_use = std_sort_images_responses(response_dir = response_dir)

    for type_of_image_to_remove in list_of_removal_types:
        
        for current_seed in list_of_seeds:
            torch.manual_seed(current_seed)
            np.random.seed(current_seed) 

            for number_of_images_being_removed in list_of_images_to_remove:
                if number_of_images_being_removed == 0:
                    full_dataset = class_dataloader.zero_dataset(
                    sorting_indices_pre= f'{sorted_indx_path}',
                    image_directory=image_dir, 
                    response_directory=response_dir,
                    sigma_value = sigma_to_use
                    )
                else: 
                    full_dataset = class_dataloader.retrieve_images(image_directory=image_dir, 
                    response_directory=response_dir,  
                    sorting_indices_pre = f'{sorted_indx_path}', 
                    kind_of_images_to_remove =type_of_image_to_remove, 
                    number_of_imgs_to_remove=number_of_images_being_removed,
                    random_idx = random_index_path,
                    ratio_to_use = ratio_num,
                    sigma_value = sigma_to_use )

                loader_train = torch.utils.data.DataLoader(full_dataset, 
                batch_size=BATCH, shuffle = True, num_workers =52, pin_memory=True)
            
                full_dataset_test = class_dataloader.test_images(image_path = test_image_npy,
                response_path= test_response_npy,sigma_value = sigma_to_use)

                full_dataset_val = class_dataloader.val_images(image_path = test_image_npy,
                response_path= test_response_npy,sigma_value = sigma_to_use)


                loader_test_unseen= torch.utils.data.DataLoader(full_dataset_test, 
                batch_size=BATCH, shuffle = False, num_workers = 38)

                loader_val_unseen= torch.utils.data.DataLoader(full_dataset_val, 
                batch_size=BATCH, shuffle = False, num_workers = 38)
            
                image_shape = np.ones([BATCH,3,112,112])
                compact = class_compact_model.Compact_Model_NN(image_shape, 
                                seed=current_seed, image_to_remove = type_of_image_to_remove,
                                number_of_images_to_remove=number_of_images_being_removed, 
                                number_of_samples =TOTAL_NUMBER_OF_IMAGES, 
                                directory_to_save=f'{r2_directory}',
                                compact_model = f'session0_neuron{picked_var_neuron}'
                                )
                
                compact = compact.to(device)
                print(compact.training_loop(compact_model=compact, train_data_loader=loader_train,
                            epochs = EPOCHS, learning_rate = LEARNING_RATE, 
                            test_data_loader = loader_test_unseen, val_data_loader= loader_val_unseen,
                            ratio_to_save = ratio_num ))
                
            
                del full_dataset
                del full_dataset_test
                del full_dataset_val
                del loader_test_unseen
                del loader_train
                del loader_val_unseen
                gc.collect()  
                torch.cuda.empty_cache() 

    break







