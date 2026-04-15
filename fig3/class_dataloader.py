import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets, transforms
import glob

#------------------------------------------------------------------------------------------------------------------------------------------------------
np.random.seed(10) 


def sequential_search_random(indices, image_files, batch_size=20000):

    first_image = np.load(image_files[0], mmap_mode='r')
    result = np.zeros((len(indices),) + first_image[0].shape, dtype=first_image.dtype) ## indices_size, 112, 112, 3

    found_indices = set()  
    index_map = {idx: i for i, idx in enumerate(indices)}  

    for file_idx, file_path in enumerate(image_files):

        start_idx = file_idx * batch_size 
        end_idx = start_idx + batch_size

        relevant_indices = [idx for idx in indices if start_idx <= idx < end_idx] 

        if relevant_indices:
            data = np.load(file_path, mmap_mode='r')

            for idx in relevant_indices:
                local_idx = idx - start_idx
                result_idx = index_map[idx]
                result[result_idx] = data[local_idx]
                found_indices.add(idx)
             
        
        if len(found_indices) == len(indices):
            break
    print(len(found_indices))
    return result


class retrieve_images(Dataset):
 
    def __init__(self, random_idx,ratio_to_use,image_directory, 
    response_directory, sigma_value, sorting_indices_pre, kind_of_images_to_remove, 
    number_of_imgs_to_remove=None,transform=None, target_transform=None ):
        image_files = sorted(glob.glob(os.path.join(image_directory, 'batch_*.npy')))
      
        response_files = sorted(glob.glob(os.path.join(response_directory, '500k_responses_batch_*.npy')))
        sorting_indices = np.load(sorting_indices_pre, mmap_mode='r')
     
        self.number_of_imgs_to_remove = number_of_imgs_to_remove
        self.transform = transform
        self.target_transform = target_transform
        self.proportion = ratio_to_use
        self.number_of_images_to_keep = 500000 - self.number_of_imgs_to_remove
        self.sigma = sigma_value
        
  
       

        if kind_of_images_to_remove== 'random': 
            random_sample_from_response_distribution = np.random.choice(sorting_indices, size=500000-(self.number_of_imgs_to_remove), replace=False)

            images =sequential_search_random(random_sample_from_response_distribution,image_files)
            responses =sequential_search_random(random_sample_from_response_distribution,response_files)
         
            self.images = torch.from_numpy(np.squeeze(images))
            self.responses = torch.from_numpy(np.squeeze(responses))

       
        elif kind_of_images_to_remove== 'minimum':
            amount_for_minimum = int(self.number_of_images_to_keep * self.proportion)
            amount_for_random = int(self.number_of_images_to_keep - amount_for_minimum)
    
            images_minimum = sorting_indices[:amount_for_minimum]
            images_random_pool =  np.setdiff1d(sorting_indices, images_minimum)
            images_random = np.random.choice(images_random_pool, size=amount_for_random, replace=False)
           
            if amount_for_random == 0:
                all_indices_to_use = images_minimum
                images = sequential_search_random(all_indices_to_use,image_files)
                responses = sequential_search_random(all_indices_to_use,response_files)
            else:
                images_min = sequential_search_random(images_minimum,image_files)
                images_rand = sequential_search_random(images_random,image_files)
                images = np.concatenate((images_min, images_rand), axis=0)

                response_min = sequential_search_random(images_minimum,response_files)
                response_rand = sequential_search_random(images_random,response_files)
                responses = np.concatenate((response_min, response_rand), axis=0)

            self.images = torch.from_numpy(np.squeeze(images))
            self.responses = torch.from_numpy(np.squeeze(responses))
        
        elif kind_of_images_to_remove== 'maximum': ### 2
            amount_for_maximum = int(self.number_of_images_to_keep * self.proportion)
            amount_for_random = int(self.number_of_images_to_keep - amount_for_maximum)

            images_maximum = sorting_indices[-(amount_for_maximum):]
            images_random_pool =  np.setdiff1d(sorting_indices, images_maximum)
            images_random = np.random.choice(images_random_pool, size=amount_for_random, replace=False)
           
            if amount_for_random == 0:
                all_indices_to_use = images_maximum
                images = sequential_search_random(all_indices_to_use,image_files)
                responses = sequential_search_random(all_indices_to_use,response_files)
            else:
                images_max = sequential_search_random(images_maximum,image_files)
                images_rand = sequential_search_random(images_random,image_files)
                images = np.concatenate((images_max, images_rand), axis=0)

                response_max = sequential_search_random(images_maximum,response_files)
                response_rand = sequential_search_random(images_random,response_files)
                responses = np.concatenate((response_max, response_rand), axis=0)

            self.images = torch.from_numpy(np.squeeze(images))
            self.responses = torch.from_numpy(np.squeeze(responses))


        elif kind_of_images_to_remove== 'two_tail': ### 1
            amount_for_maximum_and_minimum = int(self.number_of_images_to_keep * self.proportion)
            amount_for_maximum = amount_for_maximum_and_minimum //2
            amount_for_minimum = amount_for_maximum_and_minimum //2
            amount_for_random = int(self.number_of_images_to_keep - amount_for_maximum_and_minimum)

            images_maximum = sorting_indices[-(amount_for_maximum):]
            images_minimum = sorting_indices[:amount_for_minimum]
            images_max_min_combined = np.concatenate([images_maximum, images_minimum])
            images_random_pool =  np.setdiff1d(sorting_indices, images_max_min_combined)
            images_random = np.random.choice(images_random_pool, size=amount_for_random, replace=False)
           

            if amount_for_random == 0:
                all_indices_to_use = np.concatenate([images_maximum, images_minimum])
                images = sequential_search_random(all_indices_to_use,image_files)
                responses = sequential_search_random(all_indices_to_use,response_files)
            else:
                images_max = sequential_search_random(images_maximum,image_files)
                images_min = sequential_search_random(images_minimum,image_files)
                images_rand = sequential_search_random(images_random,image_files)
                images = np.concatenate((images_max, images_min, images_rand), axis=0)
         
                response_max = sequential_search_random(images_maximum,response_files)
                response_min = sequential_search_random(images_minimum,response_files)
                response_rand = sequential_search_random(images_random,response_files)
                responses = np.concatenate((response_max, response_min, response_rand), axis=0)

            self.images = torch.from_numpy(np.squeeze(images))
            self.responses = torch.from_numpy(np.squeeze(responses))

        elif kind_of_images_to_remove== 'middle': ### 4
            amount_for_middle_in_general_before_each_side = int(self.number_of_images_to_keep * self.proportion) 
            amount_for_middle = amount_for_middle_in_general_before_each_side//2 
            amount_for_random = int(self.number_of_images_to_keep -amount_for_middle_in_general_before_each_side)

            middle_of_the_sorted_list = len(sorting_indices) // 2
            images_middle = sorting_indices[middle_of_the_sorted_list-amount_for_middle:middle_of_the_sorted_list+amount_for_middle]
            images_random_pool =  np.setdiff1d(sorting_indices, images_middle)
            images_random = np.random.choice(images_random_pool, size=amount_for_random, replace=False)
           
            if amount_for_random == 0:
                all_indices_to_use = images_middle
                images = sequential_search_random(all_indices_to_use,image_files)
                responses = sequential_search_random(all_indices_to_use,response_files)
            else:
                images_mid = sequential_search_random(images_middle,image_files)
                images_rand = sequential_search_random(images_random,image_files)
                images = np.concatenate((images_mid, images_rand), axis=0)

                
                response_middle = sequential_search_random(images_middle,response_files)
                response_rand = sequential_search_random(images_random,response_files)
                responses = np.concatenate((response_middle, response_rand), axis=0)
                
            self.images = torch.from_numpy(np.squeeze(images))
            self.responses = torch.from_numpy(np.squeeze(responses))
     
   
    def __len__(self):
        return len(self.responses)

    def __getitem__(self, idx): 
       
        images = self.images[idx]
        responses = (self.responses[idx])
        
        images = np.transpose(images, (2, 0, 1))

        epsilon = torch.randn_like(responses)  
        responses_train = responses + 0.2 * self.sigma * epsilon 
    
        if self.transform:
            images = self.transform(images)
        if self.target_transform:
            label = self.target_transforma(label)

        return images, responses_train


class val_images(Dataset):
 
    def __init__(self, image_path,response_path, sigma_value,transform=None, target_transform=None):
        
        self.images = np.load(image_path, allow_pickle=True)[10000:]
  
        self.responses = np.load(response_path, allow_pickle=True)[10000:]

        
        self.transform = transform
        self.target_transform = target_transform
        self.sigma = sigma_value


    def __len__(self):
        return len(self.responses)

    def __getitem__(self, idx): 
        image = self.images[idx]
        response = self.responses[idx]

        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) 
        response = torch.tensor(response, dtype=torch.float32)

        epsilon = torch.randn_like(response)  
        responses_train = response + 0.2 * self.sigma * epsilon 

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            response = self.target_transform(response)

        return image, responses_train



class test_images(Dataset):
 
    def __init__(self, image_path,response_path,sigma_value, transform=None, target_transform=None):
        
        self.images = np.load(image_path, allow_pickle=True)[:10000]
  
        self.responses = np.load(response_path, allow_pickle=True)[:10000]
     
    
        self.transform = transform
        self.target_transform = target_transform
        self.sigma = sigma_value
     
    def __len__(self):
        return len(self.responses)

    def __getitem__(self, idx): 
        image = self.images[idx]
        response = self.responses[idx]

        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) 
        response = torch.tensor(response, dtype=torch.float32)

        epsilon = torch.randn_like(response)  
        responses_train = response + 0.2 * self.sigma * epsilon 
     
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            response = self.target_transform(response)

        return image, responses_train
