import numpy as np
import os
from scipy.spatial.distance import cdist


def get_all_compact_responses_together(root_dir ):

    # num_neurons, num_images, feature_dim = 219, 100, 512 ## if you are using CLIP embeddings
    num_neurons, num_images, feature_dim = 219, 100, 34 ## if you are using interpretable embeddings

    F_max_all = np.zeros((num_neurons, num_images, feature_dim))
    for n, filename in enumerate(os.listdir(root_dir)):
        arr = np.load(f'{root_dir}{filename}')
        if arr.shape == (feature_dim, num_images):
            arr = arr.T
        
        F_max_all[n] = arr
    
    return F_max_all



max_features = get_all_compact_responses_together(f'./pref_from_500k/')
print(max_features.shape) ## (219, 100, 34)
    

min_features = get_all_compact_responses_together(f'./anti_pref_from_500k/')
print(min_features.shape) ## (219, 100, 34)



def distance_from_one_to_other(F_all, F_picked,selected_neuron):
    N, T, D = F_all.shape

    D_min = np.zeros(N)
    Xi = F_picked
    for j in range(N):
        Xj = F_all[j] 
    
        dist_mat = cdist(Xi, Xj, metric='euclidean')  
        D_min[j] = np.min(dist_mat)

    D_min[selected_neuron] = np.inf   

    return D_min

def compute_overlap(x_max_test, x_min_test, x_max_subsample, x_min_subsample, select_neuron, k=5 ):
    distance_max = distance_from_one_to_other(F_all=x_max_subsample, F_picked=x_max_test,selected_neuron=select_neuron)
    distance_min = distance_from_one_to_other(F_all=x_min_subsample, F_picked=x_min_test,selected_neuron=select_neuron)

    sorted_distance_max = (np.argsort(distance_max))[:k]
    sorted_distance_min = (np.argsort(distance_min))[:k]
    overlap = np.mean(np.intersect1d(sorted_distance_max, sorted_distance_min).size / k)

    return overlap, sorted_distance_max, sorted_distance_min


def shuffle_function(x_max_subsample, x_min_subsample, shuffle_mode):
    N, T, D = x_max_subsample.shape

    if shuffle_mode == 'within_shuffle_case':
        
        x_max_subsample_temp = x_max_subsample.copy()
        x_min_subsample_temp = x_min_subsample.copy()

        r = np.random.permutation(N)
        r = r[:int(np.floor(N/2))]
        x_max_subsample_temp[r] = x_min_subsample[r] 
        x_min_subsample_temp[r] = x_max_subsample[r]

    if shuffle_mode == 'across_shuffle_case':
        random_perm = np.random.permutation(N)
        random_perm_another = np.random.permutation(N)
        x_max_subsample_temp = x_max_subsample.copy()
        x_min_subsample_temp = x_min_subsample.copy()

        x_max_subsample_temp =x_max_subsample[random_perm,:,:] 
        x_min_subsample_temp=x_min_subsample[random_perm_another,:,:] 

    return x_max_subsample_temp, x_min_subsample_temp


def full_shuffle_function(x_max_subsample, x_min_subsample):
    x_max_subsample_temp, x_min_subsample_temp= shuffle_function(x_max_subsample, x_min_subsample, shuffle_mode='within_shuffle_case')

    full_x_max_subsample_temp, full_x_min_subsample_temp= shuffle_function(x_max_subsample_temp, x_min_subsample_temp, shuffle_mode='across_shuffle_case')

    return full_x_max_subsample_temp, full_x_min_subsample_temp


all_neurons, _,_ = max_features.shape
k =10

all_overlap = []
shuffle_within = []
shuffle_across = []
shuffle_full = []
for i_neuron in range(all_neurons):
    
    x_pref_test=max_features[i_neuron,:,:] ## one neuron's preferred features
    x_antipref_test=min_features[i_neuron,:,:] ## one neuron's anti-preferred features

    x_pref_subsample_features = max_features.copy() ## all neuron features for preferred
    x_antipref_subsample_features = min_features.copy() ## all neuron features for anti-preferred
   
    x_pref_subsample_features[i_neuron, :,:] = np.inf ## set the selected neuron's features in pref. to infinity
    x_antipref_subsample_features[i_neuron, :,:] = np.inf ## set the selected neuron's features in anti-pref. to infinity

 
    x_pref_subsample_features_within, x_antipref_subsample_features_within = shuffle_function(x_max_subsample=x_pref_subsample_features
                                                                                ,x_min_subsample= x_antipref_subsample_features,
                                                                                shuffle_mode='within_shuffle_case')

    x_pref_subsample_features_full, x_antipref_subsample_features_full = full_shuffle_function(x_max_subsample=x_pref_subsample_features
                                                                                ,x_min_subsample= x_antipref_subsample_features)

    x_pref_subsample_features_across, x_antipref_subsample_features_across = shuffle_function(x_max_subsample=x_pref_subsample_features
                                                                                ,x_min_subsample= x_antipref_subsample_features,
                                                                                shuffle_mode='across_shuffle_case')


    print('........',x_pref_test.shape)
    overlap,_,_ = compute_overlap(x_max_test=x_pref_test, 
                              x_min_test=x_antipref_test, 
                              x_max_subsample=x_pref_subsample_features, 
                              x_min_subsample=x_antipref_subsample_features, 
                              select_neuron=i_neuron,
                              k=k)
    
    overlap_full,_,_ = compute_overlap(x_max_test=x_pref_test, 
                              x_min_test=x_antipref_test, 
                              x_max_subsample=x_pref_subsample_features_full, 
                              x_min_subsample=x_antipref_subsample_features_full, 
                              select_neuron=i_neuron,
                              k=k)
    
    overlap_across,_,_ = compute_overlap(x_max_test=x_pref_test, 
                              x_min_test=x_antipref_test, 
                              x_max_subsample=x_pref_subsample_features_across, 
                              x_min_subsample=x_antipref_subsample_features_across, 
                              select_neuron=i_neuron,
                              k=k)
    
    overlap_within,_,_ = compute_overlap(x_max_test=x_pref_test, 
                              x_min_test=x_antipref_test, 
                              x_max_subsample=x_pref_subsample_features_within, 
                              x_min_subsample=x_antipref_subsample_features_within, 
                              select_neuron=i_neuron,
                              k=k)

    all_overlap.append(overlap)
    shuffle_within.append(overlap_within)
    shuffle_across.append(overlap_across)
    shuffle_full.append(overlap_full)
    



