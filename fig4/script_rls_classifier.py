## script used for recursive least squares classifier in Fig.3e-f from Gondur et al., ICLR2026


import numpy as np  
import os
from PIL import Image
import numpy as np  
import io
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.svm import SVC
import class_image_stats
from sklearn.linear_model import LogisticRegression
import clip
import torch


dnn_embeds = True ## if you want to use CLIP feature set this True, if you want 34-dim interpretable features, set it to False

def filename_to_int(filename):
    match = re.search(r'session(\d+)_neuron(\d+)', filename)
    if match:
        session, neuron = match.groups()
        return session, neuron
    return None  



if dnn_embeds:
    def extract_features(image):
        img_np = image.astype(np.uint8)
        img_pil = Image.fromarray(img_np[0])
        img_pil = img_pil.resize((224, 224))
        
        device =  "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        img_pil = preprocess(img_pil).unsqueeze(0).to(device)  

        with torch.no_grad():
            image_features = model.encode_image(img_pil)
        return image_features.squeeze()
else: 

    ImageStatsClass = class_image_stats.ImageStatsClass()

    def extract_features(image):
        color_intensities, color_names = ImageStatsClass.get_color_intensities(image)

        metrics = {
            'Luminance': ImageStatsClass.get_luminance,
            'Contrast': ImageStatsClass.get_contrast,
            'Edge Intensity': ImageStatsClass.get_edge_intensities,
            'Small Curve Intensity': lambda imgs: ImageStatsClass.get_curve_intensities(imgs)[0, :],
            'Small Dot': lambda imgs: ImageStatsClass.get_dot_intensities(imgs)[0, :],
            'Medium Dot': lambda imgs: ImageStatsClass.get_dot_intensities(imgs)[1, :],
            'Large Dot': lambda imgs: ImageStatsClass.get_dot_intensities(imgs)[2, :],
            'Medium Curve Intensity': lambda imgs: ImageStatsClass.get_curve_intensities(imgs)[1, :],
            'Large Curve Intensity': lambda imgs: ImageStatsClass.get_curve_intensities(imgs)[2, :],
            'Spatial Low Freq': lambda imgs: ImageStatsClass.get_spatial_frequencies(imgs)[0],
            'Spatial Med Freq': lambda imgs: ImageStatsClass.get_spatial_frequencies(imgs)[1],
            'Spatial High Freq': lambda imgs: ImageStatsClass.get_spatial_frequencies(imgs)[2],
            'Vertical Orientation': lambda imgs: ImageStatsClass.get_spatial_orientations(imgs)[0],
            'Horizontal Orientation': lambda imgs: ImageStatsClass.get_spatial_orientations(imgs)[1],
            'Positive 45 Orientation': lambda imgs: ImageStatsClass.get_spatial_orientations(imgs)[2],
            'Negative 45 Orientation': lambda imgs: ImageStatsClass.get_spatial_orientations(imgs)[3],
            'X Vertical': lambda imgs: ImageStatsClass.get_line_intensities(imgs)[0],
            'X Horizontal': lambda imgs: ImageStatsClass.get_line_intensities(imgs)[1],
            'X Diagonal Postive 45': lambda imgs: ImageStatsClass.get_line_intensities(imgs)[2],
            'X Diagonal Negative 45': lambda imgs: ImageStatsClass.get_line_intensities(imgs)[3]
        }

        for i, color_name in enumerate(color_names):
                metrics[f'Color Intensity ({color_name})'] = lambda imgs, i=i: ImageStatsClass.get_color_intensities(imgs)[0][i, :]

        features = []
        
        for metric_name, metric_func in metrics.items():
            feature_value1 = metric_func(image)
            feature_value = np.mean(feature_value1)
                
            features.append(feature_value)
        
        features_array = np.array(features).reshape(1, -1)

        return features_array


if dnn_embeds: 
    csv_path = './clip_features_true_labels.csv'
else: 
    csv_path = './our_features_true_labels.csv'
meta_info_path = './all_tasks_info_only_compact.csv'
n_trials = 100
window_size = 20

if dnn_embeds:
    def load_and_process_image(img_path):
        img = Image.open(img_path).convert("RGB").resize((224,224))
        return np.asarray(img) 

else:
    def load_and_process_image(img_path):
        img = Image.open(img_path).convert("RGB").resize((112,112))
        return np.asarray(img) 

feature_labels = [
    'Luminance','Contrast','Edge Intensity','Small Curve Intensity',
    'Small Dot','Medium Dot','Large Dot',
    'Medium Curve Intensity',
    'Large Curve Intensity',
    'Spatial Low Freq',
    'Spatial Med Freq',
    'Spatial High Freq',
    'Vertical Orientation',
    'Horizontal Orientation',
    'Positive 45 Orientation',
    'Negative 45 Orientation', 'X Vertical','X Horizontal',
    'X Diagonal Postive 45','X Diagonal Negative 45',
    'white', 'red', 'orange', 'yellow', 'chartreuse_green', 'green', 'spring_green', 'cyan', 'azure', 'blue', 'violet', 'magenta', 'rose', 'black']




with open('./userX/itask_sequence.txt', 'r') as f:
    list_of_all_tasks  = [line.strip() for line in f if line.strip() != '']

final_u_dictionary = {}


for current_task in list_of_all_tasks:
    print(f"\nRunning task: {current_task}")
    
    max_dir = f"./task_data/{current_task}/images_max"
    min_dir = f"./task_data/{current_task}/images_min"
    responses_max = np.load(f"./task_data/{current_task}/responses_max.npy")
    responses_min = np.load(f"./task_data/{current_task}/responses_min.npy")
    binned_response = np.load(f"./task_data/{current_task}/bin_numbers_delta_response.npy")

    df_info = pd.read_csv(meta_info_path)
    task_info = df_info[df_info["folder_name"] == current_task]
    model_type = task_info['imodel'].values[0] 
    neuron_number = task_info['iunit'].values[0]
    prior_label = task_info['iprior'].values[0]


    reference_dir = f"./task_data/{current_task}/references"
    use_reference = False
    sum_pos_ref, sum_neg_ref = None, None

    try:
        reference_files = sorted([f for f in os.listdir(reference_dir) if f.endswith('.png')])
        use_reference = (len(reference_files) > 0)
        reference_features = []
        reference_labels_list = []

        if use_reference:
            responses_max_min = []
            for i, ref_file in enumerate(reference_files):
                ref_path = os.path.join(reference_dir, ref_file)
                ref_img = load_and_process_image(ref_path)
                f_ref = extract_features(np.expand_dims(ref_img, axis=0))
                # print(f_ref.shape)
                reference_features.append(f_ref)
                if prior_label == 'maxprior':
                    reference_labels_list.append(1) 
                elif prior_label == 'minprior':
                    reference_labels_list.append(0) 
                elif prior_label == 'maxminprior':
                    match = re.search(r'response_([-\d.]+)\.png', ref_file)
                    val = float(match.group(1)) if match else 0
                    responses_max_min.append((i, ref_file, val))
            
            reference_features = np.vstack(reference_features)

            if prior_label == 'maxminprior':
                reference_labels = [None] * len(reference_features)
                responses_max_min.sort(key=lambda x: x[2])  
             
                for item in responses_max_min[:18]:
                    orig_idx = item[0] 
                    reference_labels[orig_idx] = 0
               
                for item in responses_max_min[-18:]:
                    orig_idx = item[0]  
                    reference_labels[orig_idx] = 1
           
                reference_labels = np.array(reference_labels)
                pos_mask = reference_labels == 1
                neg_mask = reference_labels== 0
                pos_feats = reference_features[pos_mask]
                neg_feats = reference_features[neg_mask]

                sum_pos_ref = pos_feats.sum(axis=0)
                sum_neg_ref = neg_feats.sum(axis=0)
             
                u = (sum_pos_ref - sum_neg_ref)
              
            elif prior_label == 'maxprior':
                u = np.mean(reference_features, axis=0)

                sum_pos_ref = np.mean(reference_features, axis=0)
                sum_neg_ref = np.zeros_like(sum_pos_ref) 
            
            elif prior_label == 'minprior':
                u = -np.mean(reference_features, axis=0)
                sum_neg_ref = np.mean(reference_features, axis=0)
                sum_pos_ref = np.zeros_like(sum_neg_ref) 
        
        else:
            if dnn_embeds:
                feature_dim = 512
            else:
                feature_dim = 34
            u = np.random.randn(feature_dim)
            sum_pos_ref = np.zeros(feature_dim)
            sum_neg_ref = np.zeros(feature_dim)
    except:
        if dnn_embeds:
            feature_dim = 512
        else:
            feature_dim = 34
        u = np.random.randn(feature_dim)
        use_reference = False


    X_diff = []
    y_diff = []
    f_max_list = []
    f_min_list = []
    raw_max_list = []
    raw_min_list = []

    for i in range(100):
        f_max =extract_features(np.expand_dims(load_and_process_image(os.path.join(max_dir, f"image{i:03}.png")), axis=0))
        f_min = extract_features(np.expand_dims(load_and_process_image(os.path.join(min_dir, f"image{i:03}.png")), axis=0))
    
    
        true_max_image = np.expand_dims(load_and_process_image(os.path.join(max_dir, f"image{i:03}.png")), axis=0)
        true_min_image = np.expand_dims(load_and_process_image(os.path.join(min_dir, f"image{i:03}.png")), axis=0)
        raw_max_list.append(true_max_image)
        raw_min_list.append(true_min_image)

        f_max_list.append(f_max.flatten())
        f_min_list.append(f_min.flatten())
        x_diff = f_max - f_min
        
        X_diff.append(x_diff)
        y_diff.append(1)


    X_diff = np.vstack(X_diff)
    y_diff = np.array(y_diff)
    X_features_max = np.vstack(f_max_list)
    X_features_min = np.vstack(f_min_list)
    imgs_max_final = np.vstack(raw_max_list)
    imgs_min_final = np.vstack(raw_min_list)
    f_max_array = np.vstack(f_max_list)
    f_min_array = np.vstack(f_min_list)

  
    results = []
    y_true, y_pred = [], []
    w = u
    P = np.eye(f_max_array.shape[-1]) * 1       
    forgetting_factor = 1
    dot_products = []
    dot_by_bin = []
    for i in range(100):
        print(f'Prior label {prior_label}')
        x_max = f_max_array[i]
        x_min = f_min_array[i]

        if np.random.random() < 0.5:
            x_i = (x_max- x_min)
            x_i = x_i / (np.linalg.norm(x_i) + 1e-10)
            y_i = 1
        else:
            x_i = (x_min- x_max)
            x_i = x_i / (np.linalg.norm(x_i) + 1e-10)
            y_i = 0
    
        y_centered = 2 * y_i - 1        

        pred_score = np.dot(w, x_i)
        pred = 1 if pred_score > 0 else 0
        y_pred.append(pred)
        dot_products.append(np.dot(w, x_i))
        
      
        Px = P @ x_i
        denom = forgetting_factor + x_i.T @ Px
        k = Px / denom

       
        error = y_centered - pred_score

        w += k * error
        P = (P - np.outer(k, x_i.T @ P)) / forgetting_factor

     
        correct = (pred == y_i)
        print(f"Trial {i}: Pred={pred}, Correct={correct}, Error={error:.4f}, Bin={binned_response[i]}")
        
        current_bin = binned_response[i]
        dot_by_bin.append((current_bin,np.dot(w, x_i)))

        results.append([
            'userX', correct, y_i, pred, neuron_number,
            current_bin, i, model_type, current_task, prior_label
        ])

        df = pd.DataFrame([results[-1]], columns=[
            "User", "Correct", "True Response", "Predicted Response",
            "Neuron", "BinDelta", "Trial", "Model", "TaskFolder", "PriorType"
        ])
        df.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path) or os.stat(csv_path).st_size == 0, index=False)

 

 

