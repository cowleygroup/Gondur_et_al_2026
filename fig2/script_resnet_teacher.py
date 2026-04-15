import os
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import zipfile
import sys
from tensorflow.keras import backend as K
import time
import class_linear_mapping_ensemble_ridgereg
from sklearn.model_selection import train_test_split

gpu_number = str(sys.argv[1])
gpu_device = gpu_number
print('using gpu ' + gpu_device)
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_device
gpus = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)



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




all_sessions =  ['190923', '211022', '210225', '201025' ]

all_data = []
all_results = []
for session in all_sessions:
    v4_neural_responses = np.load(f'./V4_neural_data/responses_repeat_averaged/responses_{session}.npy').T
    image_dir = "./V4_neural_data/images/"
    image_zipfiles = zipfile.ZipFile(f'{image_dir}images_{session}.zip' )
    images = (get_images_from_zip(directory_path=image_zipfiles))

    def predict_neural_response(layers_to_use = ['conv4_block4_add'], image_data=images, responses=v4_neural_responses, type_of_relu='none'):
        
        all_r2s = []
        tuple_r2s = []
        for layer in layers_to_use:
            R2s_over_sessions = []
            X = np.load(f'./embeddings.npy', allow_pickle=True)

            print('Starting . . . ')
            start_time = time.time()
            batch_predictions_before_transpose_pre = X
        
            LM = class_linear_mapping_ensemble_ridgereg.LinearMappingClass()
            
            batch_predictions_after_transpose =batch_predictions_before_transpose_pre.reshape(batch_predictions_before_transpose_pre.shape[0],-1)
            if type_of_relu == 'none':
                embeds = batch_predictions_after_transpose
            if type_of_relu == 'neg_regular_relu':
                embeds_pre = batch_predictions_after_transpose
                embeds = np.maximum(0, -embeds_pre)
            if type_of_relu == 'regular_relu':
                embeds_pre = batch_predictions_after_transpose
                embeds = np.maximum(0, embeds_pre)
           

            Xtrain, Xtest_temp, Ytrain, Ytest_temp = train_test_split(
            embeds, responses, test_size=0.2, random_state=314)

            Xtest, Xval, Ytest, Yval = train_test_split(
            Xtest_temp, Ytest_temp, test_size=0.5, random_state=314)

            y_mean = Ytrain.mean(axis=0, keepdims=True)
            y_std  = Ytrain.std(axis=0, keepdims=True) + 1e-8  

            ytrain_z = (Ytrain - y_mean) / y_std

            alphanum = LM.choose_alpha(Xtrain, ytrain_z)
            Y_hat_val, Y_hat_test, Yhat_train, ridgecoef, ridgeinter =LM.get_ridge_regression(Xtrain, ytrain_z, Xval, Xtest, alpha=alphanum)
           
            v4_neural_responses_raw = (np.load(f'./V4_neural_data/responses_raw/responses_{session}.npy'))
            v4_neural_responses_raw  = np.transpose(v4_neural_responses_raw , (1, 0, 2)) 
            Xtrain, Xtest_temp, Ytrain, Ytest_temp = train_test_split(
            embeds,  v4_neural_responses_raw, test_size=0.2, random_state=314)

            Xtest, Xval, Ytest, Yval = train_test_split(
            Xtest_temp, Ytest_temp, test_size=0.5, random_state=314)
            
            Ytest =  np.transpose( Ytest  , (1,0,2)) 
            R2s_over_sessions = LM.compute_r2_ER(Ytest, Y_hat_test)

        
            print(f'medianR2 = {np.median(R2s_over_sessions)}')
            all_r2s.append(np.median(R2s_over_sessions))
            tuple_r2s.append((layer, np.median(R2s_over_sessions)))
       
        return np.median(R2s_over_sessions), R2s_over_sessions
   
    relu, R2s_over_sessions = predict_neural_response(type_of_relu='none')

    all_results.append((session, np.array(R2s_over_sessions)))
    
print(all_results)








