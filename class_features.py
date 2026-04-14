import os
from os import walk
import numpy as np
import tensorflow as tf
import io
import sys
gpu_device = sys.argv[1]
print('using gpu ' + gpu_device)
import os
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_device
from tensorflow.keras.layers import AveragePooling2D, GlobalAveragePooling2D

from tensorflow.keras import backend as K

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_resnet50

from tensorflow.keras.applications.nasnet import NASNetMobile
from tensorflow.keras.applications.nasnet import preprocess_input as preprocess_input_nasnetmobile

from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, Flatten, Dense
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

from tensorflow.keras.models import load_model

from tensorflow.keras import layers
from PIL import Image


class FeaturesClass:

 
### INIT FUNCTION

	def __init__(self, for_prototyping_flag=False):
		# DOCUMENT
		# INPUT:
		#	for_prototyping_flag: (True or False), if True, Resnet embedding features will be (num_images,num_channels)...it only considers central pixel (for speed)
		#				if False, takes 3x3 downsample of ResNet features

		# do not load model on onset---b/c we have to call a bunch of classes with different models
		self.model = []

		self.for_prototyping_flag = for_prototyping_flag


### CALL FUNCTIONS

	def load_model(self, layer_id='chosen', weights='imagenet'):
		
		K.clear_session()  # clear the session

		del self.model

		self.model = ResNet50(weights=weights)
	
		input_shape = (112,112, 3)

		
		self.model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
	
		if layer_id == 'chosen': 
			layer_id = 'conv4_block4_out'
		
		x = self.model.get_layer(layer_id).output

		print(f'X is {x.shape}')
		x =  GlobalAveragePooling2D()(x)   
	
		print(f'X is {x.shape}')
		x = Flatten()(x)
		self.model = Model(inputs=self.model.input, outputs=x)
		


	def get_features_from_imgs(self, imgs_raw):
		# assume imgs are raw (no preprocessing)
		# Make sure you call load_model first!
		# returns features (num_features, num_images)
		# DOCUMENT
		# INPUT:
		# 
		# OUTPUT:
		#	features: (num_feature_vars, num_images), embedding features from ResNet50

		imgs = np.copy(imgs_raw)
		imgs = preprocess_input_resnet50(imgs)
		# print(imgs.shape)

		features = self.model.predict(imgs)
		# print(features.shape)

		return features.T # return features as (num_features, num_images) array






