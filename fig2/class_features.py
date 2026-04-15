
# DOCUMENT
#
# returns features from middle layer of pre-trained DNN from raw images
#
# first load model
# then you can get features

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

from tensorflow.keras import backend as K
# gpus = tf.config.list_physical_devices('GPU')
# print(gpus)
# tf.config.experimental.set_memory_growth(gpus[0], True)
# tf.keras.utils.disable_interactive_logging()

# to see available models for TF 1.14: 
#    https://github.com/tensorflow/docs/tree/r1.14/site/en/api_docs/python/tf/keras/applications

from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input as preprocess_input_vgg19


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

	def __init__(self, for_prototyping_flag=True):
		# DOCUMENT
		# INPUT:
		#	for_prototyping_flag: (True or False), if True, Resnet embedding features will be (num_images,num_channels)...it only considers central pixel (for speed)
		#				if False, takes 3x3 downsample of ResNet features

		# do not load model on onset---b/c we have to call a bunch of classes with different models
		self.model = []

		self.for_prototyping_flag = for_prototyping_flag
		


### CALL FUNCTIONS

	def load_model(self, layer_id='chosen', weights='imagenet' ,taskdriven_DNN =None):
		self.taskdriven_DNN = taskdriven_DNN
		del self.model

		if self.taskdriven_DNN == 'VGG19':
			self.model = VGG19(weights=weights)
			if layer_id == 'chosen': 
				#layer_id = 'block4_conv4' ## (None, 28, 28, 512)
				layer_id = 'block4_pool'####None, 14, 14, 512)
		
		else:
		
			self.model = ResNet50(weights=weights)

			if layer_id == 'chosen': 
				layer_id = 'conv4_block4_add' ## pre-relu
				##layer_id = 'conv4_block4_out' # 'activation_33'
		x = self.model.get_layer(layer_id).output
		
		input_shape = (112,112, 3)
		K.clear_session()  # clear the session


		if self.for_prototyping_flag == True:
			x = AveragePooling2D(pool_size=(5,5), strides=(4,4), padding='valid')(x)

			print(f'Shape of x is {x.shape}')

		else:  
			x = AveragePooling2D(pool_size=(2,2), strides=2, padding='valid')(x)
		
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
		features = self.model.predict(imgs)

		return features.T # return features as (num_features, num_images) array

	def get_features_from_imgs_vgg(self, imgs_raw):
		# assume imgs are raw (no preprocessing)
		# Make sure you call load_model first!
		# returns features (num_features, num_images)
		# DOCUMENT
		# INPUT:
		# 
		# OUTPUT:
		#	features: (num_feature_vars, num_images), embedding features from ResNet50

		imgs = np.copy(imgs_raw)
		imgs =  preprocess_input_vgg19(imgs)

		features = self.model.predict(imgs)

		return features.T # return features as (num_features, num_images) array

