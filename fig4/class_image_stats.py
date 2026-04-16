
# This class contains a number of statistics to describe
#  the contents of an image (luminance, contrast, spatial frequency, ...).

# assumes input images are 112 x 112 x 3  (num_pixels,num_pixels,rgb)


import numpy as np
from scipy import ndimage
from skimage.measure import shannon_entropy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class ImageStatsClass:

	def __init__(self):
		print('IMAGES SHOULD BE 112x112x3!!!')
		pass


	
	def get_all_visual_features(self,imgs):
		# DOCUMENT ME!

		v_features = []; feature_names = []

		x = self.get_luminance(imgs)
		v_features.append(x)
		feature_names.append('luminance')

		x = self.get_contrast(imgs)
		v_features.append(x)
		feature_names.append('contrast')

		colors_ints, color_names = self.get_color_intensities(imgs)
		for icolor in range(colors_ints.shape[0]):
			v_features.append(colors_ints[icolor])
			feature_names.append(color_names[icolor])

		x_low, x_mid, x_high = self.get_spatial_frequencies(imgs)
		v_features.append(x_low)
		feature_names.append('low_spatial_freq')
		v_features.append(x_mid)
		feature_names.append('mid_spatial_freq')
		v_features.append(x_high)
		feature_names.append('high_spatial_freq')

		x_vert, x_horiz, x_diag45, x_diagneg45 = self.get_spatial_orientations(imgs)
		v_features.append(x_vert)
		feature_names.append('ori_vert')
		v_features.append(x_horiz)
		feature_names.append('ori_horiz')
		v_features.append(x_diag45)
		feature_names.append('ori_diag45')
		v_features.append(x_diagneg45)
		feature_names.append('ori_diag-45')

		x = self.get_edge_intensities(imgs)
		v_features.append(x)
		feature_names.append('edge_intensities')

		x_vert, x_horiz, x_diag45, x_diagneg45 = self.get_line_intensities(imgs)
		v_features.append(x_vert)
		feature_names.append('line_vert')
		v_features.append(x_horiz)
		feature_names.append('line_horiz')
		v_features.append(x_diag45)
		feature_names.append('line_diag45')
		v_features.append(x_diagneg45)
		feature_names.append('line_diag-45')

		curve_ints = self.get_curve_intensities(imgs)
		v_features.append(curve_ints[0])
		feature_names.append('curve_int_small')
		v_features.append(curve_ints[1])
		feature_names.append('curve_int_med')
		v_features.append(curve_ints[2])
		feature_names.append('curve_int_large')

		dot_ints = self.get_dot_intensities(imgs)
		v_features.append(dot_ints[0])
		feature_names.append('dot_int_small')
		v_features.append(dot_ints[1])
		feature_names.append('dot_int_med')
		v_features.append(dot_ints[2])
		feature_names.append('dot_int_large')

		return np.array(v_features)#, feature_names
			# (num_features,num_images),  (num_features,)





	@staticmethod
	def get_luminance(imgs):
		# returns luminance (average intensity)
		#
		# INPUT:
		#	imgs: (num_images,112,112,3), raw images with pixel intensities between 0 and 255
		# OUTPUT:
		#	luminances: (num_images,), luminance (average pixel intensity), between 0 and 1

		luminances = np.mean(imgs,axis=(1,2,3)) / 255.
		return luminances

	@staticmethod
	def get_shannon_entropy(imgs):
		# returns entropy
		#
		# INPUT:
		#	imgs: (num_images,112,112,3), raw images with pixel intensities between 0 and 255
		# OUTPUT:
		#	shannon entropy: (num_images,), -sum(pk*log(pk))

		imgs_grayscale = np.mean(imgs, axis=-1)
		
		shannon_entropy_val = shannon_entropy(imgs_grayscale)

		return shannon_entropy_val
	

	@staticmethod
	def get_contrast(imgs):
		# returns contrast (RMS)
		#
		# INPUT:
		#	imgs: (num_images,112,112,3), raw images with pixel intensities between 0 and 255
		# OUTPUT:
		#	contrasts: (num_images,), root-mean-square contrast

		num_pixels = imgs.shape[1]
		max_std = 128  # given values between 0 and 255

		imgs_grayscale = np.mean(imgs, axis=-1)

		contrasts = np.std(imgs_grayscale, axis=(1,2)) / max_std

		return contrasts


	@staticmethod
	def get_color_intensities(imgs):
		# returns red intensity
		#
		# INPUT:
		#	imgs: (num_images,112,112,3), raw images with pixel intensities between 0 and 255
		# OUTPUT:
		#	color_intensities: (num_colors,num_images), fraction of pixels per image for given colors (see 'color_names')
		#			so color_intensities[:,iimage] will sum to 1
		#	color_names: (list of strings), color names considered, where color_names[icolor] corresponds to color_intensities[icolor,:]

		num_colors = 14
		num_images = imgs.shape[0]
		num_pixels = imgs.shape[1]

		color_names = ['white', 'red', 'orange', 'yellow', 'chartreuse_green', 'green', 'spring_green', 'cyan', 'azure', 'blue', 'violet', 'magenta', 'rose', 'black']
		rgb_values = np.array([
			[255,255,255], # white
			[255,0,0], # red
			[255,127,0], # orange
			[255,255,0], # yellow
			[127,255,0], # chartreuse green
			[0,255,0], # green
			[0,255,127], # spring green
			[0,255,255], # cyan
			[0,127,255], # azure
			[0,0,255], # blue
			[127,0,255], # violet
			[255,0,255], # magenta
			[255,0,127], # rose
			[0,0,0] # black
			])

		distances = np.zeros((num_images,num_pixels, num_pixels, num_colors))

		# compute nearest neighbor color for each pixel
		for icolor in range(num_colors):
			distances[:,:,:,icolor] = np.sqrt(np.sum((imgs - rgb_values[icolor][np.newaxis,np.newaxis,np.newaxis,:])**2,axis=-1))

		chosen_colors = np.argmin(distances, axis=-1)  # choose lowest distance
		
		# now compute fraction of colors
		color_intensities = np.zeros((num_colors, num_images))
		for icolor in range(num_colors):
			color_intensities[icolor,:] = np.sum(chosen_colors==icolor,axis=(1,2)) / (112*112)

		return color_intensities, color_names


	@staticmethod
	def get_spatial_frequencies(imgs):
		# returns spatial frequencies (low, medium, high), normalized across the three
		#
		# INPUT:
		#	imgs: (num_images,112,112,3), raw images with pixel intensities between 0 and 255
		# OUTPUT:
		#	low_frequencies: (num_images,), low spatial frequency index, between 0 and 1
		#			1 --> image mostly has low spatial frequencies
		#	med_frequencies: (num_images,), med spatial frequency index, between 0 and 1
		#	high_frequencies: (num_images,), high spatial frequency index, between 0 and 1
		#
		# NOTE: If an image has all types of spatial frequencies, then low_freq ~= med_freq ~= high_freq ~= 1/3  (all equal intensities)

		num_images = imgs.shape[0]
		num_pixels = imgs.shape[1]

		low_frequencies = np.zeros((num_images,))
		med_frequencies = np.zeros((num_images,))
		high_frequencies = np.zeros((num_images,))

		for iimg in range(num_images):
			# compute 2-d fourier transform
			img_grayscale = np.mean(imgs[iimg],axis=-1)
			img_fourier = np.fft.fft2(img_grayscale)
			img_fourier = np.fft.fftshift(img_fourier)
			magnitude_spectrum = np.abs(img_fourier)

			# consider three "bins": low, med, and high spatial frequencies
			#	(to test: pass in many images, take image with max index for each)
			x = np.arange(num_pixels)
			xx, yy = np.meshgrid(x, x)
			x_center = num_pixels // 2
			dists_from_center = np.sqrt((xx - x_center)**2 + (yy - x_center)**2)

			# inds_low = (dists_from_center > 0) * (dists_from_center <= num_pixels * 0.05)
			# inds_med = (dists_from_center > num_pixels * 0.05) * (dists_from_center <= num_pixels * 0.1)
			# inds_high = (dists_from_center > num_pixels * 0.1) * (dists_from_center <= num_pixels * 0.3)

			inds_low = (dists_from_center > 0) * (dists_from_center <= num_pixels * 0.05)
			inds_med = (dists_from_center > num_pixels * 0.05) * (dists_from_center <= num_pixels * 0.2)
			inds_high = (dists_from_center > num_pixels * 0.2) #* (dists_from_center <= num_pixels * 0.5)

			low_freq_mag = np.mean(magnitude_spectrum[inds_low])
			med_freq_mag = np.mean(magnitude_spectrum[inds_med])
			high_freq_mag = np.mean(magnitude_spectrum[inds_high])

			low_frequencies[iimg] = low_freq_mag
			med_frequencies[iimg] = med_freq_mag
			high_frequencies[iimg] = high_freq_mag

			# # normalize
			# low_freq_index = low_freq_mag / (low_freq_mag + med_freq_mag + high_freq_mag)
			# med_freq_index = med_freq_mag / (low_freq_mag + med_freq_mag + high_freq_mag)
			# high_freq_index = high_freq_mag / (low_freq_mag + med_freq_mag + high_freq_mag)

		# summer = low_frequencies + med_frequencies + high_frequencies + 1e-5
		low_frequencies = low_frequencies / 1e5
		med_frequencies = med_frequencies / 0.5e5
		high_frequencies = high_frequencies / 1.5e3

		low_frequences = np.clip(low_frequencies, a_min=0., a_max=1.)
		med_frequencies = np.clip(med_frequencies, a_min=0., a_max=1.)
		high_frequencies = np.clip(high_frequencies, a_min=0., a_max=1.)

		return low_frequencies, med_frequencies, high_frequencies


	@staticmethod
	def get_spatial_orientations(imgs):
		# returns spatial orientations (horiz,vert,45,-45), normalized across the four
		#
		# INPUT:
		#	imgs: (num_images,112,112,3), raw images with pixel intensities between 0 and 255
		# OUTPUT:
		#	orientations_horizontal: (num_images,), overall orientation intensity along the horizontal direction [0,1]
		#	orientations_vertical: (num_images,), overall orientation intensity along the vertical direction [0,1]
		#	orientations_diag_45: (num_images,), overall orientation intensity along the diagonal 45deg (bottom left to top right) [0,1]
		#	orientations_diag_neg45: (num_images,), overall orientation intensity along the diagonal neg45deg (top left to bottom right) [0,1]
		#
		# NOTE: If an image has all types of orientation intensities, then all orientations ~= 1/4

		num_images = imgs.shape[0]
		num_pixels = imgs.shape[1]

		imgs_grayscale = np.mean(imgs,axis=-1)
		imgs_grayscale = imgs_grayscale - 128

		# vertical edges
		if True:
			kernel = np.zeros((5,5))
			kernel[:,:2] = -1.
			kernel[:,-2:] = 1.
			kernel = kernel[np.newaxis,:,:]

			Z_conv = ndimage.convolve(imgs_grayscale, kernel, mode='constant', cval=0.0)
			Z_conv = np.quantile(Z_conv**2,q=0.85,axis=(1,2))
			orientations_vertical = Z_conv

		# horizontal edges
		if True:
			kernel = np.zeros((5,5))
			kernel[:2,:] = -1.
			kernel[-2:,:] = 1.
			kernel = kernel[np.newaxis,:,:]

			Z_conv = ndimage.convolve(imgs_grayscale, kernel, mode='constant', cval=0.0)
			Z_conv = np.quantile(Z_conv**2,q=0.85,axis=(1,2))
			orientations_horizontal = Z_conv

		# diagonal edges 45deg
		if True:
			kernel = np.zeros((5,5))
			kernel[np.triu_indices(5,k=1)] = -1.
			kernel[np.tril_indices(5,k=-1)] = 1.

			kernel = np.rot90(kernel)
			kernel = kernel[np.newaxis,:,:]

			Z_conv = ndimage.convolve(imgs_grayscale, kernel, mode='constant', cval=0.0)
			Z_conv = np.quantile(Z_conv**2,q=0.85,axis=(1,2)) 
			orientations_diag_45 = Z_conv

		# diagonal edges -45deg
		if True:
			kernel = np.zeros((5,5))
			kernel[np.triu_indices(5,k=1)] = -1.
			kernel[np.tril_indices(5,k=-1)] = 1.
			kernel = kernel[np.newaxis,:,:]

			Z_conv = ndimage.convolve(imgs_grayscale, kernel, mode='constant', cval=0.0)
			Z_conv = np.quantile(Z_conv**2,q=0.85,axis=(1,2))
			orientations_diag_neg45 = Z_conv

		# normalize orientation intensities
		if True:
			summer = orientations_vertical + orientations_horizontal + orientations_diag_45 + orientations_diag_neg45 + 1e-5
			orientations_vertical = orientations_vertical / summer
			orientations_horizontal = orientations_horizontal / summer
			orientations_diag_45 = orientations_diag_45 / summer
			orientations_diag_neg45 = orientations_diag_neg45 / summer

		return orientations_vertical, orientations_horizontal, orientations_diag_45, orientations_diag_neg45


	@staticmethod
	def get_edge_intensities(imgs):
		# returns strength of edges, regardless of orientation (horiz,vert,45,-45)
		#
		# INPUT:
		#	imgs: (num_images,112,112,3), raw images with pixel intensities between 0 and 255
		# OUTPUT:
		#	edge_intensities: (num_images,), overall edge intensity along any direction [0,1]
		#			normalized by images with maximum edge intensities

		num_images = imgs.shape[0]
		num_pixels = imgs.shape[1]

		imgs_grayscale = np.mean(imgs,axis=-1)

		# vertical edges
		if True:
			kernel = np.zeros((5,5))
			kernel[:,:2] = -1.
			kernel[:,-2:] = 1.
			kernel = kernel[np.newaxis,:,:]

			Z_conv = ndimage.convolve(imgs_grayscale, kernel, mode='constant', cval=0.0)
			Z_conv = np.median(Z_conv**2,axis=(1,2))
			orientations_vertical = Z_conv

		# horizontal edges
		if True:
			kernel = np.zeros((5,5))
			kernel[:2,:] = -1.
			kernel[-2:,:] = 1.
			kernel = kernel[np.newaxis,:,:]

			Z_conv = ndimage.convolve(imgs_grayscale, kernel, mode='constant', cval=0.0)
			Z_conv = np.median(Z_conv**2,axis=(1,2))
			orientations_horizontal = Z_conv

		# diagonal edges 45deg
		if True:
			kernel = np.zeros((5,5))
			kernel[np.triu_indices(5,k=1)] = -1.
			kernel[np.tril_indices(5,k=-1)] = 1.
			kernel = np.rot90(kernel)
			kernel = kernel[np.newaxis,:,:]

			Z_conv = ndimage.convolve(imgs_grayscale, kernel, mode='constant', cval=0.0)
			Z_conv = np.median(Z_conv**2,axis=(1,2))
			orientations_diag_45 = Z_conv

		# diagonal edges -45deg
		if True:
			kernel = np.zeros((5,5))
			kernel[np.triu_indices(5,k=1)] = -1.
			kernel[np.tril_indices(5,k=-1)] = 1.
			kernel = kernel[np.newaxis,:,:]

			Z_conv = ndimage.convolve(imgs_grayscale, kernel, mode='constant', cval=0.0)
			Z_conv = np.median(Z_conv**2,axis=(1,2))
			orientations_diag_neg45 = Z_conv

		# get edge intensity
		edge_intensities = orientations_vertical + orientations_horizontal + orientations_diag_45 + orientations_diag_neg45 + 1e-5

		edge_intensities = edge_intensities / 1.05e6  # normalize based on image with maximal edges

		edge_intensities = np.clip(edge_intensities, a_min=0., a_max=1.)
		return edge_intensities


	@staticmethod
	def get_line_intensities(imgs):
		# returns spatial orientations (horiz,vert,45,-45), normalized across the four
		#
		# INPUT:
		#	imgs: (num_images,112,112,3), raw images with pixel intensities between 0 and 255
		# OUTPUT:
		#	orientations_horizontal: (num_images,), overall orientation intensity along the horizontal direction [0,1]
		#	orientations_vertical: (num_images,), overall orientation intensity along the vertical direction [0,1]
		#	orientations_diag_45: (num_images,), overall orientation intensity along the diagonal 45deg (bottom left to top right) [0,1]
		#	orientations_diag_neg45: (num_images,), overall orientation intensity along the diagonal neg45deg (top left to bottom right) [0,1]
		#
		# NOTE: If an image has all types of orientation intensities, then all orientations ~= 1/4

		num_images = imgs.shape[0]
		num_pixels = imgs.shape[1]

		imgs_grayscale = np.mean(imgs,axis=-1)

		imgs_grayscale = imgs_grayscale - 128

		# vertical edges
		if True:
			kernel = -1.25 * np.ones((5,5))
			kernel[:,1:4] = 0.25
			kernel[:,2] = 2.
			kernel = kernel[np.newaxis,:,:]

			Z_conv = ndimage.convolve(imgs_grayscale, kernel, mode='constant', cval=0.0)
			Z_conv = np.quantile(Z_conv**2,q=0.9,axis=(1,2))
			orientations_vertical = Z_conv

		# horizontal edges
		if True:
			kernel = -1.25 * np.ones((5,5))
			kernel[1:4,:] = 0.25
			kernel[2,:] = 2.
			kernel = kernel[np.newaxis,:,:]

			Z_conv = ndimage.convolve(imgs_grayscale, kernel, mode='constant', cval=0.0)
			Z_conv = np.quantile(Z_conv**2,q=0.9,axis=(1,2))
			orientations_horizontal = Z_conv

		# diagonal edges 45deg
		if True:
			kernel = -1 * np.ones((5,5))
			kernel = kernel + np.diag(1.25*np.ones((5,)),k=1)[:5,:5]
			kernel = kernel + np.diag(1.25*np.ones((5,)),k=-1)[:5,:5]
			kernel = kernel + np.diag(3*np.ones((5,)),k=0)

			kernel = np.rot90(kernel)
			kernel = kernel[np.newaxis,:,:]

			Z_conv = ndimage.convolve(imgs_grayscale, kernel, mode='constant', cval=0.0)
			Z_conv = np.quantile(Z_conv**2,q=0.95,axis=(1,2)) 
			orientations_diag_45 = Z_conv

		# diagonal edges -45deg
		if True:
			kernel = -1 * np.ones((5,5))
			kernel = kernel + np.diag(1.25*np.ones((5,)),k=1)[:5,:5]
			kernel = kernel + np.diag(1.25*np.ones((5,)),k=-1)[:5,:5]
			kernel = kernel + np.diag(3*np.ones((5,)),k=0)

			kernel = kernel[np.newaxis,:,:]

			Z_conv = ndimage.convolve(imgs_grayscale, kernel, mode='constant', cval=0.0)
			Z_conv = np.quantile(Z_conv**2,q=0.95,axis=(1,2))
			orientations_diag_neg45 = Z_conv

		# normalize orientation intensities
		if True:
			# summer = orientations_vertical + orientations_horizontal + orientations_diag_45 + orientations_diag_neg45 + 1e-5
			orientations_vertical = orientations_vertical / 2e6
			orientations_horizontal = orientations_horizontal / 2e6
			orientations_diag_45 = orientations_diag_45 / 2e5
			orientations_diag_neg45 = orientations_diag_neg45 / 2e5

			orientations_vertical = np.clip(orientations_vertical, a_min=0., a_max=1.)
			orientations_horizontal = np.clip(orientations_horizontal, a_min=0., a_max=1.)
			orientations_diag_45 = np.clip(orientations_diag_45, a_min=0., a_max=1.)
			orientations_diag_neg45 = np.clip(orientations_diag_neg45, a_min=0., a_max=1.)

		return orientations_vertical, orientations_horizontal, orientations_diag_45, orientations_diag_neg45


	@staticmethod
	def get_curve_intensities(imgs):
		# returns curvature intensities (three different sizes of curves)
		#
		# INPUT:
		#	imgs: (num_images,112,112,3), raw images with pixel intensities between 0 and 255 for 112 pixels
		# OUTPUT:
		#	curve_intensities: (3,num_images), overall curve intensity for three different "sizes" of curves
		#		where curve_intensities[0,:] is smallest, curve_intensities[2,:] is largest

		num_images = imgs.shape[0]
		num_pixels = imgs.shape[1]

		imgs_grayscale = (np.mean(imgs,axis=-1) - 127)/128

		kernel_width = 9
		downsample_strides = [1,2,4]

		curve_intensities = np.zeros((len(downsample_strides), num_images))

		for istride, downsample_stride in enumerate(downsample_strides):

			imgs_new = np.copy(imgs_grayscale)
			imgs_new = imgs_new[:,::downsample_stride,::downsample_stride]

			# bottom right
			if True:
				xx, yy = np.meshgrid(np.arange(kernel_width),np.arange(kernel_width))
				zz = np.sqrt(xx**2 + yy**2)
				kernel = np.ones(zz.shape)
				kernel[zz < 5] = -1.
				kernel = kernel - np.mean(kernel)
				kernel = kernel / np.sqrt(np.sum(kernel**2))

				kernel = kernel[np.newaxis,:,:]
				Z_conv1 = ndimage.convolve(imgs_new, kernel, mode='nearest', cval=0.0)**2

			# bottom left
			if True:
				xx, yy = np.meshgrid(np.arange(kernel_width),np.arange(kernel_width))
				zz = np.sqrt(xx**2 + yy**2)
				kernel = np.ones(zz.shape)
				kernel[zz < 5] = -1.
				kernel = kernel - np.mean(kernel)
				kernel = kernel / np.sqrt(np.sum(kernel**2))
				kernel = np.rot90(kernel)
				kernel = kernel[np.newaxis,:,:]

				Z_conv2 = ndimage.convolve(imgs_new, kernel, mode='nearest', cval=0.0)**2

			# top left
			if True:
				xx, yy = np.meshgrid(np.arange(kernel_width),np.arange(kernel_width))
				zz = np.sqrt(xx**2 + yy**2)
				kernel = np.ones(zz.shape)
				kernel[zz < 5] = -1.
				kernel = kernel - np.mean(kernel)
				kernel = kernel / np.sqrt(np.sum(kernel**2))
				kernel = np.rot90(np.rot90(kernel))
				kernel = kernel[np.newaxis,:,:]

				Z_conv3 = ndimage.convolve(imgs_new, kernel, mode='nearest', cval=0.0)**2

			# top right
			if True:
				xx, yy = np.meshgrid(np.arange(kernel_width),np.arange(kernel_width))
				zz = np.sqrt(xx**2 + yy**2)
				kernel = np.ones(zz.shape)
				kernel[zz < 5] = -1.
				kernel = kernel - np.mean(kernel)
				kernel = kernel / np.sqrt(np.sum(kernel**2))
				kernel = np.rot90(np.rot90(np.rot90(kernel)))
				kernel = kernel[np.newaxis,:,:]

				Z_conv4 = ndimage.convolve(imgs_new, kernel, mode='nearest')**2

			Z_conv = Z_conv1 + Z_conv2 + Z_conv3 + Z_conv4

			Z_conv = np.reshape(Z_conv, (num_images,-1))
			energies = np.mean(np.sort(Z_conv, axis=-1)[:,-int(np.floor(Z_conv.shape[1]*0.15)):], axis=-1)   # take top 15%

			curve_intensities[istride,:] = energies

		curve_intensities[0,:] = curve_intensities[0,:] / 80.
		curve_intensities[1,:] = curve_intensities[1,:] / 80.
		curve_intensities[2,:] = curve_intensities[2,:] / 80.

		curve_intensities = np.clip(curve_intensities, a_min=0, a_max=1.)

		return curve_intensities



	@staticmethod
	def get_dot_intensities(imgs):
		# NOT TESTED
		# returns curvature intensities (three different sizes of curves)
		#
		# INPUT:
		#	imgs: (num_images,num_pixels,num_pixels,3), raw images with pixel intensities between 0 and 255
		# OUTPUT:
		#	dot_intensities: (3,num_images), overall curve intensity for three different "sizes" of curves
		#		where dot_intensities[0,:] is smallest, dot_intensities[2,:] is largest

		num_images = imgs.shape[0]
		num_pixels = imgs.shape[1]

		imgs_grayscale = np.mean(imgs,axis=-1) - 127

		downsample_strides = [1,2,4]

		dot_intensities = np.zeros((len(downsample_strides), num_images))

		# dot detector
		if True:
			xx, yy = np.meshgrid(np.arange(-5,6),np.arange(-5,6))
			zz = np.sqrt(xx**2 + yy**2)
			kernel = np.ones(zz.shape) * 2
			kernel[zz > 2.5] = -1.
			# kernel = kernel - np.mean(kernel)
			kernel = kernel / np.sqrt(np.sum(kernel**2))

		for istride, downsample_stride in enumerate(downsample_strides):
			imgs_new = np.copy(imgs_grayscale)
			imgs_new = imgs_new[:,::downsample_stride,::downsample_stride]

			Z_conv = ndimage.convolve(imgs_new/128, kernel[np.newaxis,:,:], mode='constant', cval=0.0)**2
				# plot Z_conv to see activity map for debugging

			Z_conv = np.reshape(Z_conv, (num_images,-1))
			energies = np.mean(np.sort(Z_conv, axis=-1)[:,-10:], axis=-1)  # take top 10 values (10 or less dots)

			dot_intensities[istride,:] = energies

		dot_intensities[0,:] = dot_intensities[0,:] / 100.
		dot_intensities[1,:] = dot_intensities[1,:] / 100.
		dot_intensities[2,:] = dot_intensities[2,:] / 100.

		dot_intensities = np.clip(dot_intensities, a_min=0, a_max=1.)

		return dot_intensities



