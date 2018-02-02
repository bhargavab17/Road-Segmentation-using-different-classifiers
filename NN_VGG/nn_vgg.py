#! /usr/bin/env python3

"""__author__ = "Pulkit Verma, AVS SAI Bhargav, and Anish Shastri"
__copyright__ = "Copyright 2017, TEAM PASS"
__license__ = "Open Source"
__version__ = "Python 2.7"
__maintainer__ = "Team Pass"
"""

# import necessary directories
import os
import re
import sys
import time
import random
import scipy.misc
import numpy as np
from glob import glob
import tensorflow as tf
from skimage import io, color
# from matplotlib import pyplot as plt


# class for Neural Network training on pretrained model VGG 16
class NN_VGG(object):
	""" Training a Neural Network
    """

	def __init__(self):
		"""Initialize the class

	    Keyword arguments:
	    None
	    """ 

	    # labels : Road or Non Road classification
		self.label_names = ['road', 'non_road']
		# Batch Size
		self.batch_size = 16
		# Number of classes : 2
		self.num_classes = len(self.label_names)
		# Number of Iterations/epochs
		self.epochs = 50
		# input image dimensions
		self.image_shape = (160, 576)


	def segment_images(self, session, logits, keep_prob, input_image, data_test_dir, image_shape):
		"""Segmenting the testing images

	    Keyword arguments:
	    session : TF active session
	    data_test_dir : directory path for the test images
	    image_shape : shape of all the testing images
	    keep_prob : 
	    logits :  
	    input_image : 
	    """

	    # Run the loop for segmenting all the test images
		for image_file in glob(os.path.join(data_test_dir, 'image_2', '*.png')):
			# read each image
			im = scipy.misc.imread(image_file)
			# resize each image to the required shape 
			image = scipy.misc.imresize(im, image_shape)
			# 
			im_softmax = session.run([tf.nn.softmax(logits)], {keep_prob: 1.0, input_image: [image]})
			#
			im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
			#
			segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
			#
			mask = np.dot(segmentation, np.array([[255, 0, 255, 255]]))
			#
			mask = scipy.misc.toimage(mask, mode="RGBA")
			#
			street_im = scipy.misc.toimage(image)
			#
			street_im.paste(mask, box=None, mask=mask)
			#
			yield os.path.basename(image_file), np.array(street_im)


	def saving_test_images(self, segmented_images_dir, data_test_dir, session,\
							 image_shape, logits, keep_prob, input_image):
		"""saving all the testing segmented images

	    Keyword arguments:
	    session : TF active session
	    segmented_images_dir : directory path for saving the segmented images
	    data_test_dir : directory path for the test images
	    image_shape : shape of all the testing images
	    keep_prob : 
	    logits :  
	    input_image : 
	    """

		print('Training Done ...')
		# get the segmented result for all images
		image_outputs = self.segment_images(session, logits, keep_prob, input_image,\
											 data_test_dir, image_shape)
		# save all the images at the required path
		for name, image in image_outputs:
			scipy.misc.imsave(os.path.join(segmented_images_dir, name), image)

	def training_network(self, session, epochs, batch_size, get_batches_fn,\
						 train_op, cross_entropy_loss, image_input, correct_label,\
						 keep_prob, learning_rate, saver):
		""" Training the network for road and non road classification 

	    Keyword arguments:
	    session : TF active session
	    epochs : number of iterations
	    batch_size : batch size
	    get_batches_fn : 
	    train_op : 
	    cross_entropy_loss : 
	    keep_prob :  
	    input_image : 
	    correct_label :
	    learning_rate :
	    saver :   
	    """

	    #
		for epoch in range(epochs):
			#
			s_time = time.time()
			#
			for image, targets in get_batches_fn(batch_size):
				#
				_, loss = session.run( [train_op, cross_entropy_loss],feed_dict = \
									   {image_input: image, correct_label: targets,\
									   keep_prob: 0.8 , learning_rate: 0.0001 })
			# Print data on the learning process
			print("Epoch: {}".format(epoch + 1), "/ {}".format(epochs), " Loss: {:.3f}".format(loss))

	# def get_accuracy(self,get_batches_fn,batch_size,):



	def gen_batch_function(self, data_folder, image_shape):
		""" get the training data 

	    Keyword arguments:
	    data_folder : 
	    image_shape : 
	    batch_size : 
	    """

		def get_batches_fn(batch_size):
			#
			image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
			#
			label_paths = {	re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
				for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}
			#
			background_color = np.array([255, 0, 0])
			#
			random.shuffle(image_paths)
			#
			for batch_i in range(0, len(image_paths), batch_size):
				#
				images = []
				#
				gt_images = []
				#
				for image_file in image_paths[batch_i:batch_i+batch_size]:
					#
					gt_image_file = label_paths[os.path.basename(image_file)]
					#
					image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
					#
					gt_image = scipy.misc.imresize(scipy.misc.imread(gt_image_file), image_shape)
					#
					gt_bg = np.all(gt_image == background_color, axis=2)
					#
					gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
					#
					gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)
					#
					images.append(image)
					#
					gt_images.append(gt_image)
				#
				yield np.array(images), np.array(gt_images)
		#
		return get_batches_fn




if "__name__" != "__main__":
	""" Main Function

	    Keyword arguments: None
	"""


	# Path to directory containing the training images
	data_train_dir = '../datasets/data_road/training'
	# Path to directory containing the testing images
	data_test_dir = '../datasets/data_road/testing'
	# Path to directory containing the segmented images
	segmented_images_dir = 'Output/images'
	# Path to directory containing the pretrained VGG model		
	vgg_model_dir = '../datasets/vgg/'
	# Sub Path containing the training images
	all_train_images = '/images_2'
	# Sub Path containing the training ground truth images
	ground_truth_train_dir = '/gt_images_2'
	
	nn_vgg = NN_VGG()


	tf.reset_default_graph()
	session = tf.Session()

	vgg_input_tensor_name = 'image_input:0'
	vgg_keep_prob_tensor_name = 'keep_prob:0'
	vgg_layer3_out_tensor_name = 'layer3_out:0'
	vgg_layer4_out_tensor_name = 'layer4_out:0'
	vgg_layer7_out_tensor_name = 'layer7_out:0'

	g = tf.get_default_graph()
	tf.saved_model.loader.load(session, ['vgg16'], vgg_model_dir)

	image_input = g.get_tensor_by_name(vgg_input_tensor_name)
	keep_prob = g.get_tensor_by_name(vgg_keep_prob_tensor_name)
	layer3_out = g.get_tensor_by_name(vgg_layer3_out_tensor_name)
	layer4_out = g.get_tensor_by_name(vgg_layer4_out_tensor_name)
	layer7_out = g.get_tensor_by_name(vgg_layer7_out_tensor_name)

	print("Loading VGG Model ...") 


	layer7_conv_1x1 = tf.layers.conv2d(layer7_out, nn_vgg.num_classes, 1, 1,
									   padding='same', kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
									   kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01))
	output = tf.layers.conv2d_transpose(layer7_conv_1x1, nn_vgg.num_classes, 5, 2,
										padding='same', kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
										kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01))
	layer4_conv_1x1 = tf.layers.conv2d(layer4_out, nn_vgg.num_classes, 1, 1,
									   padding='same', kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
									   kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01))
	output = tf.add(output, layer4_conv_1x1)
	output = tf.layers.conv2d_transpose(output, nn_vgg.num_classes, 5, 2,
										padding='same', kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
										kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01))
	layer3_conv_1x1 = tf.layers.conv2d(layer3_out, nn_vgg.num_classes, 1, 1,
									   padding='same', kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
									   kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01))
	output = tf.add(output, layer3_conv_1x1)
	output = tf.layers.conv2d_transpose(output, nn_vgg.num_classes, 16, 8,
										padding='same', kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
										kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01))   


	learning_rate = tf.placeholder(dtype = tf.float32)
	correct_label = tf.placeholder(dtype = tf.float32, shape = (None, None, None, nn_vgg.num_classes))

	get_batches_fn_train = nn_vgg.gen_batch_function(data_train_dir, nn_vgg.image_shape)
	get_batches_fn_test = nn_vgg.gen_batch_function(data_test_dir,nn_vgg.image_shape)

	logits = tf.reshape(output, (-1, nn_vgg.num_classes))
	labels = tf.reshape(correct_label, (-1, nn_vgg.num_classes))
	cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels))
	train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)

	saver = tf.train.Saver()
	session.run(tf.global_variables_initializer())

	nn_vgg.training_network(session, nn_vgg.epochs, nn_vgg.batch_size, get_batches_fn_train, train_op, cross_entropy_loss, image_input,
				 correct_label, keep_prob, learning_rate, saver)

	nn_vgg.saving_test_images(segmented_images_dir, data_test_dir, session, nn_vgg.image_shape, logits, keep_prob, image_input)
	session.close()


