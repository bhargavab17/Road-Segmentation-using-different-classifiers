#! /usr/bin/env python

"""__author__ = "Pulkit Verma, AVS SAI Bhargav, and Anish Shastri"
__copyright__ = "Copyright 2017, TEAM PASS"
__license__ = "Open Source"
__version__ = "Python 2.7"
__maintainer__ = "Team Pass"
"""

# import necessary directories
import os
import glob
import numpy as np
from skimage import io
from skimage import color
from skimage import exposure
from sklearn import svm, metrics
from sklearn.externals import joblib
from skimage.util import img_as_float
from skimage.segmentation import slic
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction import image
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from skimage.feature import hog, greycomatrix, greycoprops

		
# class for Road Segmentation training on different Classifiers
class ROAD_SEGMENTATION(object):
	""" training of Road segmentation class
    """

	def __init__(self):
		"""Initialize the class

	    Keyword arguments:
	    None
	    """ 
	    # Variable to create train and test vector:- 1:create new vectors; 0: Used saved vectors
		self.create_vectors_again = 0
		# Variable to start training on a model:- 1:train a new model; 0: Used saved model
		self.start_training_model = 0
		# Select the classifier to use for the training
		self.select_classifier = 'nb'

		# Global List initializations
		self.image = []
		self.image_greyscale = []
		self.label_image = []
		self.superpixel = []
		self.superpixel_labels = []
		self.superpixel_location = []
		self.superpixel_color = []
		self.superpixel_size = []
		self.superpixel_hog = []
		self.superpixel_color_histogram = []
		self.superpixel_texture = []

	def readImageToSuperpixels(self, image_path):
		"""Read an Image and create super pixel

	    Keyword arguments:
	    image_path : path for the images to read
	    """

	    # read the image
		self.image = img_as_float(io.imread(image_path))
		# image conversion to gray scale
		self.image_greyscale = color.rgb2gray(self.image)
		# create super pixel of the image read
		self.superpixel = slic(self.image, n_segments=300, compactness=20, sigma=1)

	def readImageLabels(self, label_path):
		"""Read an Image and find the labels

	    Keyword arguments:
	    label_path : path for the images to read
	    """

	    # read the image
		self.label_image = io.imread(label_path)
		# find the labels of the image read
		self.superpixel_labels = self.findSpLabel(self.superpixel, self.label_image, 0.5)


	def findAllFeatures(self):
		"""Find all the feature of the image and create a feature vector

		Features Selected: Location of the Superpixel
						   Mean color of the Superpixel
						   Hog features of the Superpixel
						   Size of the Superpixel
						   Texture of the Superpixel
		Feature vector is created by stacking the features vertically

	    Keyword arguments:
	    None
	    """

	    # Find SuperPixel Location
		self.superpixel_location = self.findSpLocation(self.superpixel)
		# Find SuperPixel Mean Color
		self.superpixel_color = self.findSpMeanColor(self.superpixel, self.image)
		# Find the hog feature of the superpixel
		self.superpixel_hog = self.findSpHOG(self.superpixel, self.image)
		# Find SuperPixel Size
		self.superpixel_size = self.findSpSize(self.superpixel)
		# Find SuperPixel texture
		self.superpixel_texture = self.findSpTexture(self.superpixel, self.image)
		# Stacking all the features in a vertical Stack
		self.featureVectors = np.vstack((self.superpixel_location.T, self.superpixel_color.T, self.superpixel_hog.T, self.superpixel_size.T, self.superpixel_texture.T)).T
	
	def findSpLocation(self, sp):
		""" Find the Location of the superpixel

	    Keyword arguments:
	    sp : superpixel for which, we need to find the location
	    """

	    #  initialize the location array
		sp_loc = []  
		# Running a loop on all superpixels 
		for i in np.unique(sp):
			# find the indexes for each superpixel
			index = np.where(sp == i)
			# find the mean x and y position
			x = np.mean(index[0])
			y = np.mean(index[1])
			# append the locations of all superpixels of an image in a list
			sp_loc.append([x,y])
		# return an array containing all locations
		return np.array(sp_loc)

	def findSpSize(self, sp):
		""" Find the Size of the superpixel

	    Keyword arguments:
	    sp : superpixel for which, we need to find the size
	    """

	    #  initialize the size array
		size = []
		# Running a loop on all superpixels 
		for i in np.unique(sp):
			# find the indexes for each superpixel
			index = np.where(sp == i)
			# find the shape
			size.append(index[0].shape)
		# return an array containing all sizes
		return np.array(size)

	def findSpMeanColor(self, sp, image):
		""" Find the color of the superpixel

	    Keyword arguments:
	    sp : superpixel for which, we need to find the color
	    image : image for which, we need to find the color
	    """

	    #  initialize the mean color array
		mean_color = []
		# Running a loop on all superpixels 
		for i in np.unique(sp):
			# find the indexes for each superpixel
			index = np.where(sp==i)
			#  find the color of the image
			clr = image[index]
			# create an array for the color of each super pixel
			mean_color.append([ np.mean(clr[:,0]), np.mean(clr[:,1]), np.mean(clr[:,2])])
		# return an array containing all color values
		return np.array(mean_color)

	def findSpHOG(self, sp, image):
		""" Find the hog of the superpixel

	    Keyword arguments:
	    sp : superpixel for which, we need to find the hog
	    image : image for which, we need to find the hog
	    """
		
		# convert the rgb image to gray image
		grayImage = color.rgb2gray(image)
	    #  initialize the gradients array
		gradients = []
		# get the hog features
		fd, hog_image = hog(grayImage, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualise=True)
		# rescaling of the hog image
		hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
		# Running a loop on all superpixels 
		for i in np.unique(sp):
			# find the indexes for each superpixel
			index = np.where(sp==i)
			# find the hog gradient of each superpixel
			gradient = [np.mean(hog_image_rescaled[index])]
			# append the gradients
			gradients.append(gradient)
		# return an array containing all gradients values
		return np.array(gradients)

	def findSpLabel(self, sp, image, thresh=0.5):
		""" Find the labels of the superpixel

	    Keyword arguments:
	    sp : superpixel for which, we need to find the labels
	    image : image for which, we need to find the labels
	    thresh: Threshold set for setting up the labels
	    """
		
		# Initialize the avg list
		avg = []
		# Collect the "B" element of the color in an image
		pixel = image[:,:,2]
		# Running a loop on all superpixels
		for i in np.unique(sp):
			# find the indexes for each superpixel
			index = np.where(sp==i)
			# collect the color value
			labels = pixel[index]
			# collect all the pink color elements and find average
			iou = 1.0 * np.sum(labels) / len(labels)
			# append in avg list
			avg.append(iou)
		# convert list in array
		avg = np.array(avg)
		# Set the label threshold
		lsp = np.array( avg > thresh)
		# return the array of all the labels collected
		return np.array(lsp)

	def findSpTexture(self, sp, image):
		""" Find the texture of the superpixel

	    Keyword arguments:
	    sp : superpixel for which, we need to find the labels
	    image : image for which, we need to find the labels
	    """

	    # Initialize the texture list
		texture = []
		num_sp = np.max(sp) + 1
		greyImage = np.around(color.rgb2gray(image) * 255, 0)
		# loop over all super pixel
		for i in xrange(0,num_sp):
			# find the indexes for each superpixel
			index = np.where(sp == i)
			# find the glcm  matrix
			glcm = greycomatrix([greyImage[index]], [5], [0], 256, symmetric=True, normed=True)
			# find the dissimilarity
			dissimilarity = greycoprops(glcm, 'dissimilarity')[0, 0]
			# find the correlation 
			correlation = greycoprops(glcm, 'correlation')[0, 0]
			# append all values in texture
			texture.append([dissimilarity, correlation])
		# return the texture values as an array
		return np.array(texture)

	def takeImageFindFeatures(self, image_path, label_path):
		""" Find all the features of an image

	    Keyword arguments:
	    image_path : image path for which, we need to find the features 
	    label_path : label path for which, we need to find the features
	    """

	    # read image and et superpixels
		self.readImageToSuperpixels(image_path)
		# get image labels
		self.readImageLabels(label_path)
		# collect all features and get a feature vector
		self.findAllFeatures()
		
	def trainClassifier(self, clf_name, X_train, Y_train):
		""" Train the classifier

	    Keyword arguments:
	    clf_name : the classifier name to be used. only can use : "nb", "svm", "knn", "rf"
	    X_train : Feature vector of train images
	    Y_train : labels of train images
	    """

	    # Select the classifier as SVM
		if clf_name == "svm":
			clf = svm.SVC()
		# Select the classifier as Naive Bayes
		elif clf_name == "nb":
			clf = GaussianNB()
		# Select the classifier as KNN
		elif clf_name == "knn":
			clf = KNeighborsClassifier(n_neighbors=7)
		# Select the classifier as Random Forest
		elif clf_name == "rf":
			clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
		# Display message for wrong classification selection
		else:
			print "Select a Proper Classifier"
		# Fit the classifier
		clf.fit(X_train, Y_train)
		# save the classifier
		joblib.dump( clf,  clf_name + '_model.pkl' )
		# return the classifier
		return clf

	def saveSegmentedImage(self, noi, im_path, predict):
		""" Save the segmented images

	    Keyword arguments:
	    noi : number of segmented images
	    im_path : path of tested images
	    predict : predictions made for the test vectors
	    """

		print("Saving all segmented Images")
		# run loop for all images
		for loop in range(noi):
			# check for an exceptions and skip it
			try:
				# read image and get superpixels
				self.readImageToSuperpixels(im_path[loop])
				# copy the images to save
				image = np.copy(R_SEG.image)
				# get the max of superpixel
				num_superpixels = np.max(R_SEG.superpixel) + 1
				# loop for all superpixels
				for i in xrange(0,num_superpixels):
					# find the indexes for each superpixel
					index = np.where(R_SEG.superpixel==i)
					# when condition is True
					if predict[i] == 1:
						# set the road color as yellow
						image[index[0],index[1],0] = 1
						image[index[0],index[1],1] = 1
						image[index[0],index[1],2] = 0
				# find the base file name
				base_name = os.path.basename(im_path[loop])
				# save the image at the required path
				io.imsave('SegmentedImages/' + base_name, image)
				# print('.')
			# handle keyboard exceptions
			except KeyboardInterrupt:
				raise
			# except any other exceptions
			except:
				pass
				# print("error")

	def predictImages(self, clf, data, labels):
		""" Predict from the classifier

	    Keyword arguments:
	    clf : the classifier to be used after fitting. 
	    data : Feature vector of test images
	    labels : labels of test images
	    """

	    # predict the labels for the test data
		predict = clf.predict(data)
		# calculate accuracy
		accuracy = metrics.accuracy_score(labels, predict)
		# calculate F1 Score
		f1 = metrics.f1_score(labels, predict)
		# calculate precision score
		precision = metrics.precision_score(labels, predict)
		# calculate Recall Score
		recall = metrics.recall_score(labels, predict)
		# print values obtained
		print("For Classifier %s on test_data: \n Accuracy : %s, \n F1_score : %s, \n Precision : %s, \n Recall : %s"
	      % (self.select_classifier, accuracy, f1, precision, recall) ) 
		# return predicted values
		return predict

if __name__ == '__main__':
	""" Main loop

	    Keyword arguments:
	    None
	"""

	# paths and names of all the train images
	image_filenames = glob.glob("../datasets/data_road/training/image_2/*.png")
	# paths and names of all the train ground truth images
	label_image_filenames = glob.glob("../datasets/data_road/training/gt_image_2/*.png")

	# paths and names of all the test images
	image_filenames_test = glob.glob("../datasets/data_road/testing/image_2/*.png")
	# paths and names of all the test ground truth images
	label_image_filenames_test = glob.glob("../datasets/data_road/testing/gt_image_2/*.png")

	# number of training images
	num_train = len(image_filenames)
	# number of test images
	num_test = len(image_filenames_test)

	# call the Class ROAD_SEGMENTATION
	R_SEG = ROAD_SEGMENTATION()

	# Run only if create_vectors_again is set
	if R_SEG.create_vectors_again == 1:
		# initialize all list used
		train_labels = []
		train_data = []
		test_labels = []
		test_data = []

		# intialize a counter to see number of faulty images (if any)
		counter = 0
		print("Extracting Train image features")
		# run of all training images
		for i in range(num_train):
			print(i)
			# check for any exception
			try:
				# read all training image and find all features 
				R_SEG.takeImageFindFeatures(image_filenames[i],label_image_filenames[i])
				# get train labels
				labels = R_SEG.superpixel_labels
				# get feature vector for train image
				feature_vectors = R_SEG.featureVectors
				# save all train vector in an array
				train_labels = np.append(train_labels, labels, 0)
				if train_data==[]:
					train_data = feature_vectors
				else:
					train_data = np.vstack((train_data,feature_vectors))
			# handle keyboard exceptions
			except KeyboardInterrupt:
				raise
			# hancle Index exceptions
			except IndexError:
				counter = counter + 1
				# print("counts : ",counter)
				# print("error handled")

		# intialize a counter to see number of faulty images (if any)
		counter = 0
		print("Extracting Test image features")
		# run of all testing images
		for i in range(num_test):
			print(i)
			# check for any exception
			try:
				# read all testing image and find all features 
				R_SEG.takeImageFindFeatures(image_filenames_test[i],label_image_filenames_test[i])
				# get test labels
				labels = R_SEG.superpixel_labels
				# get feature vector for test image
				feature_vectors = R_SEG.featureVectors
				# save all test vector in an array
				test_labels = np.append(test_labels, labels, 0)
				if test_data==[]:
					test_data = feature_vectors
				else:
					test_data = np.vstack((test_data,feature_vectors))
			# handle keyboard exceptions
			except KeyboardInterrupt:
				raise
			# hancle Index exceptions
			except IndexError:
				counter = counter + 1
				# print("counts : ",counter)
				# print("error handled")

		# save all the train and test vectors
		joblib.dump(train_data, 'train_data.pkl') 
		joblib.dump(test_data, 'test_data.pkl') 
		joblib.dump(train_labels, 'train_labels.pkl') 
		joblib.dump(test_labels, 'test_labels.pkl') 
	else:
		# run only if create_vectors_again is unset
		# load all the train and test vectors
		train_data = joblib.load('train_data.pkl')
		test_data = joblib.load('test_data.pkl')
		train_labels = joblib.load('train_labels.pkl')
		test_labels = joblib.load('test_labels.pkl')

	# fit the model again if start_training_model is set
	if R_SEG.start_training_model == 1:
		clf = R_SEG.trainClassifier( R_SEG.select_classifier, train_data, train_labels )
	else:
		# load the fitted model if start_training_model is unset
		clf = joblib.load( R_SEG.select_classifier + '_model.pkl' )
	# find all the predictions on all test images
	predictions = R_SEG.predictImages(clf, test_data, test_labels)
	# save all the images
	R_SEG.saveSegmentedImage( num_test, image_filenames_test, predictions )

#************************************ end of code ************************************#