import cv2
import numpy as np
import math
from sklearn import mixture
from scipy import linalg
import itertools
import matplotlib.pyplot as plt
from scipy import linalg
import matplotlib as mpl
from time import time
from scipy import infty
from sklearn import preprocessing
from sklearn.utils import shuffle
from matplotlib import colors as mcolors
from scipy.misc import imfilter, imread
from skimage import color, data, restoration
from scipy.signal import convolve2d as conv2
from itertools import chain
from skimage import feature
import os

color_iter = itertools.cycle(['navy', 'red', 'cornflowerblue', 'gold', 'darkorange','b','cyan'])
color_code = {
	1: (171,166, 27),
    2: (112, 26, 91,),
    3: (61, 42,   61), 
    4: (19, 118, 140),
    5: (227, 25, 227),
    6: (139, 69,   19),
    7: (56, 161,  48)
	}

def test(imagetest,gmm):
	pre = gmm.predict(imagetest)
	# plot_results(imagetest,pre,gmm.means_,gmm.covariances_,'Gaussian Mixture')
	# print gmm.means_
	# plt.show()

	return pre
def train(num_patches,img,n_samples,w,h):
	imtrain = shuffle(img)
	imtrain = imtrain[:1000]
	# imtrain = img
	gmm = mixture.GaussianMixture(n_components=7, covariance_type='full', 
       		tol=0.01, reg_covar=1e-06, max_iter=1200, n_init=1, init_params='kmeans', 
       		warm_start=True).fit(imtrain)

	return gmm

def segmented(image,samples,label,num_comp):
	labels = np.expand_dims(label, axis = 0)
	labels = np.transpose(labels)

	for i in range (1,num_comp):
		indices = np.where(np.all(labels == i, axis =-1))
		indices = np.unravel_index(indices,(w,h), order= 'C')
		type(indices)
		indices = np.transpose(indices)

		l = chain.from_iterable(zip(*indices))

		for j, (lowercase, uppercase) in enumerate(l):
        	# set the colour accordingly

			image[lowercase,uppercase] = color_code[(i)] 
	return image

# local binary pattern descriptor
def createFeature(image, n_samples):
	numpoints = 24
	radius = 8
	img_src = cv2.GaussianBlur(image,(5,5),0)
	# imtest = img_src
	imtest = cv2.cvtColor(img_src,cv2.COLOR_BGR2LAB)

	# blur = cv2.bilateralFilter(img_src,9,75,75)
	# blurthresh=100
	# imtest = np.fix(imtest, blurthresh)
	img_gray = cv2.cvtColor(img_src,cv2.COLOR_BGR2GRAY)
	lbp = feature.local_binary_pattern(img_gray,numpoints,radius, method="uniform")
	lbp = np.reshape(lbp,(n_samples,1))
	imtest = np.reshape(imtest,(n_samples,d))
	# print np.shape(lbp)
	data = np.column_stack((imtest, lbp))
	data = preprocessing.normalize(imtest, norm='l2')
	data= preprocessing.scale(data)

	return data, imtest

img = cv2.imread('um_000000.png')
gt_image = cv2.imread('um_lane_000000.png')
b = gt_image[:,:,0] < 255
g = gt_image[:,:,1] 
r = gt_image[:,:,2] 
gt_image[b] = 0
gt_image[~b] = 1
img_src = cv2.multiply(gt_image,img)


w, h, d = tuple(img_src.shape)

# Number of samples per component
n_samples = w*h
#Number of sets of training samples
num_patches=100;

#print w,h

samples, imtest=createFeature(img_src, n_samples)
gmm = train(num_patches,samples,n_samples,w,h)
print gmm.means_


image_test = cv2.imread('umm_000033.png')
test_samples, im=createFeature(image_test, n_samples)
pre = test(test_samples,gmm)
# image_test = cv2.cvtColor(image_test, cv2.COLOR_BGR2LAB)
seg1 = segmented(image_test,test_samples,pre,7)
cv2.imwrite('segmentation.png', seg1)
k= cv2.waitKey(0)
if k ==27:
	cv2.destroyAllWindows()




