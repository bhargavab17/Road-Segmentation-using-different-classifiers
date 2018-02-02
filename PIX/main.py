#! /usr/bin/env python

"""__author__ = "Anish Shastri, AVS SAI Bhargav, and Pulkit Verma"
__copyright__ = "Copyright 2017, TEAM PASS"
__license__ = "Open Source"
__version__ = "Python 2.7"
__maintainer__ = "Team Pass"
"""

# import necessary directories
from skimage.feature import local_binary_pattern
from skimage.feature import hog
from sklearn.cluster import KMeans
from mahotas.features import haralick,lbp
from skimage import feature, io
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import data, color, exposure, filter 
from PIL.Image import Image
from PIL import Image as Img
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.externals import joblib
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# Set or unset variable for creating test and train patterns
create_feature_again = 1
# Set or unset variable for fitting model
run_train=1

# function to classify and fit the data
def Classify(train_data, train_labels, run_train):
    if(run_train ==1):
        # call the classifier
        clf = GaussianNB()
        # fit the data
        clf.fit(X_train,Y_train)
        # save the model
        _=joblib.dump(clf,'NB_joblib.pkl',compress=9)
    else:
        # load the model
        clf=joblib.load('NB_joblib.pkl')
    # return the classifier
    return clf

# function to predict
def prediction(clf, X_test, Y_test):
    # predict the labels for the test data
    predict = clf.predict(X_test)
    # calculate accuracy
    accuracy = metrics.accuracy_score(Y_test, predict)
    # calculate F1 Score
    f1 = metrics.f1_score(Y_test, predict)
    # calculate precision score
    precision = metrics.precision_score(Y_test, predict)
    # calculate Recall Score
    recall = metrics.recall_score(Y_test, predict)
    # print values obtained
    print("Accuracy : %s, \n F1_score : %s, \n Precision : %s, \n Recall : %s"
      % (accuracy, f1, precision, recall) ) 

    return predict


if __name__ == '__main__':
    # take the path arguments for test and train
    arg_train = sys.argv[1]
    arg_test = sys.argv[2]

    # create full path of test and train
    Xtrain_path = arg_train + '/image_2/'
    Ytrain_path = arg_train + '/gt_image_2/'
    Xtest_path = arg_test + '/image_2/'
    Ytest_path = arg_test + '/gt_image_2/'

    # initialize lists
    xtrain = []
    ytrain = []
    xtest = []
    ytest = []

    # take all training and testing images and ground truths
    train_images = sorted(os.listdir(Xtrain_path))
    gt_train_images = sorted(os.listdir(Ytrain_path))
    test_images = sorted(os.listdir(Xtest_path))
    gt_test_images = sorted(os.listdir(Ytest_path))

    # set image size
    w = 1242
    h = 375

    # noi of images to test
    noi = 100

    # initialize arrays
    xt_arr = np.zeros([noi* 465750, 8]) 
    yt_arr = np.zeros([noi* 465750, 1])

    # Create test and train feature vectors
    if create_feature_again == 1:
        # for all images
        for loop in range(noi):
            print(Xtrain_path + train_images[loop])
            # read image
            im = Img.open(Xtrain_path + train_images[loop])
            im1=np.array(im)
            a = np.asarray(list(im.getdata()))
            # convert to hsv
            hsv = Image.convert(im, 'HSV')
            b = np.asarray(list(hsv.getdata()))

            # convert to gray
            gray = Image.convert(im, 'L')
            gray_hist=exposure.equalize_hist(gray,nbins=2)
            gray_hist=np.reshape(gray_hist, (w*h, 1))

            # convert to cbr
            cbr=color.rgb2grey(im1)
            cbr_a = np.reshape(cbr, (w*h, 1))

            # convert to lbp
            lbp = local_binary_pattern(gray_hist, 24, 8, 'uniform')
            lbp = np.reshape(lbp, (w*h, 1))
            
            # stack all features to create a vector
            d = np.hstack([a,b,lbp,cbr_a])

            # stack all vectors to get all train features vectors
            if loop == 0:
                xt_arr = d
            else:
                xt_arr = np.vstack([xt_arr, d])

        # loop to get all labels for train images
        for loop in range(noi):
            print(loop)
            # initialize array
            train = np.zeros([w*h, 1])
            # open image
            im = Img.open(Ytrain_path + gt_train_images[loop])
            a = np.asarray(list(im.getdata()))
            # find the pink color occurences and indexes
            index = np.where(a[:, 2] > 220)
            train[index] = 1
            # save all labels
            if loop == 0:
                yt_arr = train
            else:
                yt_arr = np.vstack([yt_arr, train])

        # initialize arrays
        xte_arr = np.zeros([1 * 465750, 5])

        noi_test = 1
        offset = 20
        # for all test images
        for loop in range(noi_test):
            try:
                print(Xtrain_path + train_images[loop+offset])
                # open image
                im = Img.open(Xtrain_path + train_images[loop+offset])
                im1=np.array(im)
                a = np.asarray(list(im.getdata()))

                # convert to hsv
                hsv = Image.convert(im, 'HSV')
                b = np.asarray(list(hsv.getdata()))

                # convert to gray
                gray = Image.convert(im, 'L')
                gray_hist=exposure.equalize_hist(gray,nbins=2)
                gray_hist=np.reshape(gray_hist, (w*h, 1))

                # convert to cbr
                cbr=color.rgb2grey(im1)
                cbr_a = np.reshape(cbr, (w*h, 1))

                # convert to lbp
                lbp = local_binary_pattern(gray_hist, 24, 8, 'uniform')
                lbp = np.reshape(lbp, (w*h, 1))

                # stack all features to create a vector
                d = np.hstack([a,b,lbp,cbr_a])
                # stack all feature vectors to get all test vectors
                if loop == 0:
                    xte_arr = d
                else:
                    xte_arr = np.vstack([xte_arr, d])

            except KeyboardInterrupt:
                raise
            except ValueError:
                print("Error handled")
 

        # loop to get all labels for test images
        for loop in range(noi_test):
            print(loop + offset)
            # initialize array
            test = np.zeros([w*h, 1])
            # open image
            im = Img.open(Ytrain_path + gt_train_images[loop + offset])
            a = np.asarray(list(im.getdata()))
            # find the pink color occurences and indexes
            index = np.where(a[:, 2] > 220)
            test[index] = 1
            # save all labels
            if loop == 0:
                yte_arr = test
            else:
                yte_arr = np.vstack([yt_arr, test])
            
        # convert in all arrays
        X_train = np.asarray(xt_arr)                                                                                                                                                                           
        Y_train = np.asarray(yt_arr)
        Y_train = np.ravel(Y_train)
        X_test = np.asarray(xte_arr)
        Y_test = np.asarray(yte_arr)
        Y_test = np.ravel(Y_test)

        # save data
        joblib.dump(X_train, 'train_data.pkl') 
        joblib.dump(X_test, 'test_data.pkl') 
        joblib.dump(Y_train, 'train_labels.pkl') 
        joblib.dump(Y_test, 'test_labels.pkl') 
    else:
        # load data
        X_train = joblib.load('train_data.pkl')
        X_test = joblib.load('test_data.pkl')
        Y_train = joblib.load('train_labels.pkl')
        Y_test = joblib.load('test_labels.pkl')

    # classify
    clf = Classify(X_train, Y_train, run_train)
    # predict
    predict = prediction(clf, X_test, Y_test)
    # read images 

    new = np.reshape(predict, (375, 1242))
    # plt.imshow(new)
    # plt.show()
    base_name = os.path.basename(Ytrain_path + gt_train_images[20])
    # save the image at the required path
    io.imsave('SegmentedImages/' + base_name, new)
