#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# import the necessary packages
from skimage.measure import compare_ssim as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pandas as pd
import os

def mse(imageA, imageB):
	# the 'Mean Squared Error' between the two images is the
	# sum of the squared difference between the two images;
	# NOTE: the two images must have the same dimension
	err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
	err /= float(imageA.shape[0] * imageA.shape[1])
	
	# return the MSE, the lower the error, the more "similar"
	# the two images are
	return err
def compare_images(imageA, imageB, title):
	# compute the mean squared error and structural similarity
	# index for the images
	m = mse(imageA, imageB)
	s = ssim(imageA, imageB)
	# setup the figure
	fig = plt.figure(title)
	plt.suptitle("MSE: %.2f, SSIM: %.2f" % (m, s))
	# show first image
	ax = fig.add_subplot(1, 2, 1)
	plt.imshow(imageA)
	plt.axis("off")
	# show the second image
	ax = fig.add_subplot(1, 2, 2)
	plt.imshow(imageB)
	plt.axis("off")
   
	# show the images
    
	plt.show()
     
# load the images -- the original, the original + contrast,
# and the original + photoshop


df = pd.read_csv("predicted_cos.txt")
parent_dir="/home/ashish/MACHINE LEARNING/ash"

path = os.path.join(parent_dir, "images")
actual = df.to_numpy()[:,0]
pred = df.to_numpy()[:,1]
a = []
# This code reads the predicted and the actual image.
# converts it to gray color and then compares the images and calculates error.
for i in range(len(pred)):
    original = cv2.imread(os.path.join(path, str("pred_{}.png".format(i))))
    contrast = cv2.imread(os.path.join(path, str("actual_{}.png".format(i))))    
    original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    contrast = cv2.cvtColor(contrast, cv2.COLOR_BGR2GRAY)
    compare_images(original, contrast, "Original vs. Contrast")
    a.append(mse(original,contrast))
    #print(mse(original,contrast))

# calculating mean, max and the standard deviation of the error in the images. 
print(np.max(a))
print(np.mean(a),np.std(a))
#sim(original,contrast,'1st')
