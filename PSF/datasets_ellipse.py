# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import glob
import cv2
import os


def load_attributes(inputPath):
	# initialize the list of column names in the CSV file and then
	# load it using Pandas
	cols = ["angle", "function"]
	df = pd.read_csv(inputPath, header=None, names=cols)

	# return the data frame
	return df

def process_angle(df, train, test):
	# initialize the column names of the continuous data
	continuous = ["angle"]
	# performin min-max scaling each continuous feature column to
	# the range [0, 1]
	cs = MinMaxScaler()
	trainContinuous = cs.fit_transform(train[continuous])
	testContinuous = cs.transform(test[continuous])
    # one-hot encode the zip code categorical data (by definition of
	# one-hot encoing, all output features are now in the range [0, 1])
	
	# construct our training and testing data points by concatenating
	# the categorical features with the continuous features
	trainX = np.hstack([trainContinuous])
	testX = np.hstack([testContinuous])
	# return the concatenated training and testing data
	return (trainX, testX)


 


