# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import glob
import cv2
import os
import datasets_ellipse
import models_ellipse

def load_attributes(inputPath):
	# initialize the list of column names in the CSV file and then load it using Pandas
	cols = ["angle"]
	df = pd.read_csv(inputPath, header=None, names=cols)

	# return the data frame
	return df


def load_images(df, inputPath):
	# initialize our images array (i.e., ellipse images )
    images = []
    
    for i in df.index.values:
        basePath = os.path.sep.join([inputPath,"{}.png".format(i)])
        image = cv2.imread(basePath)
        image = cv2.resize(image, (64, 64))   # resizing each of the images
        images.append(image)
    
    return np.array(images)  
