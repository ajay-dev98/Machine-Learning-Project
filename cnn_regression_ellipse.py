# import the necessary packages
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import datasets_ellipse
import models_ellipse
import numpy as np
import argparse
import locale
import os

# path to the main folder
loc="/home/dev/Desktop/9th Semester/ML/Project/keras-regression-cnns/cnn_modified_ellipse"
# construct the path to the input .txt file that contains information and then load the dataset
print("[INFO] loading attributes...")
inputPath = os.path.sep.join([loc, "label.txt"]) 
df = datasets_ellipse.load_attributes(inputPath)

# load the  images and then scale the pixel intensities to the range [0, 1]
print("[INFO] loading images...")
images = datasets_ellipse.load_images(df, loc)

images = images / 255.0

# partition the data into training and testing splits using 75% of the data for training and the remaining 25% for testing
split = train_test_split(df, images, test_size=0.25, random_state=42)
(trainAttrX, testAttrX, trainImagesX, testImagesX) = split

# find the largest angle in the training set and use it to scale our angles to the range [0, 1] (will lead to better training and convergence)
maxAngle = trainAttrX["angle"].max()
trainY = trainAttrX["angle"] / maxAngle
testY = testAttrX["angle"] / maxAngle

# create our Convolutional Neural Network and then compile the model using mean squared error as our loss
model = models_ellipse.create_cnn(64, 64, 3, regress=True)
opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss="mean_squared_error", optimizer=opt, metrics=['mean_squared_error'])

# train the model
print("[INFO] training model...")
model.fit(x=trainImagesX, y=trainY, 
    validation_data=(testImagesX, testY),
    epochs=200, batch_size=16)

# make predictions on the testing data
print("[INFO] predicting house prices...")
preds = model.predict(testImagesX)

# compute the difference between the *predicted* angle and the *actual* angle, then compute the percentage difference and the absolute percentage difference
diff = preds.flatten() - testY
percentDiff = (diff / testY) * 100
absPercentDiff = np.abs(percentDiff)

# compute the mean and standard deviation of the absolute percentage difference
mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)

# finally, show some statistics on our model
locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
print("[INFO] avg. angle: {}, std angle: {}".format(
	locale.currency(df["angle"].mean(), grouping=True),
	locale.currency(df["angle"].std(), grouping=True)))
print("[INFO] mean: {:.2f}%, std: {:.2f}%".format(mean, std))
