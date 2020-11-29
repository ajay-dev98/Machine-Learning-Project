#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# USAGE
# python cnn_regression.py 

# import the necessary packages
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import datasets_ellipse
import models_ellipse
import numpy as np
import argparse
import locale
import os


# mention the parent directory    
loc="/home/ashish/MACHINE LEARNING/ash"
# construct the path to the input .txt file that contains information
# and then load the dataset
print("[INFO] loading attributes...")
inputPath = os.path.sep.join([loc, "label_cos.txt"]) 
df = datasets_ellipse.load_attributes(inputPath)

# construct a training and testing split with 75% of the data used
# for training and the remaining 25% for evaluation
print("[INFO] constructing training/testing split...")
(train, test) = train_test_split(df, test_size=0.25, random_state=42) \




# find the largest label value in the training set and use it to
# scale the labels to the range [0, 1] (this will lead to
# better training and convergence)
max_value = train["function"].max()
trainY = train["function"] / max_value
testY = test["function"] / max_value

 #process the attributes data by performing min-max scaling
# on continuous features, one-hot encoding on categorical features,
# and then finally concatenating them together
print("[INFO] processing data...")
(trainX, testX) = datasets_ellipse.process_angle(df, train, test)


# create our MLP and then compile the model using mean absolute
# percentage error as our loss, implying that we seek to minimize
# the absolute percentage difference between our price *predictions*
# and the *actual prices*
model = models_ellipse.create_mlp(trainX.shape[1], regress=True)
opt = Adam(lr=1e-4, decay=1e-3 / 200)
model.compile(loss="mean_squared_error", optimizer=opt)
# train the model
print("[INFO] training model...")
history = model.fit(x=trainX, y=trainY, 
	validation_data=(testX, testY),
	epochs=300, batch_size=16)


print("[INFO] predicting angle...")
preds = model.predict(testX)
# compute the difference between the *predicted* value and the
# *actual* value, then compute the percentage difference and
# the absolute percentage difference
diff = preds.flatten() - testY
percentDiff = (diff / testY) * 100
absPercentDiff = np.abs(percentDiff)
predicted = preds.flatten()
actual = testY.to_numpy()
# compute the mean and standard deviation of the absolute percentage
# difference
mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)
# finally, show some statistics on our model
locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
print("[INFO] avg.prediction: {}, std_prediction: {}".format(
	locale.currency(df["function"].mean(), grouping=True),
	locale.currency(df["function"].std(), grouping=True)))
print("[INFO] mean: {:.2f}%, std: {:.2f}%".format(mean, std))


# open a file named 'predicted_--.txt' 
# and write the predicted and actual values in it.
with open('predicted_cos.txt', 'w') as file:
    for i in range(len(predicted)):
        file.write('{},{}\n'.format(actual[i],predicted[i]))
        


training_loss = history.history['loss']
test_loss = history.history['val_loss']

# Create count of the number of epochs
epoch_count = range(1, len(training_loss) + 1)

# Visualize loss history
#plt.xscale('log')
plt.yscale('log')
plt.plot(epoch_count, training_loss, 'r--')
plt.plot(epoch_count, test_loss, 'b-')
plt.title("f = cos(x)")
plt.legend(['Training Loss', 'Test Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig("f_cos")
plt.show();
# shows the error value relative to each epoch. 
##%

# plots the histogram of the error.
y2,bins2,z2 = plt.hist(np.log(absPercentDiff),bins='auto',label = 'Testing Loss')
plt.xlabel('loss', fontsize='15')
plt.ylabel('f(loss)',fontsize='15')
plt.title("f = cos(x)",fontsize='18')
plt.legend()
plt.show()
