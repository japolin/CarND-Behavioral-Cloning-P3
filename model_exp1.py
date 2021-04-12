#  experiment 1 CNN model
#  import libraries

import os
import csv
import cv2
import math
import scipy
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# create empty lists to receive lines
lines = []

# reading in lines from csv file
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    # skipping heading
    next(reader, None)
    for line in reader:
        lines.append(line)
        
# create empty lists to receive images and steering angles
images = []
measurements = []
car_images=[]
steering_angles =[] 
X_train=[]
y_train=[]
augmented_images = []
augmented_angles = []

# step through the lines one by one
for line in lines:
    image_center_path = line[0]
    image_left_path = line[1]
    image_right_path = line[2]
    
    image_center_filename = image_center_path.split('/')[-1]
    image_left_filename = image_left_path.split('/')[-1]
    image_right_filename = image_right_path.split('/')[-1]
    
    current_image_center_path = './data/IMG/' + image_center_filename
    current_image_left_path = './data/IMG/' + image_left_filename
    current_image_right_path = './data/IMG/' + image_right_filename
    
    image_center = mpimg.imread(str(current_image_center_path))
    image_left = mpimg.imread(str(current_image_left_path))
    image_right = mpimg.imread(str(current_image_right_path))
    
    # add images to data set
    car_images.append(image_center)
    car_images.append(image_left)
    car_images.append(image_right)
    
    # add steering angle corrections to the left and right
    correction = 0.30
    steering_center = float(line[3])
    steering_left = steering_center + correction
    steering_right = steering_center - correction
    
    # add angles to data set
    steering_angles.append(steering_center)
    steering_angles.append(steering_left)
    steering_angles.append(steering_right)
    
# step through the images and steering angles to augment the dataset
for image, angle in zip(car_images, steering_angles):
    augmented_images.append(image)
    augmented_angles.append(angle)
    augmented_images.append(cv2.flip(image,1))
    augmented_angles.append(angle*(-1.0))
    
# grap the augmented imageas and steering angles    
X_train = np.array(augmented_images)
y_train = np.array(augmented_angles)  

# convolutional neural network design
#  import keras libraries
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D

# create a regression network
# starting with an empty model
model = Sequential()

# add new layers for regression network
# adding lambda layer to parallelize image normalization
# to ensure the mode will normalize input images
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

# trimming image to only see section with road
model.add(Cropping2D(cropping=((70,25),(0,0))))

# conv2D layer 1 - filters = 24, filter size= 5x5, subsample= 2x2, activiation = relu, dropout = 0.25
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
# dropout layer to avoid overfitting
model.add(Dropout(0.25))

# conv2D layer 2 - filters = 36, filter size= 5x5, subsample= 2x2, activiation = relu, dropout = 0.25
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
# dropout layer to avoid overfitting
model.add(Dropout(0.25))

# conv2D layer 3 - filters = 48, filter size= 3x3, subsample= 2x2, activiation = relu, dropout = 0.25
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
# dropout layer to avoid overfitting
model.add(Dropout(0.25))

# conv2D layer 4 - filters = 64, filter size= 3x3, subsample= 2x2, activiation = relu, dropout = 0.25
model.add(Convolution2D(64,3,3,subsample=(2,2),activation="relu"))
# dropout layer to avoid overfitting
model.add(Dropout(0.25))

#flatten images from 2D array to 1D array
model.add(Flatten())

# fully connected layer 5
model.add(Dense(500))   

# fully connected layer 6
model.add(Dense(100))

# fully connected layer 7
model.add(Dense(50))

# fully connected layer 8
model.add(Dense(10))

# fully connected layer 9
model.add(Dense(1))
          
# show model summary
model.summary()

# compile the model
# using mean squared error since the problem is regression
model.compile(loss='mse', optimizer='adam', metrics=['mae'] 
              
# fit the mode
history = model.fit(X_train, y_train, shuffle=True, epochs=20, validation_split=0.20, verbose=1) 

# save the model
model.save('model_exp1.h5')