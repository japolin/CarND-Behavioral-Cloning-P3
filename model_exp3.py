#  experiment 3 CNN model
#  import libraries

import os
import csv

# create empty lists to receive lines
lines = [] 

# reading in lines from csv file
with open('./data/driving_log.csv') as csvfile: 
    reader = csv.reader(csvfile)
    # skipping heading
    next(reader, None) 
    for line in reader:
        lines.append(line)

#  import libraries
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

# splitting the dataset to train and validation 
train_samples, validation_samples = train_test_split(samples,test_size=0.15) 

#  import libraries
import cv2
import numpy as np
import sklearn
import matplotlib.pyplot as plt

# generator to work with large dataset
def generator(lines, batch_size=32):
    num_samples = len(lines)
    while 1: 
        #shuffling the total images
        shuffle(lines)
        for offset in range(0, num_samples, batch_size):
            batch_samples = lines[offset:offset+batch_size]
            images = []
            angles = []
            # step through the batch_sample one by one
            for batch_sample in batch_samples:
                    # taking images in the the order of center, left, and right
                    for i in range(0,3):
                        name = './data/IMG/'+batch_sample[i].split('/')[-1]
                        center_image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB) 
                        center_angle = float(batch_sample[3])
                        images.append(center_image)
                        
                        # add steering angle corrections to the left and righ
                        if(i==0):
                            angles.append(center_angle)
                        elif(i==1):
                            angles.append(center_angle+0.2)
                        elif(i==2):
                            angles.append(center_angle-0.2)
                        
                        # step through the images and steering angles to augment the dataset
                        images.append(cv2.flip(center_image,1))
                        if(i==0):
                            angles.append(center_angle*-1)
                        elif(i==1):
                            angles.append((center_angle+0.2)*-1)
                        elif(i==2):
                            angles.append((center_angle-0.2)*-1)
                            
            # grap the augmented imageas and steering angles             
            X_train = np.array(images)
            y_train = np.array(angles)
            
            # hold dataset until the generator runs
            yield sklearn.utils.shuffle(X_train, y_train)
            
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# convolutional neural network design
#  import keras libraries
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Activation, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers import Lambda, Cropping2D

# create a regression network
# starting with an empty model
model = Sequential()

# add new layers for regression network
# adding lambda layer to parallelize image normalization
# to ensure the mode will normalize input images
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

# trim image to only see section with road
model.add(Cropping2D(cropping=((70,25),(0,0)))) 

# conv2D layer 1 - filters = 24, filter size= 5x5, strides= 2x2, activiation = elu
model.add(Conv2D(24,(5,5),strides=[2, 2]))
model.add(Activation('elu'))

# conv2D layer 2 - filters = 36, filter size= 5x5, strides= 2x2, activiation = elu
model.add(Conv2D(36,(5,5),strides=[2, 2]))
model.add(Activation('elu'))

# conv2D layer 3 - filters = 48, filter size= 5x5, subsample= 2x2, activiation = relu, dropout = 0.25
model.add(Conv2D(48,(5,5),strides=[2, 2]))
model.add(Activation('elu'))

# conv2D layer 4 - filters = 64, filter size= 3x3, activiation = elu
model.add(Conv2D(64,(3,3)))
model.add(Activation('elu'))

# conv2D layer 5 - filters = 64, filter size= 3x3, activiation = elu
model.add(Conv2D(64,(3,3)))
model.add(Activation('elu'))

# flatten image from 2D to side by side
model.add(Flatten())

# fully connected layer 5
model.add(Dense(100))
model.add(Activation('elu'))

# dropout layer to avoid overfitting
model.add(Dropout(0.25))

# fully connected layer 7
model.add(Dense(50))
model.add(Activation('elu'))

# fully connected layer 8
model.add(Dense(10))
model.add(Activation('elu'))

# fully connected layer 9
model.add(Dense(1))

# show model summary
model.summary()

# compile the model
# using mean squared error since the problem is regression
model.compile(loss='mse',optimizer='adam')

# fit the model
history = model.fit_generator(train_generator, steps_per_epoch= len(train_samples),epochs = 5, verbose =1,validation_data=validation_generator,validation_steps =len(validation_samples))

# save the model
model.save('model_exp3.h5')