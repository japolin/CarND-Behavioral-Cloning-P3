# **Using Deep Learning to Clone Driving Behavior** 

The overall goal of this project is to train a deep neural
network to drive a car like a human.

The objectives of this project were the following:

* Files & Code Quality
* Model Architecture and Training Strategy
* Architecture and Training Documentation
* Simulation Testing


[//]: # (Image References)

[image1]: ./examples/model_exp1_loss.png "Exp1 Model"
[image2]: ./examples/model_exp2_loss.png "Exp2 Model"
[image3]: ./examples/model_exp3_loss.png "Exp3 Model"
[image4]: ./examples/model_exp1.png "Model Visualization"
[image5]: ./examples/image5.jpg "Representative Image"
[image6]: ./examples/image6.jpg  "Normal Image"
[image7]: ./examples/image7.jpg  "Recovery Image"
[image8]: ./examples/image8.jpg "Back Image"
[image9]: ./examples/image9.jpg "Track 2 Image"
[image10]: ./examples/image10.png "Normal Image"
[image11]: ./examples/image11.png "Flipped Image"

___

### Behavioral Cloning Project Specifications

### I considered the [rubric](https://review.udacity.com/#!/rubrics/1968/view) points individually.  Consequently, the writeup below describes how I addressed each point in my implementation. 

---
### Abstract

#### Purpose


* Use the simulator to collect data of good driving behavior
* Design and build a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test models that successfully drives around track one without leaving the road
* Summarize the results with a written report

Collect data of good driving behavior using a simulator.  Design, build, and train a convolutional neural network to automatically drive a car in the simulator.     

#### Method

More than 24,000 image samples were collected from the simulator.  Also, steering angles, throttling, braking, and vehicle speed of the car were collected.  The data collected were preprocessed using Python 3.0, OpenCV, and Numpy. Data augmentation were implemented on image samples.  Three experiments to design, train, and validate a convolutional neural network (CNN) were implemented using Keras.

#### Results

Three experimental CNNs were trained and validated on a driving simulation dataset.  Data augmentation resulted in over 49,000 image samples.  For experiments one and two, the training and validation loss metrics were achieved for 20 (epoch) passes over the training set while experiment three was for 5 epochs.  The training loss decreased in all experiments while the validation loss only decreased in the third experiment.  The training times per epoch for experiments one, two, and three were 1.01 minutes, 1.11 minutes, and 31.8 minutes, respectively.

#### Conclusion

After testing the models in the simulator, no tire left the drivable portion of the track.  While good driving behavior has shown to be ideal in all test simulations, the training time necessary for running many experiments to improve model overfitting and achieve convergence along the training set appeared to be impractical.  Consequently, determining the optimal architecture of the CNN for this given task remains mainly empirical. The codes to this project can be found using this [github link](git@github.com:japolin/git@github.com:japolin/CarND-Behavioral-Cloning-P3.git).

---
### Files & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model_exp1.py, model_exp2.py, and model_exp3.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model_exp1.h5, model_exp2.h5, and model_exp3.h5 containing a trained convolution neural network 
* Behavioral Cloning Project.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
#### 3. Submission code is usable and readable

The model files (model_exp1.py, model_exp2.py, and model_exp3.py) contain the code for training and saving the convolution neural network. The files show the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The deep neural network architectures used for this project consists of a convolutional neural network and fully connected layers.  The body of the network implemented the convolutional part of the network.  This body section works as a feature extractor similar to the function of a classical computer vision technique.  The head implements the fully-connected section with the last layer where the steering angle being the single output.  The head section of this network is considered the regressor similar to the function of a classical machine learning regressor.

The models I experimented with consisted of a convolutional neural network with 5x5 or 3x3 filter sizes and depths between 24 and 64 (model_exp1.py lines 98-116; model_exp2.py lines 142-159; model_exp3.py lines 96-114). 

The model includes RELU or ELU layers to introduce nonlinearity (model_exp1.py lines 98-116; model_exp2.py lines 142-159; model_exp3.py lines 96-114), and the data is normalized in the model using a Keras lambda layer (model_exp1.py line 93; model_exp2.py line 136; model_exp3.py line 91). 

#### 2. Attempts to reduce overfitting in the model

All experimental models contained dropout layers in order to reduce overfitting (model_exp1.py lines 101-116; model_exp2.py lines 144-159; model_exp3.py line 124).

Augmentations of the dataset also attempted to prevent overfitting of the training data (model_exp1.py lines 70-74; model_exp2.py lines 112-117; model_exp3.py lines 58-64). 

The models were trained and validated on different datasets to ensure that the models were not overfitting (model_exp1.py line 144; model_exp2.py line 187; model_exp3.py line 145). The models were tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

All models used an adam optimizer.  So the learning rate was not tuned manually (model_exp1.py line 141; model_exp2.py line 184; model_exp3.py line 142).

#### 4. Appropriate training data

For all experiments, I used the training data to keep the vehicle driving on the road. I used a combination of center lane driving, flipping the images to augment the data, and collected data from the second track for the second experimental model. 

For details about how I created the training data, see the next section. 

### Architecture and Training Documentation

#### 1. Solution Design Approach

Using Keras, the overall strategy was to derive a model architecture that could take images that simulated good driving behavior as the input into the convolutional neural network and map the raw pixel values to steering angles commands to test driving behavior in the simulator.

My first step was to use a convolutional neural network model similar to the network architecture used by Bojarski et al. [1].  I thought this model might be appropriate because the model appeared to learn steering angles using road features.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a decreasing training loss but the validation loss remained plateaued and slightly increased (see Figure 1). This implied that the model was overfitting.  It did well on the training set but not so well on the test set.  The training time was 1.01 minutes per epoch for 20 epochs.

![alt text][image1]

To combat the overfitting, I collected more data from the second track to see if this would improve the model.  I found that the first model still had a decreasing training loss but even worse validation loss that remained plateaued and slightly increased (see Figure 2). This also implied that the model was overfitting. The training time was 1.11 minutes per epoch for 20 epochs.  

![alt text][image2]

Then I modified the model and train the model with only the data from the first track.  I found that the second model had a decreasing training and validation loss but the validation loss slightly increased (see Figure 3). This also implied that the model was overfitting. Also, I expected that the test set to converge along with the training set but they were not similar.  The training time was 31.8 minutes per epoch for 5 epochs.  

![alt text][image3]

The final step was to run the simulator to see how well the car was driving around track one. Test simulations based on experiments one and two performed better than experiment three.  Compared to experiments one and two, there were a few spots where the vehicle fell off the track in experiment three.  To improve the driving behavior in experiment three, training the model with more epochs may improve the driving behavior but the training time is too expensive.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture chosen was model_exp1.py (lines 98-116).  It consisted of a convolutional neural network with the following layers and layer sizes.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 color images  				        |
| Lambda                | normalization function                        |
| Cropping              | image trimming                                |
| Convolution 5x5     	| 24 filters, 2x2 stride, RELU activiation, outputs=78.5x158.5x24     |
| Dropout   	      	| rate = 0.25                                   |
| Convolution 5x5     	| 36 filters, 2x2 stride, RELU activiation, outputs=37.75x77.75x36     |
| Dropout   	      	| rate = 0.25                                   |
| Convolution 3x3     	| 48 filters, 2x2 stride, RELU activiation, outputs=18.375x38.375x48     |
| Dropout   	      	| rate = 0.25                                   |
| Convolution 3x3     	| 64 filters, 2x2 stride, RELU activiation, outputs=8.6875x18.6875x64     |
| Dropout   	      	| rate = 0.25                                   |
| Flatting              | input = 8.6875x18.6875x64, output = 1204.69				|
| Fully Connection      | input = 1204.69 , output = 500				|
| Fully Connection      | input = 500 , output = 100				    |
| Fully Connection      | input = 100 , output = 50				        |
| Fully Connection      | input = 50 , output = 10		                |
| Fully Connection (Output)      | input = 10 , output = 1				|

Here is a visualization of the architecture...

![alt text][image4]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image5]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to the center. These images show what recovery looks like starting from the top image to the bottom.

![alt text][image6]

![alt text][image7]

![alt text][image8]

Then I repeated this process on track two in order to get more data points.  See a representative image below.

![alt text][image9]

To augment the dataset, I also flipped images and angles thinking that this would increase additional diversity of the data without actually collecting new data.  This helps to increase the generalization of the model. For example, here is an image that has then been flipped where the normal image is on the top and flipped image on the bottom.

![alt text][image10]

![alt text][image11]

After the collection process, I had 38,572 training samples and 9,644 samples for validation. I then preprocessed this data by using the lambda function to normalize the input images.  The images were trimmed to only see the section of the road.  While fitting the training set, I randomly shuffled the training set and put 20% of the data into a validation set.  The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 20 as evidenced of decreasing training loss but the validation loss didn't by remaining plateaued and slightly increasing (see Figure 2). I used an adam optimizer so that manually training the learning rate wasn't necessary.

[1] Bojarksi M, Firner B, Fleep B, Jackel L, Muller U, Zieba K, Del Testa D. End-to-End Deep Learning for Self-Driving Cars. *Nvidia Developer Blog* 2016. https://developer.nvidia.com/blog/deep-learning-self-driving-cars/.
