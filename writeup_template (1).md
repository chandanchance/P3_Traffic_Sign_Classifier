# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)


[image1]: ./Screenshots/dataLoad.PNG "Data Loading"
[image2]: ./Screenshots/distribution.PNG "distribution"
[image3]: ./Screenshots/dataPreparation.jpg "dataPreparation"
[image4]: ./Screenshots/testForNewData.PNG "testForNewData"
[image5]: ./Screenshots/PredictionForNewData.PNG "PredictionForNewData"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

The data which has been loaded :

![alt text][image1]


Here is an exploratory visualization of the data set. It is a bar chart showing how the data has been distributed.

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step,  `preprocess` function has been defined which converts the images into gray scale and normalise the images.

The image which has been normalised is the below example: 

![alt text][image3]


Data augmentation was not used as the accuracy which I got without augmenting the data is itself high. Coverting the 3 channels into gray channel and normalising the data was itself sufficient to get high accuracy.

Ofcourse to increase the results or considering the real world deployment, there might be need for data augmentation.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I changed few features of the actual LeNet.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 Gray image   							| 
| Convolution 5x5     	| 1x1 stride, VALID padding, outputs 28x28x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				|
| Convolution 5x5	    | 1x1 stride, VALID padding, outputs 10x10x64	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16   				|
| Flattened     		|outputs 1600									|
| Fully Connected Layer1| input 1600 output 240							|
| RELU					|												|
| Dropout				|0.5 for Training set							|
| Fully Connected Layer2| input 240 output 84							|
| RELU					|												|
| Dropout				|0.5 for Training set							|
| Output Layer          | input 84 output 43							|

 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the above architecture with `AdamOptimizer` as optimizer,  `0.001` as the learning rate, batch size is `64` and epoch is `20`.
I also used 0.5 as the dropout (default no given while building the architecture). This was done to avoid over fitting.


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
Validation Accuracy = 0.982
Training Accuracy = 0.999
test Accuracy = 0.957



* What was the first architecture that was tried and why was it chosen?
I first tried with the actual LeNet with increasing the number of epochs, followed by varing the number of batch size.

* What were some problems with the initial architecture?
I did not find the accuracy to incease much so I thought of increasing the depth of the output from every convolution layer. 

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

I increased the number of k (depth) given as output by the convolution layers. As I had come accross in the class that decreasing the size and increasing the depth of the image would help in capturing the features.

 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4]



#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| No entry       		| No entry    									| 
| 60kmph     			| 100kmph 										|
| Left turn				| Left turn										|
| Stop          		| Stop      					 				|
| Yield     			| Yield              							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 95% which means that still every feature is not captured so all the images are not rightly classified.


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 47th cell of the Ipython notebook.

For the first image, the model is completely sure that this is a noentry (probability of 1.), and the image does contain a noentry.
For the second image, the model is relatively sure that this is a Speed limit (100km/h) (probability of 0.7), and the image does not contain a 100kmph speedlimit but contain a 60kmh.
For the third image, the model is relatively sure that this is a left_turn (probability of 0.7), and the image does contain a left_turn.
For the fourth image, the model is almost sure that this is a stop_sign (probability of 0.9), and the image does contain a stop_sign.
For the fifth image, the model is completely sure that this is a yield_sign (probability of 1.0), and the image does contain a yield_sign.

Image: noentry
Probabilities:
   1.000000 : 17 - No entry
   0.000000 : 40 - Roundabout mandatory
   0.000000 : 14 - Stop
   0.000000 : 39 - Keep left
   0.000000 : 33 - Turn right ahead

Image: 60_kmh
Probabilities:
   0.779798 : 7 - Speed limit (100km/h)
   0.067348 : 10 - No passing for vehicles over 3.5 metric tons
   0.040164 : 12 - Priority road
   0.026542 : 2 - Speed limit (50km/h)
   0.022773 : 9 - No passing

Image: left_turn
Probabilities:
   0.693886 : 34 - Turn left ahead
   0.299388 : 37 - Go straight or left
   0.005847 : 38 - Keep right
   0.000823 : 1 - Speed limit (30km/h)
   0.000047 : 14 - Stop

Image: stop_sign
Probabilities:
   0.915584 : 14 - Stop
   0.050765 : 1 - Speed limit (30km/h)
   0.030840 : 2 - Speed limit (50km/h)
   0.002743 : 33 - Turn right ahead
   0.000026 : 37 - Go straight or left

Image: yield_sign
Probabilities:
   1.000000 : 13 - Yield
   0.000000 : 3 - Speed limit (60km/h)
   0.000000 : 35 - Ahead only
   0.000000 : 34 - Turn left ahead
   0.000000 : 38 - Keep right


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


