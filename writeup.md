# **Traffic Sign Recognition** 

## Table of Contents ##
- [Data Set Exploration](#data-set-exploration)
- [Preprocessing](#preprocessing)
- [Model Building](#model-building)
- [Model Evaluation](#model-evaluation)
- [Suggestion for Improvements](#suggestion-for-improvements)

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./traffic_external_test_images/class01_speedlimit30/class01_speedlimit30_image01.jpg "Traffic Sign 1"
[image5]: ./traffic_external_test_images/class03_speedlimit60/class03_speedlimit60_image01.jpg "Traffic Sign 2"
[image6]: ./traffic_external_test_images/class04_speedlimit70/class04_speedlimit70_image01.jpg "Traffic Sign 3"
[image7]: ./traffic_external_test_images/class11_rightofwaynextintersection/class11_rightofwaynextintersection_image01.jpg "Traffic Sign 4"
[image8]: ./traffic_external_test_images/class12_priorityroad/class12_priorityroad_image01.jpg "Traffic Sign 5"
[trainging_set_stat_image]: ./report_images/training_set_stats.png "Training Set Stat"
[sampled_class_00]: ./report_images/sampled_class00.png "Class 00 sampled images"
[sampled_class_09]: ./report_images/sampled_class09.png "Class 09 sampled images"
[sampled_class_17]: ./report_images/sampled_class17.png "Class 17 sampled images"

[hist_eq_1_before]: ./report_images/histogramEqualize/image1.png "histogram equalize before image 1"
[hist_eq_1_after]: ./report_images/histogramEqualize/histEq_image1.png "histogram equalize after image 1"
[hist_eq_3_before]: ./report_images/histogramEqualize/image3.png "histogram equalize before image 3"
[hist_eq_3_after]: ./report_images/histogramEqualize/histEq_image3.png "histogram equalize after image 3"


Here is a link to my [project code](Traffic_Sign_Classifier.ipynb)

### Data Set Exploration ###
###### back to [Table of Contents](#table-of-contents)
The trafffic sign data sets that are downloaded from the project page consists of training, validation and testing set files. Each file is a pickled numpy arrays of RGB images that have already been cropped,and resizezd to 32x32 pixels. Many of the images have been centered while a few are not. After checking that the data set size will fit into memory, I load the entire dataset and use numpy to find to calculate summary statistics of the traffic data set:
* The size of training set is : 34799 
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

Traffic sign identification is a multiclass classification problem which takes a fixed size input (32x32x3) numpy array and output softmax probabilities of all 43 classes. Before building and training the model, I determine whether there's class imbalance in the training set by computing the number of training images per classe. Here's the bar chart that summary the number of training images in each class.

The number of training images varies widely by 10 fold from < 200 images for some classes () to > 2000 images for others. I use the training set as provide for the first few rounds of model training, but I do take note that the class imbalance in this training set might impact the overall accuracy of the model. 

![Training Set Stat][trainging_set_stat_image]

###Design and Test a Model Architecture

### Preprocessing ###
###### back to [Table of Contents](#table-of-contents)

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

![sampled_class_00][sampled_class_00]
Sample training set images in class 0. 

![sampled_class_17][sampled_class_17]
Sample training set images in class 17. 

From visual inspection, I found that images in each classes are very different in average illumination across all channels and the pixel intensities in many of the images do not span the entire dynamic range. To adjust the dynamic range of the images, I used cv2's histogram equalization on RGB channels of each images separately and restack the channels. I think there's might be signal in the different channel so I use all the color channels as input to the model.

```python
def equalizeHistRbg(image): 
    eq = image.copy()
    for dim in range(image.shape[-1]):
        eq[:, :, dim] = cv2.equalizeHist(image[:, :, dim])    
    return eq
```

Here is an example of a traffic sign image before and after histogram equalization.

![hist_eq_1_before][hist_eq_1_before]
![hist_eq_1_after][hist_eq_1_after]

![hist_eq_3_before][hist_eq_3_before]
![hist_eq_3_after][hist_eq_3_after]

As a last step, I normalized the image data to make sure that the input values range from 0 - 1. The input images are 8-bit i.e. the intensity values range from 0 - 128. Here's my implementation of the image normalization. 

```python
def normalize(image):
    return (image - 128)/image
```
In the first iteration of the model training, I train a baseline model with the imbalance training set as I'm interested in how this base line model perform without additional data. To add more data to the the data set, I am planning to use `keras`'s `datagen` module. I have implemented the code for generating more dataset here. 

```python
datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=False
)
datagen.fit(X_train.astype(float))
```

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 

### Preprocessing ###
###### back to [Table of Contents](#table-of-contents)

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


