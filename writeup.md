# **Traffic Sign Recognition** 

## Table of Contents ##
- [Data Set Exploration](#data-set-exploration)
- [Preprocessing](#preprocessing)
- [Model Architecture](#model-architecture)
- [Model Training and Evaluation](#model-training-and-evaluation)
    - [Training Experiments](#training-experiments)
    - [Testing On New Images](#testing-on-new-images)
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
[traffic_sign_1]: ./traffic_external_test_images/class11_rightofwaynextintersection/class11_rightofwaynextintersection_image01.jpg "Traffic Sign 1"
[traffic_sign_2]: ./traffic_external_test_images/class12_priorityroad/class12_priorityroad_image01.jpg "Traffic Sign 2"
[traffic_sign_3]: ./traffic_external_test_images/class12_priorityroad/class12_priorityroad_image02.jpg "Traffic Sign 3"
[traffic_sign_4]: ./traffic_external_test_images/class13_yield/class13_yield_image01.jpg "Traffic Sign 4"
[traffic_sign_5]: ./traffic_external_test_images/class13_yield/class13_yield_image01.jpg "Traffic Sign 5"
[traffic_sign_6]: ./traffic_external_test_images/class14_stop/class14_stop_image01.jpg "Traffic Sign 6"
[traffic_sign_7]: ./traffic_external_test_images/class14_stop/class14_stop_image02.jpg "Traffic Sign 7"
[traffic_sign_8]: ./traffic_external_test_images/class14_stop/class17_noentry_image01.jpg "Traffic Sign 8"
[traffic_sign_9]: ./traffic_external_test_images/class32_endofallspeedpassinglimit/class32_endofallspeedpassinglimit_image01.jpg "Traffic Sign 9"
[traffic_sign_10]: ./traffic_external_test_images/class38_keepright/class38_keepright_image01.jpg "Traffic Sign 10"
[traffic_sign_11]: ./traffic_external_test_images/class40_roundabout/class40_roundabout_image01.jpg "Traffic Sign 11"
[traffic_sign_12]: ./traffic_external_test_images/class40_roundabout/class40_roundabout_image02.jpg "Traffic Sign 12"
[all_test_traffic_signs]: ./report_images/all_test_images.png "all test images"

[confusion_matrix]: ./report_images/confusion_matrix.png "confusion matrix of the current model"
[testing_accuracy_vs_training_set_size]: ./report_images//TestAccuracyVSTrainingSetSize.png "confusion matrix of the current model"

[trainging_set_stat_image]: ./report_images/training_set_stats.png "Training Set Stat"
[sampled_class_00]: ./report_images/sampled_class00.png "Class 00 sampled images"
[sampled_class_09]: ./report_images/sampled_class09.png "Class 09 sampled images"
[sampled_class_17]: ./report_images/sampled_class17.png "Class 17 sampled images"

[hist_eq_1_before]: ./report_images/histogramEqualize/image1.png "histogram equalize before image 1"
[hist_eq_1_after]: ./report_images/histogramEqualize/histEq_image1.png "histogram equalize after image 1"
[hist_eq_3_before]: ./report_images/histogramEqualize/image3.png "histogram equalize before image 3"
[hist_eq_3_after]: ./report_images/histogramEqualize/histEq_image3.png "histogram equalize after image 3"


[web_result_1]: ./report_images/webImagePredictions/test_result1.png
[web_result_2]: ./report_images/webImagePredictions/test_result2.png
[web_result_3]: ./report_images/webImagePredictions/test_result3.png
[web_result_4]: ./report_images/webImagePredictions/test_result4.png
[web_result_5]: ./report_images/webImagePredictions/test_result5.png
[web_result_6]: ./report_images/webImagePredictions/test_result6.png
[web_result_7]: ./report_images/webImagePredictions/test_result7.png
[web_result_8]: ./report_images/webImagePredictions/test_result8.png
[web_result_9]: ./report_images/webImagePredictions/test_result9.png
[web_result_10]: ./report_images/webImagePredictions/test_result10.png
[web_result_11]: ./report_images/webImagePredictions/test_result11.png
[web_result_12]: ./report_images/webImagePredictions/test_result12.png


Here is a link to my [project code](Traffic_Sign_Classifier.ipynb) and the [additional work](Traffic_Sign_Classifier-Backup.ipynb)

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

As a last step, I normalized the image data to make sure that the input values range from -0.5 - 0.5. The input intensity array are 8-bit (`dtype=uint8`) i.e. the intensity values range from 0 - 256. To apply rescaling, element-wise subtract 128 from the input intensity value then divide by 256. Here's my implementation of the image normalization. 

```python
def normalize(image):
    return (image - 128)/256
```
In the first iteration of the model training, I train a baseline model with the imbalance training set as I'm interested in how this base line model perform without generating additional data. To add more data to the the data set, I am planning to use `keras`'s `datagen` module. I have implemented the code for generating more dataset here. 

```python
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False,
        fill_mode='nearest')
```

### Model Architecture ###
###### back to [Table of Contents](#table-of-contents)

My baseline model architecture is inspired by LeNet. Since there are more classes in Traffic Sign data sets than MNIST, I add a one more Convolution-Relu-MaxPool stack and two more fully connected layers to make the model deeper. Here's the final architecture of my baseline model. Having more convolutional layers allows for learning hierarchically higher level features that composed of low-level gradient features (directional 1D edges).

| Layer                 		|     Description	        					| 
|:-----------------------------:|:---------------------------------------------:| 
| Input         		        | 32x32x3 RGB image   							| 
| Stack 1: Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x16 	|
| Stack 1: RELU Activation		|												|
| Stack 1: Max pooling	      	| 2x2 stride, outputs 14x14x16 		    		|
| Stack 2: Convolution 5x5     	| 1x1 stride, same padding, outputs 10x10x64 	|
| Stack 2: RELU					|												|
| Stack 2: Max pooling	      	| 2x2 stride, outputs 5x5x64 		    		|
| Stack 3: Convolution 5x5     	| 1x1 stride, same padding, outputs 5x5x256 	|
| Stack 3: RELU					|												|
| Stack 3: Max pooling	      	| 2x2 stride,  outputs 2x2x256   				|
| Flatten Layer 	        	| outputs 1024 									|
| Fully connected 1	        	| outputs 512									|
| Fully connected 2		        | outputs 256 									|
| Fully connected 3		        | outputs 128 									|
| Softmax				        | outputs 43 									|
 

### Model Training And Evaluation ###
###### back to [Table of Contents](#table-of-contents)
#### Training Experiments ####
To train the model, I set the placeholders for the inputs and the the labels as array of 32-bit float array and one hot encoded 32-bit int array respectively. During the training, the model weights are updated iteratively to minimize the softmax cross entropy loss of the predicted and the target classes. For the cross entropy minimization, I use tensorflow's [AdamOptimizer](https://arxiv.org/abs/1412.6980v8) (Adaptive Moment Estimation) algorithm - a variant of mini-batch stochastic gradient descent with adaptive learning rate and momentum which has been shown to work well for supervised computer vision tasks.

I started off using the LeNet architecture with the output size set to 43 (instead of 10 in the MNIST example). Without any modification to the model architecture, LeNet was able to achieve around 0.6 validation accuracy which gives me reassurance that this architecture is a good starting point, but might not be deep enough for Traffic Sign classification. While experimenting with the model architecture, I initially set the batch size to be small (starting from 128). With this initial model training, I start to get a sense of how quickly the validation accuracy improves over training interations and how these changaes as I add more layers and increase the training batch size. For the final training, I chose the batch size to as large as it would fit into my personal computer memory and I found 1024 to work well. After finalizing the model architecture and batch size, I experimented with varying the learning rate from 0.1 to 0.00001. I found that the optimal value to be around 0.001 and anything below 0.0001 to be too slow. With the learning rate = 0.001, I reproducibly observe that athe validation accuracy rapidly increases from ~0.3 in the first iteration to 0.9 after ~20 iterations and gradually incrementally increases after 30 iterations and in most training the validation accuracy does not improve after 50 iteractions. 

Here is the summary of my approaches to tune the model architecture while using the rate of increase in validation accuracy as a proxy of model training improvements: 
* Start with LeNet Architecture
* Increase the number of units
* Add Convolutional Layers
* Add Fully Connected Layers
* Lower the learning rate
* Increase batch size

With the final architecture discussed above, my modified LeNet model was able to achieve the following accuracy:
* training set accuracy of **1.000**
* validation set accuracy of **0.947**
* test set accuracy of **0.910**

I'm planning to add dropout layers in the next iteration of the model, but I'm running out of time. 

#### Testing On New Images ####
For testing on new images, I downloaded 12 German traffic signs from on the Google images, cropped and resized them to 32x32x3. Here are the example of the images after cropping and resizing:

![all_test_traffic_signs][all_test_traffic_signs] 

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook and is also included here
```python
import operator
import json

def top_n_class(prob, num_top_n=2):    
    top_n = [(signname_dict[i], prob[i]) for i in np.argsort(prob)[::-1]][:num_top_n]
    return top_n

with tf.Session() as sess:
    sess = tf.get_default_session()
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    soft_max_ops = tf.nn.softmax(logits)
    predicted_probabilities = soft_max_ops.eval(feed_dict={x: X_test_external.astype('uint32')})
    data_with_predictions['top_5_prediction'] = [top_n_class(p, num_top_n=5) for p in predicted_probabilities]
    data_with_predictions['Truth_SignName'] = data_with_predictions['y_true'].apply(lambda x: signname_dict[x])
```
The model was able to correctly guess 10 of the 12 traffic signs, which gives an accuracy of **83.33%**, while the testing set accuracy quite a bit higher (**91.0%**). The top class model predicted probabilities is summarized in the table below

| Truth                                 | Prediction                            | Probability | 
|:-------------------------------------:|:-------------------------------------:|:-----------:|  
| Right-of-way at the next intersection | Right-of-way at the next intersection | 1.0         | 
| Priority road                         | Priority road                         | 1.0         |
| Priority road                         | Priority road                         | 1.0         | 
| Yield                                 | Yield                                 | 1.0         |
| Yield                                 | Priority road                         | 0.650       | 
| Stop                                  | Stop                                  | 0.876       |
| Stop                                  | No entry                              | 1.0         | 
| No entry                              | No entry                              | 1.0         |
| End of all speed and passing limits   | End of all speed and passing limits   | 0.999       | 
| Keep right                            | Keep right                            | 1.0         |
| Roundabout mandatory                  | Roundabout mandatory                  | 1.0         |
| Roundabout mandatory                  | Roundabout mandatory                  | 1.0         |

For 11 of out 12 images, the model is relatively sure that about its prediction with only one image in which the model predicts a probability of 0.65 (the model predictions the probabilities > 0.8 for all the other images). The first misclassified image is a yield sign with a white rectangle on the bottom. The model classifies it as "Priority Road" with probability 0.65, i.e. the model is not entirely sure of its prediction. 
![web_result_5][web_result_5]

The second misclassified image is a rotated stop sign in which the model prediction it's a "No entry" with probability of 1.0. This is an manifestation that the model does not seem to generalize well beyond the provided training and validation sets as simple rotation of the sign can "fool" the network to misclassify the input. The image of ths misclassified rotated Stop sign is shown here. 
![web_result_7][web_result_7]

To gain a more comprehensive understanding of the current model performance, I plot the confusion matrix below. Most of the off diagonal entries have very low probabilities, with a small number of these having probabilities between 0.1 and 0.5.  
![confusion_matrix][confusion_matrix]

I'm also interested in understanding whether the imbalance training set affects the testing set accuracy. The plot of testing set accuracy as a function of training set size shows that all the classes with testing set accuracy < 0.75 all have training set size < 1000 images. On the other hands, not all classes with small training set size < 1000 images per class have high accuracy. My hypothesis is that the images within classes that have low testing set accuracy are confusible among themselves.

![testing_accuracy_vs_training_set_size][testing_accuracy_vs_training_set_size]

### Suggestion for Improvements ###
There 3 areas I would like to improve on the current model 
1. Balance training set size by generating more data with Image Augmentation
If I have more time to work on the project, I will use Keras `Data Augmentation` library to generate more images per classes with augmentation including rotation, flipping, and blurrying. Having more images per classes should provide the model with more examples at training time and might mitigate overfitting. Balancing the training set by feeding the model with the same number of training images per classes help train the model to not bias toward any particular classes. More importantly the augmentation/aberration should help the model to generalize better. 

2. Experimenting with adding Dropout layer
Adding Dropout layer to the model has been shown to help deep neural network to avoid overfitting.

3. Allow model to predict on Traffic sign

###### back to [Table of Contents](#table-of-contents)

