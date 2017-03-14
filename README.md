##VEHICLE DETECTION

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car.png
[image1a]: ./examples/noncar.png
[image2]: ./examples/hog.jpg
[image2a]: ./examples/original.png
[image3]: ./examples/test1.jpg
[image3a]: ./examples/rectangle.jpg
[image4]: ./examples/heatmap.jpg
[image5]: ./examples/output.jpg
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the second code cell of the IPython notebook in the form of a function
`get_hog_features`.  

The function has been called within another function `extract_features` in fifth code cell of the notebook. This function has been used under
the section `training the model...` where it is fed with the image data with and without cars to train the model.
I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes respectively:

![image1]
![image1a]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

HOG
![image2]

Original
![image2a]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters starting with using `RGB` as the color space. The results were fairly impressive, however I realised that the model was not able to perform well when a dark vehicle was in shadow. `HLS` color space seemed to perform better than `RGB` and had a higher validation accuracy with the model, however it lead to a lot of false positives in the predictions. `YCrCb` had the highest and I decided to use it as the color space.

The other parameters are same as what was used during the course work and it seems to perform well.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).


I trained a linear SVM by extracting color histogram feature and HOG feature and combining them together. The code for extracting the features them and structuring them in the form that is used by the training model is in the fifth code cell of the ipython notebook as part of the function `extract_features`. This method is later used in the section `training the model...` in order to extract features for all the data set that is available for training the model. The output of both these features are combined and normalized (code cell 9) using the `StandardScaler` function.

The data set from images with cars and without cars is read seperately so it is easier to assign the labels.

The normalized data is then split into training and validation set and fed to the model to be trained with the following results after training.


`Using: 9 orientations 8 pixels per cell and 2 cells per block
Feature vector length: 8460
13.15 Seconds to train SVC...
Test Accuracy of SVC =  0.9944`



###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search has been implemented as part of the function `find_cars`(code cell 11) and is applied to every frame of the video to identify for cars.


Even before the sliding window search, the image fed to the model is first cropped so the relevant part of the data is being fed to the model. Hog sub sampling is done so as to optimize the pipeline and the overlap is specified in terms of cells steps to be taken as suggested in the course work. The output of the method is shown below.

![image3]
![image3a]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Since I could see a few false positives in the images that were tested on the model, I decided to increase the threshold of the heat map. This removed most of the false positives. The image was cropped so that the model would not spend a lot of its time trying to learn from irrelevant data.
Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![image4]
![image5]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

The above images represent the same.





---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The compromise between stable identification of all the vehicles and false positives was difficult to resolve. Even now the model is not completely successful in 100 percent vehicle identification with 0% false positives. This pipeline could be more robust by training it with more diverse data and also other classification models could be tested to see if they perform well.  

