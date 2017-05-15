**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[car_not_car]: ./output_images/car_not_car.png
[hog]: ./output_images/hog.png
[sws]: ./output_images/sliding_window_search.png
[pipe_eg]: ./output_images/pipeline_examples.png
[heatmaps]: ./output_images/heatmaps.png
[labels_map]: ./output_images/frame955_res_heat.png
[output_bboxes]: ./output_images/frame955_out_bboxes.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

## **Code Structure**
- The code is decoupled into components in separated files.
- The main script `detect_vehicle.py` is located at the root of project folder, every Python script other then that is placed in `utils/`.
- The `Classifier` class is a wrapper class for models.
    - The `ManualFeatureClassifier` takes care of the feature extraction and scale handling in this project when processing videos. Currently, it has two subclasses, namely `SVMClassifier` and `ANNClassifier`. The first one is used in the submission.
    - The `CNNClassifier` is a separated class as CNN extracts features for itself.
- The training scripts are `train_*.py` where `*` is the name of model. They are used to train the corresponding model given a dataset and output file arguments.
- Helper functions are contained in the scripts `feature_extraction.py` and `utils.py`.
- The trained linear SVM model is saved as `model.svm` in the project root.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for feature extraction is present in different locations. The `hog` function in `skimage.feature` is used to extract HOG features and a wrapper is provided in lines 6 through 23 in `utils/feature_extraction.py`.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![Car and not-Car][car_not_car]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `RGB` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![HOG features][hog]

#### 2. Explain how you settled on your final choice of HOG parameters.

A thorough search through the combination of the following set of parameters is conducted by training a linear SVM and generating the corresponding detection video using `test_video.mp4`.

- `color_space = 'RGB', 'HSV', 'HLS', 'YUV', 'LUV', 'YCrCb'` for all three feature types
- `orient = 8, 9, 12` for HOG features
- `spatial_size = (16, 16), (32, 32)` for spatial binned color
- `hist_bins = 16, 32` for color histogram

It turns out the following set of parameters works best:

- `color_space = 'YCrCb'`
- `orient = 9`
- `spatial_size = (16, 16)`
- `hist_bins = 16`

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the spatial binned color, color histogram and the HOG features, with parameters described above. The code for extracting different features is in `utils/feature_extraction.py`. The training code is included in `utils/train_svm.py`. This script accepts a dataset directory and an output model filename which should end in `.svm` for later processing in the vehicle detection part.

The training code runs as follows:
```
1. Extract features for each `car` and `noncar` data point
2. Generate labels: `car` = 1, `noncar` = 0
3. Split dataset (randomly) into training and validation sets (ratio = 4:1)
4. Fit a scaler, and scale the data to be of zero mean and unit variance
5. Train a linear SVM classifier using 3-fold cross-validation.
6. Calculate and present training and test accuracies
7. Save the model together with the parameters and scaler.
```

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search for the linear SVM is included in the method `predict_all()` (lines 29 through 116) in the class `ManualFeatureClassifier` in `utils/manualfeatureclassifier.py`.

This method accepts five parameters:

- `image`: current frame image
- `ystart`: start of y-position to be searched
- `ystop`: end of y-position to be searched
- `scale`: scale of search windows
- `pix_per_step`: pixel to skip per step

The parameters specify where in the image the classifier should perform the search and the scale of the search windows relative to the dimension accepted by the model. The method first extract the search area and scale it by `1/scale`. By doing this one can use the dimension accepted by the model directly in extracting the content of the search windows as this is effectively equivalent to having each search window scaled by `scale`. Then, the HOG feature map is computed on the scaled search area for sub-sampling. The search window starts from the top-left of the search area and goes all the way to the right. This horizontal search is repeated at **four** levels where each of them differ by `pix_per_step = 16` pixels (in the scaled search area image). This configuration is due to the fact that farther cars (smaller in the image) will only be found near the top of the search area, and those closer ones (larger in the image) will only be found near the bottom. Therefore, one does not need to search for cars near the bottom with small search window.

The method then extract the features according to the parameters included in the model file (which ends in `.svm` for SVM models).

Finally, it feeds the features of each sub-image to the SVM classifier for prediction. A heat map is then constructed to indicate the search windows with positive reponses from the model. This is achieved by first to creating a zero array with the same size as the scaled search area image, setting the values covered by search windows with positive response to 1, then resizing it back to the original search area size, and finally placing it in the corresponding position in a zero array with size the same as the original image. Please refer to the code in lines 90 through 97 in `utils/manualfeatureclassifier.py`.

The scales are determined by actually measuring the size of cars in the video frames (in terms of pixels). Eight scales from 0.5 to 2.25 are chosen for the search (line 13 in `detect_vehicle.py`)

The overlapping percentage is selected to be 75% so that a fair amount of new information is present in the next window while keeping the computational cost in a reasonable range.

Here is an illustration of the search windows:

![Sliding Windows][sws]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on eight scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. The pipeline employs the HOG sub-sampling technique so that the HOG features of different sliding windows can be extracted directly from a precomputed HOG feature map. The extracted features of sliding windows are fed to the classifier at the same time to reduce call overheads and take advantage of vectorization in the sklearn library. Here are some example images:

![Pipeline example][pipe_eg]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./ycrcb_out.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video (lines 107-115 in `manualfeatureclassifier.py`).  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions (lines 79-83 in `detect_vehicle.py`). There are two thresholds for the heatmap to go through (lines 50-66 in `detect_vehicle.py`). The heatmap will first go through a lower threshold to trim out excessive boundaries of the detections. Then, `scipy.ndimage.measurements.label()` is used to identify individual blobs in the heatmap. The heat value of each blob is then set to the maximum value in that component. The heatmap will next go through the second, higher, threshold to filter out components with insufficient votes. Each remaining blob is then assumed to correspond to a vehicle. Bounding boxes are constructed to cover the each of each blob remained.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes drawn on one of the frames:

### Here are six frames and their corresponding heatmaps:

![Heatmaps][heatmaps]

### Here is the output of `scipy.ndimage.measurements.label()` on the heatmap from frame 955:
![Labels map][labels_map]

### Here the resulting bounding boxes are drawn onto the frame above:
![Output bboxes][output_bboxes]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

One of the problems of this model is that the performance of classification is color-dependent. This problem can be identified in the video where the bounding box of the white car on the right does not seem to be very stable (0:21~). However, the bounding box on the black car stays there perfectly. As the color may vary in an unexpected way, one should not rely on those factor in recognition. Instead, it would be better to focus on the shape of the objects. This can be achieved in various ways, such as using Sobel operator to extract gradient information, or one can also use grayscale images when performing feature extraction.

Another problem of this pipeline is that it searches for vehicles frame by frame independently (due to time limitation). As a result, the false positive handling using heatmap double-thresholding is not working as effective as expected (0:21, 0:40, 0:42). A modification to the pipeline would be heatmap tracking. Each heatmap can be saved for several frames and the resulting bounding boxes should come from the resulting heatmap that includes the votes from a number of the most current frames. This may cause an amount of delay in the detection, but the effect is negligible if the ratio of number of voting frames to number of frames per second is sufficiently small.

