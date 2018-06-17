## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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
[features_hog]: ./output_images/features_hog.png
[features_spatial]: ./output_images/features_spatial.png
[features_color]: ./output_images/features_color.png
[find_cars]: ./output_images/find_cars.png
[search_1]: ./output_images/search_1.png
[search_2]: ./output_images/search_2.png
[search_3]: ./output_images/search_3.png
[search_4]:  ./output_images/search_4.png
[search_comb]: ./output_images/search_comb.png
[heatmap_1]: ./output_images/heatmap_1.png
[heatmap_thres]: ./output_images/heatmap_thres.png
[heatmap_lbl]: ./output_images/heatmap_lbl.png
[drawn_img]: ./output_images/drawn_img.png
[test_images]: ./output_images/test_images.png


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `BGR` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][features_hog]

I also explored features other than HOG. Specifically bin spatial features, here is a sample

![alt text][features_spatial]

... and color histogram features

![alt text][features_color]

In the end these feature vectores where stacked together and scaled using the sci-kit learn scaler.


#### 2. Explain how you settled on your final choice of HOG parameters.

I run a thorough exploration of all possible combinations of parameters for the HOG feature extractor. I compared them using the prediction accuracy, using a separate test set, but also the time spent to extract features and train the classifier. One of my first observations was that feature extraction time varied significantly depending on parameters, up to more than 5x the minimum observed value.

Here is a table of the top 20 results based on accuracy.

|Accuracy | ColorSpace | Orientations | Pixels Per Cell | Cells Per Block | HOG Channel | Extract Time | Train time |
| :-----: | :--------: | :----------: | :-------------: | :-------------: | :---------: | :----------: | :---------:|
| 0.9977 | HSV | 10 | 16 | 2 | ALL | 123.87 | 297.52 |
| 0.9972 | YUV | 11 | 16 | 2 | ALL | 64.13 | 176.86 |
| 0.9972 | YUV | 10 | 16 | 2 | ALL | 109.9 | 267.53 |
| 0.9969 | YUV | 11 | 16 | 2 | ALL | 61.0 | 162.85 |
| 0.9969 | HSV | 11 | 16 | 2 | ALL | 139.28 | 317.89 |
| 0.9969 | HSV | 10 | 16 | 2 | ALL | 142.65 | 260.55 |
| 0.9969 | HLS | 9 | 16 | 2 | ALL | 115.59 | 323.52 |
| 0.9966 | LUV | 11 | 8 | 2 | 0 | 104.33 | 453.97 |
| 0.9966 | HSV | 9 | 16 | 2 | ALL | 139.19 | 278.51 |
| 0.9963 | YCrCb | 11 | 16 | 2 | ALL | 81.49 | 213.96 |
| 0.9963 | HLS | 11 | 16 | 2 | ALL | 136.07 | 339.64 |
| 0.9961 | HLS | 9 | 16 | 2 | ALL | 144.19 | 418.22 |
| 0.9958 | LUV | 11 | 8 | 2 | 0 | 104.52 | 440.59 |
| 0.9958 | HSV | 9 | 16 | 2 | ALL | 256.13 | 375.77 |
| 0.9958 | HSV | 9 | 16 | 2 | 0 | 60.7 | 299.34 |
| 0.9958 | HSV | 11 | 8 | 2 | 1 | 104.09 | 410.69 |
| 0.9958 | HSV | 11 | 16 | 2 | 1 | 70.34 | 356.35 |
| 0.9958 | HSV | 10 | 16 | 2 | 2 | 74.9 | 366.51 |
| 0.9955 | LUV | 10 | 8 | 2 | 0 | 126.63 | 512.38 |
| 0.9955 | HSV | 9 | 16 | 2 | 1 | 54.13 | 265.88 |

The top result (`HSV, 10, 16,2, ALL`) gave a 99.77% accuracy in ~124s feature extaction time and ~298s train time, for a total of ~421s. However the second result (`YUV, 11, 16, 2, ALL`) gave a 99.72% accuracy in half the time ~241. Moreover, the next top 2 results were given with similar parameters, and the `YUV` colorspace, thus I opted to chose the following:

* colorspace: **YUV**
* channels: **ALL**
* orientations: **11**
* pixels per cell: **16**
* cells per block: **2**


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a SVM classifier with a sbf kernel and chose `auto` for the gamma parameter. For the C parameter I eventually settled at 40 as a value that gave consistently good results. Exploring the parameters of the classifier was done manually.

Here is a sample result of my `find_cars` method:

![alt_text][find_cars]

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

First I constrained the search to the bottom half of the image (4/7 specifically). Then I explored several different scales at specific regions of that bottom half.

##### scale 1.0

For scale 1.0, very little boxes, it made sense to search only further away from the car. Here is visualization of the search:

![alt text][search_1]

##### scale 1.5

![alt text][search_2]

##### scale 2.0

![alt text][search_3]

##### scale 3.0

![alt text][search_4]

##### Combined scales

Overall the pipeline combined all of the above scales. Here is the final result over a test image

![alt text][search_comb]


#### 2. Heatmaps

To address the overlapping detection I employed heatmaps. I used the detected boxes to generate a heatmap:

![alt text][heatmap_1]

Then applied a threshold over it

![alt text][heatmap_thres]

Next, I labelled the heatmaps melting overalpping boxes into singular regions

![alt text][heatmap_lbl]

Finally, I calculated bounding boxes out of the labels, and drew them onto the original image

![alt text][drawn_img]


#### 3. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Here is my pipeline applied to all the test images

![alt text][test_images]


---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_video_out_cache_mp.mp4)


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The biggest issue with this project was computational performance. Unfortunately sklearn is CPU only and quite slow. To optimize computational speed I employed multiprocessing as much as possible. My pipeline does feature extraction at various scales; each extraction is a completely independent task. Thus multiprocessing is an ideal optimization method, as it provides true parallelism (sidestepping the Python GIL issue) for higher interprocess communication cost. Independent tasks means no interprocess communication maximizing performance gains, contrary to multithreading which provides partial parallelism (GIL) and very fast cross-thread communication.

