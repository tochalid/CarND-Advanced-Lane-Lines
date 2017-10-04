## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./examples/01_undist/orig_dst0.jpg "Road Transformed"
[image3]: ./examples/02_thres/GxSc_org_0.jpg "Binary Example"
[image4]: ./examples/02_thres/GxyMagDir_org_0.jpg "Binary Example"
[image5]: ./output_images/warp_1.jpeg "Warp Example"
[image51]: ./output_images/warp_2.jpeg "Warp Example"
[image6]: ./output_images/Lane_detected_1.jpeg "Fit Visual"
[image7]: ./output_images/Lane_detected_2.jpeg "Fit Visual"
[image8]: ./output_images/result_1506643786.4190717.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

---

### Writeup

#### 1. [README](https://github.com/tochalid/CarND-Advanced-Lane-Lines/tree/master/writeup_template.md) ...you're actually reading it!

### Camera Calibration

#### 1. Computation of the camera matrix and distortion coefficients.

The code for this step is contained in lines #1 through #39 of the file called my `./camera_cal/CameraCalibration.py`.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

For additional result examples please visit [Camera Calibration](https://github.com/tochalid/CarND-Advanced-Lane-Lines/tree/master/camera_cal)

### Pipeline (single images)

#### 1. Example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![alt text][image2]

#### 2. How color transforms, gradients or other methods to create a thresholded binary image are used.

I used a combination of color channel and absolut gradient orientation thresholds to generate a binary image (thresholding steps at lines #21 through #49 in `./mySolution/AdvancedLaneLinesDetection.py`).  Here are two examples of my output for this step. The first has been used for the video, the second shows a way using magnitude of gradient instead absolut gradient, but not choosen due to its weaker result.

![alt text][image3]

![alt text][image4]

#### 3. Performing a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`, which appears in lines #51 through #67 in the file `./mySolution/AdvancedLaneLinesDetection.py`.  The `warp()` function takes as inputs an image (`img`). I chose to hardcode the source and destination points in the following manner:

```python
    src = np.float32(
        [[(img_size[0] / 2) - 100, img_size[1] / 2 + 120],
         [((img_size[0] / 6) - 10), img_size[1]],
         [(img_size[0] * 5 / 6) + 60, img_size[1]],
         [(img_size[0] / 2 + 105), img_size[1] / 2 + 120]])
    dst = np.float32(
        [[(img_size[0] / 4), 0],
         [(img_size[0] / 4), img_size[1]],
         [(img_size[0] * 3 / 4), img_size[1]],
         [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 554, 465      | 320, 0        |
| 213, 690      | 320, 690      |
| 1117,690      | 960, 690      |
| 721, 465      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image5]
![alt text][image51]

#### 4. Identifying lane-line pixels and fit their positions with a polynomial?

The code is included in a function called `detect_lanes` in line #74 trough #209. The function expects a binary warped image as paramter. To identify the lane pixels in an image I analyse it generating a histogram, that i split in left and right half an based on the maximum peaks I can detect coordinates with most probable lane line pixels.

Based on 2 hyperparameter (num of windows = 20, margin = 35 px) I use a sliding window detecting non-zero pixels in the thresholded binary image in line #131 to 183 to identify the pixels of each lane (left/right). After extracting the pixel positions a 2nd order polynomial can be fitted and coordinates for plotting extracted.

If there has been a previously detected lane, than the non-zero pixels of the current lane are detected scanning around the previously lane using margin. See code #107 to #130. Again pixels positions, polynomial fit and plotting coordinates are extracted.

In line #186 to #193 two lane instances (#282) are initialized with detected information for reuse in the next frame.

![alt text][image6]
![alt text][image7]


#### 5. Calculating the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in function `measures_and_plotting` in lines #205 through #265 in my code in `AdvancedLaneLinesDetection.py`. In order to transform the pixel measure into real world measure (eg meters) I use a rough approximation ratio of (ym_per_pix = 30 / 720, xm_per_pix = 3.7 / 70) provided via the lane class variables (see `LineClass.py` #9.

#### 6. Example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines #252 through #258 in my code in `AdvanceLaneLineDetection.py` in the function `measures_and_plotting`.  Here is an example of my result on a test image:

![alt text][image8]

---

### Pipeline (video)

Here's a link to my video result [project_video_detected.mp4](./project_video_detected.mp4)

---

### Discussion

#### 1. Briefly discuss

From the implementation point of view I used a sequence of functions to separate the transformations and calculations. In order to share information accross those functions I used the LineClass instances. The code could be optimized refactoring names and splitting some of the functions to increase granularity, eg measuring and plotting.

I didn't apply smoothing over a set of previously detected lanes, what would help for a more robust detection if the detection is lost in a series of frames initiating histogram search again. However, starting with histogram doesn't help if the series is to long. Analysing via frames doesnt consider time and the continuum of the real world what can be mitigated somehow modeling with smoothing. There is a moment (ca. 20-35s) in the video where the pipeline nearly breaks, but that can be fixed with smoothing.

For the filtering I test-wise applied CLAHE filter on L-channel what improved contrast and could be helpful also in the challenge videos.

For curvy roads a model with a probabilistic scores for pixels being part of the lane could be helpful, see [Peter Moran's](http://petermoran.org/robust-lane-tracking/) post.
