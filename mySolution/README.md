## Advanced Lane Finding

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

[image1]: ./examples/test_undist.jpg "Calibration"
[image2]: ./examples/und.jpg "Compare Undistorted"
[image3]: ./examples/c_b.jpg "Compare Channels"
[image4]: ./examples/bin.jpg "Binary Combined"
[image5]: ./examples/cop.jpg "Warp Compare"
[image6]: ./examples/lan_1.jpg "Histogram Detection"
[image7]: ./examples/lan_2.jpg "Reuse Previous"
[image8]: ./examples/res.jpg ""
[video1]: ./examples/project_video_detected.mp4 "Video"
[video2]: ./examples/project_video_detected_noS.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

---

### Writeup

#### 1. My README.md
... you're actually reading it!

Udacity [README](https://github.com/tochalid/CarND-Advanced-Lane-Lines/tree/master/writeup_template.md)

### Camera Calibration

#### 1. Computation of the camera matrix and distortion coefficients.

The code for this step is contained in lines #1 through #39 of the file called my `./camera_cal/CameraCalibration.py`.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

For additional result examples please visit [Camera Calibration](https://github.com/tochalid/CarND-Advanced-Lane-Lines/tree/master/camera_cal)

### Pipeline (single images)

The pipeline code is in line #438

#### 1. Example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![alt text][image2]

#### 2. How color transforms, gradients or other methods to create a thresholded binary image are used.

I used a combination of 3 color channels (L of LUV, B o LAB, S of HLS) and absolut gradient orientation to generate a combined binary image (thresholding steps at lines #94 through #133 in `./mySolution/_AdvancedLaneLinesDetection.py`) encapsulated in function `color_and_gradient`.  Here are examples of my output for this step. The second image is the final result of the combined binary.

![alt text][image3]

![alt text][image4]

#### 3. Performing a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp()`, which appears in lines #136 through #156 in the file `./mySolution/_AdvancedLaneLinesDetection.py`.  The `warp` function takes as inputs an image (`img`). I chose to hardcode the source and destination points in the following manner:

```python
    src = np.float32(
        [[(img_size[0] / 2) - 50, img_size[1] / 2 + 100],  # -50.100
         [((img_size[0] / 6) + 58), img_size[1]],          # 58
         [(img_size[0] * 5 / 6) + 95, img_size[1]],        # 95
         [(img_size[0] / 2 + 61), img_size[1] / 2 + 100]]) # 61.100
    dst = np.float32(
        [[(img_size[0] / 4), 0],
         [(img_size[0] / 4), img_size[1]],
         [(img_size[0] * 3 / 4), img_size[1]],
         [(img_size[0] * 3 / 4), 0]])

```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 590, 460      | 320, 0        |
| 271, 720      | 320, 720      |
| 1162,720      | 960, 720      |
| 701, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image5]

#### 4. Identifying lane-line pixels and fit their positions with a polynomial?

The code is included in a function called `detect_lanes` in line #161 trough #344. The function expects a binary warped image as paramter. To identify the lane pixels in an image I analyse it generating a histogram, that is split in left and right half. Based on the maximum peaks that can be detected, coordinates are identified with most probable pixels belonging to lane lines.

Based on 2 hyperparameter (num of windows = 20, margin = 35 px) I use a sliding window detecting non-zero pixels in the thresholded binary image in line #253 to 292 to identify the pixels of each lane (left/right). After extracting the pixel positions a 2nd order polynomial can be fitted and coordinates for plotting extracted.

If there has been a previously detected lane, than the non-zero pixels of the current lane are detected scanning around the previously lane using margin. See code #209 to #251. Again pixels positions, polynomial fit and plotting coordinates are extracted.

In code line #212+3 two lane line instances (code line #467+8) are initialized with previously detected information for reuse in the current frame.

In code line #329-338 the polynomial coefficient are averaged (see also Class.Q_MAX_LEN_BFIT = 5) to stabilize the fitting with 15 previously detected fittings.

The firt image shows lane fitting by detection using histogram, the second reuses previously detected fittings.

![alt text][image6]

![alt text][image7]


#### 5. Calculating the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in function `measures_and_plotting` in lines #375 through #435 in my code in `_AdvancedLaneLinesDetection14_sub2.py`. In order to transform the pixel measure into real world measure (eg meters) I use a rough approximation ratio of (ym_per_pix = 30 / 720, xm_per_pix = 3.7 / 640) provided via the lane class variables (see `LineClass.py` #9. The src/dst warp points correlates with this measure, what otherwise would scramble the accuracy of the curvature estimation. The offset of the car from the center line is calculated based on histogram using `argmax` and averaged over last 5 values (code #200 to #221).

#### 6. Example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines #414 through #434 in my code in `AdvanceLaneLineDetection.py` in the function `measures_and_plotting`.  Here is an example of my result on a test image:

![alt text][image8]

---

### Pipeline (video)

Here's a link to my video result
[project_video_detected.mp4](./examples/project_video_detected.mp4)

Here are some output samples
[output_samples.zip](./output_images/run7.zip)

---

### Discussion

From the implementation point of view I used a sequence of functions to separate the transformations and calculations. In order to share information across those functions I used the LineClass instances. The code could be further optimized refactoring names and splitting some of the functions to increase granularity, eg measuring and plotting.

I do apply smoothing over a set of previously detected lanes, helps for robust detection if the detection is lost in a series of frames initiating histogram search again. However, starting with histogram doesn't help if the series is to long. Analysing via frames doesnt consider time and the continuum of the real world what can be mitigated somehow modeling with smoothing.

Here is a video without the S-channel that has caused trouble with the shadows and changing light conditions.

[project_video_detected_noS.mp4](./examples/project_video_detected_noS.mp4)

For the filtering I test-wise applied CLAHE filter on L-channel what improved contrast and could be helpful also in the challenge videos. However, the filter generates alos noise in the detection area depending on the margin value, thus not used.

For curvy roads a model with a probabilistic scores for pixels being part of the lane could be helpful, see [Peter Moran's](http://petermoran.org/robust-lane-tracking/) post.
