## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


Detailed solution is described in ipython notebook `P4_advanced_lane_finding.ipynb`. The output can be found via this [YouTube link](https://www.youtube.com/watch?v=nnAKGXw9ZR4).

The Project
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

The images for camera calibration are stored in the folder called `camera_cal`.  The images in `test_images` are for testing your pipeline on single frames. `test_images/channels_test_images` are related to exploration of various colorspaces. `test_images/thresholding_test_images` are related to multi-modal thresholding. And finally `pipeline_output_test_images` are related to example lane detection pipeline outputs showing several crucial intermediate steps.

The challenge videos will be attempted soon !
