import glob
import pickle
from collections import deque
from itertools import product

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from moviepy.editor import VideoFileClip
from natsort import natsorted
from sklearn.linear_model import ElasticNet

sns.set_style("whitegrid", {'axes.grid': False})


def get_obj_img_pts(img_cal_names, num_x=9, num_y=6):
    '''Generate object/image points given filenames of calibration images

    params:
        num_x = the number of inner corner points along the x-axis of the test grid
        num_y = the number of inner corner points along the y-axis of the test grid
    '''

    # generate object points
    obj_pt = np.array(list(product(range(num_y), range(num_x), range(1))), np.float32)
    obj_pt[:, [0, 1]] = obj_pt[:, [1, 0]]

    obj_pts = []
    img_pts = []
    img_cals = []
    img_cal_names_ret = []

    for idx, img_cal_name in enumerate(img_cal_names):
        img_cal = mpimg.imread(img_cal_name)
        img_gray = cv2.cvtColor(img_cal, cv2.COLOR_RGB2GRAY)

        ret, img_pt = cv2.findChessboardCorners(img_gray, (num_x, num_y), None)
        if ret:
            print('corners_found: {}'.format(img_cal_name))
            obj_pts.append(obj_pt)
            img_pts.append(img_pt)

            # visualize the image points on calibration images
            cv2.drawChessboardCorners(img_cal, (num_x, num_y), img_pt, ret)

            img_cals.append(img_cal)
            img_cal_names_ret.append(img_cal_name)

    return obj_pts, img_pts, img_cals, img_cal_names_ret


def correct_dist(img, obj_pts, img_pts):
    '''Undistort an image given object/image points
    '''

    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_pts, img_pts, img_size, None, None)

    return cv2.undistort(img, mtx, dist_coeffs, None, mtx)


def img_subplots(imgs, img_names=None, f_size=(12, 10), f_cols=4):
    '''Create subplots of images and return figure handle
    '''

    assert (len(imgs) == len(img_names))

    f_rows = np.ceil(len(imgs) / f_cols).astype('int')

    fig, f_axes = plt.subplots(f_rows, f_cols, figsize=f_size)
    fig.set_tight_layout(True)

    for idx, f_ax in enumerate(f_axes.reshape(-1)):
        f_ax.axis("off")
        if idx < len(imgs):
            img = imgs[idx]

            color_map = "gray" if len(img.shape) == 2 else None
            f_ax.imshow(img, cmap=color_map)

            if img_names is not None:
                f_ax.set_title(img_names[idx])

    return fig


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def region_of_interest(img, roi_vertex_scales):
    """Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.

    vertices shall have the shape (num_of_polygon, num_of_vertices, 2)
    eg: vertices = np.array([[(wd*.45, ht*.53),(wd*.05, ht), (wd*.98, ht), (wd*.55, ht*.53)]], dtype=np.int32)
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        mask_color = (255,) * channel_count
        ht, wd, _ = img.shape
    else:
        mask_color = 255
        ht, wd = img.shape

    vertices = np.int32([[(wd * wd_scale, ht * ht_scale) for (wd_scale, ht_scale) in roi_vertex_scales]])

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)

    return masked_image


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap, line_thickness=2, line_color=(0, 255, 0)):
    """Returns an image with hough lines drawn.

    `img` should be the output of a Canny transform.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((*img.shape, 3), dtype=np.uint8)
    draw_lines(line_img, lines, thickness=line_thickness, color=line_color)

    return line_img


def draw_lines(img, lines, thickness=2, color=(255, 0, 0)):
    """ Draw interpolated lanes on img

    """

    lane_1st, lane_2nd = [], []
    height, width, _ = img.shape

    # separate the line segments based on slope and their position in the image
    for line in lines:
        for x1, y1, x2, y2 in line:
            if ((x2 - x1) != 0) and ((y2 - y1) / (x2 - x1) < 0) and ((x1 + x2) / 2 / width < 0.55):
                lane_1st.append(line)
            elif ((x2 - x1) != 0) and ((y2 - y1) / (x2 - x1) > 0) and ((x1 + x2) / 2 / width > 0.55):
                lane_2nd.append(line)

    # fit the left and right lane separately with ElasticNet
    x_pred = np.arange(img.shape[1]).reshape(-1, 1)
    for lane in [np.array(lane_1st), np.array(lane_2nd)]:
        lane = lane.reshape(lane.shape[0] * 2, 2)
        X, y = lane[:, 0], lane[:, 1]
        reg = ElasticNet().fit(X.reshape(-1, 1), y)
        y_pred = np.hstack((x_pred, reg.predict(x_pred).reshape(-1, 1)))

        cv2.polylines(img, np.int32([y_pred]), False, color, thickness)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""

    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def select_color(img, colors):
    ''' Return img with specified color selected

        colors is a list of (color_lower, color_upper) tuples
    '''

    img_color_select = np.zeros_like(img)

    for color_lower, color_upper in colors:
        color_mask = cv2.inRange(img, color_lower, color_upper)
        img_color_select += cv2.bitwise_and(img, img, mask=color_mask)

    return img_color_select


def sobel_thresh(img, th=(30, 100), kernel_size=3, op_dirs=(1, 0), debug=False):
    '''Absolute gradient thresholding
    '''

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    img_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, *op_dirs, ksize=kernel_size))
    img_sobel_scaled = np.uint8(255 * img_sobel / np.max(img_sobel))
    img_bin = img2binary(img_sobel_scaled, th)

    if debug:
        return img_sobel_scaled, img_bin
    else:
        return img_bin


def mag_thresh(img, th=(30, 100), kernel_size=3, debug=False):
    '''Gradient magnitude thresholding
    '''

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    img_sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=kernel_size)
    img_sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=kernel_size)
    img_sobel_mag = np.sqrt(img_sobel_x ** 2 + img_sobel_y ** 2)
    img_sobel_scaled = np.uint8(255 * img_sobel_mag / np.max(img_sobel_mag))
    img_bin = img2binary(img_sobel_scaled, th)

    if debug:
        return img_sobel_scaled, img_bin
    else:
        return img_bin


def img2binary(img, th=(75, 225)):
    '''Covert an image to a binary mask given thresholds

    '''
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    img_bin = np.zeros_like(img)
    img_bin[(img > th[0]) & (img <= th[1])] = 1

    return img_bin


def threshold_multi(img, roi_vertex_scales, colors_rgb, colors_hls, sobel_th=(80, 150), debug=False):
    img = gaussian_blur(img, kernel_size=3)

    img_rgb = img
    img_hls = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HLS)
    img_yuv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)

    # color selection in RGB and HLS space
    img_rgb_bin = img2binary(select_color(img_rgb, colors_rgb), th=[0, 255])
    img_hls_bin = img2binary(select_color(img_hls, colors_hls), th=(0, 255))
    img_color_bin = img_rgb_bin | img_hls_bin

    # U abs gradient th (YUV)
    img_channel = img_yuv[:, :, 1]
    img_u_sobel = sobel_thresh(img_channel, th=sobel_th, kernel_size=9, op_dirs=[1, 0])

    # combine thresholded binary images
    img_bin_combined = img_color_bin | img_u_sobel
    img_bin_combined_roi = region_of_interest(img_bin_combined, roi_vertex_scales)

    if debug:
        return img_color_bin, img_u_sobel, img_bin_combined, img_bin_combined_roi
    else:
        return img_bin_combined_roi


def get_perspective_matrix(img, src_scales, dst_scales):
    if len(img.shape) == 3:
        ht, wd, _ = img.shape
    elif len(img.shape) == 2:
        ht, wd = img.shape
    else:
        raise Exception("Only 2D images are supported.")

    src = np.float32([(wd * wd_scale, ht * ht_scale) for (wd_scale, ht_scale) in src_scales])
    dst = np.float32([(wd * wd_scale, ht * ht_scale) for (wd_scale, ht_scale) in dst_scales])

    M = cv2.getPerspectiveTransform(src, dst)
    inv_M = cv2.getPerspectiveTransform(dst, src)

    return M, inv_M, src


def get_binary_lane(img_strips, window_scale, offset=0.10):
    '''Return a segmented lane using the sliding window method'''
    lane = []
    window = np.array(window_scale) * img_strips[0].shape[1]
    for img_strip in reversed(img_strips):

        img_windowed = np.zeros_like(img_strip)
        img_windowed[:, window[0]:window[1]] = img_strip[:, window[0]:window[1]]

        lane_pts_x = np.where(np.sum(img_windowed, axis=0))
        if len(lane_pts_x[0]) > 5:
            lane.append(img_windowed)
            lane_mean = np.mean(lane_pts_x)
            lane_offset = offset * img_strip.shape[1]
            window = [int(lane_mean - lane_offset), int(lane_mean + lane_offset)]

        else:
            lane.append(np.zeros_like(img_windowed))

    return np.vstack(reversed(lane))


def fit_lane_pts(pts, y_fit_range=None, num_pts_y_fit=300):
    '''Return fitted points or coefficeints of 2nd order fitting x = F(y).

    params:
        pts: tuple of x_array and y_array `(x_array, y_array)`
    '''

    pts_x, pts_y = reversed(pts)
    coeffs = np.polyfit(pts_y, pts_x, 2)

    if y_fit_range is not None:
        pts_y_fit = np.linspace(0, y_fit_range, num=num_pts_y_fit)
        pts_x_fit = np.polyval(coeffs, pts_y_fit)

        return pts_x_fit, pts_y_fit

    else:
        return coeffs

# def fit_lane_pts_with_coeffs(coeffs, y_fit_range, num_pts_y_fit=300):
#     '''Return fitted points given coefficients
#
#     params:
#         coeffs: 2nd order fitting coefficients with the highest order first
#     '''
#
#     pts_y_fit = np.linspace(0, y_fit_range, num=num_pts_y_fit)
#     pts_x_fit = np.polyval(coeffs, pts_y_fit)
#
#     return pts_x_fit, pts_y_fit


def calc_curvature(pts, xm_per_pix=3.7 / 700, ym_per_pix=30 / 720):
    '''Calculate curvature given scales from pixel space to real physical space'''

    pts = np.array(pts).T * np.array([ym_per_pix, xm_per_pix])
    pts = (pts[:, 0], pts[:, 1])
    coeffs = fit_lane_pts(pts)

    y_eval = np.max(pts[1])
    curve_radius = ((1 + (2 * coeffs[0] * y_eval + coeffs[1]) ** 2) ** 1.5) / np.absolute(2 * coeffs[0])

    return curve_radius


def lane_detection(img, ROI_vertex_scales,
                   src_scales, dst_scales,
                   colors_rgb,
                   colors_hls,
                   sobel_th=(80, 150),
                   num_bins=20,
                   window_L=(0.1, 0.45),
                   window_R=(0.6, 0.90),
                   draw_lane_color=(0, 255, 0),
                   debug=False):
    img_corr = correct_dist(img, obj_pts, img_pts)

    ht, wd, _ = img.shape
    M, inv_M, pts_src = get_perspective_matrix(img, src_scales, dst_scales)
    img_warped = cv2.warpPerspective(img_corr, M, (wd, ht), flags=cv2.INTER_LINEAR)

    # thresholding corrected image and
    # perspective transformation of the resulting binary image
    img_bin = threshold_multi(img_corr, ROI_vertex_scales, colors_rgb=colors_rgb, colors_hls=colors_hls, sobel_th=sobel_th)
    img_bin_warped = cv2.warpPerspective(img_bin, M, (wd, ht), flags=cv2.INTER_LINEAR)

    # split perpective transformed binary image into multiple horizontal strips
    # img_bin_blurred = gaussian_blur(img_bin_warped, kernel_size=blur_kernel_size)
    img_bin_splits = np.vsplit(img_bin_warped, num_bins)

    # isolate the left and right lane with sliding windows
    lane_L = get_binary_lane(img_bin_splits, window_L)
    lane_R = get_binary_lane(img_bin_splits, window_R)

    pts_L = np.where(lane_L)
    pts_R = np.where(lane_R)

    if (len(pts_L[0]) < 3) | (len(pts_R[0]) < 3):
        return img_corr

    # calculate curvature for left/right lane
    pts_fit_L = fit_lane_pts(pts_L, y_fit_range=img_bin.shape[0], num_pts_y_fit=300)
    curve_radius_L = calc_curvature(pts_L)

    pts_fit_R = fit_lane_pts(pts_R, y_fit_range=img_bin.shape[0], num_pts_y_fit=300)
    curve_radius_R = calc_curvature(pts_R)

    # [curve_radius_L, curve_radius_R]

    # create an image to draw the lines on
    lane_warped_color = np.zeros_like(img_corr, dtype=np.uint8)

    # draw fitted points to a lane image
    pts_draw = np.hstack([pts_fit_L, np.fliplr(pts_fit_R)]).T.reshape(-1, 1, 2).astype(np.int)
    cv2.fillPoly(lane_warped_color, [pts_draw], draw_lane_color)

    # inverse perspective transform of lane image
    lane_color = cv2.warpPerspective(lane_warped_color, inv_M, (wd, ht), flags=cv2.INTER_LINEAR)

    # overlay detected lanes with the undistorted image
    img_combined = cv2.addWeighted(img_corr, 1, lane_color, 0.3, 0)

    if debug:
        print("The left  curvature is {:.1f} m".format(curve_radius_L))
        print("The right curvature is {:.1f} m".format(curve_radius_R))
        print("")

        pts_warp_roi = np.int32(pts_src.reshape([-1, 1, 2]))
        pts_roi = np.int32([[(wd * wd_scale, ht * ht_scale) for (wd_scale, ht_scale) in ROI_vertex_scales]])

        img_warp_roi = cv2.polylines(img_corr, [pts_warp_roi], True, (0, 255, 0), 5)  # draw the warp region in green
        img_warp_roi = cv2.polylines(img_warp_roi, [pts_roi], True, (0, 0, 255), 5)  # draw the roi selection in blue

        return img_warp_roi, img_warped, img_bin, img_bin_warped, lane_L, lane_R, lane_warped_color, img_combined
    else:
        return img_combined


# Define a class to receive the characteristics of each line detection
class LaneDetector:
    def __init__(self, N=30, TOLERANCE_CURVATURE=2, TOLERANCE_PTS=100):

        # append to history if the current curvature is less than TOLERANCE_CURVATURE
        # compared to the average curvature over N confident frames
        self.TOLERANCE_CURVATURE = TOLERANCE_CURVATURE

        # proceed with lane curve fitting if detected points are greater than TOLERANCE_PTS
        self.TOLERANCE_PTS = TOLERANCE_PTS

        # x,y values and fitted polynomial coeffs of the last n fits
        # assuming 30 frames per second
        self.N = N
        self.pts_fit_L_last_n = deque(maxlen=self.N)
        self.pts_fit_R_last_n = deque(maxlen=self.N)

        # average x,y values of the fitted lanes over the last n fit
        self.pts_L_last = None
        self.pts_R_last = None

        # radius of curvature of the line in some units
        self.curve_radius = 0
        self.curve_radius_last_n = deque(maxlen=self.N)
        self.curve_radius_avg = 0
        self.curve_radius_diff = 0

        # distance in meters of vehicle center from the line
        self.vehicle_offset = None
        self.vehicle_offset_last_n = deque(maxlen=self.N)
        self.vehicle_offset_avg = None

        # # difference in fit coefficients between last and new fits
        # self.coeffs_L_last_n = deque(maxlen=self.N)
        # self.coeffs_R_last_n = deque(maxlen=self.N)
        # self.coeffs_L_avg = None
        # self.coeffs_R_avg = None
        # self.fit_coeffs_diffs = np.array([0, 0, 0], dtype='float')

        # lane mask
        self.lane_mask = None
        self.lane_masks = []

        # problematic frames
        self.frame_N = 0
        self.error_frame_N = 0
        self.error_frames = []

    def get_binary_lane(self, img_strips, window_scale, offset=0.10):
        '''Return a segmented lane using the sliding window method'''

        lane = []
        img_window_masks = []
        window = (np.array(window_scale) * img_strips[0].shape[1]).astype(np.int)
        for img_strip in reversed(img_strips):

            img_windowed = np.zeros_like(img_strip)
            img_windowed[:, window[0]:window[1]] = img_strip[:, window[0]:window[1]]

            img_window_mask = np.zeros_like(img_strip)
            img_window_mask[:, window[0]:window[1]] = 1
            img_window_masks.append(img_window_mask)

            lane_pts_x = np.where(np.sum(img_windowed, axis=0))
            if len(lane_pts_x[0]) > 5:
                lane.append(img_windowed)
                lane_mean = np.mean(lane_pts_x)
                lane_offset = offset * img_strip.shape[1]
                window = [int(lane_mean - lane_offset), int(lane_mean + lane_offset)]

            else:
                lane.append(np.zeros_like(img_windowed))

        return np.vstack(reversed(lane)), np.vstack(reversed(img_window_masks))

    def calc_curvature(self, pts, xm_per_pix=3.7 / 700, ym_per_pix=30 / 720):
        '''Calculate curvature given scales from pixel space to real physical space'''

        pts = np.array(pts).T * np.array([ym_per_pix, xm_per_pix])
        pts = (pts[:, 0], pts[:, 1])
        coeffs = fit_lane_pts(pts)

        y_eval = np.max(pts[1])
        curve_radius = ((1 + (2 * coeffs[0] * y_eval + coeffs[1]) ** 2) ** 1.5) / np.absolute(2 * coeffs[0])

        return curve_radius, coeffs

    def lane_detection(self, img, ROI_vertex_scales,
                       src_scales, dst_scales,
                       colors_rgb,
                       colors_hls,
                       sobel_th=(80, 150),
                       num_bins=20,
                       window_L=(0.1, 0.45),
                       window_R=(0.6, 0.90),
                       draw_lane_color=(0, 255, 0),
                       debug=False):

        img_corr = correct_dist(img, obj_pts, img_pts)

        ht, wd, _ = img.shape
        M, inv_M, pts_src = get_perspective_matrix(img, src_scales, dst_scales)
        img_warped = cv2.warpPerspective(img_corr, M, (wd, ht), flags=cv2.INTER_LINEAR)

        # thresholding corrected image and
        # perspective transformation of the resulting binary image
        img_bin = threshold_multi(img_corr, ROI_vertex_scales, colors_rgb=colors_rgb, colors_hls=colors_hls, sobel_th=sobel_th)
        img_bin_warped = cv2.warpPerspective(img_bin, M, (wd, ht), flags=cv2.INTER_LINEAR)

        # split perpective transformed binary image into multiple horizontal strips
        img_bin_splits = np.vsplit(img_bin_warped, num_bins)

        # isolate the left and right lane with masks generated with sliding windows
        # if lane_mask is not defined, search the lane lines from scratch
        # else use the previous window for lane lines detection
        if not self.lane_mask:
            lane_L, mask_L = self.get_binary_lane(img_bin_splits, window_L)
            lane_R, mask_R = self.get_binary_lane(img_bin_splits, window_R)
            self.lane_mask = [mask_L, mask_R]

        else:
            mask_L, mask_R = self.lane_mask
            lane_L = cv2.bitwise_and(img_bin_warped, mask_L)
            lane_R = cv2.bitwise_and(img_bin_warped, mask_R)

        # get (i,j) coordinates for the lane points
        pts_L = np.where(lane_L)
        pts_R = np.where(lane_R)

        # if the number of lane points detected is less than TOLERANCE_PTS for either lane,
        # use the detected points from the last and current frame for subsequent fitting
        if (len(pts_L[0]) < self.TOLERANCE_PTS) | (len(pts_R[0]) < self.TOLERANCE_PTS):
            self.lane_mask = None
            self.error_frame_N += 1
            self.error_frames.append(img)

            if self.pts_L_last is not None:
                # concatenate (i,j) coordinates of points detected for the last and current frame
                pts_L = [pts_last + pts for (pts_last, pts) in zip(self.pts_L_last, pts_L)]
                pts_R = [pts_last + pts for (pts_last, pts) in zip(self.pts_R_last, pts_R)]
            else:
                return img_corr
        else:
            self.pts_L_last = pts_L
            self.pts_R_last = pts_R

        # calculate curvature for left/right lane
        # the curve radius is estimated as the mean of left/right lane, which is smoothed over the last n frames
        pts_fit_L = fit_lane_pts(pts_L, y_fit_range=img_bin.shape[0], num_pts_y_fit=ht)
        curve_radius_L, coeffs_L = self.calc_curvature(pts_L)

        pts_fit_R = fit_lane_pts(pts_R, y_fit_range=img_bin.shape[0], num_pts_y_fit=ht)
        curve_radius_R, coeffs_R = self.calc_curvature(pts_R)

        self.curve_radius = np.mean([curve_radius_L, curve_radius_R])
        self.curve_radius_diff = np.abs((self.curve_radius - self.curve_radius_avg) / self.curve_radius_avg)

        # if the lane curve difference is less than TOLERANCE_CURVATURE or is the first frame
        # append the current curvature and coefficients to their respective double ended queue
        if (self.curve_radius_diff < self.TOLERANCE_CURVATURE) or (self.frame_N == 0):
            self.curve_radius_last_n.append(self.curve_radius)
            self.curve_radius_avg = np.mean(self.curve_radius_last_n)

            # self.coeffs_L_last_n.append(coeffs_L)
            # self.coeffs_R_last_n.append(coeffs_R)
            # self.coeffs_L_avg = np.mean(self.coeffs_L_last_n, axis=0)
            # self.coeffs_R_avg = np.mean(self.coeffs_R_last_n, axis=0)

        else:
            self.lane_mask = None

        # estimate vehicle offset from the center of the road
        # using the x coordinates of the last 10 points from the bottom of the frame
        xm_per_pix = 3.7 / 700  # meters per pixel

        # here a negative sign is needed to measure offsets with respect to the center of the road
        self.vehicle_offset = -xm_per_pix * (np.mean(pts_fit_L[0][-10:]) + np.mean(pts_fit_R[0][-10:]) - wd) / 2
        self.vehicle_offset_last_n.append(self.vehicle_offset)
        self.vehicle_offset_avg = np.mean(self.vehicle_offset_last_n)

        # create an image to draw fitted points on
        lane_warped_color = np.zeros_like(img_corr, dtype=np.uint8)

        # draw fitted points to a lane image
        pts_draw = np.hstack([pts_fit_L, np.fliplr(pts_fit_R)]).T.reshape(-1, 1, 2).astype(np.int)
        cv2.fillPoly(lane_warped_color, [pts_draw], draw_lane_color)

        # inverse perspective transform of lane image
        lane_color = cv2.warpPerspective(lane_warped_color, inv_M,
                                         (wd, ht), flags=cv2.INTER_LINEAR)
        lane_color = region_of_interest(lane_color, ROI_vertex_scales)

        # overlay detected lanes with the undistorted image
        img_combined = cv2.addWeighted(img_corr, 1, lane_color, 0.3, 0)

        # draw text onto the image
        img_txt = "Radius of curvature: {:7.1f}m  Offset from road center: {:7.3f}m  Errors: {:3.0f} /{:5.0f}".format(self.curve_radius_avg,
                                                                                                                      self.vehicle_offset_avg,
                                                                                                                      self.error_frame_N,
                                                                                                                      self.frame_N)
        img_txt_offset = (int(wd * 0.01), int(ht * 0.04))

        pts_txt_bounding_box = np.int32([(0, 0), (wd, 0), (wd, ht * 0.05), (0, ht * 0.05)]).reshape([-1, 1, 2])
        img_combined = cv2.fillPoly(img_combined, [pts_txt_bounding_box], (43, 43, 43))

        cv2.putText(img_combined,
                    img_txt,
                    img_txt_offset,
                    cv2.FONT_HERSHEY_COMPLEX, 0.8,
                    (250, 250, 250), 1)

        self.frame_N += 1

        if debug:
            print("The left  curvature is {:.1f} m".format(curve_radius_L))
            print("The right curvature is {:.1f} m".format(curve_radius_R))
            print("")

            # draw perspective warp and ROI bounding box
            pts_warp_roi = np.int32(pts_src.reshape([-1, 1, 2]))
            pts_roi = np.int32([[(wd * wd_scale, ht * ht_scale) for (wd_scale, ht_scale) in ROI_vertex_scales]])

            img_warp_roi = cv2.polylines(img_corr, [pts_warp_roi], True, (0, 255, 0), 5)  # blue for perspective transform bounding box
            img_warp_roi = cv2.polylines(img_warp_roi, [pts_roi], True, (0, 0, 255), 5)  # green for ROI bounding box

            return img_warp_roi, img_warped, img_bin, img_bin_warped, lane_L, lane_R, lane_warped_color, img_combined
        else:
            return img_combined


def process_image(img):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # the solution is detailed in the function lane_detection above

    return lane_detector.lane_detection(img, ROI_vertex_scales, src_scales, dst_scales,
                                        colors_rgb, colors_hls,
                                        window_L=window_L, window_R=window_R, debug=False)


NUM_X = 9
NUM_Y = 6
LOAD_OBJ_IMG_PTS = True
img_cal_names = natsorted(glob.glob('camera_cal/*.jpg'))

if not LOAD_OBJ_IMG_PTS:
    obj_pts, img_pts, img_cals, img_cal_names_ret = get_obj_img_pts(img_cal_names, num_x=NUM_X, num_y=NUM_Y)

    with open(r"obj_img_pts", "wb") as file_output:
        pickle.dump([obj_pts, img_pts], file_output)

else:
    with open(r"obj_img_pts", "rb") as file_input:
        obj_pts, img_pts = pickle.load(file_input)

ROI_vertex_scales = [(0.48, 0.59), (0.52, 0.59), (0.65, 0.65), (0.95, 1), (0.05, 1), (0.35, 0.65)]

src_x_top, src_x_bot = 0.42, 0.065  # counting from the left edge
src_y_top, src_y_bot = 0.67, 0

dst_x_top, dst_x_bot = 0.2, 0.2  # counting from the left edge
dst_y_top, dst_y_bot = 0.2, 0

src_scales = [(src_x_top, src_y_top), (1 - src_x_top, src_y_top),
              (1 - src_x_bot, 1 - src_y_bot), (src_x_bot, 1 - src_y_bot)]
dst_scales = [(dst_x_top, dst_y_top), (1 - dst_x_top, dst_y_top),
              (1 - dst_x_top, 1 - dst_y_top), (dst_x_top, 1 - dst_y_bot)]

colors_rgb = [(np.uint8([190, 190, 190]), np.uint8([255, 255, 255]))]

colors_hls = [(np.uint8([0, 120, 150]), np.uint8([75, 255, 255])),
              (np.uint8([75, 180, 0]), np.uint8([120, 255, 35]))]

window_L = (0.1, 0.45)
window_R = (0.6, 0.90)

lane_detector = LaneDetector()

# img_test_names = natsorted(glob.glob("test_images/*.jpg"))
# for img_test_name in img_test_names[0:2]:
#     print(img_test_name)
#     img = mpimg.imread(img_test_name)
#
#     imgs = lane_detector.lane_detection(img, ROI_vertex_scales, src_scales, dst_scales,
#                                         colors_rgb, colors_hls,
#                                         window_L=window_L, window_R=window_R, debug=True)
#
#     img_names = ["undistorted", "perspective warped", "thresholded", "thresholded warped",
#                  "left lane", "right lane", "warped detected lanes", "combined with detected lanes"]
#
#     fig = img_subplots(imgs, img_names, f_size=(10, 12), f_cols=2)
#     fig.suptitle(img_test_name, y=1.05, fontsize=16)
#     plt.axis("on")
#
# plt.show()

# clip_files = ["test.mp4", "challenge_video.mp4", "harder_challenge_video.mp4"]
clip_files = ["project_video.mp4", "challenge_video.mp4", "harder_challenge_video.mp4"]
for clip_file in clip_files[0:1]:
    clip = VideoFileClip(clip_file)
    clip_out = clip.fl_image(process_image)
    clip_out.write_videofile("z_sol_" + clip_file, audio=False)
    print("======================================================")
