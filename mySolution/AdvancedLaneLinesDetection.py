import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mySolution.LineClass import LaneLine
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import shutil
import os
import datetime
from collections import deque

# Switches and Directories to control output during testing
LOG_LEVEL = False
GLOBAL_SAVE = True # just a switch to save images to disk
GLOBAL_PREDEL = False    # just a switch to delete files
print('PREDEL=',GLOBAL_PREDEL,' SAVE=',GLOBAL_SAVE,' LOG=', LOG_LEVEL)
OUT_PATH = '../output_images'
DIR_RUN = '/run6'
DIR_BIN = '/bin' ; FN_BIN = '/bin_'; dir_bin = OUT_PATH+DIR_RUN+DIR_BIN+FN_BIN
DIR_COP = '/cop' ; FN_COP = '/cop_'; dir_cop = OUT_PATH+DIR_RUN+DIR_COP+FN_COP
DIR_LAN = '/lan' ; FN_LAN = '/lan_'; dir_lan = OUT_PATH+DIR_RUN+DIR_LAN+FN_LAN
DIR_RES = '/res' ; FN_RES = '/res_'; dir_res = OUT_PATH+DIR_RUN+DIR_RES+FN_RES
DIR_WRP = '/wrp' ; FN_WRP = '/wrp_'; dir_wrp = OUT_PATH+DIR_RUN+DIR_WRP+FN_WRP
DIR_C_B = '/c_b' ; FN_C_B = '/c_b_'; dir_c_b = OUT_PATH+DIR_RUN+DIR_C_B+FN_C_B
DIR_UND = '/und' ; FN_UND = '/und_'; dir_und = OUT_PATH+DIR_RUN+DIR_UND+FN_UND


def delete_all_files(delete=GLOBAL_PREDEL):
    if delete:
        shutil.rmtree(OUT_PATH+DIR_RUN)
        os.makedirs(OUT_PATH+DIR_RUN)
        os.makedirs(OUT_PATH+DIR_RUN+DIR_BIN)
        os.makedirs(OUT_PATH+DIR_RUN+DIR_COP)
        os.makedirs(OUT_PATH+DIR_RUN+DIR_LAN)
        os.makedirs(OUT_PATH+DIR_RUN+DIR_RES)
        os.makedirs(OUT_PATH+DIR_RUN+DIR_WRP)
        os.makedirs(OUT_PATH+DIR_RUN+DIR_C_B)
        os.makedirs(OUT_PATH+DIR_RUN+DIR_UND)
        print('All files deleted')
    return
delete_all_files()


# plots 2 images and can save as <timestamp>.jpg
def compare_saving(img1, img2, dir, save=GLOBAL_SAVE):
    if save:
        f001, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6)); f001.tight_layout()
        ax1.set_title('before', fontsize=20)
        ax1.imshow(img1, cmap='gray')
        ax2.set_title('after', fontsize=20)
        ax2.imshow(img2, cmap='gray')
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.savefig(dir+str(datetime.datetime.now())+'.jpg', dpi='figure', format='jpg');
    return

def image_saving(img, dir, save=GLOBAL_SAVE):
    if save: plt.imsave(dir+str(datetime.datetime.now())+'.jpg', img, cmap='gray', format='jpg')
    return

# plots 2 images and can save as <timestamp>.jpg
def compare_warp(img1, img2, dir, src, dst, save=GLOBAL_SAVE):
    if save:
        ax1 = plt.subplot(1, 2, 1)
        ax1.set_title('before', fontsize=12)
        ax1.imshow(img1, cmap='gray')
        plt.plot(src[0][0], src[0][1], '.')
        plt.plot(src[1][0], src[1][1], '.')
        plt.plot(src[2][0], src[2][1], '.')
        plt.plot(src[3][0], src[0][1], '.')
        ax2 = plt.subplot(1, 2, 2)
        ax2.set_title('after', fontsize=12)
        ax2.imshow(img2, cmap='gray')
        plt.plot(dst[0][0], dst[0][1], '.')
        plt.plot(dst[1][0], dst[1][1], '.')
        plt.plot(dst[2][0], dst[2][1], '.')
        plt.plot(dst[3][0], dst[0][1], '.')
        # plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.savefig(dir + str(datetime.datetime.now()) + '.jpg', dpi='figure', format='jpg')
    return

def image_saving_warp(img, dir, pts, save=GLOBAL_SAVE):
    plt.imshow(img, cmap='gray', aspect='auto')
    plt.plot(pts[0][0], pts[0][1], '.')
    plt.plot(pts[1][0], pts[1][1], '.')
    plt.plot(pts[2][0], pts[2][1], '.')
    plt.plot(pts[3][0], pts[0][1], '.')
    if save: plt.savefig(dir+str(datetime.datetime.now())+'.jpg', dpi='figure', format='jpg')
    return


def color_and_gradient(img, b_thres=(165, 255), l_thres=(205, 255), s_thres=(175, 195), sx_thres=(40, 120), save=GLOBAL_SAVE, dir=dir_c_b):

    def get_binary_from_channel(img, color_cvt, channel_idx, thres=(0,255)):
        color_space = cv2.cvtColor(img, color_cvt)
        channel = color_space[:,:,channel_idx]
        binary = np.zeros_like(channel)
        binary[(channel >= thres[0]) & (channel <= thres[1])] = 1
        return binary

    # Binary with threshold from channel
    b_binary = get_binary_from_channel(img, cv2.COLOR_RGB2LAB, 2, (b_thres[0], b_thres[1]))
    l_binary = get_binary_from_channel(img, cv2.COLOR_RGB2LUV, 0, (l_thres[0], l_thres[1]))
    s_binary = get_binary_from_channel(img, cv2.COLOR_RGB2HLS, 2, (s_thres[0], s_thres[1]))

    # Absolut Sobel x for verticals detection
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thres[0]) & (scaled_sobel <= sx_thres[1])] = 1

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(b_binary == 1) | (l_binary == 1) |(s_binary == 1) | (sxbinary == 1)] = 1

    # Todo: separate in independent function
    if save:
        fs = 20
        f003, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(16, 6)); f003.tight_layout()
        ax1.set_title('B of LAB', fontsize=fs); ax1.imshow(b_binary, cmap='gray')
        ax2.set_title('L of LUV', fontsize=fs); ax2.imshow(l_binary, cmap='gray')
        ax3.set_title('S of HLS', fontsize=fs); ax3.imshow(s_binary, cmap='gray')
        ax4.set_title('Sobel X', fontsize=fs); ax4.imshow(sxbinary, cmap='gray')
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.savefig(dir+str(datetime.datetime.now())+'.jpg', dpi='figure', format='jpg')

    return combined_binary


def warp(img):
    img_size = (img.shape[1], img.shape[0])
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
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    lll.Minv = rll.Minv = cv2.getPerspectiveTransform(dst, src)

    return warped, src, dst


def sanity2_lane_distance():
# Todo: implement if lines are parallel
    return


def detect_lanes(img): # Assuming you have created a warped binary image called "img"
    # Set the width of the windows +/- margin
    margin = 35 # 35 #70 #60 #55 #30 #55 #65 #65 #55
    # Choose the number of sliding windows
    nwindows = 20 # 20 #15 #25 #50 #9
    # Set minimum number of pixels found to recenter window
    minpix = 750 # 770 #250 #70 #60 #65 #55 #150 #35 #70
    # lane statistics
    leftx = None
    lefty = None
    left_fit = None
    left_fitx = None

    rightx = None
    righty = None
    right_fit = None
    right_fitx = None

    ploty = None

    detected = eval(lll.detected[-1]) & eval(rll.detected[-1])

    # Take a histogram of the bottom half of the image
    histogram = np.sum(img[img.shape[0] // 2:, :], axis=0)
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Calculate the offset of the vehicle from the lane center
    offset = (img.shape[1] - leftx_base - rightx_base)/2.0
    offset_m = offset * LaneLine.xm_per_pix
    lll.line_base_pos = rll.line_base_pos = offset_m

    # Create an image to draw
    out = np.dstack((img, img, img)) * 255

    if detected:
        # print(' IF  .. is reusing detected lanes')
        # Assume you now have a new warped binary image from the next frame of video (also called "binary_warped")
        left_fit = lll.current_fit # from previous image
        right_fit = rll.current_fit # from previous image
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin))
                          & (nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin))
                           & (nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds] # set new indices for img
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        # Generate x and y values for plotting
        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

        # Create an image to draw on and an image to show the selection window
        win = np.zeros_like(out)
        # Color in left and right line pixels
        out[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]  # paint left lane pixels in red
        out[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]  # paint right lane pixels in blue
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))
        # Draw the lane onto the warped blank image
        cv2.fillPoly(win, np.int_([left_line_pts]), (0, 255, 0))  # marking left lane in yellow as polygon with margin
        cv2.fillPoly(win, np.int_([right_line_pts]), (0, 255, 0))  # marking right lane in yellow as polygon with margin
        out = cv2.addWeighted(out, 1, win, 0.3, 0)

    else:
        # print(' ELSE ... you are running offroad ... initializing histogram')
        # Set height of windows
        window_height = np.int(img.shape[0] / nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = img.shape[0] - (window + 1) * window_height
            win_y_high = img.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # Draw the windows on the visualization image
            cv2.rectangle(out,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
            cv2.rectangle(out,(win_xright_low,win_y_low),(win_xright_high,win_y_high), (0,255,0), 2)

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
           # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix: leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix: rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        # Generate x and y values for plotting
        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        out[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]  # paint left lane pixels in red
        out[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]  # paint right lane pixels in blue

    # set the lane statistics
    lll.allx = leftx
    lll.ally = lefty
    lll.current_fit = left_fit
    lll.best_fits.append(left_fit)

    rll.allx = rightx
    rll.ally = righty
    rll.current_fit = right_fit
    rll.best_fits.append(right_fit)

    lll.ploty = rll.ploty = ploty
    # get last n average polynomial coefficients and put in flat list for slicing and averaging

    def calculate_average_coefficients(lane):
        flat = []
        for i in range(0, len(lane.best_fits)):
            for j in range(0, len(lane.best_fits[i])):
                flat.append(lane.best_fits[i][j])
        # calculate average and initialize lane statistics
        lane.best_fit_avg = np.array([np.average(flat[0::3]),np.average(flat[1::3]),np.average(flat[2::3])])
        return
    calculate_average_coefficients(lll)
    calculate_average_coefficients(rll)

    # correct lane using average of last n polynomial coefficients
    lll.best_fitx = lll.best_fit_avg[0] * ploty ** 2 + lll.best_fit_avg[1] * ploty + lll.best_fit_avg[2]
    rll.best_fitx = rll.best_fit_avg[0] * ploty ** 2 + rll.best_fit_avg[1] * ploty + rll.best_fit_avg[2]

    return out


def sanity3_curvature(line, curvature, dev=0.50): #0.2 #0.8 #0.6 #0.75
    # initialize the curvature queue
    if line.radius_of_curvature[0] == 0.0:
        line.radius_of_curvature.clear()
        line.radius_of_curvature.append(curvature)
    else:
        avrg = np.average(line.radius_of_curvature)
        # print('\navgr ', avrg)
        # print('curv', curvature)
        # print('prev ', line.radius_of_curvature[-1])
        lower_limit = (1-dev)*avrg
        upper_limit = (1+dev)*avrg
        if lower_limit <= curvature <= upper_limit:
            # append detected curvature
            line.radius_of_curvature.append(curvature)
            # update detection status for lane
            line.detected.append('True')
            # print('\nCurvature in range ... status: ', line.detected)
        else:
            # append previously detected curvature
            line.radius_of_curvature.append(line.radius_of_curvature[-1])
            # update detection status for corrected lane
            line.detected.append('False')
            # print('\nCurvature NOT in range ... status: ', line.detected)

    return np.average(line.radius_of_curvature)


def measures_and_plotting(org, img):

    # set variables from detect_lanes call
    leftx = lll.allx
    lefty = lll.ally
    left_fit = lll.current_fit
    left_fitx = lll.best_fitx

    rightx = rll.allx
    righty = rll.ally
    right_fit = rll.current_fit
    right_fitx = rll.best_fitx

    ploty = lll.ploty

    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = np.max(ploty)

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty * LaneLine.ym_per_pix, left_fitx * LaneLine.xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * LaneLine.ym_per_pix, right_fitx * LaneLine.xm_per_pix, 2)

    # Calculate the new radiant of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * LaneLine.ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * LaneLine.ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])

    avrg_left_curverad = sanity3_curvature(lll, left_curverad)
    avrg_right_curverad = sanity3_curvature(rll, right_curverad)

    # Now our radius of curvature is in meters
    avrg_curve_rad_m = (avrg_left_curverad + avrg_right_curverad) / 2
    # Example values: 632.1 m    626.2 m

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Create output images to draw on and visualize the results
    poly = np.dstack((img, img, img)) * 0
    lane_pixel = np.dstack((img, img, img)) * 0
    lane_pixel[lefty, leftx] = [255, 0, 0]  # paint left lane pixels in red
    lane_pixel[righty, rightx] = [0, 0, 255]  # paint right lane pixels in blue

    # Draw the lane onto the warped blank image
    warped = cv2.fillPoly(poly, np.int_([pts]), (0, 255, 0))

    # Warp back to original image space using inverse perspective matrix (Minv)
    unwarped_poly = cv2.warpPerspective(warped, lll.Minv, (lane_pixel.shape[1], lane_pixel.shape[0]))
    unwarped_lanes = cv2.warpPerspective(lane_pixel, lll.Minv, (lane_pixel.shape[1], lane_pixel.shape[0]))

    # Combine the result with the original image
    result = cv2.addWeighted(org, 1, unwarped_poly, 0.3, 0)
    result = cv2.addWeighted(result, 1, unwarped_lanes, 2, 0)

    cv2.putText(result, 'Radius of Curvature: ' + str("%.0f" % avrg_curve_rad_m +' m'),
                org=(70, 70), fontFace=2, fontScale=2, color=(255,255,255), thickness=2)
    cv2.putText(result, 'Veh. Center Offset : ' + str("%.2f" % lll.line_base_pos + ' m'),
                org=(70, 130), fontFace=2, fontScale=2, color=(255,255,255), thickness=2)
    return result


def pipeline(i):
    org = np.copy(i)
    u = cv2.undistort(i, mtx, dist, None, mtx)
    compare_saving(org, u, dir_und)
    w, s, d = warp(u)
    compare_warp(u, w, dir_cop, src=s, dst=d)
    # image_saving_warp(u, dir_wrp, pts=s)

    b = color_and_gradient(w, dir=dir_c_b)
    image_saving(b, dir_bin)
    compare_saving(w, b, dir_wrp)

    l = detect_lanes(b)
    compare_saving(w, l, dir_lan)

    r = measures_and_plotting(org, b)
    compare_saving(r, w, dir_res)

    if LOG_LEVEL: lll.print_line_status('LLL')
    if LOG_LEVEL: rll.print_line_status('RLL')
    return r


# Read the saved camera matrix and distortion coefficients calculated using CameraCalibration.py -> cv2.calibrateCamera()
dist_pickle = pickle.load(open( "../camera_cal/9x6cal_dist_pickle.p", "rb"))
mtx = dist_pickle["mtx"]; dist = dist_pickle["dist"]

# instantiate 2 line class to store detected lane lines
lll = LaneLine() # lll: left_lane_line
rll = LaneLine() # rll: right_lane_line

# Project Video
# clip1 = VideoFileClip("../project_video.mp4").subclip(13,14) # good for warp check
# clip1 = VideoFileClip("../project_video.mp4").subclip(40,42) # good for binary check
# clip1 = VideoFileClip("../project_video.mp4s").subclip(23,26) # good for binary check
# clip1 = VideoFileClip("../project_video.mp4").subclip(21,25) # good for latency check
# clip1 = VideoFileClip("../project_video.mp4").subclip(38,42) # good for latency check
# clip1 = VideoFileClip("../project_video.mp4").subclip(21,51)
# clip1 = VideoFileClip('../project_video.mp4')
clip1 = VideoFileClip('../challenge_video.mp4')
# clip1 = VideoFileClip('../harder_challenge_video.mp4')
output_clip = '../output_videos/project_video_detected_'+str(datetime.datetime.now())+'.mp4'
white_clip = clip1.fl_image(pipeline) #NOTE: function expects color images
white_clip.write_videofile(output_clip, audio=False)

