import numpy as np
from collections import deque

# Define a class to receive the characteristics of each line detection
# You can create an instance of the Line() class for the left and right lane lines
# to keep track of recent detections and to perform sanity checks.

class LaneLine():
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = (3.7 / 640)  # meters per pixel in x dimension

    Q_MAX_LEN_CURVE = 25 # controls the number of values averaged #25..50 #smaler less latency, higher stabilizes line better
    Q_MAX_LEN_BASE = 5 # controls the number of values averaged #25..50 #smaler less latency, higher stabilizes line better
    Q_MAX_LEN_BFIT = 5 # controls number of values averaged for best fit #5
    Q_MAX_LEN_DETC = 1 # controls number of remembered detected flags

    def __init__(self):
        #was the line detected in the last iteration?
        self.detected = deque(['False'] * LaneLine.Q_MAX_LEN_DETC, maxlen=LaneLine.Q_MAX_LEN_DETC) #set

        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None

        # x values for polynom plotting
        self.best_fitx = None #set
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.empty(3)] #set
        #polynomial coefficients averaged over the last n iterations
        self.best_fit_avg = None
        #polynomial coefficients of last n iterations
        self.best_fits = deque(maxlen=LaneLine.Q_MAX_LEN_BFIT)

        #radius of curvature of the line in some units
        self.radius_of_curvature = deque(np.array([0.0], dtype='float'), maxlen=LaneLine.Q_MAX_LEN_CURVE) # set
        #distance in meters of vehicle center from the line
        self.line_bases = deque(np.array([0.0], dtype='float'), maxlen=LaneLine.Q_MAX_LEN_BASE) # set
        #distance in meters of vehicle center from the line
        self.line_base_pos = None # set

        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')

        #x values for detected line pixels
        self.allx = None #set
        #y values for detected line pixels
        self.ally = None #set
        # line space
        self.ploty = None #set

        # inverse matrix to unwarp
        self.Minv = None #set

    def print_line_status(self, name='line instance'):
        # print('ym_per_pix: ', self.ym_per_pix)
        # print('xm_per_pix: ', self.xm_per_pix)
        # print('Q_MAX_LEN_CURVE: ', self.Q_MAX_LEN_CURVE)
        # print('Q_MAX_LEN_FITX: ', self.Q_MAX_LEN_FITX)
        print('\n\n------ ', name, ' ------')
        print('detected: ', self.detected)
        print('recent_xfitted: ', self.recent_xfitted)
        print('bestx: ', self.bestx)
        # print('Fitx: ', self.fitx)
        print('current_fit: ', self.current_fit, '- shape', self.current_fit.shape)
        print('avg best_fit: ', self.best_fit_avg)
        print('best_fits: ', self.best_fits)
        print('radius_of_curvature: ', self.radius_of_curvature)
        print('line_base_pos: ', self.line_base_pos)
        print('diffs: ', self.diffs, '- shape', self.diffs.shape)
        print('allx: ', self.allx, '- shape', self.allx.shape)
        print('ally: ', self.ally, '- shape', self.ally.shape)
        # print('ploty: ', self.ploty, '- shape', self.ploty.shape)  # array with numbers 0..719
        # print('Minv: ', self.Minv, '- shape', self.Minv.shape)
        return
