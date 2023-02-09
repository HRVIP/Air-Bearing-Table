# Initial version of camera pose estimate for use with air-bearing table.
# References:
# Ali Yasin Eser https://aliyasineser.medium.com/aruco-marker-tracking-with-opencv-8cb844c26628#:~:text=ArUco%20markers%20have%20sizes%20from,detect%20them%20in%20the%20image.
# Josh Day https://github.com/jwday/ComputerVision/blob/master/utilities/calibration_checkerboard.py
# Peter F. Kelmperer http://www.peterklemperer.com/blog/2017/11/30/three-dimensional-aruco-boards/

### Imports
import numpy as np
import cv2
import cv2.aruco as aruco
import os

### Define useful constants
marker_side_length = 0.050     # marker side length in m #TODO: determine this value
calibLoc = '../images/calib_images/calib.yaml'  # calibration file location
imageLoc = '../images/test3.jpg'                # location of image to process
width = int(1280)              # output image width
height = int(720)              # output image height

### Import calibration parameters
calibFile = cv2.FileStorage(calibLoc, cv2.FILE_STORAGE_READ)    # load in camera calibration file
cameraMatrix = calibFile.getNode("camera_matrix").mat()         # camera calibration matrix
distCoeffs = calibFile.getNode("dist_coeff").mat()              # camera distortion matrix

### Read image and draw without annotations
cap = cv2.imread(imageLoc)                      # read image from file path
cv2.namedWindow('source', cv2.WINDOW_NORMAL)    # create window to display image
cv2.startWindowThread()
cv2.resizeWindow('source', width, height)       # set window size to predefined dimensions
cv2.imshow('source', cap)                       # show image
while not (cv2.waitKey(1) & 0xFF == ord('q')):  # wait until user presses quit
    pass
cv2.destroyAllWindows()                         # close window

### Apply grayscale and blur to image
blur = cv2.GaussianBlur(cap, (11, 11), 0)       # smooth image and remove Gaussian noise
gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)   # convert to grayscale

### Set up Aruco detector
arucoDict = cv2.aruco.getPredefinedDictionary(0)         # small dictionary of 4x4 markers
arucoParams = cv2.aruco.DetectorParameters()             # default aruco parameters
detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

### Create Aruco board
# In boardCorners, each row is a numpy array corresponding to a single marker on the board
# Each column is coordinates of a corner
# Each column of those columns corresponds to x, y, z
# This is currently populated with a single example marker.
boardCorners = [np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]])]
boardIDs = np.array([[0]], dtype=np.int32)                      # boardIDs has Aruco IDs of each marker, as np.array( [[id1], [id2], [id3], [id4]])
board = aruco.Board_create(board_corners, arucoDict, boardIDs)  # Create the board

### Detect markers and dislpay detected markers
corners, ids, rejectedImgPoints = detector.detectMarkers(gray)  # marker detection
cap_marked = aruco.drawDetectedMarkers(cap, corners, ids)       # draw markers on image
cv2.namedWindow('marker', cv2.WINDOW_NORMAL)                    # create window to display image
cv2.resizeWindow('marker', width, height)                       # resize window to predetermined dimensions
cv2.imshow('marker', cap_marked)                                # display image
while not(cv2.waitKey(1) & 0xFF == ord('q')):                   # wait for user input
    pass
cv2.destroyAllWindows()

### Estimate board pose
ret, rvec, tvec = aruco.estimatePoseBoard(corners, ids, board, cameraMatrix, distCoeffs)
print("Rvec: {}".format(rvec))
print("Tvec: {}".format(tvec))
print("Camera Matrix: {}".format(cameraMatrix))
print("Distortion Coefficients: {}".format(distCoeffs))
capWithAxes = cv2.drawFrameAxes(cap, cameraMatrix, distCoeffs, rvec, tvec, 0.1)
cv2.namedWindow('axes', cv2.WINDOW_NORMAL)      # create a new window
cv2.resizeWindow('axes', width, height)         # resize window to predefined dimensions
cv2.imshow('axes', capWithAxes)                 # display image with axes
while not (cv2.waitKey(1) & 0xFF == ord('q')):
    pass
cv2.destroyAllWindows()                         # wait for user input then close windows