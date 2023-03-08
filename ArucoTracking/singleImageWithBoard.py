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
marker_side_length = 0.040       # marker side length in m #TODO: determine this value
calibLoc = '../images/calibImages/calib.yaml'   # calibration file location
imageLoc = '../images/image3.jpg'          # location of image to process
width = int(1920)               # output image width
height = int(1080)              # output image height           # output image height

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
arucoDict = cv2.aruco.getPredefinedDictionary(0)        		# small dictionary of 4x4 markers
arucoParams = cv2.aruco.DetectorParameters_create()             		# default aruco parameters

### Create Aruco board
w = 0.040			# marker width (m)
h = 0.040			# marker height (m)
dx = 0.002			# distance between markers (m)
origin = 0.0

marker1 = np.array([[origin, h+h+dx, origin], [w, h+h+dx, origin], [w, h+dx, origin], [origin, h+dx, origin]], dtype=np.float32)
marker2 = np.array([[w+dx, h+h+dx, origin], [w+w+dx, h+h+dx, origin], [w+w+dx, h+dx, origin], [w+dx, h+dx, origin]], dtype=np.float32)
marker3 = np.array([[w+dx, h, origin], [w+w+dx, h, origin], [w+w+dx, origin, origin], [w+dx, origin, origin]], dtype=np.float32)
marker4 = np.array([[origin, h, origin], [w, h, origin], [w, origin, origin], [origin, origin, origin]], dtype=np.float32)
boardCorners = np.array([marker1, marker2, marker4, marker3])
boardIDs = np.array([[1], [2], [3], [4]], dtype=np.int32)
board = aruco.Board_create(boardCorners, arucoDict, boardIDs)

### Detect markers and dislpay detected markers
corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, arucoDict, parameters=arucoParams, cameraMatrix=cameraMatrix, distCoeff=distCoeffs) # marker detection
cap_marked = aruco.drawDetectedMarkers(cap, corners, ids)       # draw markers on image
cv2.namedWindow('marker', cv2.WINDOW_NORMAL)                    # create window to display image
cv2.resizeWindow('marker', width, height)                       # resize window to predetermined dimensions
cv2.imshow('marker', cap_marked)                                # display image
while not(cv2.waitKey(1) & 0xFF == ord('q')):                   # wait for user input
    pass
cv2.destroyAllWindows()

### Estimate board pose
rvec_init, tvec_init, objpoints = aruco.estimatePoseSingleMarkers(corners[0], marker_side_length, cameraMatrix, distCoeffs)
ret, rvec, tvec = aruco.estimatePoseBoard(corners, ids, board, cameraMatrix, distCoeffs, rvec_init, tvec_init)
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