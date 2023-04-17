# Pi Camera real-time video marker detection - for lab reunion demo!


### Imports
import numpy as np
import cv2
import cv2.aruco as aruco
from picamera.array import PiRGBArray
from picamera import PiCamera
import os
import time
import datetime
import ctypes
import csv

### Define useful variables
marker_side_length = 0.040       				# marker side length in m
calibLoc = '../images/calibImages/calib_april_11_low_res.yaml'   # calibration file location
width = int(640)               					# output image width (px)
height = int(480)              					# output image height (px)
framerate = 40									# camera framerate

### Import calibration parameters
print("Welcome to live marker detection!")
print("Press q to quit at any time.")
print("This video will time out after 10 minutes; press the green arrow at the top to restart.")
print("The camera will detect and mark axes on the ArUco marker (next to keyboard!)")
calibFile = cv2.FileStorage(calibLoc, cv2.FILE_STORAGE_READ)    # load in camera calibration file
cameraMatrix = calibFile.getNode("camera_matrix").mat()         # camera calibration matrix
distCoeffs = calibFile.getNode("dist_coeff").mat()              # camera distortion matrix

### Camera setup
camera = PiCamera()
camera.resolution = (width, height)						# TODO: determine relationship between resolution and runtime (tradeoff with accuracy?)
camera.framerate = framerate							# frames per second
rawCapture = PiRGBArray(camera, size=(width, height))	# 3D RGB array
time.sleep(1)											# allow camera to warm up

### Set up Aruco detector
arucoDict = cv2.aruco.getPredefinedDictionary(0)		# small dictionary of 4x4 markers
arucoParams = cv2.aruco.DetectorParameters_create()     # default aruco parameters

### Create Aruco board
w = 0.040			# marker width (m)
h = 0.040			# marker height (m)
dx = 0.002			# distance between markers (m)
origin = 0.0		# "origin" of ArUco board, in board coordinate system (0, 0, 0)
rvec_init = np.empty(3)		# initial estimate of board rotation - can be empty or initial guess
tvec_init = np.empty(3)		# initial estimate of board position - can be empty or initial guess

# Marker corners: top left, top right, bottom right, bottom left, with (x, y, z) for each
marker1 = np.array([[origin, h+h+dx, origin], [w, h+h+dx, origin], [w, h+dx, origin], [origin, h+dx, origin]], dtype=np.float32)
marker2 = np.array([[w+dx, h+h+dx, origin], [w+w+dx, h+h+dx, origin], [w+w+dx, h+dx, origin], [w+dx, h+dx, origin]], dtype=np.float32)
marker3 = np.array([[origin, h, origin], [w, h, origin], [w, origin, origin], [origin, origin, origin]], dtype=np.float32)
marker4 = np.array([[w+dx, h, origin], [w+w+dx, h, origin], [w+w+dx, origin, origin], [w+dx, origin, origin]], dtype=np.float32)
boardCorners = np.array([marker1, marker2, marker3, marker4])
boardIDs = np.array([[1], [2], [3], [4]], dtype=np.int32)		# ArUco ID of each marker (must correspond to order in which markers are defined!)
board = aruco.Board_create(boardCorners, arucoDict, boardIDs)	# actual board object
    
### For demo, we don't need output files or multiprocessing (yay!)

# TODO: exit condition (user press q) doesn't seem to be working, so for now, we use a temporary exit condition of running for a certain time
runtime = 600 # just in case exit code fails :)
startTime = time.time()
exampleText="Position: X=-0.00, Y=-0.00, Z=-0.00"
labelCoordsTop = (20, 50)
labelCoordsBottom = (20, 80)
labelfont = cv2.FONT_HERSHEY_SIMPLEX
labelColor = (255,255,255)
bgColor = (0,0,0)
labelThickness = 1
labelScale = 0.8
textSize, _ = cv2.getTextSize(exampleText, labelfont, labelScale, labelThickness)
text_w, text_h = textSize
xt, yt = labelCoordsTop
xb, yb = labelCoordsBottom
cv2.namedWindow('video', cv2.WINDOW_NORMAL)
cv2.resizeWindow('video', 640, 480)
for image in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    
    frame = image.array			# capture frame as an array
    blur = cv2.GaussianBlur(frame, (11, 11), 0) 	# smooth image and remove Gaussian noise
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)   # convert to grayscale
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, arucoDict, parameters=arucoParams, cameraMatrix=cameraMatrix, distCoeff=distCoeffs) # marker detection
    ret, rvec, tvec = aruco.estimatePoseBoard(corners, ids, board, cameraMatrix, distCoeffs, rvec_init, tvec_init)	# estimate board pose using markers
    if (len(corners) != 0):
        labelTextTop = "Position: X={:.2f}, Y={:.2f}, Z={:.2f} ".format(tvec[0], tvec[1], tvec[2])
        labelTextBottom =  "Rotation: X={:.2f}, Y={:.2f}, Z={:.2f}".format(rvec[0], rvec[1], rvec[2])
    else:
        labelTextTop = exampleText
        labelTextBottom = exampleText
    capWithAxes = cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, rvec, tvec, 0.1)		# real-time visualization: draw axes
    cv2.rectangle(capWithAxes, labelCoordsTop, (xt + text_w, yt - text_h), bgColor, -1)
    cv2.rectangle(capWithAxes, labelCoordsBottom,(xb + text_w, yb - text_h), bgColor, -1)
    capWithText = cv2.putText(capWithAxes, labelTextTop, labelCoordsTop, labelfont, labelScale, labelColor,labelThickness, cv2.LINE_AA)
    capWithText = cv2.putText(capWithText, labelTextBottom, labelCoordsBottom, labelfont, labelScale, labelColor,labelThickness, cv2.LINE_AA)
    cv2.imshow('video', capWithText)														# display image with axes
    rawCapture.truncate(0)		# clear and prepare for next frame
    
    # Stop video if user quits
    if cv2.waitKey(1) & 0xFF == ord('q') or time.time() - startTime > runtime:                 
        break

# Cleanup
print("Cleaning up...")
cv2.destroyAllWindows()
time.sleep(1)
cv2.destroyAllWindows()
print("Done!")