# Initial version of camera pose estimate for use with air-bearing table - RPi Camera
# References:
# Ali Yasin Eser https://aliyasineser.medium.com/aruco-marker-tracking-with-opencv-8cb844c26628#:~:text=ArUco%20markers%20have%20sizes%20from,detect%20them%20in%20the%20image.
# Josh Day https://github.com/jwday/ComputerVision/blob/master/utilities/calibration_checkerboard.py

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
height = int(1080)              # output image height


### Draws image without annotations
# imageLoc --> image file location
# cap --> image read in by CV2
def readAndDrawImage(imageLoc):
    cap = cv2.imread(imageLoc)                      # read image from file path
    cv2.namedWindow('source', cv2.WINDOW_NORMAL)    # create window to display image
    cv2.startWindowThread()
    cv2.resizeWindow('source', width, height)       # set window size to predefined dimensions
    cv2.imshow('source', cap)                       # show image
    while not (cv2.waitKey(1) & 0xFF == ord('q')):  # wait until user presses quit
        pass
    cv2.destroyAllWindows()                         # close window
    return cap

### Import calibration parameters
# calibFile --> calibration file location
# cameraMatrix --> camera calibration matrix
# distCoeffs --> camera distortion matrix
def loadCalibrationParameters(calibFile):
    calibFile = cv2.FileStorage(calibLoc, cv2.FILE_STORAGE_READ)    # load in camera calibration file
    cameraMatrix = calibFile.getNode("camera_matrix").mat()         # camera calibration matrix
    distCoeffs = calibFile.getNode("dist_coeff").mat()              # camera distortion matrix
    return cameraMatrix, distCoeffs

### Read image, remove blur, and convert to grayscale
# cap --> image
# gray --> smoothed grayscale image
def preProcessImage(cap):
    blur = cv2.GaussianBlur(cap, (11, 11), 0)       # smooth image and remove Gaussian noise
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)   # convert to grayscale
    return gray

### Detect Aruco markers.
# gray --> grayscale image with blur
# corners --> marker x-y coordinates
# ids --> identifiers encoded in aruco markers
# rejected --> potential markers which were detected but rejected because they could not be parsed
def detectMarkers(gray, cameraMatrix, distCoeffs):                          # image grayscale + blur
    arucoDict = cv2.aruco.getPredefinedDictionary(0)         # small dictionary of 4x4 markers
    arucoParams = cv2.aruco.DetectorParameters()             # default aruco parameters
    detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

    corners, ids, rejectedImgPoints = detector.detectMarkers(gray) # marker detection
    print("Detected markers: {}".format(corners))
    print("Rejected markers: {}".format(rejectedImgPoints))
    return corners, ids, rejectedImgPoints

### Display Aruco markers.
# cap --> image (no blur/grayscale effects)
# corners --> marker x-y coordinates
# ids --> marker ids
def drawMarkers(cap, corners, ids):
    cap_marked = aruco.drawDetectedMarkers(cap, corners, ids)   # draw markers on image
    cv2.namedWindow('marker', cv2.WINDOW_NORMAL)                # create window to display image
    cv2.resizeWindow('marker', width, height)                   # resize window to predetermined dimensions
    cv2.imshow('marker', cap_marked)                            # display image
    while not(cv2.waitKey(1) & 0xFF == ord('q')):               # wait for user input
        pass
    cv2.destroyAllWindows()

### Estimate pose.
# cap --> image (without gray/blur)
# corners --> x-y marker coordinates
# rvec, tvec --> rotation and translation vectors
def estimatePose(cap, corners):
    rvec, tvec, objpoints = aruco.estimatePoseSingleMarkers(corners, marker_side_length, cameraMatrix, distCoeffs)
    print("Rvec: {}".format(rvec))
    print("Tvec: {}".format(tvec))
    print("Camera Matrix: {}".format(cameraMatrix))
    print("Distortion Coefficients: {}".format(distCoeffs))
    for i in range (0, len(rvec)):
        cap = cv2.drawFrameAxes(cap, cameraMatrix, distCoeffs, rvec[i, :, :], tvec[i, :, :], 0.1)
    # cap_drawn = cv2.drawFrameAxes(cap, cameraMatrix, distCoeffs, rvec, tvec, 2*marker_side_length) # draw axes
    cv2.namedWindow('axes', cv2.WINDOW_NORMAL)  # create a new window
    cv2.resizeWindow('axes', width, height)     # resize window to predefined dimensions
    cv2.imshow('axes', cap)               # display image with axes
    while not (cv2.waitKey(1) & 0xFF == ord('q')):
        pass
    cv2.destroyAllWindows()                     # wait for user input then close windows
    return rvec, tvec

if os.path.isfile(imageLoc):
    cap = readAndDrawImage(imageLoc)
    if os.path.isfile(calibLoc):
        cameraMatrix, distCoeffs = loadCalibrationParameters(calibLoc)
        gray = preProcessImage(cap)
        corners, ids, rejected = detectMarkers(gray, cameraMatrix, distCoeffs)
        drawMarkers(cap, corners, ids)
        rvec, tvec = estimatePose(cap, corners)
    else:
        print('Cannot find calibration file.')
else:
    print('Cannot find image.')
