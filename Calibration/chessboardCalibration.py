### Calibrate camera using pre-saved chessboard images

### Imports
import numpy as np
import cv2
import glob

WAIT_TIME = 1000    # wait time in ms

# termination criteria
# stop calibration after 30 iterations occur, or when convergence metric reaches 0.001
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

### Calibrate camera
# calibFiles --> directory to find calibration images
# width --> number of intersection points of squares in long side of calibration board
# height --> number of intersection points of squares in short size of calibratoin board
# squareSize --> side of chessboard square
def calibrateCamera(calibFiles, width, height, squareSize):

    ### Prepare checkerboard matrix
    objp = np.zeros((height*width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1,2)
    objp = objp * squareSize
    ### Arrays to store object and image points
    objpoints = [] # 3D point in real world space
    imgpoints = [] # 2D points in image plane

    ### Open directory for calibration images
    images = glob.glob(calibFiles + '/*.jpg')
    found = 0
    i = 0

    for fname in images:
        i = i + 1
        ### Find chessboard corners in a single image
        img = cv2.imread(fname)   # read an image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert to grayscale
        ret, corners = cv2.findChessboardCorners(gray, (width, height), None) # find internal chessboard corners

        ### If corners are found, save points and display to user
        if ret == True:
            found = found + 1
            print("Found {} of {} images".format(found, i))
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            img = cv2.drawChessboardCorners(img, (width, height), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()
    print("Generating calibration matrix...")
    shape = tuple([1920, 1080])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, shape, None, None)

    return ret, mtx, dist, rvecs, tvecs

### Save calibration data
# mtx --> camera matrix
# dist --> distortion matrix
# path --> where to save calibration data
def saveCalibrationData(mtx, dist, path):
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    cv_file.write("camera_matrix", mtx)
    cv_file.write("dist_coeff", dist)
    cv_file.release()

### Main function
width = 21            # horizontal intersections
height = 20           # vertical intersections
squareSize = 0.0254   # side length (m)
calibFiles = "images/piCameraCalibImages"
destination = calibFiles + '/calib_py_april_7.yaml'
ret, mtx, dist, rvecs, tvecs = calibrateCamera(calibFiles, width, height, squareSize)
saveCalibrationData(mtx, dist, destination)
