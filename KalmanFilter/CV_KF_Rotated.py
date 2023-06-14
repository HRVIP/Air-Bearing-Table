# Pi Camera real-time video marker detection
# This version uses multiprocessing for improved performance
# One process for video recording, one for data saving, and one for pose estimation

### Imports
import numpy as np
import cv2
import cv2.aruco as aruco
from picamera.array import PiRGBArray
from picamera import PiCamera
import os
import time
import datetime
from multiprocessing import Process, Pipe, Manager, Value
import ctypes
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
import csv
import pandas as pd
import matplotlib.pyplot as plt

### Define useful variables
marker_side_length = 0.040       				# marker side length in m
calibLoc = '../images/calibImages/calib_april_11_imperial_units.yaml'   # calibration file location
width = int(640)               					# output image width (px)
height = int(480)              					# output image height (px)
manager = Manager()								# manages values shared between processes
finishedRecording = manager.Value(ctypes.c_bool, False)		# exit condition
framerate = 32									# camera framerate

### Import calibration parameters
print("Import calibration parameters...")
calibFile = cv2.FileStorage(calibLoc, cv2.FILE_STORAGE_READ)    # load in camera calibration file
cameraMatrix = calibFile.getNode("camera_matrix").mat()         # camera calibration matrix
distCoeffs = calibFile.getNode("dist_coeff").mat()              # camera distortion matrix

### Camera setup
print("Setting up camera...")
camera = PiCamera()
camera.resolution = (width, height)						# TODO: determine relationship between resolution and runtime (tradeoff with accuracy?)
camera.framerate = framerate							# frames per second
rawCapture = PiRGBArray(camera, size=(width, height))	# 3D RGB array
time.sleep(1)											# allow camera to warm up

### Set up Aruco detector
print("Setting up Aruco parameters...")
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
    
### Set up output files
size = (width, height)				# video output size
now = datetime.datetime.now()		# time at which we save data
outputDir = '/home/pi/OnboardStateEstimate/Air-Bearing-Table/KalmanFilter/Outputs'		# data saving directory
if os.path.isdir(outputDir) == False:		# create data saving directory only if it doesn't already exist
    os.mkdir(outputDir)
outfile = outputDir + "/CV_KF_"+now.strftime("%m.%d.%y_%H.%M.%S")+"_{}FPS".format(framerate)
outfileCSV = outfile + ".csv"			# for pose estimate output
outfileHeader = ['Time', 'x_meas', 'y_meas', 'theta_meas', 'x_out', 'vx_out', 'y_out', 'vy_out', 'theta_out', 'w_out']					# header for CSV pose output

### Runs Kalman Filter
def kalmanFilterProcess(x_meas, y_meas, theta_meas, sendSaveData, done):
    ### KALMAN FILTER SETUP
    print("Setting up Kalman Filter...")
    dt = 0.200    # placeholder for elapsed time
    f = KalmanFilter(dim_x=6, dim_z=3)
    t_start = datetime.datetime.now()
    print("Created Kalman Filter! Setting up initial state...")
    f.x = np.array([0., 0., 0., 0., 0., 0.])    # initial state (x, vx, y, vy, theta, wz)
    f.F = np.array([[1., dt, 0., 0., 0., 0.],
                    [0., 1., 0., 0., 0., 0.],
                    [0., 0., 1., dt, 0., 0.],
                    [0., 0., 0., 1., 0., 0.],
                    [0., 0., 0., 0., 1., dt],
                    [0., 0., 0., 0., 0., 1.]])        # state transition matrix
    print("Setting up measurement matrix...")
    f.H = np.array([[1., 0., 0., 0., 0., 0.],
                    [0., 0., 1., 0., 0., 0.],
                    [0., 0., 0., 0., 1., 0.]])  # measurement matrix (map states to measurements)
    print("Setting up noise matrices...")
    f.P = 10 * np.identity(6) # TODO: this is a placeholder for cov matrix
    f.R = 8 * np.identity(3) # TODO: placeholder for measurement noise
    f.Q = np.identity(6) # TODO: placeholder for process noise

    prev = datetime.datetime.now()
    while done.value != True:

        # Get elapsed time for prediction
        now = datetime.datetime.now()
        dt = (now - prev).total_seconds()
        prev = now

        # Update state transition matrix using dt
        f.F = np.array([[1., dt, 0., 0., 0., 0.],
                        [0., 1., 0., 0., 0., 0.],
                        [0., 0., 1., dt, 0., 0.],
                        [0., 0., 0., 1., 0., 0.],
                        [0., 0., 0., 0., 1., dt],
                        [0., 0., 0., 0., 0., 1.]])

        # Read sensors, predict, and update
        z = np.array([x_meas.value, y_meas.value, theta_meas.value])
        f.predict()
        f.update(z.T)

        # Save results
        allData = np.concatenate((np.array([(now - t_start).total_seconds()]), f.z, f.x))
        sendSaveData.send(allData)

### Saves pose estimation data
def saveData(receiveSaveData, done):
    
    # Set up CSV file
    with open(outfileCSV, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(outfileHeader)	# write header
        csvfile.close()
        
    # Save data
    while done.value != True:
        data = receiveSaveData.recv()		# receive pose data
        with open(outfileCSV, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(data)	# write pose data as a row in CSV file
            csvfile.close()
    
# Start processes
print("Starting processses...")
x_meas = manager.Value('d', 0.0)
y_meas = manager.Value('d', 0.0)
theta_meas = manager.Value('d', 0.0)
receiveSaveData, sendSaveData = Pipe()			# sends pose estimate data from pose estimation process to data saving process
process_KF = Process(target=kalmanFilterProcess, args=(x_meas, y_meas, theta_meas, sendSaveData, finishedRecording,))   # estimates pose
process_CSV = Process(target=saveData, args=(receiveSaveData, finishedRecording,))						# saves pose data to CSV
process_KF.start()
process_CSV.start()

### MAIN LOOP
print("Entering main loop!")

### Set up output window
cv2.namedWindow('video', cv2.WINDOW_NORMAL)
cv2.startWindowThread()
cv2.resizeWindow('video', width, height)
runtime = 20
startTime = time.time()

### Detect markers
print("Starting marker detection...")
for image in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):

    frame = image.array			# capture frame as an array
    blur = cv2.GaussianBlur(frame, (11, 11), 0) 	# smooth image and remove Gaussian noise
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)   # convert to grayscale
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, arucoDict, parameters=arucoParams, cameraMatrix=cameraMatrix, distCoeff=distCoeffs) # marker detection
    ret, rCameraFrame, tCameraFrame = aruco.estimatePoseBoard(corners, ids, board, cameraMatrix, distCoeffs, rvec_init, tvec_init)	# estimate board pose using markers
    capWithAxes = cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, rCameraFrame, tCameraFrame, 0.1)		# real-time visualization: draw axes
    cv2.imshow('video', capWithAxes)
    
    # rotate into marker reference frame
    # NOTE: current testing is with the cubesat marker. this rotation is just between that reference frame and the camera frame
    rCameraFrameMatrix = cv2.Rodrigues(rCameraFrame)[0]				# convert rotation vector to matrix
    rMarkerFrameMatrix = np.matrix(rCameraFrameMatrix).T			# transpose of rotation matrix is rotation of camera relative to markers
    rMarkerFrame = cv2.Rodrigues(rMarkerFrameMatrix)[0]				# camera rotation relative to markers as a vector
    tMarkerFrame = np.dot(rMarkerFrameMatrix, np.matrix(-1 * tCameraFrame).T)	# camera position relative to markers

    # update MP values
    # for air table: x-y is flat (corresponds to aruco x-z), z is up (corresponds to aruco y)
    x_meas.value = float(tMarkerFrame[0][0])		# horizontal placement of camera
    y_meas.value = float(tMarkerFrame[2][0])		# distance of camera from marker (x-y is "flat" for our purposes)
    theta_meas.value = float(rMarkerFrame[1][0])  	# display image with axes
    rawCapture.truncate(0)		# clear and prepare for next frame

    # Stop video if user quits
    if cv2.waitKey(1) & 0xFF == ord('q') or time.time() - startTime > runtime:
        print("Stopping marker detection.")
        finishedRecording.value = True
        break

### Cleanup

# Clean up and shut down
print("Cleaning up...")
time.sleep(1)
cv2.destroyAllWindows()
process_KF.terminate()
process_KF.join()
process_CSV.terminate()
process_CSV.join()
print("Done collecting data!")

### plot results!
outdata = pd.read_csv(outfileCSV)
deltaT = outdata.Time
x_meas = outdata.x_meas
y_meas = outdata.y_meas
theta_meas = outdata.theta_meas
x_out = outdata.x_out
y_out = outdata.y_out
theta_out = outdata.theta_out
plt.plot(deltaT, theta_meas, label="measured")
plt.plot(deltaT, theta_out, label="filtered")
plt.legend()
plt.show()
