### IMPORTS
import numpy as np
import cv2
import cv2.aruco as aruco
from picamera.array import PiRGBArray
from picamera import PiCamera
import os
import sys
import datetime
import time
from multiprocessing import Process, Pipe, Manager, Value
from filterpy.kalman import KalmanFilter
import csv

### OUTPUT FILE SETUP
now = datetime.datetime.now()		# time at which we save data
datetimeStr = "{:02d}-{:02d}-{:02d}_{:02d}{:02d}{:02d}".format(now.month, now.day, now.year%1000, now.hour, now.minute, now.second)
outputDir = '/home/pi/OnboardStateEstimate/Air-Bearing-Table/ArucoTracking/Outputs'		# data saving directory
outfileCSV = outputDir + "/OnboardKF_CV_" + datetimeStr + ".csv"                        # for CSV output
if os.path.isdir(outputDir) == False:		# create data saving directory only if it doesn't already exist
    try:
        os.mkdir(outputDir)
    except:
        print("Failed to create output directory. Saving in current working directory.")
        outfileCSV = datetimeStr + ".csv"
outfileHeader = ['Time', 'x_meas', 'y_meas', 'theta_meas', 'x_filtered', 'vx_filtered', 'vy_filtered', 'theta_z_filtered', 'wz_filtered']	# header for KF CSV output

### Define useful variables
marker_side_length = 0.040       				# marker side length in m
calibLoc = '../images/calibImages/calib_april_11_low_res.yaml'   # calibration file location
width = int(640)               					# output image width (px)
height = int(480)              					# output image height (px)
framerate = 40									# camera framerate

if os.path.isfile(calibLoc):
    print("Importing calibration parameters...")
    calibFile = cv2.FileStorage(calibLoc, cv2.FILE_STORAGE_READ)    # load in camera calibration file
    cameraMatrix = calibFile.getNode("camera_matrix").mat()         # camera calibration matrix
    distCoeffs = calibFile.getNode("dist_coeff").mat()              # camera distortion matrix
else:
    print("Calibration file path is invalid.")
    sys.exit()

print("Setting up camera...")
camera = PiCamera()
camera.resolution = (width, height)						# TODO: determine relationship between resolution and runtime (tradeoff with accuracy?)
camera.framerate = framerate							# frames per second
rawCapture = PiRGBArray(camera, size=(width, height))	# 3D RGB array
time.sleep(1)											# allow camera to warm up

print("Setting up Aruco parameters...")
arucoDict = cv2.aruco.getPredefinedDictionary(0)		# small dictionary of 4x4 markers
arucoParams = cv2.aruco.DetectorParameters_create()     # default aruco parameters

print("Setting up Aruco board...")
w = 0.040			# marker width (m)
h = 0.040			# marker height (m)
dx = 0.002			# distance between markers (m)
origin = 0.0		# "origin" of ArUco board, in board coordinate system (0, 0, 0)
rvec_init = np.empty(3)		# initial estimate of board rotation - can be empty or initial guess
tvec_init = np.empty(3)		# initial estimate of board position - can be empty or initial guess
marker1 = np.array([[origin, h+h+dx, origin], [w, h+h+dx, origin], [w, h+dx, origin], [origin, h+dx, origin]], dtype=np.float32)
marker2 = np.array([[w+dx, h+h+dx, origin], [w+w+dx, h+h+dx, origin], [w+w+dx, h+dx, origin], [w+dx, h+dx, origin]], dtype=np.float32)
marker3 = np.array([[origin, h, origin], [w, h, origin], [w, origin, origin], [origin, origin, origin]], dtype=np.float32)
marker4 = np.array([[w+dx, h, origin], [w+w+dx, h, origin], [w+w+dx, origin, origin], [w+dx, origin, origin]], dtype=np.float32)
boardCorners = np.array([marker1, marker2, marker3, marker4])
boardIDs = np.array([[1], [2], [3], [4]], dtype=np.int32)		# ArUco ID of each marker (must correspond to order in which markers are defined!)
board = aruco.Board_create(boardCorners, arucoDict, boardIDs)	# actual board object

### SENSOR: CV READING
# Read CV system to get x-y position and z-rotation
def ReadCV(x_meas, y_meas, theta_meas, rawCapture, camera):
    print("Detection Started!")
    cv2.namedWindow('video', cv2.WINDOW_NORMAL)
    cv2.startWindowThread()
    cv2.resizeWindow('video', width, height)
    while True:
        image = camera.capture(rawCapture, format="bgr", use_video_port=True)

        # Process image
        frame = image.array
        blur = cv2.GaussianBlur(frame, (11, 11), 0)
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)

        # Detect markers
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, arucoDict, parameters=arucoParams, cameraMatrix=cameraMatrix, distCoeff=distCoeffs) # marker detection
        ret, rvec, tvec = aruco.estimatePoseBoard(corners, ids, board, cameraMatrix, distCoeffs, rvec_init, tvec_init)	# estimate board pose using markers
        x_meas.value = tvec[0]
        y_meas.value = tvec[1]
        theta_meas.value = rvec[2]
        print("%f, %f, %f").format(tvec[0], tvec[1], rvec[2])
        capWithAxes = cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, rvec, tvec, 0.1)		# real-time visualization: draw axes
        cv2.imshow('video', capWithAxes)		
        rawCapture.truncate(0)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

### DATA SAVING
# Receives data from main process and saves to a CSV
def SaveCSV(receiveSaveData):

    # Set up CSV file
    try:
        with open(outfileCSV, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(outfileHeader)
            csvfile.close()
    except:
        print("Failed to set up CSV header.")

    # Save data continuously as updated states are calculated
    while True:
        data = receiveSaveData.recv()    # receive data from main process
        try:
            with open(outfileCSV, 'a', newline='') as csvfile:  # append to CSV
                writer = csv.writer(csvfile)
                writer.writerow(data)
                csvfile.close()
        except:
            print("Failed to write the following CSV data:")
            print(data)

### MULTIPROCESSING
print("Setting up multiprocessing manager...")
manager = Manager()     # manages multiprocesing variables

# Measured variables
print("Setting up multiprocessing variables...")
x_meas = manager.Value('d', 0.0)
y_meas = manager.Value('d', 0.0)
theta_meas = manager.Value('d', 0.0)

# Pipe to send values to data saving
# Data saving uses a pipe instead of manager values since it must wait for updated values from KF
# whereas sensor readings are updated continuously - shouldn't "wait" on anything like a pipe
print("Setting up multiprocessing pipe...")
receiveSaveData, sendSaveData = Pipe()        # contains filtered/updated state from KF

# Start processes
print("Defining processes..")
process_CV = Process(target=ReadCV, args=(x_meas, y_meas, theta_meas, rawCapture, camera,))
process_CSV = Process(target=SaveCSV, args=(receiveSaveData,))
print("Starting processes...")
process_CV.start()
process_CSV.start()

### KALMAN FILTER SETUP
print("Setting up Kalman Filter...")
dt = 0.200    # placeholder for elapsed time
f = KalmanFilter(dim_x=6, dim_z=3)
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
f.P = np.identity(6) # TODO: this is a placeholder for cov matrix
f.R = np.identity(3) # TODO: placeholder for measurement noise
f.Q = np.identity(6) # TODO: placeholder for process noise

### MAIN LOOP
print("Entering main loop!")
prev = datetime.datetime.now()
start = prev
runtime = 20
while (prev - start).total_seconds() < runtime:
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
    z = np.array([x_meas.value, y_meas.value, theta_meas.value]).T
    f.predict()
    f.update(z)

    # Save results
    allData = np.concatenate((np.array([now]), f.z, f.x))
    sendSaveData.send(allData)

    # exit condition
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Clean up and shut down
print("Cleaning up...")
time.sleep(1)
process_CV.terminate()
process_CV.join()
process_CSV.terminate()
process_CSV.join()
print("Done!")
