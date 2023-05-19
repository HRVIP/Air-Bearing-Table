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
import csv

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

### Processes an image
def getCVMeasurement(x_meas, y_meas, theta_meas, done):
    
    ### Set up output window
    cv2.namedWindow('video', cv2.WINDOW_NORMAL)
    cv2.startWindowThread()
    cv2.resizeWindow('video', width, height)
    print("Starting marker detection...")
    
    ### Detect markers
    while done.value != True:

        frame = camera.capture(rawCapture, format="bgr", use_video_port=True)							# receive raw frame from Pi camera
        blur = cv2.GaussianBlur(frame, (11, 11), 0) 	# smooth image and remove Gaussian noise
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)   # convert to grayscale
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, arucoDict, parameters=arucoParams, cameraMatrix=cameraMatrix, distCoeff=distCoeffs) # marker detection
        ret, rvec, tvec = aruco.estimatePoseBoard(corners, ids, board, cameraMatrix, distCoeffs, rvec_init, tvec_init)	# estimate board pose using markers
        capWithAxes = cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, rvec, tvec, 0.1)		# real-time visualization: draw axes
        cv2.imshow('video', capWithAxes)
        rawCapture.truncate(0)
        x_meas.value = tvec[0]
        y_meas.value = tvec[1]
        theta_meas.value = rvec[2]

    ### Cleanup
    cv2.destroyAllWindows()
    print("Stopping marker detection.")

### Saves pose estimation data
def saveData(connRecv, done):
    
    # Set up CSV file
    with open(outfileCSV, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(outfileHeader)	# write header
        csvfile.close()
        
    # Save data
    while done.value != True:
        data = connRecv.recv()		# receive pose data
        with open(outfileCSV, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(data)	# write pose data as a row in CSV file
            csvfile.close()
    
# Start processes
x_meas = manager.Value('d', 0.0)
y_meas = manager.Value('d', 0.0)
theta_meas = manager.Value('d', 0.0)
receiveSaveData, sendSaveData = Pipe()			# sends pose estimate data from pose estimation process to data saving process
process_CV = Process(target=getCVMeasurement, args=(x_meas, y_meas, theta_meas, finishedRecording,))	# estimates pose
process_Data = Process(target=saveData, args=(receiveSaveData, finishedRecording,))						# saves pose data to CSV
process_CV.start()
process_Data.start()

runtime = 20
startTime = time.time()

### KALMAN FILTER SETUP
print("Setting up Kalman Filter")
dt = 0.200    # placeholder for elapsed time
f = KalmanFilter(dim_x=6, dim_z=3)
f.x = np.array([0., 0., 0., 0., 0., 0.])    # initial state (x, vx, y, vy, theta, wz)
f.F = np.array([1., dt, 0., 0., 0., 0.],
               [0., 1., 0., 0., 0., 0.],
               [0., 0., 1., dt, 0., 0.],
               [0., 0., 0., 1., 0., 0.],
               [0., 0., 0., 0., 1., dt],
               [0., 0., 0., 0., 0., 1.])        # state transition matrix
f.H = np.array([[1., 0., 0., 0., 0., 0.],
                [0., 0., 1., 0., 0., 0.],
                [0., 0., 0., 0., 1., 0.]])  # measurement matrix (map states to measurements)
f.P = np.identity(8) # TODO: this is a placeholder for cov matrix
f.R = np.identity(6) # TODO: placeholder for measurement noise
f.Q = np.identity(8) # TODO: placeholder for process noise

print("Starting filter...")
prev = datetime.datetime.now()
while True:
    # Get elapsed time for prediction
    now = datetime.datetime.now()
    dt = (now - prev).total_seconds()
    prev = now

    # Update state transition matrix using dt
    f.F = np.array([1., dt, 0., 0., 0., 0.],
                   [0., 1., 0., 0., 0., 0.],
                   [0., 0., 1., dt, 0., 0.],
                   [0., 0., 0., 1., 0., 0.],
                   [0., 0., 0., 0., 1., dt],
                   [0., 0., 0., 0., 0., 1.])

    # Read sensors, predict, and update
    z = np.array([x_meas.value, y_meas.value, theta_meas.value]).T
    f.predict()
    f.update(z)

    # Save results
    allData = np.concatenate((np.array([now]), f.z.T, f.x.T))
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
process_Data.terminate()
process_Data.join()
print("Done!")
