### IMPORTS
import numpy as np
import cv2
import cv2.aruco as aruco
from picamera.array import PiRGBArray
from picamera import PiCamera
import os
import datetime
import board
import adafruit_ds3231
import adafruit_icm20x
import time
from multiprocessing import Process, Pipe, Manager, Value
from filterpy.kalman import KalmanFilter
import csv

### I2C DEVICE SETUP
i2c = board.I2C()                           # initialize I2C
icm = adafruit_icm20x.ICM20948(i2c, 0x69)   # connect to IMU
rtc = adafruit_ds3231.DS3231(i2c, 0x68)     # connect to RTC

### OUTPUT FILE SETUP
now = rtc.datetime		# time at which we save data
datetimeStr = "{:02d}-{:02d}-{:02d}_{:02d}-{:02d}-{:02d}".format(now.tm_mon, now.tm_mday, now.tm_year%1000, now.tm_hour, now.tm_min, now.tm_sec)
outputDir = '/home/pi/OnboardStateEstimate/Air-Bearing-Table/ArucoTracking/Outputs'		# data saving directory
if os.path.isdir(outputDir) == False:		# create data saving directory only if it doesn't already exist
    os.mkdir(outputDir)
outfileCSV = outputDir + "/OnboardKF_" + datetimeStr + ".csv"               # for CSV output
outfileHeader = ['Time', 'x', 'y', 'vx', 'vy', 'ax', 'ay', 'theta_z', 'wz']	# header for KF CSV output

### CV SETUP
marker_side_length = 0.040       				# marker side length in m
calibLoc = '../images/calibImages/calib_april_11_imperial_units.yaml'   # calibration file location
width = int(640)               					# output image width (px)
height = int(480)              					# output image height (px)
framerate = 32									# camera framerate

print("Importing calibration parameters...")
calibFile = cv2.FileStorage(calibLoc, cv2.FILE_STORAGE_READ)    # load in camera calibration file
cameraMatrix = calibFile.getNode("camera_matrix").mat()         # camera calibration matrix
distCoeffs = calibFile.getNode("dist_coeff").mat()              # camera distortion matrix

print("Setting up camera...")
camera = PiCamera()                                     # connect to Pi camera
camera.resolution = (width, height)						# specify resolution - should be the same as in calibration file
camera.framerate = framerate							# frames per second
rawCapture = PiRGBArray(camera, size=(width, height))	# 3D RGB array
time.sleep(1)				                            # allow camera to warm up

print("Setting up Aruco parameters...")
arucoDict = cv2.aruco.getPredefinedDictionary(0)		# small dictionary of 4x4 markers
arucoParams = cv2.aruco.DetectorParameters_create()     # default aruco parameters

print("Setting up Aruco board...")
w = marker_side_length			# marker width (m)
h = marker_side_length			# marker height (m)
dx = 0.002			            # distance between markers (m)
origin = 0.0		        # "origin" of ArUco board, in board coordinate system (0, 0, 0)
marker1 = np.array([[origin, h+h+dx, origin], [w, h+h+dx, origin], [w, h+dx, origin], [origin, h+dx, origin]], dtype=np.float32)
marker2 = np.array([[w+dx, h+h+dx, origin], [w+w+dx, h+h+dx, origin], [w+w+dx, h+dx, origin], [w+dx, h+dx, origin]], dtype=np.float32)
marker3 = np.array([[origin, h, origin], [w, h, origin], [w, origin, origin], [origin, origin, origin]], dtype=np.float32)
marker4 = np.array([[w+dx, h, origin], [w+w+dx, h, origin], [w+w+dx, origin, origin], [w+dx, origin, origin]], dtype=np.float32)
boardCorners = np.array([marker1, marker2, marker3, marker4])
boardIDs = np.array([[1], [2], [3], [4]], dtype=np.int32)		# ArUco ID of each marker (must correspond to order in which markers are defined!)
board = aruco.Board_create(boardCorners, arucoDict, boardIDs)	# actual board object
rvec_init = np.empty(3)		# initial estimate of board rotation - can be empty or initial guess
tvec_init = np.empty(3)		# initial estimate of board position - can be empty or initial guess

### SENSOR: IMU READING
# Reads IMU acceleration and gyro
def ReadIMU(ax_meas, ay_meas, wz_meas):
    while True:
        # Directly read sensor values
        ax_meas.value = icm.acceleration[0]
        ay_meas.value = icm.acceleration[1]
        wz_meas.value = icm.gyro[2]

### SENSOR: CV READING
# Read CV system to get x-y position and z-rotation
def ReadCV(x_meas, y_meas, theta_meas):
    while True:
        for image in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
            frame = rawCapture.array
            blur = cv2.GaussianBlur(frame, (11, 11), 0)
            gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, arucoDict, parameters=arucoParams, cameraMatrix=cameraMatrix, distCoeff=distCoeffs) # marker detection
            ret, rCamToMarker, tCamToMarker = aruco.estimatePoseBoard(corners, ids, board, cameraMatrix, distCoeffs, rvec_init, tvec_init)	# estimate board pose using markers
            rCamToMarkerMatrix = cv2.Rodrigues(rCamToMarker)[0]     # convert rotation vector to matrix
            rMarkerToCamMatrix = np.matrix(rCamToMarkerMatrix).T    # transpose of rotation matrix is rotation of camera relative to board
            rMarkerToCam = cv2.Rodrigues(rMarkerToCamMatrix)[0]     # camera rotation as a vector
            tMarkerToCam = np.dot(rMarkerToCamMatrix, np.matrix(-1 * tCamToMarker).T)  # camera position relative to marker/board
            x_meas.value = tMarkerToCam[0]
            y_meas.value = tMarkerToCam[1]
            theta_meas.value = rMarkerToCam[2]

### DATA SAVING
# Receives data from main process and saves to a CSV
def SaveCSV(receiveFilteredState, receiveKFTimestamp):

    # Set up CSV file
    with open(outfileCSV, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(outfileHeader)
        csvfile.close()

    # Save data continuously as updated states are calculated
    while True:
        time = receiveKFTimestamp.recv()    # receive timestamp
        state = receiveFilteredState.recv() # receive state
        savedata = []
        savedata.extend(time)
        savedata.extend(state)
        with open(outfileCSV, 'a', newline='') as csvfile:  # append to CSV
            writer = csv.writer(csvfile)
            writer.writerow(savedata)
            csvfile.close()

### MULTIPROCESSING
manager = Manager()     # manages multiprocesing variables

# Measured variables
x_meas = manager.Value('d', 0.0)
y_meas = manager.Value('d', 0.0)
ax_meas = manager.Value('d', 0.0)
ay_meas = manager.Value('d', 0.0)
theta_meas = manager.Value('d', 0.0)
wz_meas = manager.Value('d', 0.0)

# Pipe to send values to data saving
# Data saving uses a pipe instead of manager values since it must wait for updated values from KF
# whereas sensor readings are updated continuously - shouldn't "wait" on anything like a pipe
receiveFilteredState, sendFilteredState = Pipe()        # contains filtered/updated state from KF
receiveKFTimestamp, sendKFTimestamp = Pipe()            # contains timestamp at which updated state was calculated

# Start processes
process_CV = Process(target=ReadCV, args=(x_meas, y_meas, theta_meas,))
process_IMU = Process(target=ReadIMU, args=(ax_meas, ay_meas, wz_meas,))
process_CSV = Process(target=SaveCSV, args=(receiveFilteredState, receiveKFTimestamp,))
process_CV.start()
process_IMU.start()
process_CSV.start()

### KALMAN FILTER SETUP
dt=0.200    # placeholder for elapsed time
f = KalmanFilter(dim_x=8, dim_z=6)
f.x = np.array([0., 0., 0., 0., 0., 0., 0., 0.])    # initial state (x, vx, ax, y, vy, ay, theta, wz)
f.F = np.array([1., dt, 0.5*dt*dt, 0., 0., 0., 0., 0.],
               [0., 1., dt, 0., 0., 0., 0., 0.],
               [0., 0., 1., 0., 0., 0., 0., 0.],
               [0., 0., 0., 1., dt, 0.5*dt*dt, 0., 0.],
               [0., 0., 0., 0., 1., dt, 0., 0.],
               [0., 0., 0., 0., 0., 1., 0., 0.],
               [0., 0., 0., 0., 0., 0., 1., dt],
               [0., 0., 0., 0., 0., 0., 0., 1.])        # state transition matrix
f.H = np.array([[1., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 1., 0., 0., 0., 0., 0.],
                [0., 0., 0., 1., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 1., 0., 0.],
                [0., 0., 0., 0., 0., 0., 1., 0.],
                [0., 0., 0., 0., 0., 0., 0., 1.]])
f.P = np.identity(8) # TODO: this is a placeholder for cov matrix
f.R = np.identity(6) # TODO: placeholder for measurement noise
f.Q = np.identity(8) # TODO: placeholder for process noise

### MAIN LOOP
prev = datetime.datetime.now()
while True:
    # Get elapsed time for prediction
    now = datetime.datetime.now()
    dt = (now - prev).total_seconds()
    prev = now

    # Update state transition matrix using dt
    f.F = np.array([1., dt, 0.5*dt*dt, 0., 0., 0., 0., 0.],
                   [0., 1., dt, 0., 0., 0., 0., 0.],
                   [0., 0., 1., 0., 0., 0., 0., 0.],
                   [0., 0., 0., 1., dt, 0.5*dt*dt, 0., 0.],
                   [0., 0., 0., 0., 1., dt, 0., 0.],
                   [0., 0., 0., 0., 0., 1., 0., 0.],
                   [0., 0., 0., 0., 0., 0., 1., dt],
                   [0., 0., 0., 0., 0., 0., 0., 1.])

    # Read sensors, predict, and update
    z = np.array([x_meas.value, ax_meas.value, y_meas.value, ay_meas.value, theta_meas.value, wz_meas.value]).T
    f.predict()
    f.update(z)

    # Save results
    sendKFTimestamp.send(now)
    sendFilteredState.send(f.x)

    # exit condition
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Clean up and shut down
print("Cleaning up...")
time.sleep(1)
process_CV.terminate()
process_CV.join()
process_IMU.terminate()
process_IMU.join()
process_CSV.terminate()
process_CSV.join()
print("Done!")
