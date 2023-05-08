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
calibLoc = '../images/calibImages/calib_april_11_low_res.yaml'   # calibration file location
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
outputDir = '/home/pi/OnboardStateEstimate/Air-Bearing-Table/ArucoTracking/Outputs'		# data saving directory
if os.path.isdir(outputDir) == False:		# create data saving directory only if it doesn't already exist
    os.mkdir(outputDir)
outfile = outputDir + "/CV_"+now.strftime("%m.%d.%y_%H.%M.%S")+"_{}FPS".format(framerate)
outfileVideo = outfile + ".h264"		# for video output
outfileCSV = outfile + ".csv"			# for pose estimate output
outfileHeader = ['Time', 'Rvec[x]', 'Rvec[y]', 'Rvec[z]', 'Tvec[x]', 'Tvec[y]', 'Tvec[z]']					# header for CSV pose output

### Processes an image
def detectPoseInFrame(connRecv, connSend, done):
    
    ### Set up output window
    cv2.namedWindow('video', cv2.WINDOW_NORMAL)
    cv2.startWindowThread()
    cv2.resizeWindow('video', width, height)
    print("Starting marker detection...")
    
    ### Detect markers
    startTime = time.time()
    while done.value != True:
        frame = connRecv.recv()							# receive raw frame from Pi camera
        blur = cv2.GaussianBlur(frame, (11, 11), 0) 	# smooth image and remove Gaussian noise
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)   # convert to grayscale
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, arucoDict, parameters=arucoParams, cameraMatrix=cameraMatrix, distCoeff=distCoeffs) # marker detection
        ret, rvec, tvec = aruco.estimatePoseBoard(corners, ids, board, cameraMatrix, distCoeffs, rvec_init, tvec_init)	# estimate board pose using markers
        capWithAxes = cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, rvec, tvec, 0.1)		# real-time visualization: draw axes
        cv2.imshow('video', capWithAxes)														# display image with axes
        now = time.time() - startTime
        connSend.send([now, rvec[0], rvec[1], rvec[2], tvec[0], tvec[1], tvec[2]])	# send data via pipe to CSV saving process
    
    ### Cleanup
    cv2.destroyAllWindows()
    print("Stopping marker detection.")

### Saves pose estimation data
def savePoseEstimate(connRecv, done):
    
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
receiveRawFrame, sendRawFrame = Pipe()			# sends raw frame from main process (Pi camera) to pose estimation process
receivePoseData, sendPoseData = Pipe()			# sends pose estimate data from pose estimation process to data saving process
process_marker_detection = Process(target=detectPoseInFrame, args=(receiveRawFrame, sendPoseData, finishedRecording,))	# estimates pose
process_data_saving = Process(target=savePoseEstimate, args=(receivePoseData, finishedRecording,))						# saves pose data to CSV
process_marker_detection.start()
process_data_saving.start()

# TODO: exit condition (user press q) doesn't seem to be working, so for now, we use a temporary exit condition of running for a certain time
runtime = 20
startTime = time.time()

print("Starting capture...")
camera.start_recording(outfileVideo)		#TODO: recording is kind of jumpy? figure out why (or do we need recording in the first place?)
for image in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    
    frame = image.array			# capture frame as an array
    sendRawFrame.send(frame)	# send for pose estimate processing
    rawCapture.truncate(0)		# clear and prepare for next frame
    
    # Stop video if user quits
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or time.time() - startTime > runtime:                 
        finishedRecording.value = True
        print("Stopping all recording.")
        break

# Cleanup
print("Cleaning up...")
time.sleep(1)
camera.stop_recording()
process_marker_detection.terminate()
process_marker_detection.join()
process_data_saving.terminate()
process_data_saving.join()
cv2.destroyAllWindows()
print("Done!")