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

### Define useful variables
marker_side_length = 0.040       # marker side length in m #TODO: determine this value
calibLoc = '../images/calibImages/calib.yaml'   # calibration file location
width = int(640)               	# output image width (px)
height = int(480)              	# output image height (px)
manager = Manager()				# manages values shared between processes
finishedRecording = manager.Value(ctypes.c_bool, False)		# exit condition
framerate = 32					# camera framerate

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
arucoDict = cv2.aruco.getPredefinedDictionary(0)        		# small dictionary of 4x4 markers
arucoParams = cv2.aruco.DetectorParameters_create()             # default aruco parameters

### Create Aruco board
w = 0.040			# marker width (m)
h = 0.040			# marker height (m)
dx = 0.002			# distance between markers (m)
origin = 0.0
rvec_init = np.empty(3)
tvec_init = np.empty(3)
marker1 = np.array([[origin, h+h+dx, origin], [w, h+h+dx, origin], [w, h+dx, origin], [origin, h+dx, origin]], dtype=np.float32)
marker2 = np.array([[w+dx, h+h+dx, origin], [w+w+dx, h+h+dx, origin], [w+w+dx, h+dx, origin], [w+dx, h+dx, origin]], dtype=np.float32)
marker3 = np.array([[w+dx, h, origin], [w+w+dx, h, origin], [w+w+dx, origin, origin], [w+dx, origin, origin]], dtype=np.float32)
marker4 = np.array([[origin, h, origin], [w, h, origin], [w, origin, origin], [origin, origin, origin]], dtype=np.float32)
boardCorners = np.array([marker1, marker2, marker4, marker3])
boardIDs = np.array([[1], [2], [3], [4]], dtype=np.int32)
board = aruco.Board_create(boardCorners, arucoDict, boardIDs)
    
### Processes an image
def detectPoseInFrame(connRecv, connSend, done):
    ### Set up output window
    cv2.namedWindow('video', cv2.WINDOW_NORMAL)
    cv2.startWindowThread()
    cv2.resizeWindow('video', width, height)
    
    print("Starting marker detection...")
    while done.value != True:
        frame = connRecv.recv()
        blur = cv2.GaussianBlur(frame, (11, 11), 0) # smooth image and remove Gaussian noise
        gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)   # convert to grayscale
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, arucoDict, parameters=arucoParams, cameraMatrix=cameraMatrix, distCoeff=distCoeffs) # marker detection
        ret, rvec, tvec = aruco.estimatePoseBoard(corners, ids, board, cameraMatrix, distCoeffs, rvec_init, tvec_init)
        capWithAxes = cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, rvec, tvec, 0.1)
        cv2.imshow('video', capWithAxes)	# display image with axes
        connSend.send(capWithAxes)			# send to video saving processing
    
    cv2.destroyAllWindows()
    print("Stopping marker detection.")
        
### Saves video
def saveVideo(connRecv, done):
    size = (width, height)
    now = datetime.datetime.now()
    outfile = "CV_"+now.strftime("%m.%d.%y_%H.%M.%S")+"_{}FPS.avi".format(framerate)
    output = cv2.VideoWriter(outfile, cv2.VideoWriter_fourcc(*'MJPG'), framerate, (width, height))
    print("Starting video saving...")
    while done.value != True:
        frame = connRecv.recv()
        output.write(frame)
    output.release()
    print("Stopping recording.")
    
# Start processes
receiveRawFrame, sendRawFrame = Pipe()
receiveProcessedFrame, sendProcessedFrame = Pipe()
process_marker_detection = Process(target=detectPoseInFrame, args=(receiveRawFrame, sendProcessedFrame, finishedRecording,))
process_video_saving = Process(target=saveVideo, args=(receiveProcessedFrame, finishedRecording,))
process_marker_detection.start()
process_video_saving.start()

# TODO: exit condition (user press q) doesn't seem to be working, so for now, we use a temporary exit condition of running for a certain time
runtime = 20
startTime = time.time()

print("Starting video...")
for image in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    
    # Process a frame and add to queue
    frame = image.array
    sendRawFrame.send(frame)
    
    # Clear and prepare for next frame
    rawCapture.truncate(0)
    
    # Stop video if user quits
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or time.time() - startTime > runtime:                 
        finishedRecording = True
        print("Stopping all recording.")
        break

# Cleanup
print("Cleaning up...")
time.sleep(1)
process_marker_detection.terminate()
process_video_saving.terminate()
process_marker_detection.join()
process_marker_detection.terminate()
cv2.destroyAllWindows()
