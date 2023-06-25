# Take a single photo (or a specified number of pictures) with Raspberry Pi camera (mostly for testing purposes)

# Imports
import cv2
import os
import picamera
from picamera.array import PiRGBArray
from datetime import datetime
import time

# Set up file saving
imageDir = '/home/pi/OnboardStateEstimate/images/'  # Default image save location
if (os.path.isdir(imageDir)) == False:
    os.mkdir(imageDir)

# Set up camera
camera = picamera.PiCamera(resolution=(1920, 1080), framerate=30)
time.sleep(2)
camera.shutter_speed = camera.exposure_speed
rawCapture = PiRGBArray(camera, size=(1920, 1080))

# Keep track of how many images are acpatured
numPictures = 10

# Capture images
camera.start_preview()
for i in range(numPictures):
    time.sleep(1)
    filename = imageDir + '/image{}.jpg'.format(i);
    camera.capture(filename)
camera.stop_preview()

print("Program ended.")
cv2.destroyAllWindows()
