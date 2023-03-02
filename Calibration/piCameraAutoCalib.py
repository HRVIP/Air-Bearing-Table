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
calibDir = imageDir + '/calibImages'
if (os.path.isdir(calibDir)) == False:
    os.mkdir(calibDir)

# Set up camera
camera = picamera.PiCamera(resolution=(1920, 1080), framerate=30)
time.sleep(2)
camera.shutter_speed = camera.exposure_speed
rawCapture = PiRGBArray(camera, size=(1920, 1080))

# Keep track of how many images are acpatured
img_int = 0
numImagesRequired = 300

# Continuously capture calibration images
for image in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    
    # Capture image
    frame = image.array
    print("Capturing image...")
    
    # Display image
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    # Save image
    img_filename = '/capture_' + str(img_int) + '.jpg'
    now = datetime.now()
    dt_string = now.strftime("%H:%M:%S")
    fileDir = calibDir + img_filename
    cv2.imwrite(fileDir, img=frame)
    img_int += 1
    
    # Process image
    print("Processing image...")
    img_read = cv2.imread(fileDir, cv2.IMREAD_ANYCOLOR)
    print("Converting RGB image to grayscale...")
    gray = cv2.cvtColor(img_read, cv2.COLOR_BGR2GRAY)
    print("Converted RGB image to grayscale...")
    print("Image {} saved at {}.".format(img_filename, dt_string))
    
    # Clear and prepare for next frame
    rawCapture.truncate(0)
    
    if img_int > numImagesRequired or key == ord("q"):
        break

print("Program ended.")
cv2.destroyAllWindows()
