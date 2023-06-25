# Capture a single image from the webcam at a set interval and save for calibration
# Useful for capturing a relatively small number of still frames (such as for capturing calibration images)

import cv2
from datetime import datetime

loc = '../images/calib_images_1_5in'  # Default location to store calibration images

def capture(loc):
    webcam = cv2.VideoCapture(0)    # set up webcam
    img_int = 0                     # loop variable

    # Capture 100 calibration images (likely will need more for accurate calibration!)
    while img_int < 100:

        # Read and display image
        check, frame = webcam.read()
        cv2.imshow("Capturing", frame)
        cv2.waitKey(500)
        print("Capturing image...")

        # Save image
        img_filename = '/capture_' + str(img_int) + '.jpg'
        now = datetime.now()
        dt_string = now.strftime("%H:%M:%S")
        fileDir = loc + img_filename
        cv2.imwrite(fileDir, img=frame)
        img_int += 1
        print("Image {} saved at {}.".format(img_filename, dt_string))

    # Turn off camera
    print("Turning off camera.")
    webcam.release()
    print("Camera off.")
    print("Program ended.")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        loc = sys.argv[1]  # Specify location of where you would like the images to be saved.
    except:
        pass  # If no argument is passed when running the file, it will default to whatever was set as the default

    capture(loc)
