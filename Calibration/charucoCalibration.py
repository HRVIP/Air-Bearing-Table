# Camera calibration using ChAruco.
# Sources:
# https://mecaruco2.readthedocs.io/en/latest/notebooks_rst/Aruco/sandbox/ludovic/aruco_calibration_rotation.html

# Import packages
import numpy as np
import cv2
from cv2 import aruco

arucoDict = cv2.aruco.getPredefinedDictionary(0)        # small dictionary of 4x4 markers

# Set up ChAruco board
def generateBoard():
    boardSize = np.array([5, 7])                            # board width (x) and height (y)
    squareLength = 0.04                                     # square length (m)
    markerLength = 0.02                                     # marker length (m)
    board = aruco.CharucoBoard(boardSize, squareLength, markerLength, arucoDict)
    imboard = board.generateImage(np.array([600, 500]))
    cv2.imwrite('charucoboard.jpg', img=imboard)
    cv2.imshow('board', imboard)
    while not(cv2.waitKey(1) & 0xFF == ord('q')):               # wait for user input
        pass
    cv2.destroyAllWindows()

# Read markers on ChAruco images
def readChessboards(images):
    allCorners = []
    allIds = []
    decimator = 0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.00001)   # criteria for corner detection

    for im in images:
        print ("Processing image {}".format(im))
        frame = cv2.imread(im)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, arucoDict)

        # If corners are detected, detect subpixels
        if len(corners) > 0:
            for corner in corners:
                cv2.cornerSubPix(gray, corner,
                                 winSize = (3,3),
                                 zeroZone = (-1, -1),
                                 criteria = criteria)
                res2 = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
                if res2[1] is not None and res2[2] is not None and len(res2[1]) > 3 and decimater%1==0:
                    allCorners.append(res2[1])
                    allIds.append(res2[2])

        decimator += 1

    imsize = gray.shape
    return allCorners, allIds, imsize

# Calibrate camera from detected markers
def calibrateCamera(allCorners, allIds, imsize):
    cameraMatrixInit = np.array([[ 1000., 0., imsize[0]/2.],
                                 [0., 1000., imsize[1]/2.],
                                 [0., 0., 1.]])
    distCoeffsInit = np.zeros((5,1))
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_ROTATIONAL_MODEL + cv2.CALIB_FIX_ASPECT_RATIO)
    (ret, cameraMatrix, distortionCoeffs0,
     rotationVectors, translationVectors,
     stdDeviationsIntrinsics, stDeviationsExtrinsics,
     perViewErrors) = cv2.aruco.calibrateCameraCharucoExtended(
        charucoCorners=allCorners,
        charucoIds=allIds,
        board=board,
        imageSize=imsize,
        cameraMatrix=cameraMatrixInit,
        distCoeffs=distCoeffsInit,
        flag=flags,
        criteria=(cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9))
    return ret, cameraMatrix, distortionCoeffs0, rotationVectors, translationVectors

