import cv2
import numpy as np

# Extracts camera pose from Aruco board pose
# If the Aruco board origin is equal to the air table origin, then the resulting camera pose is relative to the table,
# Reference: https://aliyasineser.medium.com/calculation-relative-positions-of-aruco-markers-eee9cc4036e3
# Parameters:
# rCamToMarker --> rotation vector of board relative to camera (3x1)
# tCamToMarker --> translation vector of board relative to camera
# Returns:
# rMarkerToCam --> camera "attitude" relative to board (3x1)
# tMarkerToCam --> camera "position" relative to board
def cameraPoseFromMarkerPose(rCamToMarker, tCamToMarker):

    rCamToMarkerMatrix = cv2.Rodrigues(rCamToMarker)[0]     # convert rotation vector to matrix
    rMarkerToCamMatrix = np.linalg.inv(rCamToMarkerMatrix)  # inverse of rotation matrix is rotation of camera relative to board
    rMarkerToCam = cv2.Rodrigues(rMarkerToCamMatrix)[0]     # camera rotation as a vector
    tMarkerToCam = np.dot(rMarkerToCamMatrix, np.matrix(-1 * tCamToMarker))     # camera position relative to marker/board
    return rMarkerToCam, tMarkerToCam

