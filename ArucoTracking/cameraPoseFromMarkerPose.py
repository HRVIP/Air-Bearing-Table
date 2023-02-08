import cv2
import numpy as np

### Extracts camera pose from a single marker pose
# Parameters:
# rCamToMarker --> rotation vector of marker relative to camera (3x1)
# tCamToMarker --> translation vector of marker relative to camera
# rTableToMarker --> known rotation of marker relative to table (3x1)
# tTableToMarker --> known position of marker relative to table
# Returns:
# rTableToCam --> camera "attitude" relative to table (3x1)
# tTableToCam --> camera "position" relative to table
def cameraPoseFromMarkerPose(rCamToMarker, tCamToMarker, rTableToMarker, tTableToMarker):

    #### Calculate camera position
    tMarkerToCam = -1 * tCamToMarker            # camera position relative to marker
    tTableToCam = tTableToMarker + tMarkerToCam # marker position relative to table

    ### Calculate camera rotation

    # Convert everything to rotation matrices (easier to work with, unfortunately)
    rTableToMarkerMatrix = cv2.Rodrigues(rTableToMarker)[0]     # marker rotation relative to table as matrix
    rCamToMarkerMatrix = cv2.Rodrigues(rCamToMarker)[0]         # marker rotation relative to camera as matrix
    rMarkerToCamMatrix = np.linalg.inv(rCamToMarkerMatrix)      # camera rotation relative to marker as matrix
    rTableToCamMatrix = np.matmul(rTableToMarkerMatrix, rMarkerToCamMatrix) # camera rotation relative to table as matrix
    rTableToCam = cv2.Rodrigues(rTableToCamMatrix)[0]   # camera rotation relative to table as vector

    return rTableToCam, tTableToCam

# test
rCamToMarker = np.array([0, 0, 5 * np.pi / 4])
rTableToMarker = np.array([0, 0, np.pi])
tCamToMarker = np.array([-1, 2, 0])
tTableToMarker = np.array([1, 3, 0])
r, t = cameraPoseFromMarkerPose(rCamToMarker, tCamToMarker, rTableToMarker, tTableToMarker)
print(r)
print(t)
