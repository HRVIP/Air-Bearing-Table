import numpy as np

# Generates array containing corners of all markers for 3D board
# w --> horizontal dimension of marker (m)
# h --> vertical dimension of marker (m)
# dx --> spacing between markers horizontally on sides A and C of table (m)
# dy --> spacing between markers horizontally on sides B and D of table (m)
# dx0 --> spacing between edge of table and first marker on sides A and C
# dy0 --> spacing between edge of table and first marker on sides B and D
# numX --> number of markers on side A or C of table
# numY --> number of markers on side B or D of table
# tableX --> length of sides A and C of table
# tableY --> length of sides B and D of table
def boardGeometry(w, h, dx, dy, dx0, dy0, numX, numY, tableX, tableY):

    # Create big array to hold all corners
    totalMarkers = 2*numX + 2*numY                          # total number of markers
    allMarkerCorners = np.zeros([totalMarkers, 4, 3])       # 4 corners per marker; 3 coordinates per corner
    allMarkerIDs = np.zeros([totalMarkers, 1], dtype=np.int32) # keep track of marker IDs (this will not be in numerical order!)
    currentMarkerID = 0                                     # keeps track of where we are in marker corners array

    # Go through each side and generate marker corners
    for sideSet in range(0, 2):

        # Define offsets (are we on the left or right side of table, or "far" or "near" side?)
        xOffsetForBD = (1 - sideSet) * tableX
        yOffsetForAC = sideSet * tableY

        # Marker corners for sides A and C (-Y/+Y sides but that gets confusing)
        for m in range(0, numX):
            topLeft = np.array([dx0 + (w * (m+1)) + (dx * m), yOffsetForAC, h])
            topRight = np.array([dx0 + (w * m) + (dx * m), yOffsetForAC, h])
            bottomRight = np.array([dx0 + (w * m) + (dx * m), yOffsetForAC, 0])
            bottomLeft = np.array([dx0 + (w * (m+1)) + (dx * m), yOffsetForAC, 0])
            markerCorners = np.array([topLeft, topRight, bottomRight, bottomLeft])
            allMarkerCorners[currentMarkerID, :, :] = markerCorners
            allMarkerIDs[currentMarkerID] = currentMarkerID
            currentMarkerID = currentMarkerID + 1

        # Marker corners for sides B and D (+X/-X sides but that gets confusing)
        for m in range(0, numY):
            topLeft = np.array([xOffsetForBD, dy0 + (w * (m+1)) + (dy * m), h])
            topRight = np.array([xOffsetForBD, dy0 + (w * m) + (dy * m), h])
            bottomRight = np.array([xOffsetForBD, dy0 + (w * m) + (dy * m), 0])
            bottomLeft = np.array([xOffsetForBD, dy0 + (w * (m+1)) + (dy * m), 0])
            markerCorners = np.array([topLeft, topRight, bottomRight, bottomLeft])
            allMarkerCorners[currentMarkerID, :, :] = markerCorners
            allMarkerIDs[currentMarkerID] = currentMarkerID
            currentMarkerID = currentMarkerID + 1

    return allMarkerCorners, allMarkerIDs

# Test case: 2m x 1m table
# 4 markers on long side, 2 on short side
# 100mm markers
# 200mm between markers on long side
# 100mm between markers on short side
# 500mm to first marker on long side
# 350mm to first marker on short side
[board, ids] = boardGeometry(0.1, 0.1, 0.2, 0.1, 0.5, 0.35, 4, 2, 2, 1)
print(board)
print(ids)
