import numpy as np
import cv2
from cv2 import aruco
from camera_calib import load_coefficients

cap = cv2.VideoCapture(0)

matrix_coefficients, distortion_coefficients = load_coefficients("webcam_chess_images/calib.yml")

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Change grayscale
    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)  # Use 4x4 dictionary to find markers
    parameters = aruco.DetectorParameters_create()  # Marker detection parameters
    # lists of ids and the corners beloning to each id
    corners, ids, rejected_img_points = aruco.detectMarkers(gray, aruco_dict,
                                                                parameters=parameters,
                                                                cameraMatrix=matrix_coefficients,
                                                                distCoeff=distortion_coefficients)
    if np.all(ids is not None):  # If there are markers found by detector
        for i in range(0, len(ids)):  # Iterate in markers
            # Estimate pose of each marker and return the values rvec and tvec---different from camera coefficients
            rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners[i], 0.025, matrix_coefficients,
                                                                           distortion_coefficients)
            (rvec - tvec).any()  # get rid of that nasty numpy value array error
            aruco.drawDetectedMarkers(frame, corners)  # Draw A square around the markers
            aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)  # Draw Axis

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

