import numpy as np
import cv2
from cv2 import aruco
from tqdm import tqdm
import pickle as pk
import os
import copy
from camera_calib import load_coefficients

images_dir = "/ps/project/datasets/AirCap_ICCV19/dji_08102020/data/images"

matrix_coefficients, distortion_coefficients = load_coefficients("/is/ps3/nsaini/projects/copenet_real_data/dji_temp_calib.yml")

markerpose_viz_dir = "/ps/project/datasets/AirCap_ICCV19/dji_08102020/data/markerpose_viz"
markerpose_dict = {}

for fl in tqdm(sorted(os.listdir(images_dir))):
    # Capture frame-by-frame
    frame = cv2.imread(os.path.join(images_dir,fl))

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
        
        markerpose_dict[fl] = []

        for i in range(0, len(ids)):  # Iterate in markers
            # Estimate pose of each marker and return the values rvec and tvec---different from camera coefficients
            rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners[i], 0.025, matrix_coefficients,
                                                                           distortion_coefficients)
            (rvec - tvec).any()  # get rid of that nasty numpy value array error
            aruco.drawDetectedMarkers(frame, corners)  # Draw A square around the markers
            aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)  # Draw Axis
            
            markerpose_dict[fl].append({"rvec":rvec,"tvec":tvec})

    # save the resulting frame
    cv2.imwrite(os.path.join(markerpose_viz_dir,fl),frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

pk.dump(markerpose_dict,open("/ps/project/datasets/AirCap_ICCV19/dji_08102020/data/markerposes.pkl","wb"))