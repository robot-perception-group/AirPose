#%% Imports
import numpy as np
import cv2
from cv2 import aruco
from tqdm import tqdm
import pickle as pk
import os
import os.path as osp
import copy
import sys
from camera_calib import load_coefficients, save_coefficients, load_coefficients
import argparse
from camera_calib import calibrate

machine_root_dir = "/ps/project/datasets/AirCap_ICCV19/cvpr_dji_24Oct20/machine_1"

#%% Extract calib images

cam_calib_vid = osp.join(machine_root_dir,"videos","calib.MP4") 
calib_images_dir = osp.join(machine_root_dir,"calib_images")
calib_file = osp.join(machine_root_dir,"camera_calib.yml")
skip_frames = 50
i = 25

# make calibration images directory
os.mkdir(calib_images_dir)

# extract calib images
cap = cv2.VideoCapture(cam_calib_vid)

while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == False:
        break
    if i%skip_frames == 0:
        cv2.imwrite(os.path.join(calib_images_dir,"{:06d}.jpg".format(i)),frame)
        print("frame: {}".format(i))
    i += 1

print("#################### camera calibration images extraction done #####################")

#%% Calibrate the camera
calib_images_dir = osp.join(machine_root_dir,"calib_images")
calib_file = osp.join(machine_root_dir,"camera_calib.yml")

ret, mtx, dist, rvecs, tvecs = calibrate(calib_images_dir,
                                        prefix = "",
                                        image_format="jpg",
                                        square_size=0.025,
                                        width=9,
                                        height=6)
save_coefficients(mtx, dist, calib_file)


#%% Extract images
vid_file_list = ["DJI_0091","DJI_0092","DJI_0093","DJI_0094"]

for vid_file_name in tqdm(vid_file_list):
    vid_path = osp.join(machine_root_dir,"videos",vid_file_name+".MP4")
    images_dir = os.path.join(machine_root_dir,"images_"+vid_file_name)

    os.mkdir(images_dir)

    cap = cv2.VideoCapture(vid_path)
    itr=0
    while(cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == False:
            break
        cv2.imwrite(os.path.join(images_dir,"{:06d}.jpg".format(itr)),frame)
        print("frame: {}".format(i))
        itr += 1

#%% Run aruco detection
import glob
machine_root_dir = "/ps/project/datasets/AirCap_ICCV19/cvpr_dji_24Oct20_downsample/machine_2"

imdirs_list = sorted(glob.glob(osp.join(machine_root_dir,"images_*")))

imdirs_list = [osp.basename(x) for x in imdirs_list]

calib_file = osp.join(machine_root_dir,"camera_calib.yml")

for images_dir in tqdm(imdirs_list):
    markerpose_viz_dir = osp.join(machine_root_dir,"markerpose_viz_"+images_dir)
    # markerpose_viz_dir = None
    markerpose_file = os.path.join(machine_root_dir, "markerposes_"+images_dir+".pkl")
    if markerpose_viz_dir is not None:
        os.mkdir(markerpose_viz_dir)

    matrix_coefficients, distortion_coefficients = load_coefficients(calib_file)

    markerpose_dict = {}

    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)  # Use 4x4 dictionary to find markers
    parameters = aruco.DetectorParameters_create()  # Marker detection parameters
    parameters.minMarkerPerimeterRate = 0.01
    parameters.polygonalApproxAccuracyRate = 0.1
    # parameters.adaptiveThreshConstant = 1
    parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    # parameters.maxErroneousBitsInBorderRate = 0.9

    for img in tqdm(sorted(os.listdir(osp.join(machine_root_dir, images_dir)))):
        frame = cv2.imread(osp.join(machine_root_dir,images_dir,img))    
        # operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Change grayscale
        
        # lists of ids and the corners beloning to each id
        corners, ids, rejected_img_points = aruco.detectMarkers(gray, aruco_dict,
                                                                    parameters=parameters,
                                                                    cameraMatrix=matrix_coefficients,
                                                                    distCoeff=distortion_coefficients)
        
        if np.all(ids is not None):  # If there are markers found by detector
            # print("marker detected: {}".format(img))

            markerpose_dict[img.split(".")[0]] = {}
            
            for i in range(0, len(ids)):  # Iterate in markers
                # Estimate pose of each marker and return the values rvec and tvec---different from camera coefficients
                rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners[i], 0.555, matrix_coefficients,
                                                                            distortion_coefficients)
                (rvec - tvec).any()  # get rid of that nasty numpy value array error
                aruco.drawDetectedMarkers(frame, corners)  # Draw A square around the markers
                aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.5)  # Draw Axis
                
                markerpose_dict[img.split(".")[0]][str(ids[i][0])] = {"rvec":rvec,"tvec":tvec}

        if markerpose_viz_dir is not None:
            # save the resulting frame
            cv2.imwrite(os.path.join(markerpose_viz_dir,img),frame)
        # cv2.imshow("im",frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    pk.dump(markerpose_dict,open(markerpose_file,"wb"))

#%% Aruco detections plotting etc
import numpy as np
import glob
from tqdm import tqdm
import pickle as pkl
import os.path as osp
from plotly.subplots import make_subplots
import plotly.graph_objects as go

data_root_dir = "/ps/project/datasets/AirCap_ICCV19/cvpr_dji_24Oct20_downsample"

pkls = sorted(glob.glob(osp.join(data_root_dir,"machine_2","markerpose*.pkl")))

colors = ["red","green","blue"]

figs_abs = []
figs_rel = []
for f in tqdm(pkls):
    markers = pkl.load(open(f,"rb"))
    max_len = int(sorted(markers.keys())[-1]) + 1
    rvecs = np.zeros([2,max_len,3])
    tvecs = np.zeros([2,max_len,3])
    tstamps = [[],[]]
    for markerid in markers:
        for key in markers[markerid].keys():
            try:
                rvecs[int(key),int(markerid)] = markers[markerid][key]["rvec"]
                tvecs[int(key),int(markerid)] = markers[markerid][key]["tvec"]
                tstamps[int(key)].append(int(markerid))
            except:
                pass

    # import ipdb; ipdb.set_trace()
    
    fig_abs = make_subplots(rows=2,cols=2,column_titles=["Rotation","Translation"],row_titles=["Marker1","Marker2"])
    for idx in range(3):
        fig_abs.add_trace(go.Scatter(x=tstamps[0],y=rvecs[0,tstamps[0],idx]*180/np.pi,marker_color=colors[idx]),row=1,col=1)
        fig_abs.add_trace(go.Scatter(x=tstamps[0],y=tvecs[0,tstamps[0],idx]*180/np.pi,marker_color=colors[idx]),row=1,col=2)
        fig_abs.add_trace(go.Scatter(x=tstamps[1],y=rvecs[1,tstamps[1],idx]*180/np.pi,marker_color=colors[idx]),row=2,col=1)
        fig_abs.add_trace(go.Scatter(x=tstamps[1],y=tvecs[1,tstamps[1],idx]*180/np.pi,marker_color=colors[idx]),row=2,col=2)
    fig_abs.update_xaxes(range = [0,max_len])
    fig_abs.write_html(f.split(".")[0]+"_abs.html")
    figs_abs.append(fig_abs)

    fig_rel = go.Figure()
    common_tstamps = np.intersect1d(tstamps[0],tstamps[1])
    if len(common_tstamps) != 0:
        tvecs_rel = tvecs[1,common_tstamps] - tvecs[0,common_tstamps]
    
        for idx in range(3):
            fig_rel.add_trace(go.Scatter(x=common_tstamps,y=tvecs_rel[:,idx]))
    fig_rel.write_html(f.split(".")[0]+"_rel.html")
    figs_rel.append(fig_rel)
    


#%% Downsample data
from tqdm import tqdm
import os.path as osp
import os
import cv2

full_data_machine_dir = "/ps/project/datasets/AirCap_ICCV19/cvpr_dji_24Oct20/machine_2"
donwsample_data_machine_dir = "/ps/project/datasets/AirCap_ICCV19/cvpr_dji_24Oct20_downsample/machine_2"
downsample_factor = 2

# os.mkdir(donwsample_data_root)
imdirs = [x for x in os.listdir(full_data_machine_dir) if "DJI" in x]

for imdir in imdirs:
    os.mkdir(osp.join(donwsample_data_machine_dir,imdir))

    for img in tqdm(os.listdir(osp.join(full_data_machine_dir,imdir))):
        im = cv2.imread(osp.join(full_data_machine_dir,imdir,img))
        cv2.imwrite(osp.join(donwsample_data_machine_dir,imdir,img),im[::downsample_factor,::downsample_factor,:])


#%% run openpose and alphapose
