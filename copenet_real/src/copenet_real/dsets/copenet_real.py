import torch
import logging
import os
import os.path as osp
import pickle as pkl
import json
from torch.utils.data import Dataset
import sys
import cv2
import numpy as np
from torchvision import transforms
from ..utils.utils import npPerspProj, resize_with_pad
import copy
from .. import constants as CONSTANTS
import torchgeometry as tgm
import h5py

# remove nose as head
op_map2smpl = np.array([8,12,9,-1,13,10,-1,14,11,-1,19,22,1,-1,-1,-1,5,2,6,3,7,4,-1,-1])
al_map2smpl = np.array([-1,11,8,-1,12,9,-1,13,10,-1,-1,-1,1,-1,-1,-1,5,2,6,3,7,4,-1,-1])
dlc_map2smpl = np.array([-1,3,2,-1,4,1,-1,5,0,-1,-1,-1,-1,-1,-1,-1,9,8,10,7,11,6,-1,-1])

def get_copenet_real_traintest(datapath="/ps/project/datasets/AirCap_ICCV19/copenet_data",train_range=range(0,7000),test_range=range(8000,15000),shuffle_cams=False,first_cam=0,kp_agrmnt_threshold=100):
    train_dset = copenet_real(datapath,train_range,shuffle_cams,first_cam,kp_agrmnt_threshold)
    test_dset = copenet_real(datapath,test_range,shuffle_cams,first_cam,kp_agrmnt_threshold)
    return train_dset, test_dset

class copenet_real(Dataset):
    def __init__(self,datapath,drange:range,shuffle_cams=False,first_cam=0,kp_agrmnt_threshold=100):
        super().__init__()

        if osp.exists(datapath):
            print("loading copenet real data...")
            
            db_im1 = [osp.join(datapath,"machine_1","images") + "/" + "{:06d}.jpg".format(i) for i in drange]
            db_im2 = [osp.join(datapath,"machine_2","images") + "/" + "{:06d}.jpg".format(i) for i in drange]

            opose_m1 = pkl.load(open(osp.join(datapath,"machine_1","openpose_res.pkl"),"rb"))
            opose_m2 = pkl.load(open(osp.join(datapath,"machine_2","openpose_res.pkl"),"rb"))
            apose_m1 = json.load(open(osp.join(datapath,"machine_1","alphapose_res.json"),"r"))
            apose_m2 = json.load(open(osp.join(datapath,"machine_2","alphapose_res.json"),"r"))
            
            if drange[0] == 0:
                pass
            elif drange[0] == 8000:
                # dlc_file_m1 = osp.join(datapath,"machine_1","yt-1-1DLC_resnet_101_fineMar5shuffle1_100000_filtered.h5")
                # dlc_file_m2 = osp.join(datapath,"machine_2","yt-2-1DLC_resnet_101_fineMar5shuffle1_100000_filtered.h5")

                # with h5py.File(dlc_file_m1,"r") as fl:
                #     dlc_m1 = np.array([fl["df_with_missing/table"][i][1].reshape(-1,3) for i in range(7000)])
                # with h5py.File(dlc_file_m2,"r") as fl:
                #     dlc_m2 = np.array([fl["df_with_missing/table"][i][1].reshape(-1,3) for i in range(7000)])
                # self.raw_dlc0 = dlc_m1
                # self.raw_dlc1 = dlc_m2
                # self.mapped_dlc0 = self.raw_dlc0[:,dlc_map2smpl]
                # self.mapped_dlc1 = self.raw_dlc1[:,dlc_map2smpl]
                # self.mapped_dlc0[:,dlc_map2smpl==-1,:] = 0
                # self.mapped_dlc1[:,dlc_map2smpl==-1,:] = 0
                pass
            
            self.raw_apose0 = apose_m1
            self.raw_apose1 = apose_m2
            

            opose = np.zeros([2,len(drange),24,3])
            apose = np.zeros([2,len(drange),24,3])
            
            count = 0
            for i in drange:
                try:
                    opose[0,count] = opose_m1["{:06d}".format(i)]["pose"][0,op_map2smpl]
                    opose[0,count][op_map2smpl==-1,:] = 0
                except:
                    pass
                try:
                    apose[0,count] = np.reshape(apose_m1["{:06d}".format(i)]["people"][0]["pose_keypoints_2d"],(18,3))[al_map2smpl]
                    apose[0,count][al_map2smpl==-1,:] = 0
                except:
                    pass
                count += 1

            count = 0
            for i in drange:
                try:
                    opose[1,count] = opose_m2["{:06d}".format(i)]["pose"][0,op_map2smpl]
                    opose[1,count][op_map2smpl==-1,:] = 0
                except:
                    pass
                try:
                    apose[1,count] = np.reshape(apose_m2["{:06d}".format(i)]["people"][0]["pose_keypoints_2d"],(18,3))[al_map2smpl]
                    apose[1,count][al_map2smpl==-1,:] = 0
                except:
                    pass
                count += 1

            self.db = {}
            self.db["im0"] = db_im1
            self.db["im1"] = db_im2

            opose = np.reshape(opose,[-1,3])
            apose = np.reshape(apose,[-1,3])

            self.opose_smpl_fmt = np.reshape(opose,[2,-1,24,3])
            self.apose_smpl_fmt = np.reshape(apose,[2,-1,24,3])

            opose[np.sqrt((opose[:,0]-apose[:,0])**2 + (opose[:,1]-apose[:,1])**2) > kp_agrmnt_threshold,2] = 0
            apose[np.sqrt((opose[:,0]-apose[:,0])**2 + (opose[:,1]-apose[:,1])**2) > kp_agrmnt_threshold,2] = 0

            self.opose = np.reshape(opose,[2,-1,24,3])
            self.apose = np.reshape(apose,[2,-1,24,3])

            cv_file = cv2.FileStorage(osp.join(datapath,"machine_1","camera_calib.yml"), cv2.FILE_STORAGE_READ)
            self.intr0 = cv_file.getNode("K").mat()
            cv_file.release()
            cv_file = cv2.FileStorage(osp.join(datapath,"machine_2","camera_calib.yml"), cv2.FILE_STORAGE_READ)
            self.intr1 = cv_file.getNode("K").mat()
            cv_file.release()

            
            pose0 = pkl.load(open(osp.join(datapath,"machine_1","markerposes_corrected_all.pkl"),"rb"))
            pose1 = pkl.load(open(osp.join(datapath,"machine_2","markerposes_corrected_all.pkl"),"rb"))
            rvecs = []
            k0 = sorted(pose0.keys())
            for i in range(len(pose1)):
                try:
                    rvecs.append(pose0[k0[i]]["0"]["rvec"])
                except:
                    rvecs.append(np.zeros(3))
            rvecs = np.array(rvecs)
            self.extr0 = tgm.angle_axis_to_rotation_matrix(torch.from_numpy(rvecs).float())

            rvecs = []
            k1 = sorted(pose1.keys())
            for i in range(len(pose1)):
                try:
                    rvecs.append(pose1[k0[i]]["0"]["rvec"])
                except:
                    rvecs.append(np.zeros(3))
            rvecs = np.array(rvecs)
            self.extr1 = tgm.angle_axis_to_rotation_matrix(torch.from_numpy(rvecs).float())
            for i in range(len(pose1)):
                self.extr0[i,:3,3] = torch.from_numpy(pose0["{:06d}".format(i)]["0"]["tvec"]).float()  
                self.extr1[i,:3,3] = torch.from_numpy(pose1["{:06d}".format(i)]["0"]["tvec"]).float()
                

            self.num_cams = 2
            self.shuffle_cams = shuffle_cams
            if shuffle_cams:
                self.first_cam = -1
            else:
                self.first_cam = first_cam
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

        else:
            sys.exit("invalid datapath !!")


    def __len__(self):
        return len(self.db["im0"])


    def __getitem__(self, idx):


        im = {}
        bb = {}
        intr = {}
        extr = {}
        crop_info = {}
        border_buffer = 50

        intr["0"] = torch.from_numpy(self.intr0).float()
        intr["1"] = torch.from_numpy(self.intr1).float()
        extr["0"] = self.extr0[idx]
        extr["1"] = self.extr1[idx]

        for i in range(self.num_cams):
            img = cv2.imread(self.db["im"+str(i)][idx])[:,:,::-1]/255.

            # xmin = np.min([np.min(self.apose[i,idx,self.apose[i,idx,:,0]!=0,0]),np.min(self.opose[i,idx,self.opose[i,idx,:,0]!=0,0])])
            # xmin = int(xmin-border_buffer) if int(xmin-border_buffer) > 0 else 0
            # ymin = np.min([np.min(self.apose[i,idx,self.apose[i,idx,:,1]!=0,1]),np.min(self.opose[i,idx,self.opose[i,idx,:,1]!=0,1])])
            # ymin = int(ymin-border_buffer) if int(ymin-border_buffer) > 0 else 0
            # xmax = np.max([np.max(self.apose[i,idx,self.apose[i,idx,:,0]!=0,0]),np.max(self.opose[i,idx,self.opose[i,idx,:,0]!=0,0])])
            # xmax = int(xmax+border_buffer) if int(xmax+border_buffer) < 1920 else 1920
            # ymax = np.max([np.max(self.apose[i,idx,self.apose[i,idx,:,1]!=0,1]),np.max(self.opose[i,idx,self.opose[i,idx,:,1]!=0,1])])
            # ymax = int(ymax+border_buffer) if int(ymax+border_buffer) < 1080 else 1080
            x = self.opose[i,idx,self.opose[i,idx,:,2]!=0,0]
            if x.size == 0:
                x = np.array([0])
            y = self.opose[i,idx,self.opose[i,idx,:,2]!=0,1]
            if y.size == 0:
                y = np.array([0])
            
            xmin = np.min(x)
            xmin = int(xmin-border_buffer) if int(xmin-border_buffer) > 0 else 0
            ymin = np.min(y)
            ymin = int(ymin-border_buffer) if int(ymin-border_buffer) > 0 else 0
            xmax = np.max(x)
            xmax = int(xmax+border_buffer) if int(xmax+border_buffer) < 1920 else 1920
            ymax = np.max(y)
            ymax = int(ymax+border_buffer) if int(ymax+border_buffer) < 1080 else 1080

            im[str(i)] = img[ymin:ymax,xmin:xmax,:]
            bb[str(i)] = (torch.tensor([(xmin+xmax)/2,(ymin+ymax)/2]).float()/intr[str(i)][:2,2] - 1).float()
            crop_info[str(i)] = torch.tensor([[ymin , xmin],[ymax , xmax]]).int()

        s = {}
        pad = {}
        for i in range(self.num_cams):
            try:
                im[str(i)],s[str(i)],pad[str(i)] = resize_with_pad(im[str(i)],size=224)
            except:
                import ipdb; ipdb.set_trace()
                print('!!!!!!!!!!!!!!'+self.db['im'+str(i)]+'!!!!!!!!!!!!!!!!!!')
                print(im[str(i)].shape)
        
        

        gt_joints_2d = {}
        gt_joints_2d_crop = {}
        for i in range(self.num_cams):
            gt_joints_2d[str(i)] = torch.from_numpy(np.concatenate([self.opose[i:i+1,idx],self.apose[i:i+1,idx]])).float()

            gt_joints_2d_crop[str(i)] = copy.deepcopy(gt_joints_2d[str(i)])
            gt_joints_2d_crop[str(i)][0,:,:2] = s[str(i)]*(gt_joints_2d[str(i)][0,:,:2] - (bb[str(i)] + 1)*intr[str(i)][:2,2])
            gt_joints_2d_crop[str(i)][1,:,:2] = s[str(i)]*(gt_joints_2d[str(i)][1,:,:2] - (bb[str(i)] + 1)*intr[str(i)][:2,2])
            
            im[str(i)] = self.normalize(torch.from_numpy(im[str(i)].transpose(2,0,1)).float())

        for i in range(self.num_cams):
            bb[str(i)] = torch.cat([bb[str(i)],torch.tensor(s[str(i)]).float().view(1)])

        if self.shuffle_cams == True:
            cam1 = np.random.randint(2)
        else:
            cam1 = self.first_cam
        cam2 = int(1 - cam1)
        cam1 = str(cam1)
        cam2 = str(cam2)

        

        return {'im0_path':self.db['im'+cam1][idx],'im1_path':self.db['im'+cam2][idx],
        'im0':im[cam1],'im1':im[cam2],
        'intr0':intr[cam1],'intr1':intr[cam2],
        'extr0':extr[cam1],'extr1':extr[cam2],
        'bb0': bb[cam1], 'bb1': bb[cam2],
        'crop_info0':crop_info[cam1], 'crop_info1':crop_info[cam2],
        'smplbetas':np.nan,'smpltrans_rel0': np.nan, 'smpltrans_rel1': np.nan,
        'smplpose_rotmat': np.nan, 'img_size':torch.tensor(CONSTANTS.IMG_SIZE).float(),
        'smplorient_rel0': np.nan,'smplorient_rel1': np.nan,
        'smpl_vertices_rel0':np.nan,'smpl_vertices_rel1':np.nan,
        'smpl_joints_rel0':np.nan,'smpl_joints_rel1':np.nan,
        'smpl_joints_2d0':gt_joints_2d[cam1],'smpl_joints_2d1':gt_joints_2d[cam2],
        'smpl_joints_2d_crop0':gt_joints_2d_crop[cam1],'smpl_joints_2d_crop1':gt_joints_2d_crop[cam2],
        'smpl_vertices': np.nan, 'smpl_joints': np.nan,
        'smpl_gender':"male","cam":int(cam1)}
        
    def get_j2d_only(self,idx):
        gt_joints_2d = {}
        for i in range(self.num_cams):
            gt_joints_2d[str(i)] = torch.from_numpy(np.concatenate([self.opose[i:i+1,idx],self.apose[i:i+1,idx]])).float()

        return {'im0_path':self.db["im0"][idx],'im1_path':self.db["im1"][idx],
                'smpl_joints_2d0':gt_joints_2d['0'],'smpl_joints_2d1':gt_joints_2d['1']}