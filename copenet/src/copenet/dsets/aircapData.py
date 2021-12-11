import sys
import os
import torch
import logging
import pickle as pk
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import numpy as np
import sys
from ..smplx.smplx import lbs
import torchgeometry as tgm
from ..utils.utils import npPerspProj, resize_with_pad, get_weak_persp_cam_full_img_input, get_weak_persp_cam_full_img_gt, transform_smpl
import torch
from .. import constants as CONSTANTS
import copy

sys.path.append("/is/ps3/nsaini/AirCap_WS/aerial-pose-tracker/aerial_pose")

from camera_and_NN import processCamsNNs


def get_copenet_real_traintest(datapath="/ps/project/datasets/AirCap_ICCV19/ICCV_28Feb_rerun2/",train_range=range(0,4000),test_range=range(4001,4615),shuffle_cams=False,first_cam=0):
    train_dset = aircapData_crop(train_range,datapath)
    test_dset = aircapData_crop(test_range,datapath)
    return train_dset, test_dset


class aircapData_crop(Dataset):
    def __init__(self, drange:range, datapath="/ps/project/datasets/AirCap_ICCV19/ICCV_28Feb_rerun2/", rottrans=False):
        super().__init__()
        
        if os.path.exists(datapath):
            self.num_cams,self.n_NNs,self.camsdata,self.NNs,self.tstamps2cam = processCamsNNs(datapath,
                                                    ["alphapose"],[0,1])
        else:
            sys.exit('database not found!!!!, create database')
        
        tstamps = np.load(os.path.join(datapath,"xsens_tstamped.npz"))["tstamps"]
        # tstamps = [x[0] for x in self.tstamps2cam if x[1]==0]
        # self.xsens_gt = np.load(os.path.join(datapath,"xsens_tstamped.npz"))["syncpose"]
        self.personpose0 = pk.load(open(os.path.join(datapath,"data/machine_1/personpose_raw.pkl"),"rb"))
        self.personpose1 = pk.load(open(os.path.join(datapath,"data/machine_2/personpose_raw.pkl"),"rb"))
        
        # if person is in both the frames
        self.tstamps = []
        for tstamp in tstamps:
            tstamp0 = self.camsdata[0].get_closest_time_stamp(tstamp)
            tstamp1 = self.camsdata[1].get_closest_time_stamp(tstamp)
            j2d0 = self.NNs[0][0].get_2d_joints_and_probs(tstamp0,self.camsdata[0].roi[tstamp0])
            j2d1 = self.NNs[0][1].get_2d_joints_and_probs(tstamp1,self.camsdata[1].roi[tstamp1])
            j2d0 = j2d0[0][j2d0[1]!=0]
            j2d1 = j2d1[0][j2d1[1]!=0]
            if len(j2d0) != 0 and len(j2d1) != 0:
                self.tstamps.append(tstamp)

        # split the dataset for train test`
        self.tstamps = np.array(self.tstamps)[drange]

        self.db_len = len(self.tstamps)

        
        # self.rottrans = rottrans
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])                 # for pre trained resnet
        self.totensor = transforms.ToTensor()                                 # converting to tensor

        self.coordinate_tfn = torch.tensor([[1,0,0],[0,0,-1],[0,1,0]]).float()

        

    def __len__(self):
        return self.db_len

    def __getitem__(self,idx):

        # tstamp0 = self.camsdata[0].timestamps[idx]
        # tstamp1 = self.camsdata[1].get_closest_time_stamp(tstamp0)
        tstamp = self.tstamps[idx]
        tstamp0 = self.camsdata[0].get_closest_time_stamp(tstamp)
        tstamp1 = self.camsdata[1].get_closest_time_stamp(tstamp)
        
        # get both images
        full_img0 = self.camsdata[0].get_frame(tstamp0)[:,:,::-1]/255.
        full_img1 = self.camsdata[1].get_frame(tstamp1)[:,:,::-1]/255.
        
        # get 2d joints
        j2d0 = self.NNs[0][0].get_2d_joints_and_probs(tstamp0,self.camsdata[0].roi[tstamp0])
        j2d1 = self.NNs[0][1].get_2d_joints_and_probs(tstamp1,self.camsdata[1].roi[tstamp1])
        j2d0 = np.concatenate([j2d0[0],np.reshape(j2d0[1],[-1,1])],1)
        j2d1 = np.concatenate([j2d1[0],np.reshape(j2d1[1],[-1,1])],1)
        j2d0 = j2d0[j2d0[:,2]!=0,:]
        j2d1 = j2d1[j2d1[:,2]!=0,:]

        # get cropping region based on 2d images
        bb0 = [np.min(j2d0[:,:2],axis=0).astype("int")-50,np.max(j2d0[:,:2],axis=0).astype("int")+50]
        bb1 = [np.min(j2d1[:,:2],axis=0).astype("int")-50,np.max(j2d1[:,:2],axis=0).astype("int")+50]
        crop_info0 = torch.tensor([[bb0[0][1],bb0[0][0]],[bb0[1][1],bb0[1][0]]]).int()
        crop_info1 = torch.tensor([[bb1[0][1],bb1[0][0]],[bb1[1][1],bb1[1][0]]]).int()
        
        # crop images based on cropping region
        img0_crop = full_img0[bb0[0][1]:bb0[1][1],bb0[0][0]:bb0[1][0]]
        img1_crop = full_img1[bb1[0][1]:bb1[1][1],bb1[0][0]:bb1[1][0]]
        
        # preprocess input images
        im0_crop_resized,s0,pad0 = resize_with_pad(img0_crop)
        im1_crop_resized,s1,pad1 = resize_with_pad(img1_crop)
        im0_crop_in = self.normalize(torch.from_numpy(im0_crop_resized).float().permute(2,0,1))
        im1_crop_in = self.normalize(torch.from_numpy(im1_crop_resized).float().permute(2,0,1))

        # calculate input "s" values
        intr0 = torch.from_numpy(self.camsdata[0].get_intrinsic()).float()
        intr1 = torch.from_numpy(self.camsdata[1].get_intrinsic()).float()
        # import ipdb; ipdb.set_trace()
        
        # get bb center
        bb0 = (torch.from_numpy(bb0[0]+bb0[1]).float()/2)/intr0[:2,2] - 1
        bb1 = (torch.from_numpy(bb1[0]+bb1[1]).float()/2)/intr1[:2,2] - 1
        
        j2d0 = torch.from_numpy(j2d0).unsqueeze(0)
        j2d1 = torch.from_numpy(j2d1).unsqueeze(0)
        

        # position0 = torch.tensor([self.camsdata[0].extrinsics[tstamp0]['position'].x,
        #             self.camsdata[0].extrinsics[tstamp0]['position'].y,
        #             self.camsdata[0].extrinsics[tstamp0]['position'].z]).float()
        # position1 = torch.tensor([self.camsdata[1].extrinsics[tstamp1]['position'].x,
        #             self.camsdata[1].extrinsics[tstamp1]['position'].y,
        #             self.camsdata[1].extrinsics[tstamp1]['position'].z]).float()
        # quat0 = torch.tensor([self.camsdata[0].extrinsics[tstamp0]['orientation'].x,
        #             self.camsdata[0].extrinsics[tstamp0]['orientation'].y,
        #             self.camsdata[0].extrinsics[tstamp0]['orientation'].z,
        #             self.camsdata[0].extrinsics[tstamp0]['orientation'].w])
        # quat1 = torch.tensor([self.camsdata[1].extrinsics[tstamp1]['orientation'].x,
        #             self.camsdata[1].extrinsics[tstamp1]['orientation'].y,
        #             self.camsdata[1].extrinsics[tstamp1]['orientation'].z,
        #             self.camsdata[1].extrinsics[tstamp1]['orientation'].w])
        # extr0 = tgm.angle_axis_to_rotation_matrix(tgm.quaternion_to_angle_axis(quat0).unsqueeze(0))
        # extr0[0,:3,3] = position0
        # extr1 = tgm.angle_axis_to_rotation_matrix(tgm.quaternion_to_angle_axis(quat1).unsqueeze(0))
        # extr1[0,:3,3] = position1
        
        extr0,_ = self.camsdata[0].get_extrinsic_and_cov(tstamp0)
        extr1,_ = self.camsdata[1].get_extrinsic_and_cov(tstamp1)
        extr0 = torch.from_numpy(extr0).float()
        extr1 = torch.from_numpy(extr1).float()

        # smpltrans0 = torch.tensor([self.personpose0[tstamp0]['position'].x,
        #                             self.personpose0[tstamp0]['position'].y,
        #                             self.personpose0[tstamp0]['position'].z,1]).float().unsqueeze(1)
        # smpltrans1 = torch.tensor([self.personpose1[tstamp1]['position'].x,
        #                             self.personpose1[tstamp1]['position'].y,
        #                             self.personpose1[tstamp1]['position'].z,1]).float().unsqueeze(1)
        
        # smpltrans0 = torch.matmul(extr0,smpltrans0).squeeze(1)
        # smpltrans1 = torch.matmul(extr1,smpltrans1).squeeze(1)

        gt_joints_2d_crop0 = copy.deepcopy(j2d0)
        gt_joints_2d_crop0[0,:,:2] = s0*(j2d0[0,:,:2] - (bb0 + 1)*intr0[:2,2])
        gt_joints_2d_crop1 = copy.deepcopy(j2d1)
        gt_joints_2d_crop1[0,:,:2] = s1*(j2d1[0,:,:2] - (bb1 + 1)*intr1[:2,2])
        bb0 = torch.cat([bb0,torch.tensor(s0).unsqueeze(0)],dim=0)
        bb1 = torch.cat([bb1,torch.tensor(s1).unsqueeze(0)],dim=0)

        # smplpose = torch.from_numpy(self.xsens_gt[idx]).type_as(extr0)
        # smplpose_rotmat = lbs.batch_rodrigues(smplpose.reshape(-1,3))
        # smplpose_rotmat[0] = torch.matmul(extr0[:3,:3],smplpose_rotmat[0])


        # return {'im0_path':self.camsdata[0].images[tstamp0],'im1_path':self.camsdata[1].images[tstamp1],
        # 'im0':im0_crop_in,'im1':im1_crop_in,
        # 'intr0':intr0,'intr1':intr1,
        # 'extr0':extr0,'extr1':extr1,
        # 'bb0':bb0, 'bb1':bb1,
        # 'smpltrans_rel0': smpltrans0[:3],
        # 'smpltrans_rel1': smpltrans1[:3],
        # 'smplorient_rel0': smplpose_rotmat[:1],
        # 'smplorient_rel1': smplpose_rotmat[:1],
        # 'smplpose_rotmat':smplpose_rotmat[1:22]}
        return {'im0_path':self.camsdata[0].images[tstamp0],'im1_path':self.camsdata[1].images[tstamp1],
        'im0':im0_crop_in,'im1':im1_crop_in,
        'intr0':intr0,'intr1':intr1,
        'extr0':extr0,'extr1':extr1,
        'bb0':bb0, 'bb1':bb1,
        'crop_info0':crop_info0, 'crop_info1':crop_info1,
        'smplbetas':np.nan,'smpltrans_rel0': np.nan, 'smpltrans_rel1': np.nan,
        'smplpose_rotmat': np.nan, 'img_size':torch.tensor(CONSTANTS.IMG_SIZE).float(),
        'smplorient_rel0': np.nan,'smplorient_rel1': np.nan,
        'smpl_vertices_rel0':np.nan,'smpl_vertices_rel1':np.nan,
        'smpl_joints_rel0':np.nan,'smpl_joints_rel1':np.nan,
        'smpl_joints_2d0':j2d0,'smpl_joints_2d1':j2d1,
        'smpl_joints_2d_crop0':gt_joints_2d_crop0,'smpl_joints_2d_crop1':gt_joints_2d_crop1,
        'smpl_vertices': np.nan, 'smpl_joints': np.nan,
        'smpl_gender':"male","cam":0}

    def get_j2d_only(self,idx):

        tstamp = self.tstamps[idx]
        tstamp0 = self.camsdata[0].get_closest_time_stamp(tstamp)
        tstamp1 = self.camsdata[1].get_closest_time_stamp(tstamp)

        j2d0 = self.NNs[0][0].get_2d_joints_and_probs(tstamp0,self.camsdata[0].roi[tstamp0])
        j2d1 = self.NNs[0][1].get_2d_joints_and_probs(tstamp1,self.camsdata[1].roi[tstamp1])
        j2d0 = torch.from_numpy(j2d0[0][j2d0[1]!=0])
        j2d1 = torch.from_numpy(j2d1[0][j2d1[1]!=0])

        return {'im0_path':self.camsdata[0].images[tstamp0],'im1_path':self.camsdata[1].images[tstamp1],
                'smpl_joints_2d0':j2d0,'smpl_joints_2d1':j2d1}

        # 'smplpose':smplpose,'smplbetas':smplbetas, 'smplorient': smplorient,
        # 'smplpose_rotmat': smplpose_rotmat, 'focal_length':torch.tensor(CONSTANTS.aircap_cam0_FOCAL_LENGTH).float(), 'img_size':torch.tensor(CONSTANTS.IMG_SIZE).float(),
        # 'smplorient_rotmat_wrt_cam0': smplorient_rotmat_wrt_cam['0'],'smplorient_rotmat_wrt_cam1': smplorient_rotmat_wrt_cam['1'],
        # 'smpl_vertices_wrt_cam0':smpl_vertices_wrt_cam['0'],'smpl_vertices_wrt_cam1':smpl_vertices_wrt_cam['1'],
        # 'smpl_joints_wrt_cam0':smpl_joints_wrt_cam['0'],'smpl_joints_wrt_cam1':smpl_joints_wrt_cam['1'],
        # 'wcam0':wcam['0'],'wcam1':wcam['1'],
        # 'wcam_in0':wcam_in['0'],'wcam_in1':wcam_in['1']}

        # # bb0_center = np.array([np.mean(bb0[0]).astype("int"),np.mean(bb0[1]).astype("int")])
        # # bb1_center = np.array([np.mean(bb1[0]).astype("int"),np.mean(bb1[1]).astype("int")])
        

        # # sx0 = bb0_center[1] - full_img0.shape[1]/2
        # # sy0 = bb0_center[0] - full_img0.shape[0]/2
        # # sz0 = 2*intr0[1,1]/(intr0[1,2]*8)
        # # s0 = torch.from_numpy(np.array([sz0,sx0,sy0])).float().to(device).unsqueeze(0)

        # # sx1 = bb1_center[1] - full_img1.shape[1]/2
        # # sy1 = bb1_center[0] - full_img1.shape[0]/2
        # # sz1 = 2*intr1[1,1]/(intr1[1,2]*8)
        # # s1 = torch.from_numpy(np.array([sz1,sx1,sy1])).float().to(device).unsqueeze(0)


        # with open(self.db[idx],'rb') as f:
        #     db = pk.load(f)
        
        # im = {}
        # for i in range(self.num_cams):
        #     im[str(i)] = cv2.imread(db['im'+str(i)])[db['bb'+str(i)][0][1]:db['bb'+str(i)][1][1],db['bb'+str(i)][0][0]:db['bb'+str(i)][1][0],:].astype(np.float32)/255.
        
        # smplpose = torch.from_numpy(db['smplpose'].reshape(63))
        # smplbetas = torch.from_numpy(db['smplshape'].reshape(10))
        # # smpltrans = torch.from_numpy(db['smpltrans'].reshape(3))
        # smplorient = torch.from_numpy(db['smplorient'].reshape(3))

        # extr = {}
        # for i in range(self.num_cams):
        #     extr[str(i)] = torch.from_numpy(db['cam'+str(i)]['extr']).float()
        
        # smpl_wrt_cam = {}
        # for i in range(self.num_cams):
        #     smpl_wrt_cam[str(i)] = transform_smpl(extr[str(i)].unsqueeze(0),
        #                                 torch.from_numpy(db['smpl_vertices_wrt_origin']),
        #                                 torch.from_numpy(db['smpl_joints_wrt_origin']),
        #                                 torch.from_numpy(db['smplorient_rotmat_wrt_origin']))
        # smpl_vertices_wrt_cam = {}
        # smpl_joints_wrt_cam = {}
        # smplorient_rotmat_wrt_cam = {}
        # for i in range(self.num_cams):
        #     smpl_vertices_wrt_cam[str(i)] = smpl_wrt_cam[str(i)][0][0]
        #     smpl_joints_wrt_cam[str(i)] = smpl_wrt_cam[str(i)][1][0]
        #     smplorient_rotmat_wrt_cam[str(i)] = smpl_wrt_cam[str(i)][2]

        # s = {}
        # pad = {}
        # for i in range(self.num_cams):
        #     try:
        #         im[str(i)],s[str(i)],pad[str(i)] = resize_with_pad(im[str(i)],size=224)
        #     except:
        #         print('!!!!!!!!!!!!!!'+db['im'+str(i)]+'!!!!!!!!!!!!!!!!!!')
        #         print(im[str(i)].shape)
        #         print(db['bb'+str(i)])
        
        # for i in range(self.num_cams):
        #     im[str(i)] = self.normalize(torch.from_numpy(im[str(i)].transpose(2,0,1)).float())
        

        # # crop_j2d0 = (db['j2d0'] - db['bb0'][0])*s0 + pad0
        # # crop_j2d1 = (db['j2d1'] - db['bb1'][0])*s1 + pad1
        # # crop_j2d2 = (db['j2d2'] - db['bb2'][0])*s2 + pad2
        # # crop_j2d3 = (db['j2d3'] - db['bb3'][0])*s3 + pad3
        # intr = {}
        # for i in range(self.num_cams):
        #     intr[str(i)] = self.totensor(db['cam'+str(i)]['intr']).float()[0]
        
        # wcam = {}
        # wcam_in = {}
        # # get weak perspective camera for the full image
        # for i in range(self.num_cams):
        #     # import ipdb; ipdb.set_trace()
        #     wcam[str(i)] = torch.from_numpy(get_weak_persp_cam_full_img_gt(intr[str(i)],smpl_joints_wrt_cam[str(i)][0].data.cpu().numpy()))
        #     wcam_in[str(i)] = torch.from_numpy(get_weak_persp_cam_full_img_input(intr[str(i)],db['bb'+str(i)]))
        
        # smplpose_rotmat = lbs.batch_rodrigues(smplpose.reshape(-1,3))


        # return {'im0_path':db['im0'],'im1_path':db['im1'],
        # 'im0':im['0'],'im1':im['1'],
        # 'intr0':intr['0'],'intr1':intr['1'],
        # 'smplpose':smplpose,'smplbetas':smplbetas, 'smplorient': smplorient,
        # 'smplpose_rotmat': smplpose_rotmat, 'focal_length':torch.tensor(CONSTANTS.FOCAL_LENGTH).float(), 'img_size':torch.tensor(CONSTANTS.IMG_SIZE).float(),
        # 'smplorient_rotmat_wrt_cam0': smplorient_rotmat_wrt_cam['0'],'smplorient_rotmat_wrt_cam1': smplorient_rotmat_wrt_cam['1'],
        # 'smpl_vertices_wrt_cam0':smpl_vertices_wrt_cam['0'],'smpl_vertices_wrt_cam1':smpl_vertices_wrt_cam['1'],
        # 'smpl_joints_wrt_cam0':smpl_joints_wrt_cam['0'],'smpl_joints_wrt_cam1':smpl_joints_wrt_cam['1'],
        # 'wcam0':wcam['0'],'wcam1':wcam['1'],
        # 'wcam_in0':wcam_in['0'],'wcam_in1':wcam_in['1']}