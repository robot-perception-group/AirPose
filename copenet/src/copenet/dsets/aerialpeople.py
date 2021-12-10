import torch
import logging
import os
import pickle as pk
from torch.utils.data import Dataset
from torchvision import transforms
import h5py
import cv2
import numpy as np
import sys
from smplx import SMPLX, lbs
import config
import torchgeometry as tgm
from utils.utils import npPerspProj, resize_with_pad, get_weak_persp_cam_full_img_input, get_weak_persp_cam_full_img_gt, transform_smpl
import torch
from utils.geometry import batch_rodrigues, perspective_projection, estimate_translation, rot6d_to_rotmat
import constants as CONSTANTS
import imgaug.augmenters as iaa
import random

def get_aerialpeople_seqsplit_old(datapath='',train_test_ratio=0.5):
    assert(train_test_ratio>0 and train_test_ratio<1)
    train_dset = aerialpeople_crop(datapath=os.path.join(datapath,'aerialpeople.pkl'))
    test_dset = aerialpeople_crop(datapath=os.path.join(datapath,'aerialpeople.pkl'))

    train_dset.db = train_dset.db[:int(train_test_ratio*train_dset.__len__())]
    train_dset.db_len = len(train_dset.db)
    test_dset.db = test_dset.db[int(train_test_ratio*test_dset.__len__()):]
    test_dset.db_len = len(test_dset.db)
    
    return train_dset, test_dset

def get_aerialpeople_seqsplit(datapath='/home/nsaini/Datasets/AerialPeople/agora_copenet_uniform_new_cropped'):
    
    train_dset = aerialpeople_crop(datapath=os.path.join(datapath,"dataset",'train_pkls.pkl'))
    test_dset = aerialpeople_crop(datapath=os.path.join(datapath,"dataset",'test_pkls.pkl'))
    
    return train_dset, test_dset

class aerialpeople_crop(Dataset):
    def __init__(self,datapath,rottrans=False):
        super().__init__()
        
        if os.path.exists(datapath):
            with open(datapath,'rb') as f:
                print('loading aerialpeople data...')
                self.db = pk.load(f)
        else:
            sys.exit('database not found!!!!, create database')
        
        self.num_cams = 2

        self.data_root = "/".join(datapath.split("/")[:-2])

        self.db_len = len(self.db)

        self.smplx_male = SMPLX(config.SMPLX_MODEL_DIR,
                         batch_size=1,
                         create_transl=False, gender="male")
        self.smplx_female = SMPLX(config.SMPLX_MODEL_DIR,
                         batch_size=1,
                         create_transl=False, gender="female")
        self.smplx_neutral = SMPLX(config.SMPLX_MODEL_DIR,
                         batch_size=1,
                         create_transl=False)

        self.transform = rottrans_tfm(0,0)
        # self.rottrans = rottrans
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])                 # for pre trained resnet
        self.totensor = transforms.ToTensor()                                 # converting to tensor

        # self.augs = [iaa.AddToBrightness((-30,30)),
        #             iaa.AddToHueAndSaturation((-50,50),per_channel=True),
        #             iaa.ChangeColorTemperature((1100,10000)),
        #             iaa.GammaContrast((0.5,2.0),per_channel=True),
        #             iaa.Grayscale(alpha=(0.0,1.0))]
                
    def __len__(self):
        return self.db_len

    def __getitem__(self,idx):

        with open(self.db[idx],'rb') as f:
            db = pk.load(f)
        
        intr = {}
        extr = {}
        for i in range(self.num_cams):
            intr[str(i)] = self.totensor(db['cam'+str(i)]['intr']).float()[0]
            extr[str(i)] = self.totensor(db['cam'+str(i)]['extr']).float()[0]

        
        im = {}
        bb = {}
        crop_info = {}
        for i in range(self.num_cams):

            ymin = (db['bb'+str(i)][0][1] - 200) if (db['bb'+str(i)][0][1] - 200) > 0 else 0
            ymax = (db['bb'+str(i)][1][1] + 200) if (db['bb'+str(i)][1][1] + 200) < 1080 else 1080
            xmin = (db['bb'+str(i)][0][0] - 200) if (db['bb'+str(i)][0][0] - 200) > 0 else 0
            xmax = (db['bb'+str(i)][1][0] + 200) if (db['bb'+str(i)][1][0] + 200) < 1920 else 1920 

            
            if db['bb'+str(i)][0][1] - ymin == 0:
                offset_ymin = 0
            else:
                offset_ymin = np.random.randint(db['bb'+str(i)][0][1] - ymin)

            if ymax - db['bb'+str(i)][1][1] == 0:
                offset_ymax = 0
            else:
                offset_ymax = np.random.randint(ymax - db['bb'+str(i)][1][1])

            if db['bb'+str(i)][0][0] - xmin == 0:
                offset_xmin = 0
            else:
                offset_xmin = np.random.randint(db['bb'+str(i)][0][0] - xmin)

            if xmax - db['bb'+str(i)][1][0] == 0:
                offset_xmax = 0
            else:
                offset_xmax = np.random.randint(xmax - db['bb'+str(i)][1][0])
            

            img = cv2.imread(os.path.join(self.data_root,db['im'+str(i)]))[:,:,::-1]/255.

            im[str(i)] = img[offset_ymin:(img.shape[0]-offset_ymax),offset_xmin:(img.shape[1]-offset_xmax),:]

            if im[str(i)].shape[0] == 0 or im[str(i)].shape[1] == 0:
                import ipdb; ipdb.set_trace()
            
            crop_info[str(i)] = torch.tensor([[ymin , xmin],[ymax , xmax]]).int()

            bb[str(i)] = (torch.tensor([xmin + offset_xmin + xmax - offset_xmax,
                                    ymin + offset_ymin + ymax - offset_ymax]).float()/2)/intr[str(i)][:2,2] - 1

        s = {}
        pad = {}
        for i in range(self.num_cams):
            try:
                im[str(i)],s[str(i)],pad[str(i)] = resize_with_pad(im[str(i)],size=224)
            except:
                print('!!!!!!!!!!!!!!'+db['im'+str(i)]+'!!!!!!!!!!!!!!!!!!')
                print(im[str(i)].shape)
                print(db['bb'+str(i)])


        smplpose = torch.from_numpy(db['smplpose'].reshape(63))
        smplbetas = torch.from_numpy(db['smplshape'].reshape(10))
        
        smpl_vertices_rel = {}
        smpl_joints_rel = {}
        smplorient_rel = {}
        smpltrans_rel = {}
        gt_joints_2d = {}
        gt_joints_2d_crop = {}
        extr = {}
        for i in range(self.num_cams):
            extr[str(i)] = self.totensor(db['cam'+str(i)]['extr']).float()
            smpl_vertices_rel[str(i)], smpl_joints_rel[str(i)], smplorient_rel[str(i)], smpltrans_rel[str(i)] = transform_smpl(extr[str(i)],
                                                                            torch.from_numpy(db['smpl_vertices_wrt_origin']),
                                                                            torch.from_numpy(db['smpl_joints_wrt_origin']),
                                                                            torch.from_numpy(db['smplorient_rotmat_wrt_origin']),
                                                                            torch.from_numpy(db['smpltrans']).float())

            gt_joints_2d[str(i)] = perspective_projection(smpl_joints_rel[str(i)],
                                                   rotation=torch.eye(3).float().unsqueeze(0),
                                                   translation=torch.zeros(1,3).float(),
                                                   focal_length=torch.tensor(CONSTANTS.FOCAL_LENGTH).float(),
                                                   camera_center=intr[str(i)][:2,2].unsqueeze(0))
        
            gt_joints_2d_crop[str(i)] = s[str(i)]*(gt_joints_2d[str(i)].squeeze(0) - (bb[str(i)] + 1)*intr[str(i)][:2,2])
            
            im[str(i)] = self.normalize(torch.from_numpy(im[str(i)].transpose(2,0,1)).float())


        smplpose_rotmat = lbs.batch_rodrigues(smplpose.reshape(-1,3))
    
        with torch.no_grad():
            if db['smplgender'].upper() == "FEMALE":
                smpl = self.smplx_female.forward(betas=smplbetas.unsqueeze(0), 
                                    body_pose=smplpose_rotmat.unsqueeze(0),
                                    global_orient=torch.eye(3).float().unsqueeze(0).unsqueeze(0).type_as(smplbetas),
                                    transl = torch.zeros(1,3).float().type_as(smplbetas),
                                    pose2rot=False)
            elif db['smplgender'].upper() == "MALE":
                smpl = self.smplx_male.forward(betas=smplbetas.unsqueeze(0), 
                                    body_pose=smplpose_rotmat.unsqueeze(0),
                                    global_orient=torch.eye(3).float().unsqueeze(0).unsqueeze(0).type_as(smplbetas),
                                    transl = torch.zeros(1,3).float().type_as(smplbetas),
                                    pose2rot=False)
            else:
                smpl = self.smplx_neutral.forward(betas=smplbetas.unsqueeze(0), 
                                    body_pose=smplpose_rotmat.unsqueeze(0),
                                    global_orient=torch.eye(3).float().unsqueeze(0).unsqueeze(0).type_as(smplbetas),
                                    transl = torch.zeros(1,3).float().type_as(smplbetas),
                                    pose2rot=False)
        
        for i in range(self.num_cams):
            bb[str(i)] = torch.cat([bb[str(i)],torch.tensor(s[str(i)]).view(1)])
        # extr0_tfm,extr1_tfm,verts_tfm,joints_tfm,orient_tfm, smpltrans_tfm = self.transform(extr['0'],
        #                                                                 extr['1'],
        #                                                                 smpl_vertices_wrt_origin,
        #                                                                 smpl_joints_wrt_origin,
        #                                                                 smplorient_rotmat_wrt_origin,
        #                                                                 smpltrans)
        
        cam1 = np.random.randint(2)
        cam2 = int(1 - cam1)
        cam1 = str(cam1)
        cam2 = str(cam2)
        return {'im0_path':os.path.join(self.data_root,db['im'+cam1]),'im1_path':os.path.join(self.data_root,db['im'+cam2]),
        'im0':im[cam1],'im1':im[cam2],
        'intr0':intr[cam1],'intr1':intr[cam2],
        'extr0':extr[cam1],'extr1':extr[cam2],
        'bb0': bb[cam1], 'bb1': bb[cam2],
        'crop_info0':crop_info[cam1], 'crop_info1':crop_info[cam2],
        'smplbetas':smplbetas,'smpltrans_rel0': smpltrans_rel[cam1].squeeze(0), 'smpltrans_rel1': smpltrans_rel[cam2].squeeze(0),
        'smplpose_rotmat': smplpose_rotmat, 'focal_length':torch.tensor(CONSTANTS.FOCAL_LENGTH).float(), 'img_size':torch.tensor(CONSTANTS.IMG_SIZE).float(),
        'smplorient_rel0': smplorient_rel[cam1],'smplorient_rel1': smplorient_rel[cam2],
        'smpl_vertices_rel0':smpl_vertices_rel[cam1],'smpl_vertices_rel1':smpl_vertices_rel[cam2],
        'smpl_joints_rel0':smpl_joints_rel[cam1],'smpl_joints_rel1':smpl_joints_rel[cam2],
        'smpl_joints_2d0':gt_joints_2d[cam1],'smpl_joints_2d1':gt_joints_2d[cam2],
        'smpl_joints_2d_crop0':gt_joints_2d_crop[cam1],'smpl_joints_2d_crop1':gt_joints_2d_crop[cam2],
        'smpl_vertices': smpl.vertices.detach(), 'smpl_joints': smpl.joints.detach(),
        'smpl_gender':db['smplgender']}

class rottrans_tfm(object):
    def __init__(self,trans_range,rot_range):
#         rot range between 0 to 360
        self.trans_range = trans_range
        self.rot_range = rot_range
    
    def __call__(self,extr0,extr1,verts,joints,orient,smpltrans):
        with torch.no_grad():
            angles = torch.rand(1,3)*self.rot_range
            trans = (torch.rand(1,3)-0.5)*self.trans_range
            
            tfm_rottrans = tgm.angle_axis_to_rotation_matrix(angles)[0]
            tfm_rottrans[:3,3] = trans[0]
            
            extr0_tfm = torch.mm(extr0,torch.inverse(tfm_rottrans))
            extr1_tfm = torch.mm(extr1,torch.inverse(tfm_rottrans))
            verts_tfm = torch.t(torch.mm(tfm_rottrans[:3,:3],torch.t(verts[0])) + tfm_rottrans[:3,3].unsqueeze(1)).unsqueeze(0)
            joints_tfm = torch.t(torch.mm(tfm_rottrans[:3,:3],torch.t(joints[0])) + tfm_rottrans[:3,3].unsqueeze(1)).unsqueeze(0)
            orient_tfm = (torch.mm(tfm_rottrans[:3,:3],orient[0])).unsqueeze(0)
            smpltrans_tfm = (torch.mm(tfm_rottrans[:3,:3],smpltrans.unsqueeze(1))).squeeze(1)
        
            # new_gt = torch.t(torch.mm(tfm_rottrans[:3,:3],torch.t(torch.tensor(sample[2]).float()))) + tfm_rottrans[:3,3]
            # new_c1 = torch.mm(tfm_rottrans,torch.tensor(np.concatenate([sample[3],[[0,0,0,1]]],axis=0)).float())[:3,:]
            # new_c2 = torch.mm(tfm_rottrans,torch.tensor(np.concatenate([sample[4],[[0,0,0,1]]],axis=0)).float())[:3,:]
    
        return (extr0_tfm,
                extr1_tfm,
                verts_tfm,
                joints_tfm,
                orient_tfm,
                smpltrans_tfm)