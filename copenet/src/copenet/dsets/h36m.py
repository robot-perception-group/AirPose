import torch
import logging
import os
import pickle as pkl
from torch.utils.data import Dataset
from torchvision import transforms
import h5py
import cv2
import numpy as np
import sys
import torchgeometry as tgm
from utils.utils import npPerspProj, resize_with_pad
import torch
from utils.utils import get_weak_persp_cam_full_img_gt, transform_smpl
import constants
import glob
import copy
from smplx.smplx import lbs

h36m_movable = [0,1,2,3,6,7,8,12,13,14,15,17,18,19,25,26,27]

# h36m2smpl = [0,6,1,11,7,2,12,8,3,9,4,13,14,17,25,18,26,19,27,20,28]

class h36m_full_train(Dataset):
    def __init__(self,datapath='/home/nsaini/Desktop/Human3.6M/',rottrans=False):
        super().__init__()
        
        if os.path.exists('dsets/h36m_db.pkl'):
            with open('dsets/h36m_db.pkl','rb') as f:
                self.db = pkl.load(f)
        else:
            sys.exit('database not found!!!!, create database')
        
        self.cam_file = os.path.join('dsets','h36m_cameras.h5')
        
        
        self.db_len = len(self.db) 
        self.transform = rottrans_tfm(100,355)
        self.shrink_factor = 4
        self.shrink_im_size = int(1000/self.shrink_factor)
        self.rottrans = rottrans
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])                 # for pre trained resnet
        self.totensor = transforms.ToTensor                                 # converting to tensor
                
    def __len__(self):
        return self.db_len
    
    def __getitem__(self,idx):
        ims = self.db[idx]
        im1 = cv2.resize(cv2.imread(ims['c1'])[:1000,:1000,:],(self.shrink_im_size,self.shrink_im_size)).transpose(2,0,1)
        im2 = cv2.resize(cv2.imread(ims['c2'])[:1000,:1000,:],(self.shrink_im_size,self.shrink_im_size)).transpose(2,0,1)
        gt = ims['gt'].reshape(-1,3)
        
        with h5py.File(self.cam_file,'r') as cam_db:
            cam1 = np.concatenate([cam_db['subject'+ims['s']]['camera1']['R'][()],
                                   cam_db['subject'+ims['s']]['camera1']['T'][()]],axis=1)
            cam2 = np.concatenate([cam_db['subject'+ims['s']]['camera2']['R'][()],
                                   cam_db['subject'+ims['s']]['camera2']['T'][()]],axis=1)
            intr1 = np.eye(3)
            intr1[0,0],intr1[1,1] = cam_db['subject'+ims['s']]['camera1']['f'][()][:,0]/self.shrink_factor
            intr1[:2,2] = cam_db['subject'+ims['s']]['camera1']['c'][()][:,0]/self.shrink_factor
            intr2 = np.eye(3)
            intr2[0,0],intr2[1,1] = cam_db['subject'+ims['s']]['camera2']['f'][()][:,0]/self.shrink_factor
            intr2[:2,2] = cam_db['subject'+ims['s']]['camera2']['c'][()][:,0]/self.shrink_factor
        
        im1 = self.normalize(torch.from_numpy(im1).float().div(255))
        im2 = self.normalize(torch.from_numpy(im2).float().div(255))
        
#         interchange cameras with equal probability
        i = [im1,im2]
        c = [cam1,cam2]
        intr = [intr1,intr2]
        p = np.random.rand()<0.5
        
        if self.rottrans:
            tr = self.transform([i[p],i[1-p],gt,c[p],c[1-p],intr[p],intr[1-p]])
            return {'im1':tr[0],'im2':tr[1],'gt':tr[2],'cam1':tr[3],'cam2':tr[4],'intr1':tr[5],'intr2':tr[6]}
        else:
            return {'im1':i[p],'im2':i[1-p],'gt':gt,'cam1':c[p],'cam2':c[1-p],'intr1':intr[p],'intr2':intr[1-p]}
        
        

        
class h36m_full_test(Dataset):
    def __init__(self,datapath='/home/nsaini/Desktop/Human3.6M/',rottrans=False):
        super().__init__()
        
        if os.path.exists('dsets/h36m_db.pkl'):
            with open('dsets/h36m_db.pkl','rb') as f:
                self.db = pkl.load(f)
        else:
            sys.exit('database not found!!!!, create database')
        
        self.cam_file = os.path.join('dsets','h36m_cameras.h5')
        
        self.transform = rottrans_tfm(100,355)
        self.db_len = len(self.db)
        self.shrink_factor = 4
        self.shrink_im_size = int(1000/self.shrink_factor)
        self.rottrans = rottrans
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])                 # for pre trained resnet
        self.totensor = transforms.ToTensor                                 # converting to tensor
                
    def __len__(self):
        return self.db_len
    
    def __getitem__(self,idx):
        ims = self.db[idx]
        im3 = cv2.resize(cv2.imread(ims['c3'])[:1000,:1000,:],(self.shrink_im_size,self.shrink_im_size)).transpose(2,0,1)
        im4 = cv2.resize(cv2.imread(ims['c4'])[:1000,:1000,:],(self.shrink_im_size,self.shrink_im_size)).transpose(2,0,1)
        gt = ims['gt'].reshape(-1,3)
        with h5py.File(self.cam_file,'r') as cam_db:
            cam3 = np.concatenate([cam_db['subject'+ims['s']]['camera3']['R'][()],
                                   cam_db['subject'+ims['s']]['camera3']['T'][()]],axis=1)
            cam4 = np.concatenate([cam_db['subject'+ims['s']]['camera4']['R'][()],
                                   cam_db['subject'+ims['s']]['camera4']['T'][()]],axis=1)
            intr3 = np.eye(3)
            intr3[0,0],intr3[1,1] = cam_db['subject'+ims['s']]['camera3']['f'][()][:,0]/self.shrink_factor
            intr3[:2,2] = cam_db['subject'+ims['s']]['camera3']['c'][()][:,0]/self.shrink_factor
            intr4 = np.eye(3)
            intr4[0,0],intr4[1,1] = cam_db['subject'+ims['s']]['camera4']['f'][()][:,0]/self.shrink_factor
            intr4[:2,2] = cam_db['subject'+ims['s']]['camera4']['c'][()][:,0]/self.shrink_factor
            
        im3 = self.normalize(torch.from_numpy(im3).float().div(255))
        im4 = self.normalize(torch.from_numpy(im4).float().div(255))
        
#         interchange cameras with equal probability
        i = [im3,im4]
        c = [cam3,cam4]
        intr = [intr3,intr4]
        p = np.random.rand()<0.5
        
        if self.rottrans:
            tr = self.transform([i[p],i[1-p],gt,c[p],c[1-p],intr[p],intr[1-p]])
            return {'im1':tr[0],'im2':tr[1],'gt':tr[2],'cam1':tr[3],'cam2':tr[4],'intr1':tr[5],'intr2':tr[6]}
        else:
            return {'im1':i[p],'im2':i[1-p],'gt':gt,'cam1':c[p],'cam2':c[1-p],'intr1':intr[p],'intr2':intr[1-p]}
        

    
    
    
    
    
class h36m_crop_train(Dataset):
    def __init__(self,datafile='/home/nsaini/Desktop/Human3.6M/dsets/h36m_db.pkl',rottrans=False):
        super().__init__()
        
        if os.path.exists(datafile):
            with open(datafile,'rb') as f:
                self.db = pkl.load(f)
        else:
            sys.exit('database not found!!!!, create database')
        
        self.cam_file = os.path.join('dsets','h36m_cameras.h5')
        
        
        self.db_len = len(self.db) 
        self.transform = rottrans_tfm(100,355)
        self.shrink_factor = 4
        self.shrink_im_size = int(1000/self.shrink_factor)
        self.rottrans = rottrans
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])                 # for pre trained resnet
        self.totensor = transforms.ToTensor                                 # converting to tensor

        from smplx.smplx import SMPLX
        smplx_path = '/is/ps3/nsaini/projects/newcopenet/data/smplx/models/smplx'
        self.smplx_model_male = SMPLX(os.path.join(smplx_path, 'SMPLX_MALE.npz')).eval()
        self.smplx_model_female = SMPLX(os.path.join(smplx_path, 'SMPLX_FEMALE.npz')).eval()
        self.smplx_model_neutral = SMPLX(os.path.join(smplx_path, 'SMPLX_NEUTRAL.npz')).eval()
                
    def __len__(self):
        return self.db_len

    def __getitem__(self,idx):
        db = self.db[idx]
        im0 = cv2.imread(db['c1'])[:1000,:1000,:]
        im1 = cv2.imread(db['c2'])[:1000,:1000,:]
        im2 = cv2.imread(db['c3'])[:1000,:1000,:]
        im3 = cv2.imread(db['c4'])[:1000,:1000,:]
        gt = db['gt'].reshape(-1,3)/1000
        
        with h5py.File(self.cam_file,'r') as cam_db:
            cam0 = np.concatenate([cam_db['subject'+db['s']]['camera1']['R'][()],
                                   cam_db['subject'+db['s']]['camera1']['T'][()]/1000],axis=1)
            cam1 = np.concatenate([cam_db['subject'+db['s']]['camera2']['R'][()],
                                   cam_db['subject'+db['s']]['camera2']['T'][()]/1000],axis=1)
            cam2 = np.concatenate([cam_db['subject'+db['s']]['camera3']['R'][()],
                                   cam_db['subject'+db['s']]['camera3']['T'][()]/1000],axis=1)
            cam3 = np.concatenate([cam_db['subject'+db['s']]['camera4']['R'][()],
                                   cam_db['subject'+db['s']]['camera4']['T'][()]/1000],axis=1)
            intr0 = np.eye(3)
            intr0[0,0],intr0[1,1] = cam_db['subject'+db['s']]['camera1']['f'][()][:,0]
            intr0[:2,2] = cam_db['subject'+db['s']]['camera1']['c'][()][:,0]
            intr1 = np.eye(3)
            intr1[0,0],intr1[1,1] = cam_db['subject'+db['s']]['camera2']['f'][()][:,0]
            intr1[:2,2] = cam_db['subject'+db['s']]['camera2']['c'][()][:,0]
            intr2 = np.eye(3)
            intr2[0,0],intr2[1,1] = cam_db['subject'+db['s']]['camera3']['f'][()][:,0]
            intr2[:2,2] = cam_db['subject'+db['s']]['camera3']['c'][()][:,0]
            intr3 = np.eye(3)
            intr3[0,0],intr3[1,1] = cam_db['subject'+db['s']]['camera4']['f'][()][:,0]
            intr3[:2,2] = cam_db['subject'+db['s']]['camera4']['c'][()][:,0]
        
        with torch.no_grad():
        
            smplpose = torch.from_numpy(db['smplpose'].reshape(63))
            smplbetas = torch.from_numpy(db['smplshape'].reshape(10))
            smpltrans = torch.from_numpy(db['smpltrans'].reshape(3))
            smplorient = torch.from_numpy(db['smplorient'].reshape(3))
            
            smplorient_rotmat = tgm.angle_axis_to_rotation_matrix(smplorient.unsqueeze(0))[:,:3,:3]
            
            if db['gender'] == 'MALE':
                smplx = self.smplx_model_male
            elif db['gender'] == 'FEMALE':
                smplx = self.smplx_model_female
            else:
                smplx = self.smplx_model_neutral

            out = smplx.forward(betas=smplbetas.view(1,10),
                                        global_orient=smplorient.view(1,3),
                                        transl=smpltrans.view(1,3),
                                        body_pose=smplpose.view(1,63))

            j2d0,extr_rot0,extr_trans0 = npPerspProj(intr0,gt,cam0)
            mincorner0 = np.array([max(min(j2d0[:,0])-50,0),max(min(j2d0[:,1])-50,0)]).astype(int)
            maxcorner0 = np.array([min(max(j2d0[:,0])+50,1000),min(max(j2d0[:,1])+50,1000)]).astype(int)
            newc0 = ((mincorner0+maxcorner0)/2).astype(int)
            
            j2d1,extr_rot1,extr_trans1 = npPerspProj(intr1,gt,cam1)
            mincorner1 = np.array([max(min(j2d1[:,0])-50,0),max(min(j2d1[:,1])-50,0)]).astype(int)
            maxcorner1 = np.array([min(max(j2d1[:,0])+50,1000),min(max(j2d1[:,1])+50,1000)]).astype(int)
            newc1 = ((mincorner1+maxcorner1)/2).astype(int)

            j2d2,extr_rot2,extr_trans2 = npPerspProj(intr2,gt,cam2)
            mincorner2 = np.array([max(min(j2d2[:,0])-50,0),max(min(j2d2[:,1])-50,0)]).astype(int)
            maxcorner2 = np.array([min(max(j2d2[:,0])+50,1000),min(max(j2d2[:,1])+50,1000)]).astype(int)
            newc2 = ((mincorner2+maxcorner2)/2).astype(int)
            
            j2d3,extr_rot3,extr_trans3 = npPerspProj(intr3,gt,cam3)
            mincorner3 = np.array([max(min(j2d3[:,0])-50,0),max(min(j2d3[:,1])-50,0)]).astype(int)
            maxcorner3 = np.array([min(max(j2d3[:,0])+50,1000),min(max(j2d3[:,1])+50,1000)]).astype(int)
            newc3 = ((mincorner3+maxcorner3)/2).astype(int)
            
            extr0 = torch.from_numpy(np.concatenate([extr_rot0,extr_trans0],axis=1)).float()
            extr1 = torch.from_numpy(np.concatenate([extr_rot1,extr_trans1],axis=1)).float()
            extr2 = torch.from_numpy(np.concatenate([extr_rot2,extr_trans2],axis=1)).float()
            extr3 = torch.from_numpy(np.concatenate([extr_rot3,extr_trans3],axis=1)).float()
            
            smpl_wrt_cam0 = transform_smpl(extr0.unsqueeze(0),
                                            out.vertices.view(1,-1,3),
                                            out.joints.view(1,-1,3)[:,:22],
                                            smplorient_rotmat)
            smpl_wrt_cam1 = transform_smpl(extr1.unsqueeze(0),
                                            out.vertices.view(1,-1,3),
                                            out.joints.view(1,-1,3)[:,:22],
                                            smplorient_rotmat)
            smpl_wrt_cam2 = transform_smpl(extr2.unsqueeze(0),
                                            out.vertices.view(1,-1,3),
                                            out.joints.view(1,-1,3)[:,:22],
                                            smplorient_rotmat)
            smpl_wrt_cam3 = transform_smpl(extr3.unsqueeze(0),
                                            out.vertices.view(1,-1,3),
                                            out.joints.view(1,-1,3)[:,:22],
                                            smplorient_rotmat)
            
            smpl_vertices_wrt_cam0 = smpl_wrt_cam0[0][0]#*torch.tensor([1,1,-1]).float()
            smpl_vertices_wrt_cam1 = smpl_wrt_cam1[0][0]#*torch.tensor([1,1,-1]).float()
            smpl_vertices_wrt_cam2 = smpl_wrt_cam2[0][0]#*torch.tensor([1,1,-1]).float()
            smpl_vertices_wrt_cam3 = smpl_wrt_cam3[0][0]#*torch.tensor([1,1,-1]).float()
            smpl_joints_wrt_cam0 = smpl_wrt_cam0[1][0]#*torch.tensor([1,1,-1]).float()
            smpl_joints_wrt_cam1 = smpl_wrt_cam1[1][0]#*torch.tensor([1,1,-1]).float()
            smpl_joints_wrt_cam2 = smpl_wrt_cam2[1][0]#*torch.tensor([1,1,-1]).float()
            smpl_joints_wrt_cam3 = smpl_wrt_cam3[1][0]#*torch.tensor([1,1,-1]).float()
            smplorient_rotmat_wrt_cam0 = smpl_wrt_cam0[2]
            smplorient_rotmat_wrt_cam1 = smpl_wrt_cam1[2]
            smplorient_rotmat_wrt_cam2 = smpl_wrt_cam2[2]
            smplorient_rotmat_wrt_cam3 = smpl_wrt_cam3[2]



            im0_resized,s0,pad0 = resize_with_pad(im0[mincorner0[1]:maxcorner0[1],mincorner0[0]:maxcorner0[0],:],224)
            im1_resized,s1,pad1 = resize_with_pad(im1[mincorner1[1]:maxcorner1[1],mincorner1[0]:maxcorner1[0],:],224)
            im2_resized,s2,pad2 = resize_with_pad(im2[mincorner2[1]:maxcorner2[1],mincorner2[0]:maxcorner2[0],:],224)
            im3_resized,s3,pad3 = resize_with_pad(im3[mincorner3[1]:maxcorner3[1],mincorner3[0]:maxcorner3[0],:],224)

            im0 = self.normalize(torch.from_numpy(im0_resized.transpose(2,0,1)/255).float())
            im1 = self.normalize(torch.from_numpy(im1_resized.transpose(2,0,1)/255).float())
            im2 = self.normalize(torch.from_numpy(im2_resized.transpose(2,0,1)/255).float())
            im3 = self.normalize(torch.from_numpy(im3_resized.transpose(2,0,1)/255).float())
            
            # crop_intr0 = copy.deepcopy(intr0[:2,2])
            # crop_intr1 = copy.deepcopy(intr1[:2,2])
            # crop_intr2 = copy.deepcopy(intr2[:2,2])
            # crop_intr3 = copy.deepcopy(intr3[:2,2])
            # crop_intr0[:2,2] = intr0[:2,2] - [mincorner0[0],mincorner0[1]]
            # crop_intr1[:2,2] = intr1[:2,2] - [mincorner1[0],mincorner1[1]]
            # crop_intr2[:2,2] = intr2[:2,2] - [mincorner2[0],mincorner2[1]]
            # crop_intr3[:2,2] = intr3[:2,2] - [mincorner3[0],mincorner3[1]]

            intr0 = torch.from_numpy(intr0).float()
            intr1 = torch.from_numpy(intr1).float()
            intr2 = torch.from_numpy(intr2).float()
            intr3 = torch.from_numpy(intr3).float()

            # gt = gt[h36m_movable]
            # gt_wrt_cam0 = torch.from_numpy(np.matmul(extr_rot0[:3,:3],gt.T).T + extr_trans0.reshape(1,3)).float()
            # gt_wrt_cam1 = torch.from_numpy(np.matmul(extr_rot1[:3,:3],gt.T).T + extr_trans1.reshape(1,3)).float()
            # gt_wrt_cam2 = torch.from_numpy(np.matmul(extr_rot2[:3,:3],gt.T).T + extr_trans2.reshape(1,3)).float()
            # gt_wrt_cam3 = torch.from_numpy(np.matmul(extr_rot3[:3,:3],gt.T).T + extr_trans3.reshape(1,3)).float()
            
            wcam0 = torch.from_numpy(get_weak_persp_cam_full_img_gt(intr0,smpl_joints_wrt_cam0[0])).float()
            wcam1 = torch.from_numpy(get_weak_persp_cam_full_img_gt(intr1,smpl_joints_wrt_cam1[0])).float()
            wcam2 = torch.from_numpy(get_weak_persp_cam_full_img_gt(intr2,smpl_joints_wrt_cam2[0])).float()
            wcam3 = torch.from_numpy(get_weak_persp_cam_full_img_gt(intr3,smpl_joints_wrt_cam3[0])).float()
            
            smplpose_rotmat = lbs.batch_rodrigues(smplpose.reshape(-1,3))
            

        # return {'im0_path':db['c1'],'im1_path':db['c2'],'im2_path':db['c3'],'im3_path':db['c4'],
        # 'im0':im0,'im1':im1,'im2':im2,'im3':im3,
        # 'intr0':intr0,'intr1':intr1,'intr2':intr2,'intr3':intr3,
        # 'smplpose':smplpose,'smplbetas':smplbetas, 'smplorient': smplorient,
        # 'smplpose_rotmat': smplpose_rotmat, 'focal_length':1146., 'img_size':[1000,1000],
        # 'smplorient_rotmat_wrt_cam0': smplorient_rotmat_wrt_cam0,'smplorient_rotmat_wrt_cam1': smplorient_rotmat_wrt_cam1,
        # 'smplorient_rotmat_wrt_cam2': smplorient_rotmat_wrt_cam2,'smplorient_rotmat_wrt_cam3': smplorient_rotmat_wrt_cam3,
        # 'smpl_vertices_wrt_cam0':smpl_vertices_wrt_cam0,'smpl_vertices_wrt_cam1':smpl_vertices_wrt_cam1,
        # 'smpl_vertices_wrt_cam2':smpl_vertices_wrt_cam2,'smpl_vertices_wrt_cam3':smpl_vertices_wrt_cam3,
        # 'smpl_joints_wrt_cam0':smpl_joints_wrt_cam0[:22],'smpl_joints_wrt_cam1':smpl_joints_wrt_cam1[:22],
        # 'smpl_joints_wrt_cam2':smpl_joints_wrt_cam2[:22],'smpl_joints_wrt_cam3':smpl_joints_wrt_cam3[:22],
        # 'wcam0':wcam0,'wcam1':wcam1,'wcam2':wcam2,'wcam3':wcam3, 'is_h36m':True}

        return {'im0_path':db['c1'],'im1_path':db['c2'],
        'im0':im0,'im1':im1,
        'intr0':intr0,'intr1':intr1,
        'smplpose':smplpose,'smplbetas':smplbetas, 'smplorient': smplorient,
        'smplpose_rotmat': smplpose_rotmat, 'focal_length':torch.tensor(1146).float(), 'img_size':torch.tensor([1000,1000]).float(),
        'smplorient_rotmat_wrt_cam0': smplorient_rotmat_wrt_cam0,'smplorient_rotmat_wrt_cam1': smplorient_rotmat_wrt_cam1,
        'smpl_vertices_wrt_cam0':smpl_vertices_wrt_cam0,'smpl_vertices_wrt_cam1':smpl_vertices_wrt_cam1,
        'smpl_joints_wrt_cam0':smpl_joints_wrt_cam0[:22],'smpl_joints_wrt_cam1':smpl_joints_wrt_cam1[:22],
        'wcam0':wcam0,'wcam1':wcam1, 'is_h36m':True}
        

        
class h36m_crop_test(Dataset):
    def __init__(self,datafile='/home/nsaini/Desktop/Human3.6M/dsets/h36m_db.pkl',rottrans=False):
        super().__init__()
        
        if os.path.exists(datafile):
            with open(datafile,'rb') as f:
                self.db = pkl.load(f)
        else:
            sys.exit('database not found!!!!, create database')
        
        self.cam_file = os.path.join('dsets','h36m_cameras.h5')
        
        self.transform = rottrans_tfm(100,355)
        self.db_len = len(self.db)
        self.shrink_factor = 4
        self.shrink_im_size = int(1000/self.shrink_factor)
        self.rottrans = rottrans
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])                 # for pre trained resnet
        self.totensor = transforms.ToTensor                                 # converting to tensor
                
    def __len__(self):
        return self.db_len
        

    def __getitem__(self,idx):
        ims = self.db[idx]
        im3 = cv2.imread(ims['c3'])[:1000,:1000,:].transpose(2,0,1)
        im4 = cv2.imread(ims['c4'])[:1000,:1000,:].transpose(2,0,1)
        gt = ims['gt'].reshape(-1,3)
        
        with h5py.File(self.cam_file,'r') as cam_db:
            cam3 = np.concatenate([cam_db['subject'+ims['s']]['camera3']['R'][()],
                                   cam_db['subject'+ims['s']]['camera3']['T'][()]],axis=1)
            cam4 = np.concatenate([cam_db['subject'+ims['s']]['camera4']['R'][()],
                                   cam_db['subject'+ims['s']]['camera4']['T'][()]],axis=1)
            intr3 = np.eye(3)
            intr3[0,0],intr3[1,1] = cam_db['subject'+ims['s']]['camera3']['f'][()][:,0]
            intr3[:2,2] = cam_db['subject'+ims['s']]['camera3']['c'][()][:,0]
            intr4 = np.eye(3)
            intr4[0,0],intr4[1,1] = cam_db['subject'+ims['s']]['camera4']['f'][()][:,0]
            intr4[:2,2] = cam_db['subject'+ims['s']]['camera4']['c'][()][:,0]
        
        j2d1 = npPerspProj(intr3,gt,cam3)
        mincorner1 = np.array([max(min(j2d1[:,0])-50,0),max(min(j2d1[:,1])-50,0)]).astype(int)
        maxcorner1 = np.array([min(max(j2d1[:,0])+50,1000),min(max(j2d1[:,1])+50,1000)]).astype(int)
        newc1 = ((mincorner1+maxcorner1)/2).astype(int)
        
        j2d2 = npPerspProj(intr4,gt,cam4)
        mincorner2 = np.array([max(min(j2d2[:,0])-50,0),max(min(j2d2[:,1])-50,0)]).astype(int)
        maxcorner2 = np.array([min(max(j2d2[:,0])+50,1000),min(max(j2d2[:,1])+50,1000)]).astype(int)
        newc2 = ((mincorner2+maxcorner2)/2).astype(int)
        
         
        im3 = torch.from_numpy(im3[:,mincorner1[1]:maxcorner1[1],mincorner1[0]:maxcorner1[0]])
        im4 = torch.from_numpy(im4[:,mincorner2[1]:maxcorner2[1],mincorner2[0]:maxcorner2[0]])
        
        intr3[:2,2] -= [mincorner1[0],mincorner1[1]]
        intr4[:2,2] -= [mincorner2[0],mincorner2[1]]

        bb1 = [mincorner1,maxcorner1]
        bb2 = [mincorner2,maxcorner2]
        
# #         interchange cameras with equal probability
#         i = [im3,im4]
#         c = [cam3,cam4]
#         intr = [intr3,intr4]
#         p = np.random.rand()<0.5
        
        return {'im1':im3,'im2':im4,'gt':gt,'cam1':cam3,'cam2':cam4,'intr1':intr3,'intr2':intr4,'bb1':bb1,'bb2':bb2}
    
    
class rottrans_tfm(object):
    def __init__(self,trans_range,rot_range):
#         rot range between 0 to 180
        self.trans_range = trans_range
        self.rot_range = rot_range
    
    def __call__(self,sample):
        with torch.no_grad():
            angles = torch.rand(1,3)*self.rot_range
            trans = (torch.rand(1,3)-0.5)*self.trans_range
            tfm_rottrans = tgm.angle_axis_to_rotation_matrix(angles)[0]
            tfm_rottrans[:3,3] = trans[0]
            new_gt = torch.t(torch.mm(tfm_rottrans[:3,:3],torch.t(torch.tensor(sample[2]).float()))) + tfm_rottrans[:3,3]
            new_c1 = torch.mm(tfm_rottrans,torch.tensor(np.concatenate([sample[3],[[0,0,0,1]]],axis=0)).float())[:3,:]
            new_c2 = torch.mm(tfm_rottrans,torch.tensor(np.concatenate([sample[4],[[0,0,0,1]]],axis=0)).float())[:3,:]
    
        return (sample[0],
                sample[1],
                new_gt.data.cpu().numpy(),
                new_c1.data.cpu().numpy(),
                new_c2.data.cpu().numpy(),
               sample[5],
               sample[6])


            
def create_db(datapath='/home/nsaini/Desktop/Human3.6m/',dest_file='dsets/h36m_db.pkl'):
    print('db file not found, reading the dataset and creating a new one ...')

    db = []
    img_dir = os.path.join(datapath,'images')
    for subject in ['S1','S5','S6','S7','S8']:
        sub_dir = os.path.join(img_dir,subject)
        exps = [exp.split('.')[0] for exp in os.listdir(sub_dir)]
        exps = list(set(exps))
        for exp in exps:
            print(subject,' ',exp)
            gtf = h5py.File(os.path.join(datapath,'h36m',subject,
                                        'MyPoses','3D_positions',exp.replace('_',' ')+'.h5'),'r')
            # import ipdb; ipdb.set_trace()
            moshf_path = os.path.join(datapath,'mosh',subject,
                            exp.replace('_',' ')+'_poses'+'.pkl')
            try:
                moshf = pkl.load(open(moshf_path,'rb'),encoding='latin1')
            except:
                print(moshf_path," doesn't exist!!!")
                continue
                # import ipdb; ipdb.set_trace()

            
            
            gt = gtf['3D_positions'][()]
            
            smplbetas = torch.from_numpy(moshf['shape_est_betas'][:10]).float().unsqueeze(0)
            smpltrans = torch.from_numpy(moshf['pose_est_trans'][:,:3]).float()
            smplorient = torch.from_numpy(moshf['pose_est_poses'][:,:3]).float()
            smplpose = torch.from_numpy(moshf['pose_est_poses'][:,3:66]).float()


            exp_dir1 = os.path.join(sub_dir,exp+'.54138969')
            p1 = sorted([f for f in os.listdir(exp_dir1) if f.endswith('.jpg')])
            exp_dir2 = os.path.join(sub_dir,exp+'.55011271')
            p2 = sorted([f for f in os.listdir(exp_dir2) if f.endswith('.jpg')])
            exp_dir3 = os.path.join(sub_dir,exp+'.58860488') # 60457274 
            p3 = sorted([f for f in os.listdir(exp_dir3) if f.endswith('.jpg')])
            exp_dir4 = os.path.join(sub_dir,exp+'.60457274') # 54138969 
            p4 = sorted([f for f in os.listdir(exp_dir4) if f.endswith('.jpg')])
            
            min_len = min(len(p1),len(p2),len(p3),len(p4),gt.shape[1])
            for l in range(min_len-10):
                db.append({'c1':os.path.join(exp_dir1,p1[l]),
                            'c2':os.path.join(exp_dir2,p2[l]),
                            'c3':os.path.join(exp_dir3,p3[l]),
                            'c4':os.path.join(exp_dir4,p4[l]),
                            's':subject[1:],
                            'gt':gt[:,l],
                            'smplshape':smplbetas.data.cpu().numpy(),
                            'smpltrans':smpltrans[4*l:4*l+1].data.cpu().numpy(),
                            'smplorient':smplorient[4*l:4*l+1].data.cpu().numpy(),
                            'smplpose':smplpose[4*l:4*l+1].data.cpu().numpy(),
                            'gender': moshf['ps']['bodymodel_fname'].split('/')[-2].upper()})


        print('Subject {} is done'.format(subject))
                
    
    with open(dest_file,'wb') as fl:
        pkl.dump(db,fl,pkl.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    create_db(datapath='/home/nsaini/Desktop/Human3.6m/',dest_file='dsets/h36m_db.pkl')