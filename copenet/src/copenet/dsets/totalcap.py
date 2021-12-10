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
from utils.utils import npPerspProj
import torch


def rotateXYZ( mesh_v, Rxyz):
    angle = np.radians(Rxyz[0])
    rx = np.array([
        [1., 0., 0.           ],
        [0., np.cos(angle), -np.sin(angle)],
        [0., np.sin(angle), np.cos(angle) ]
    ])

    angle = np.radians(Rxyz[1])
    ry = np.array([
        [np.cos(angle), 0., np.sin(angle)],
        [0., 1., 0.           ],
        [-np.sin(angle), 0., np.cos(angle)]
    ])

    angle = np.radians(Rxyz[2])
    rz = np.array([
        [np.cos(angle), -np.sin(angle), 0. ],
        [np.sin(angle), np.cos(angle), 0. ],
        [0., 0., 1. ]
    ])
    # return rotateZ(rotateY(rotateX(mesh_v, Rxyz[0]), Rxyz[1]), Rxyz[2])
    return rz.dot(ry.dot(rx.dot(mesh_v.T))).T
    # return rx.dot(mesh_v.T).T


class totalcap_full(Dataset):
    def __init__(self,datapath='/home/nsaini/Datasets/totalcapture',rottrans=False):
        super().__init__()
        
        if os.path.exists('dsets/totalcap_db.pkl'):
            with open('dsets/totalcap_db.pkl','rb') as f:
                print('loading totalcap data...')
                self.db = pkl.load(f)['db']
        else:
            sys.exit('database not found!!!!, create database')
        
        with open(os.path.join(datapath,'cameras.pkl'),'rb') as f:
            self.cams = pkl.load(f)
        
        
        self.db_len = len(self.db) 
        self.transform = rottrans_tfm(100,355)
        self.shrink_factor = 4
        self.shrink_im_size_height = int(1079/self.shrink_factor)
        self.shrink_im_size_width = int(1919/self.shrink_factor)

        # change intrinsics according to shrinked image
        self.cams['cam1']['intr'] /= self.shrink_factor
        self.cams['cam2']['intr'] /= self.shrink_factor
        self.cams['cam3']['intr'] /= self.shrink_factor
        self.cams['cam4']['intr'] /= self.shrink_factor
        self.cams['cam5']['intr'] /= self.shrink_factor
        self.cams['cam6']['intr'] /= self.shrink_factor
        self.cams['cam7']['intr'] /= self.shrink_factor
        self.cams['cam8']['intr'] /= self.shrink_factor

        self.rottrans = rottrans
        # self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                          std=[0.229, 0.224, 0.225])                 # for pre trained resnet
        self.totensor = transforms.ToTensor                                 # converting to tensor

                
    def __len__(self):
        return self.db_len
    
    def __getitem__(self,idx):
        db = self.db[idx]
        im1 = cv2.resize(cv2.imread(db['im1']),(self.shrink_im_size_height,self.shrink_im_size_width)).transpose(2,0,1)
        im2 = cv2.resize(cv2.imread(db['im2']),(self.shrink_im_size_height,self.shrink_im_size_width)).transpose(2,0,1)
        im3 = cv2.resize(cv2.imread(db['im3']),(self.shrink_im_size_height,self.shrink_im_size_width)).transpose(2,0,1)
        im4 = cv2.resize(cv2.imread(db['im4']),(self.shrink_im_size_height,self.shrink_im_size_width)).transpose(2,0,1)
        im5 = cv2.resize(cv2.imread(db['im5']),(self.shrink_im_size_height,self.shrink_im_size_width)).transpose(2,0,1)
        im6 = cv2.resize(cv2.imread(db['im6']),(self.shrink_im_size_height,self.shrink_im_size_width)).transpose(2,0,1)
        im7 = cv2.resize(cv2.imread(db['im7']),(self.shrink_im_size_height,self.shrink_im_size_width)).transpose(2,0,1)
        im8 = cv2.resize(cv2.imread(db['im8']),(self.shrink_im_size_height,self.shrink_im_size_width)).transpose(2,0,1)

        smplpose = db['poses']
        smplbetas = db['betas']
        smpltrans = db['trans']
        
        
        cam1 = np.concatenate([self.cams['cam1']['extr'],self.cams['cam1']['trans']],axis=1)
        cam2 = np.concatenate([self.cams['cam2']['extr'],self.cams['cam2']['trans']],axis=1)
        cam3 = np.concatenate([self.cams['cam3']['extr'],self.cams['cam3']['trans']],axis=1)
        cam4 = np.concatenate([self.cams['cam4']['extr'],self.cams['cam4']['trans']],axis=1)
        cam5 = np.concatenate([self.cams['cam5']['extr'],self.cams['cam5']['trans']],axis=1)
        cam6 = np.concatenate([self.cams['cam6']['extr'],self.cams['cam6']['trans']],axis=1)
        cam7 = np.concatenate([self.cams['cam7']['extr'],self.cams['cam7']['trans']],axis=1)
        cam8 = np.concatenate([self.cams['cam8']['extr'],self.cams['cam8']['trans']],axis=1)

        intr1 = self.cams['cam1']['intr']
        intr2 = self.cams['cam2']['intr']
        intr3 = self.cams['cam3']['intr']
        intr4 = self.cams['cam4']['intr']
        intr5 = self.cams['cam5']['intr']
        intr6 = self.cams['cam6']['intr']
        intr7 = self.cams['cam7']['intr']
        intr8 = self.cams['cam8']['intr']
        
        return {'im1':im1,'im2':im2,'im3':im3,'im4':im4,'im5':im5,'im6':im6,'im7':im7,'im8':im8,
        'cam1': cam1,'cam2': cam2,'cam3': cam3,'cam4': cam4,'cam5': cam5,'cam6': cam6,'cam7': cam7,'cam8': cam8,
        'intr1':intr1,'intr2':intr2,'intr3':intr3,'intr4':intr4,'intr5':intr5,'intr6':intr6,'intr7':intr7,'intr8':intr8,
        'smplpose':smplpose,'smplbetas':smplbetas,'smpltrans':smpltrans}
            
        

    
class totalcap_crop(Dataset):
    def __init__(self,datapath='/home/nsaini/Datasets/totalcapture',rottrans=False):
        super().__init__()
        
        if os.path.exists('dsets/totalcap_db.pkl'):
            with open('dsets/totalcap_db.pkl','rb') as f:
                print('loading totalcap data...')
                self.db = pkl.load(f)['db']
        else:
            sys.exit('database not found!!!!, create database')
        
        with open(os.path.join(datapath,'cameras.pkl'),'rb') as f:
            self.cams = pkl.load(f)
        
        
        self.db_len = len(self.db) 
        self.transform = rottrans_tfm(100,355)

        # self.rottrans = rottrans
        # self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                          std=[0.229, 0.224, 0.225])                 # for pre trained resnet
        self.totensor = transforms.ToTensor                                 # converting to tensor
                
    def __len__(self):
        return self.db_len

    def __getitem__(self,idx):
        db = self.db[idx]
        im1 = cv2.imread(db['im1']).transpose(2,0,1)[:,db['bb1'][0][0]:db['bb1'][1][0],db['bb1'][0][1]:db['bb1'][1][1]]
        im2 = cv2.imread(db['im2']).transpose(2,0,1)[:,db['bb2'][0][0]:db['bb2'][1][0],db['bb2'][0][1]:db['bb2'][1][1]]
        im3 = cv2.imread(db['im3']).transpose(2,0,1)[:,db['bb3'][0][0]:db['bb3'][1][0],db['bb3'][0][1]:db['bb3'][1][1]]
        im4 = cv2.imread(db['im4']).transpose(2,0,1)[:,db['bb4'][0][0]:db['bb4'][1][0],db['bb4'][0][1]:db['bb4'][1][1]]
        im5 = cv2.imread(db['im5']).transpose(2,0,1)[:,db['bb5'][0][0]:db['bb5'][1][0],db['bb5'][0][1]:db['bb5'][1][1]]
        im6 = cv2.imread(db['im6']).transpose(2,0,1)[:,db['bb6'][0][0]:db['bb6'][1][0],db['bb6'][0][1]:db['bb6'][1][1]]
        im7 = cv2.imread(db['im7']).transpose(2,0,1)[:,db['bb7'][0][0]:db['bb7'][1][0],db['bb7'][0][1]:db['bb7'][1][1]]
        im8 = cv2.imread(db['im8']).transpose(2,0,1)[:,db['bb8'][0][0]:db['bb8'][1][0],db['bb8'][0][1]:db['bb8'][1][1]]
        
        smplpose = db['poses']
        smplbetas = db['betas']
        smpltrans = db['trans']
        
        import ipdb; ipdb.set_trace()
        cam1 = np.concatenate([self.cams['cam1']['extr'],self.cams['cam1']['trans']],axis=1)
        cam2 = np.concatenate([self.cams['cam2']['extr'],self.cams['cam2']['trans']],axis=1)
        cam3 = np.concatenate([self.cams['cam3']['extr'],self.cams['cam3']['trans']],axis=1)
        cam4 = np.concatenate([self.cams['cam4']['extr'],self.cams['cam4']['trans']],axis=1)
        cam5 = np.concatenate([self.cams['cam5']['extr'],self.cams['cam5']['trans']],axis=1)
        cam6 = np.concatenate([self.cams['cam6']['extr'],self.cams['cam6']['trans']],axis=1)
        cam7 = np.concatenate([self.cams['cam7']['extr'],self.cams['cam7']['trans']],axis=1)
        cam8 = np.concatenate([self.cams['cam8']['extr'],self.cams['cam8']['trans']],axis=1)

        intr1 = self.cams['cam1']['intr']
        intr2 = self.cams['cam2']['intr']
        intr3 = self.cams['cam3']['intr']
        intr4 = self.cams['cam4']['intr']
        intr5 = self.cams['cam5']['intr']
        intr6 = self.cams['cam6']['intr']
        intr7 = self.cams['cam7']['intr']
        intr8 = self.cams['cam8']['intr']
        

        return {'im1':im1,'im2':im2,'im3':im3,'im4':im4,'im5':im5,'im6':im6,'im7':im7,'im8':im8,
        'cam1': cam1,'cam2': cam2,'cam3': cam3,'cam4': cam4,'cam5': cam5,'cam6': cam6,'cam7': cam7,'cam8': cam8,
        'intr1':intr1,'intr2':intr2,'intr3':intr3,'intr4':intr4,'intr5':intr5,'intr6':intr6,'intr7':intr7,'intr8':intr8,
        'smplpose':smplpose,'smplbetas':smplbetas,'smpltrans':smpltrans}
        
        
    
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


def totalcap_ds(ratio = [0.8,0.2],dstype='crop',**kwargs):
    if dstype == 'crop':
        dset = totalcap_crop(**kwargs)
        train_size = int(0.8 * len(dset))
        test_size = len(dset) - train_size
        train_dset, test_dset = torch.utils.data.random_split(dset,[train_size,test_size])
    else:
        dset = totalcap_full(**kwargs)
        train_size = int(0.8 * len(dset))
        test_size = len(dset) - train_size
        train_dset, test_dset = torch.utils.data.random_split(dset,[train_size,test_size])

    return train_dset, test_dset