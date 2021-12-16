"""
This file defines the core research contribution   
"""
import os
import time
import sys
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from argparse import ArgumentParser
import numpy as np
import torchgeometry as tgm

import copy
from .models import model_copenet
from .dsets import aerialpeople
import cv2
import torchvision
from .smplx.smplx import SMPLX, lbs
import pickle as pk
from .utils.renderer import Renderer
from . import constants as CONSTANTS
from .utils.utils import transform_smpl, add_noise_input_cams,add_noise_input_smpltrans
from .utils.geometry import batch_rodrigues, perspective_projection, estimate_translation, rot6d_to_rotmat

import pytorch_lightning as pl

from .config import device

smplx = None
smplx_test = None

def create_smplx(copenet_home,train_batch_size,val_batch_size):
    global smplx 
    smplx = SMPLX(os.path.join(copenet_home,"src/copenet/data/smplx/models/smplx"),
                         batch_size=train_batch_size,
                         create_transl=False)

    global smplx_test 
    smplx_test = SMPLX(os.path.join(copenet_home,"src/copenet/data/smplx/models/smplx"),
                         batch_size=val_batch_size,
                         create_transl=False)




class copenet_twoview(pl.LightningModule):

    def __init__(self, hparams):
        super(copenet_twoview, self).__init__()

        global smplx
        global smplx_test

        # not the best model...
        self.save_hyperparameters(hparams)
        self.model = model_copenet.getcopenet(os.path.join(self.hparams.copenet_home,"src/copenet/data/smpl_mean_params.npz"))

        create_smplx(self.hparams.copenet_home,self.hparams.batch_size,self.hparams.val_batch_size)
        
        smplx.to(device)
        smplx_test.to(device)

        smplx_hand_idx = pk.load(open(os.path.join(self.hparams.copenet_home,"src/copenet/data/smplx/MANO_SMPLX_vertex_ids.pkl"),'rb'))
        smplx_face_idx = np.load(os.path.join(self.hparams.copenet_home,"src/copenet/data/smplx/SMPL-X__FLAME_vertex_ids.npy"))
        self.register_buffer("body_only_mask",torch.ones(smplx.v_template.shape[0],1))
        self.body_only_mask[smplx_hand_idx['left_hand'],:] = 0
        self.body_only_mask[smplx_hand_idx['right_hand'],:] = 0
        self.body_only_mask[smplx_face_idx,:] = 0
        
        self.mseloss = nn.MSELoss(reduction='none')

        self.focal_length = CONSTANTS.FOCAL_LENGTH
        self.renderer = Renderer(focal_length=self.focal_length, img_res=CONSTANTS.IMG_SIZE, faces=smplx.faces)

    def forward(self, **kwargs):
        return self.model(**kwargs)
        

    def get_loss(self,input_batch, 
                    pred_smpltrans0,
                    pred_smpltrans1, 
                    pred_rotmat0,
                    pred_rotmat1, 
                    pred_betas0,
                    pred_betas1, 
                    pred_output_cam0,
                    pred_output_cam1, 
                    pred_joints_2d_cam0,
                    pred_joints_2d_cam1):

        gt_smplpose_rotmat = input_batch['smplpose_rotmat'] # SMPL pose rotation matrices
        gt_smpltrans_rel0 = input_batch['smpltrans_rel0'] # SMPL trans parameters
        gt_smpltrans_rel1 = input_batch['smpltrans_rel1'] # SMPL trans parameters
        gt_smplorient_rel0 = input_batch['smplorient_rel0'] # SMPL orientation parameters
        gt_smplorient_rel1 = input_batch['smplorient_rel1'] # SMPL orientation parameters
        gt_vertices0 = input_batch['smpl_vertices'].squeeze(1)
        gt_vertices1 = input_batch['smpl_vertices'].squeeze(1)
        gt_joints0 = input_batch['smpl_joints'].squeeze(1)
        gt_joints1 = input_batch['smpl_joints'].squeeze(1)
        gt_joints_2d_cam0 = input_batch['smpl_joints_2d0'].squeeze(1)
        gt_joints_2d_cam1 = input_batch['smpl_joints_2d1'].squeeze(1)
        
        loss_keypoints = (self.mseloss(pred_joints_2d_cam0[:,:22], gt_joints_2d_cam0[:,:22])).mean() + \
                            (self.mseloss(pred_joints_2d_cam1[:,:22], gt_joints_2d_cam1[:,:22])).mean()

    
        loss = self.mseloss(pred_output_cam0.joints[:,:22], gt_joints0[:,:22]) + \
                self.mseloss(pred_output_cam1.joints[:,:22], gt_joints1[:,:22]) + \
                    self.mseloss(pred_output_cam0.joints[:,:22],pred_output_cam1.joints[:,:22])
        loss[:,[4,5,18,19]] *= self.hparams.limbs3d_loss_weight
        loss[:,[7,8,20,21]] *= self.hparams.limbs3d_loss_weight**2
        loss_keypoints_3d = loss.mean()

        loss_regr_shape = self.mseloss(pred_output_cam0.vertices, gt_vertices0).mean() + \
                            self.mseloss(pred_output_cam1.vertices, gt_vertices1).mean() + \
                                self.mseloss(pred_output_cam0.vertices, pred_output_cam1.vertices).mean()

        loss_regr_trans = self.mseloss(pred_smpltrans0, gt_smpltrans_rel0).mean() + \
                            self.mseloss(pred_smpltrans1, gt_smpltrans_rel1).mean()
        
        loss_rootrot = self.mseloss(pred_rotmat0[:,:1], gt_smplorient_rel0).mean() + \
                        self.mseloss(pred_rotmat1[:,:1], gt_smplorient_rel1).mean()

        loss_rotmat = self.mseloss(pred_rotmat0[:,1:], gt_smplpose_rotmat) + \
                        self.mseloss(pred_rotmat1[:,1:], gt_smplpose_rotmat) + \
                            self.mseloss(pred_rotmat0[:,1:], pred_rotmat1[:,1:])

        # one index less than actual limbs because the root joint is not there in rotmats
        loss_rotmat[:,[3,4,17,18]] *= self.hparams.limbstheta_loss_weight
        loss_rotmat[:,[6,7,19,20]] *= self.hparams.limbstheta_loss_weight**2
        loss_regr_pose =  loss_rotmat.mean()
        
        loss_regul_betas = torch.mul(pred_betas0,pred_betas0).mean() + \
                            torch.mul(pred_betas1,pred_betas1).mean() + \
                                self.mseloss(pred_betas0,pred_betas1).mean()
        
        # Compute total loss
        loss = self.hparams.trans_loss_weight * loss_regr_trans + \
                self.hparams.keypoint2d_loss_weight * loss_keypoints + \
                self.hparams.keypoint3d_loss_weight * loss_keypoints_3d + \
                self.hparams.shape_loss_weight * loss_regr_shape + \
                self.hparams.rootrot_loss_weight * loss_rootrot + \
                self.hparams.pose_loss_weight * loss_regr_pose + \
                self.hparams.beta_loss_weight * loss_regul_betas

        loss *= 60

        losses = {'loss': loss.detach().item(),
                  'loss_regr_trans': loss_regr_trans.detach().item(),
                  'loss_keypoints': loss_keypoints.detach().item(),
                  'loss_keypoints_3d': loss_keypoints_3d.detach().item(),
                  'loss_regr_shape': loss_regr_shape.detach().item(),
                  'loss_rootrot': loss_rootrot.detach().item(),
                  'loss_regr_pose': loss_regr_pose.detach().item(),
                  'loss_regul_betas': loss_regul_betas.detach().item()}

        return loss, losses


    def fwd_pass_and_loss(self,input_batch,is_val=False,is_test=False):
        
        with torch.no_grad():
            # Get data from the batch
            im0 = input_batch['im0'].float() # input image
            im1 = input_batch['im1'].float() # input image
            gt_smpltrans_rel0 = input_batch['smpltrans_rel0'] # SMPL trans parameters
            gt_smpltrans_rel1 = input_batch['smpltrans_rel1'] # SMPL trans parameters
            bb0 = input_batch['bb0']
            bb1 = input_batch['bb1']
            intr0 = input_batch['intr0']
            intr1 = input_batch['intr1']
            
        
            batch_size = im0.shape[0]
            
            if is_test and (self.hparams.testdata.lower() == "aircapdata") :
                in_smpltrans0 = gt_smpltrans_rel0
                in_smpltrans1 = gt_smpltrans_rel1
            elif self.hparams.smpltrans_noise_sigma is None:
                in_smpltrans0 = torch.from_numpy(np.array([0,0,10])).float().expand(batch_size, -1).clone().type_as(gt_smpltrans_rel0)
                in_smpltrans1 = torch.from_numpy(np.array([0,0,10])).float().expand(batch_size, -1).clone().type_as(gt_smpltrans_rel1)
            else:
                in_smpltrans0, _ = add_noise_input_smpltrans(gt_smpltrans_rel0,self.hparams.smpltrans_noise_sigma)
                in_smpltrans1, _ = add_noise_input_smpltrans(gt_smpltrans_rel1,self.hparams.smpltrans_noise_sigma)

            # noisy input pose
            # gt_theta0 = torch.cat([gt_smplorient_rel0,gt_smplpose_rotmat],dim=1)[:,:,:,:2] 
            # init_theta0 = gt_theta0 + self.hparams.theta_noise_sigma * torch.randn(batch_size,22,3,2).type_as(gt_smplorient_rel0)
            # init_theta0 = init_theta0.reshape(batch_size,22*6)
            # gt_theta1 = torch.cat([gt_smplorient_rel1,gt_smplpose_rotmat],dim=1)[:,:,:,:2] 
            # init_theta1 = gt_theta1 + self.hparams.theta_noise_sigma * torch.randn(batch_size,22,3,2).type_as(gt_smplorient_rel1)
            # init_theta1 = init_theta1.reshape(batch_size,22*6)

            # distance scaling
            distance_scaling = True
            if distance_scaling:
                trans_scale = 0.05
                in_smpltrans0 *= trans_scale
                in_smpltrans1 *= trans_scale
        
        pred_pose0, pred_betas0, pred_pose1, pred_betas1,  = self.forward(x0 = im0,
                                                            x1 = im1,
                                                            bb0 = bb0,
                                                            bb1 = bb1,
                                                            init_position0 = in_smpltrans0,
                                                            init_position1 = in_smpltrans1,
                                                            iters = self.hparams.reg_iters)
                                                                        

        pred_smpltrans0 = pred_pose0[:,:3]
        pred_smpltrans1 = pred_pose1[:,:3]
        if distance_scaling:
            pred_smpltrans0 /= trans_scale
            pred_smpltrans1 /= trans_scale
            in_smpltrans0 /= trans_scale
            in_smpltrans1 /= trans_scale
        # import ipdb; ipdb.set_trace()
        pred_rotmat0 = rot6d_to_rotmat(pred_pose0[:,3:]).view(batch_size, 22, 3, 3)
        pred_rotmat1 = rot6d_to_rotmat(pred_pose1[:,3:]).view(batch_size, 22, 3, 3)
        
        # # Sanity check
        # #################################
        # gt_smplpose_rotmat = input_batch['smplpose_rotmat'] # SMPL pose rotation matrices
        # gt_smplorient_rel0 = input_batch['smplorient_rel0'] # SMPL orientation parameters
        # gt_smplorient_rel1 = input_batch['smplorient_rel1'] # SMPL orientation parameters
        # pred_smpltrans0 = gt_smpltrans_rel0
        # pred_smpltrans1 = gt_smpltrans_rel1
        # pred_rotmat0 = torch.cat([gt_smplorient_rel0,gt_smplpose_rotmat],dim=1).view(batch_size,22,3,3)
        # pred_rotmat1 = torch.cat([gt_smplorient_rel1,gt_smplpose_rotmat],dim=1).view(batch_size,22,3,3)
        # #################################
        
        if is_val or is_test:
            pred_output_cam0 = smplx_test.forward(betas=pred_betas0, 
                                    body_pose=pred_rotmat0[:,1:],
                                    global_orient=torch.eye(3,device=self.device).float().unsqueeze(0).repeat(batch_size,1,1).unsqueeze(1),
                                    transl = torch.zeros(batch_size,3).float().type_as(pred_betas0),
                                    pose2rot=False)
            transf_mat0 = torch.cat([pred_rotmat0[:,:1].squeeze(1),
                                pred_smpltrans0.unsqueeze(2)],dim=2)
            pred_vertices_cam0,pred_joints_cam0,_,_ = transform_smpl(transf_mat0,
                                                pred_output_cam0.vertices.squeeze(1),
                                                pred_output_cam0.joints.squeeze(1))

            pred_output_cam1 = smplx_test.forward(betas=pred_betas1, 
                                    body_pose=pred_rotmat1[:,1:],
                                    global_orient=torch.eye(3,device=self.device).float().unsqueeze(0).repeat(batch_size,1,1).unsqueeze(1),
                                    transl = torch.zeros(batch_size,3).float().type_as(pred_betas0),
                                    pose2rot=False)
            transf_mat1 = torch.cat([pred_rotmat1[:,:1].squeeze(1),
                                pred_smpltrans1.unsqueeze(2)],dim=2)
            pred_vertices_cam1,pred_joints_cam1,_,_ = transform_smpl(transf_mat1,
                                                pred_output_cam1.vertices.squeeze(1),
                                                pred_output_cam1.joints.squeeze(1))
            if is_test:
                pred_output_cam_in0 = smplx_test.forward(betas=torch.zeros(batch_size,10).float().type_as(pred_betas0), 
                                        body_pose=pred_rotmat0[:,1:],
                                        global_orient=torch.eye(3,device=self.device).float().unsqueeze(0).repeat(batch_size,1,1).unsqueeze(1),
                                        transl = torch.zeros(batch_size,3).float().type_as(pred_betas0),
                                        pose2rot=False)
                transf_mat_in0 = torch.cat([torch.eye(3,device=self.device).float().unsqueeze(0).repeat(batch_size,1,1),
                                    in_smpltrans0.unsqueeze(2)],dim=2)
                pred_vertices_cam_in0,_,_,_ = transform_smpl(transf_mat_in0,
                                                    pred_output_cam_in0.vertices.squeeze(1),
                                                    pred_output_cam_in0.joints.squeeze(1))

                pred_output_cam_in1 = smplx_test.forward(betas=torch.zeros(batch_size,10).float().type_as(pred_betas1), 
                                        body_pose=pred_rotmat1[:,1:],
                                        global_orient=torch.eye(3,device=self.device).float().unsqueeze(0).repeat(batch_size,1,1).unsqueeze(1),
                                        transl = torch.zeros(batch_size,3).float().type_as(pred_betas1),
                                        pose2rot=False)
                transf_mat_in1 = torch.cat([torch.eye(3,device=self.device).float().unsqueeze(0).repeat(batch_size,1,1),
                                    in_smpltrans1.unsqueeze(2)],dim=2)
                pred_vertices_cam_in1,_,_,_ = transform_smpl(transf_mat_in1,
                                                    pred_output_cam_in1.vertices.squeeze(1),
                                                    pred_output_cam_in1.joints.squeeze(1))
        else:
            pred_output_cam0 = smplx.forward(betas=pred_betas0, 
                                body_pose=pred_rotmat0[:,1:],
                                global_orient=torch.eye(3,device=self.device).float().unsqueeze(0).repeat(batch_size,1,1).unsqueeze(1),
                                transl = torch.zeros(batch_size,3).float().type_as(pred_betas0),
                                pose2rot=False)

            transf_mat0 = torch.cat([pred_rotmat0[:,:1].squeeze(1),
                                pred_smpltrans0.unsqueeze(2)],dim=2)

            pred_vertices_cam0,pred_joints_cam0,_,_ = transform_smpl(transf_mat0,
                                                pred_output_cam0.vertices.squeeze(1),
                                                pred_output_cam0.joints.squeeze(1))
            
            pred_output_cam1 = smplx.forward(betas=pred_betas1, 
                                body_pose=pred_rotmat1[:,1:],
                                global_orient=torch.eye(3,device=self.device).float().unsqueeze(0).repeat(batch_size,1,1).unsqueeze(1),
                                transl = torch.zeros(batch_size,3).float().type_as(pred_betas1),
                                pose2rot=False)

            transf_mat1 = torch.cat([pred_rotmat1[:,:1].squeeze(1),
                                pred_smpltrans1.unsqueeze(2)],dim=2)

            pred_vertices_cam1,pred_joints_cam1,_,_ = transform_smpl(transf_mat1,
                                                pred_output_cam1.vertices.squeeze(1),
                                                pred_output_cam1.joints.squeeze(1))
        
        pred_joints_2d_cam0 = perspective_projection(pred_joints_cam0,
                                                   rotation=torch.eye(3).float().unsqueeze(0).repeat(batch_size,1,1).type_as(pred_betas0),
                                                   translation=torch.zeros(batch_size, 3).type_as(pred_betas0),
                                                   focal_length=self.focal_length,
                                                   camera_center=intr0[:,:2,2].unsqueeze(0))

        pred_joints_2d_cam1 = perspective_projection(pred_joints_cam1,
                                                   rotation=torch.eye(3).float().unsqueeze(0).repeat(batch_size,1,1).type_as(pred_betas1),
                                                   translation=torch.zeros(batch_size, 3).type_as(pred_betas1),
                                                   focal_length=self.focal_length,
                                                   camera_center=intr1[:,:2,2].unsqueeze(0))
        
        
        if is_test:
            loss, losses = None, None
                
            pred_angles0 = tgm.rotation_matrix_to_angle_axis(torch.cat([pred_rotmat0,torch.zeros(batch_size,22,3,1).float().type_as(pred_betas0)],dim=3).view(-1,3,4)).view(batch_size,22,3)
            pred_angles1 = tgm.rotation_matrix_to_angle_axis(torch.cat([pred_rotmat1,torch.zeros(batch_size,22,3,1).float().type_as(pred_betas0)],dim=3).view(-1,3,4)).view(batch_size,22,3)
            gt_angles0 = tgm.rotation_matrix_to_angle_axis(torch.cat([torch.cat([input_batch['smplorient_rel0'],input_batch['smplpose_rotmat']],dim=1),torch.zeros(batch_size,22,3,1).float().type_as(pred_betas0)],dim=3).view(-1,3,4)).view(batch_size,22,3)
            gt_angles1 = tgm.rotation_matrix_to_angle_axis(torch.cat([torch.cat([input_batch['smplorient_rel1'],input_batch['smplpose_rotmat']],dim=1),torch.zeros(batch_size,22,3,1).float().type_as(pred_betas1)],dim=3).view(-1,3,4)).view(batch_size,22,3)

            output = {'pred_vertices_cam0': pred_vertices_cam0.detach().cpu(),
                        'pred_vertices_cam1': pred_vertices_cam1.detach().cpu(),
                        "pred_vertices_cam_in0": pred_vertices_cam_in0.detach().cpu(),
                        "pred_vertices_cam_in1": pred_vertices_cam_in1.detach().cpu(),
                        "pred_j2d_cam0": pred_joints_2d_cam0.detach().cpu(),
                        "pred_j2d_cam1": pred_joints_2d_cam1.detach().cpu(),
                        "pred_j3d_cam0": pred_joints_cam0.detach().cpu(),
                        "pred_j3d_cam1": pred_joints_cam1.detach().cpu(),
                        'pred_smpltrans0': pred_smpltrans0.detach().cpu(),
                        'pred_smpltrans1': pred_smpltrans1.detach().cpu(),
                        "pred_angles0": pred_angles0.detach().cpu(),
                        "pred_angles1": pred_angles1.detach().cpu(),
                        "pred_betas0": pred_betas0.detach().cpu(),
                        "pred_betas1": pred_betas1.detach().cpu(),
                        'in_smpltrans0': in_smpltrans0.detach().cpu(),
                        'in_smpltrans1': in_smpltrans1.detach().cpu(),
                        "gt_angles0": gt_angles0.detach().cpu(),
                        "gt_angles1": gt_angles1.detach().cpu(),
                        'gt_smpltrans0': gt_smpltrans_rel0.detach().cpu(),
                        'gt_smpltrans1': gt_smpltrans_rel1.detach().cpu(),
                        'smplorient_rel0': input_batch['smplorient_rel0'].detach().cpu(),
                        'smplorient_rel1': input_batch['smplorient_rel1'].detach().cpu(),
                        'smplpose_rotmat': input_batch['smplpose_rotmat'].detach().cpu()}
        else:
            loss, losses = self.get_loss(input_batch,
                                pred_smpltrans0,
                                pred_smpltrans1,
                                pred_rotmat0,
                                pred_rotmat1,
                                pred_betas0,
                                pred_betas1,
                                pred_output_cam0,
                                pred_output_cam1,
                                pred_joints_2d_cam0,
                                pred_joints_2d_cam1)
            # Pack output arguments for tensorboard logging
            output = {'pred_vertices_cam0': pred_vertices_cam0.detach(),
                        'pred_vertices_cam1': pred_vertices_cam1.detach(),
                        'pred_smpltrans0': pred_smpltrans0.detach(),
                        'pred_smpltrans1': pred_smpltrans1.detach(),
                        'in_smpltrans0': in_smpltrans0.detach(),
                        'in_smpltrans1': in_smpltrans1.detach(),
                        'gt_smpltrans0': gt_smpltrans_rel0.detach(),
                        'gt_smpltrans1': gt_smpltrans_rel1.detach()}
        

        return output, losses, loss

    def training_step(self, batch, batch_idx):
        
        
        output, losses, loss = self.fwd_pass_and_loss(batch,is_val=False, is_test=False)

        with torch.no_grad():
        # logging
            if batch_idx % self.hparams.summary_steps == 0:
                train_summ_pred_image, train_summ_in_image = self.summaries(batch, output, losses, is_test=False)
                self.logger.experiment.add_image('train_pred_shape_cam', train_summ_pred_image, self.global_step)
                self.logger.experiment.add_image('train_input_images',train_summ_in_image, self.global_step)
                for loss_name, val in losses.items():
                    self.logger.experiment.add_scalar(loss_name + '/train', val, self.global_step)

        return {"loss" : loss}

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        with torch.no_grad():
            output, losses, loss = self.fwd_pass_and_loss(batch,is_val=True, is_test=False)

            if batch_idx % self.hparams.val_summary_steps == 0:
                val_summ_pred_image, val_summ_in_image = self.summaries(batch, output, losses, is_test=False)

                # logging
                self.logger.experiment.add_image('val_pred_shape_cam', val_summ_pred_image, self.global_step)
                self.logger.experiment.add_image('val_input_images',val_summ_in_image, self.global_step)
        
        return {'val_losses': losses,"val_loss":loss}

    def validation_epoch_end(self, outputs):
        for loss_name, val in outputs[0]["val_losses"].items():
            val_list = []
            for x in outputs:
                val_list.append(x["val_losses"][loss_name])
            mean_val = np.mean(val_list)
            self.logger.experiment.add_scalar(loss_name + '/val', mean_val, self.global_step)
        self.log("val_loss", np.mean([x["val_loss"].cpu().numpy() for x in outputs]))
        return {"val_loss":np.mean([x["val_loss"].cpu().numpy() for x in outputs])}

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        optimizer = torch.optim.Adam(self.model.parameters(), 
                                lr=self.hparams.lr,
                                weight_decay=0,
                                amsgrad=True)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

        return optimizer#, [scheduler]

    def train_dataloader(self):
        # REQUIRED
        train_dset, _ = aerialpeople.get_aerialpeople_seqsplit(self.hparams.datapath)
        return DataLoader(train_dset, batch_size=self.hparams.batch_size,
                            num_workers=self.hparams.num_workers,
                            pin_memory=self.hparams.pin_memory,
                            shuffle=self.hparams.shuffle_train,
                            drop_last=True)

    def val_dataloader(self):
        # OPTIONAL
        _, val_dset = aerialpeople.get_aerialpeople_seqsplit(self.hparams.datapath)
        return DataLoader(val_dset, batch_size=self.hparams.val_batch_size,
                            num_workers=self.hparams.num_workers,
                            pin_memory=self.hparams.pin_memory,
                            shuffle=self.hparams.shuffle_train,
                            drop_last=True)

    def summaries(self, input_batch,output, losses, is_test):
        batch_size = input_batch['im0'].shape[0]
        skip_factor = 4    # number of samples to be logged
        img_downsize_factor = 5
        
        
        with torch.no_grad():
            im0 = input_batch['im0'][::int(batch_size/skip_factor)] * torch.tensor([0.229, 0.224, 0.225], device=self.device).reshape(1,3,1,1)
            im0 = im0 + torch.tensor([0.485, 0.456, 0.406], device=self.device).reshape(1,3,1,1)
            images0 = []
            for i in range(0,batch_size,int(batch_size/skip_factor)):
                blank_img = torch.zeros(3,1080,1920).float()
                blank_img[:,input_batch["crop_info0"][i,0,0]:input_batch["crop_info0"][i,1,0],
                    input_batch["crop_info0"][i,0,1]:input_batch["crop_info0"][i,1,1]] = torch.from_numpy(cv2.imread(input_batch['im0_path'][i])[:,:,::-1]/255.).float().permute(2,0,1)
                images0.append(blank_img)
            images0 = torch.stack(images0)
            pred_vertices_cam0 = output['pred_vertices_cam0'][::int(batch_size/skip_factor)]
            # pred_vertices_cam = input_batch['smpl_vertices_rel'][::int(batch_size/skip_factor)].squeeze(1)
            
            images_pred_cam0 = self.renderer.visualize_tb(pred_vertices_cam0,
                                                        torch.zeros(batch_size,3,device=self._device).float(),
                                                        torch.eye(3,device=self._device).float().unsqueeze(0).repeat(batch_size,1,1),
                                                        images0)
            
            summ_pred_image0 = images_pred_cam0[:,::img_downsize_factor,::img_downsize_factor]
            
            summ_in_image0 = torchvision.utils.make_grid(im0)

            im1 = input_batch['im1'][::int(batch_size/skip_factor)] * torch.tensor([0.229, 0.224, 0.225], device=self.device).reshape(1,3,1,1)
            im1 = im1 + torch.tensor([0.485, 0.456, 0.406], device=self.device).reshape(1,3,1,1)

            images1= []
            for i in range(0,batch_size,int(batch_size/skip_factor)):
                blank_img = torch.zeros(3,1080,1920).float()
                blank_img[:,input_batch["crop_info1"][i,0,0]:input_batch["crop_info1"][i,1,0],
                    input_batch["crop_info1"][i,0,1]:input_batch["crop_info1"][i,1,1]] = torch.from_numpy(cv2.imread(input_batch['im1_path'][i])[:,:,::-1]/255.).float().permute(2,0,1)
                images1.append(blank_img)
            images1 = torch.stack(images1)
            pred_vertices_cam1 = output['pred_vertices_cam1'][::int(batch_size/skip_factor)]
            
            
            images_pred_cam1 = self.renderer.visualize_tb(pred_vertices_cam1,
                                                        torch.zeros(batch_size,3,device=self._device).float(),
                                                        torch.eye(3,device=self._device).float().unsqueeze(0).repeat(batch_size,1,1),
                                                        images1)
            
            summ_pred_image1 = images_pred_cam1[:,::img_downsize_factor,::img_downsize_factor]
            
            summ_in_image1 = torchvision.utils.make_grid(im1)

            summ_in_image = torch.cat((summ_in_image0, summ_in_image1),1)
            summ_pred_image = torch.cat((summ_pred_image0, summ_pred_image1),1)
            # import ipdb; ipdb.set_trace()
            if is_test and self.hparams.testdata.lower() == "aircapdata":
                import ipdb; ipdb.set_trace()
            
            return summ_pred_image, summ_in_image


    def test_dataloader(self):
        # OPTIONAL
        if self.hparams.testdata.lower() == "aircapdata":
            aircap_dset = aircapData.aircapData_crop()
            return DataLoader(aircap_dset, batch_size=self.hparams.val_batch_size,
                                num_workers=self.hparams.num_workers,
                                pin_memory=self.hparams.pin_memory,
                                drop_last=False)
        else:
            train_dset, val_dset = aerialpeople.get_aerialpeople_seqsplit(self.hparams.datapath)
            train_dloader = DataLoader(train_dset, batch_size=self.hparams.val_batch_size,
                                        num_workers=self.hparams.num_workers,
                                        pin_memory=self.hparams.pin_memory,
                                        shuffle=False,
                                        drop_last=True)
            test_dloader = DataLoader(val_dset, batch_size=self.hparams.val_batch_size,
                                        num_workers=self.hparams.num_workers,
                                        pin_memory=self.hparams.pin_memory,
                                        shuffle=False,
                                        drop_last=True)

            return [test_dloader, train_dloader]


    def test_step(self, batch, batch_idx, dset_idx=0):
        # OPTIONAL
        output, losses, loss = self.fwd_pass_and_loss(batch,is_val=True,is_test=True)
        # self.viz_3d(batch,output)
        # if self.hparams.testdata.lower() == "aircapdata":
        #     train_summ_pred_image, train_summ_in_image = self.summaries(batch, output, losses, is_test=True)
        # cv2.imwrite(train_summ_pred_image.permute(1,2,0).data.numpy())
        
        return {"test_loss" : loss,
                "output" : output}

    def test_epoch_end(self, outputs):
        # OPTIONAL
        global smplx
        test_err_smpltrans0 = np.array([(x["output"]["pred_smpltrans0"] - 
            x["output"]["gt_smpltrans0"]).cpu().numpy() for x in outputs[0]]).reshape(-1,3)
        mean_test_err_smpltrans0 = np.mean(np.sqrt(np.sum(test_err_smpltrans0**2,1)))
        test_err_smpltrans1 = np.array([(x["output"]["pred_smpltrans1"] - 
            x["output"]["gt_smpltrans1"]).cpu().numpy() for x in outputs[0]]).reshape(-1,3)
        mean_test_err_smpltrans1 = np.mean(np.sqrt(np.sum(test_err_smpltrans1**2,1)))

        train_err_smpltrans0 = np.array([(x["output"]["pred_smpltrans0"] - 
            x["output"]["gt_smpltrans0"]).cpu().numpy() for x in outputs[1]]).reshape(-1,3)
        mean_train_err_smpltrans0 = np.mean(np.sqrt(np.sum(train_err_smpltrans0**2,1)))
        train_err_smpltrans1 = np.array([(x["output"]["pred_smpltrans1"] - 
            x["output"]["gt_smpltrans1"]).cpu().numpy() for x in outputs[1]]).reshape(-1,3)
        mean_train_err_smpltrans1 = np.mean(np.sqrt(np.sum(train_err_smpltrans1**2,1)))


        smplpose_rotmat = torch.stack([x["output"]["smplpose_rotmat"].to(device) for x in outputs[0]])
        smplorient_rel0 = torch.stack([x["output"]["smplorient_rel0"].to(device) for x in outputs[0]])  
        smplorient_rel1 = torch.stack([x["output"]["smplorient_rel1"].to(device) for x in outputs[0]])

        pred_angles0_test = torch.cat([x["output"]["pred_angles0"].to(device) for x in outputs[0]])
        pred_angles1_test = torch.cat([x["output"]["pred_angles1"].to(device) for x in outputs[0]])
        pred_rotmat0_test = tgm.angle_axis_to_rotation_matrix(pred_angles0_test.view(-1,3)).view(smplpose_rotmat.shape[0],smplpose_rotmat.shape[1],22,4,4)
        pred_rotmat1_test = tgm.angle_axis_to_rotation_matrix(pred_angles1_test.view(-1,3)).view(smplpose_rotmat.shape[0],smplpose_rotmat.shape[1],22,4,4)

        # pred_angles0_train = torch.cat([x["output"]["pred_angles0"].to(device) for x in outputs[1]])
        # pred_angles1_train = torch.cat([x["output"]["pred_angles1"].to(device) for x in outputs[1]])
        # pred_rotmat0_train = tgm.angle_axis_to_rotation_matrix(pred_angles0_train.view(-1,3)).view(pred_angles0_train.shape[0],22,4,4)
        # pred_rotmat1_train = tgm.angle_axis_to_rotation_matrix(pred_angles1_train.view(-1,3)).view(pred_angles1_train.shape[0],22,4,4)
    

        joints3d = []
        from tqdm import tqdm
        for i in tqdm(range(smplpose_rotmat.shape[0])):
            out_gt0 = smplx.forward(body_pose=smplpose_rotmat[i],
                                    global_orient= smplorient_rel0[i],pose2rot=False)
            out_gt1 = smplx.forward(body_pose=smplpose_rotmat[i],
                                    global_orient= smplorient_rel1[i],pose2rot=False)
            out_pred0 = smplx.forward(body_pose=pred_rotmat0_test[i,:,1:22,:3,:3],
                                    global_orient= pred_rotmat0_test[i,:,0:1,:3,:3],pose2rot=False)
            out_pred1 = smplx.forward(body_pose=pred_rotmat1_test[i,:,1:22,:3,:3],
                                    global_orient= pred_rotmat1_test[i,:,0:1,:3,:3],pose2rot=False)
            
            joints3d.append(np.stack([out_gt0.joints.detach().cpu().numpy(),
                        out_gt1.joints.detach().cpu().numpy(),
                        out_pred0.joints.detach().cpu().numpy(),
                        out_pred1.joints.detach().cpu().numpy()]).transpose(1,0,2,3))

        j3d = np.stack(joints3d).reshape(-1,4,127,3)[:,:,:22]
        print("test_mpjpe0: {}".format(np.mean(np.sqrt(np.sum((j3d[:,0] - j3d[:,2])**2,2))[:,:22])))
        print("test_mpjpe1: {}".format(np.mean(np.sqrt(np.sum((j3d[:,1] - j3d[:,3])**2,2))[:,:22])))
        print("test_mpe00: {}".format(mean_test_err_smpltrans0))
        print("test_mpe1: {}".format(mean_train_err_smpltrans0))

        # print("train_mpe00: {}".format(mean_test_err_smpltrans1))
        # print("train_mpjpe0: {}".format(mean_test_err_smplangles1))
        # print("train_mpe1: {}".format(mean_train_err_smpltrans1))
        # print("train_mpjpe1: {}".format(mean_train_err_smplangles1))

        # import ipdb; ipdb.set_trace()
        return {"outputs":outputs}


    def viz_3d(self,batch,output,viz_idx=[0]):
        import meshcat
        import meshcat.geometry as g

        if not hasattr(self,"vizualizer_3d"):
            self.vizualizer_3d = meshcat.Visualizer()

        import ipdb; ipdb.set_trace()
        pred_vertices0_wrt_origin,_,_,_ = transform_smpl(torch.inverse(batch["extr0"].squeeze(1)),output["pred_vertices_cam0"])
        pred_vertices1_wrt_origin,_,_,_ = transform_smpl(torch.inverse(batch["extr1"].squeeze(1)),output["pred_vertices_cam1"])

        for idx in viz_idx:
            self.vizualizer_3d["mesh0_"+str(idx)].set_object(g.TriangularMeshGeometry(pred_vertices0_wrt_origin.cpu().numpy()[idx],smplx.faces),
                            g.MeshLambertMaterial(
                                color=0xff22dd,
                                reflectivity=0.8))

            self.vizualizer_3d["mesh1_"+str(idx)].set_object(g.TriangularMeshGeometry(pred_vertices1_wrt_origin.cpu().numpy()[idx],smplx.faces),
                            g.MeshLambertMaterial(
                                color=0xff22dd,
                                reflectivity=0.8))
        
        import ipdb;ipdb.set_trace()


    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        req = parser.add_argument_group('Required')
        req.add_argument('--name', required=True, help='Name of the experiment')
        req.add_argument('--version', required=True, help='Version of the experiment')

        gen = parser.add_argument_group('General')
        gen.add_argument('--time_to_run', type=int, default=np.inf, help='Total time to run in seconds. Used for training in environments with timing constraints')
        gen.add_argument('--num_workers', type=int, default=30, help='Number of processes used for data loading')
        pin = gen.add_mutually_exclusive_group()
        pin.add_argument('--pin_memory', dest='pin_memory', action='store_true')
        pin.add_argument('--no_pin_memory', dest='pin_memory', action='store_false')
        gen.set_defaults(pin_memory=True)

        train = parser.add_argument_group('Training Options')
        train.add_argument('--datapath', type=str, default="/home/nsaini/Datasets/AerialPeople/agora_copenet_uniform_new_cropped/", help='Path to the dataset')
        train.add_argument('--model', type=str, default=None, required=True, help='model type')
        train.add_argument('--copenet_home', type=str, default="/is/ps3/nsaini/projects/copenet", help='copenet repo home')
        train.add_argument('--log_dir', default='/is/ps3/nsaini/projects/copenet/airpose_logs', help='Directory to store logs')
        train.add_argument('--testdata', type=str, default="aerialpeople", help='test dataset')
        train.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
        train.add_argument('--batch_size', type=int, default=30, help='Batch size')
        train.add_argument('--val_batch_size', type=int, default=30, help='Validation data batch size')
        train.add_argument('--summary_steps', type=int, default=500, help='Summary saving frequency')
        train.add_argument('--val_summary_steps', type=float, default=50, help='validation summary frequency')
        train.add_argument('--checkpoint_steps', type=int, default=10000, help='Checkpoint saving frequency')
        train.add_argument('--img_res', type=int, default=224, help='Rescale bounding boxes to size [img_res, img_res] before feeding them in the network') 
        train.add_argument('--shape_loss_weight', default=50, type=float, help='Weight of per-vertex loss') 
        train.add_argument('--keypoint2d_loss_weight', default=0.002, type=float, help='Weight of 2D keypoint loss')
        train.add_argument('--keypoint3d_loss_weight', default=1, type=float, help='Weight of 3D keypoint loss')
        train.add_argument('--limbs3d_loss_weight', default=3., type=float, help='Weight of limbs 3D keypoint loss')
        train.add_argument('--limbstheta_loss_weight', default=1., type=float, help='Weight of limbs rotation angle loss')
        train.add_argument('--cam_noise_sigma', default=[0,0], type=list, help='noise sigma to add to gt cams (trans and rot)')
        train.add_argument('--smpltrans_noise_sigma', default=None, help='noise sigma to add to smpltrans')
        train.add_argument('--theta_noise_sigma', default=0.2, type=float, help='noise sigma to add to smpl thetas')
        train.add_argument('--trans_loss_weight', default=10, type=float, help='Weight of SMPL translation loss') 
        train.add_argument('--rootrot_loss_weight', default=1, type=float, help='Weight of SMPL root rotation loss') 
        train.add_argument('--pose_loss_weight', default=50, type=float, help='Weight of SMPL pose loss') 
        train.add_argument('--beta_loss_weight', default=1, type=float, help='Weight of SMPL betas loss')
        train.add_argument('--cams_loss_weight', default=1, type=float, help='Weight of cams params loss')
        train.add_argument('--reg_iters', default=3, type=int, help='number of regressor iterations')
        train.add_argument('--gt_train_weight', default=1., help='Weight for GT keypoints during training')
        train.add_argument('--train_reg_only_epochs', default=-1, help='number of epochs the regressor only part to be initially trained')  

        shuffle_train = train.add_mutually_exclusive_group()
        shuffle_train.add_argument('--shuffle_train', dest='shuffle_train', action='store_true', help='Shuffle training data')
        shuffle_train.add_argument('--no_shuffle_train', dest='shuffle_train', action='store_false', help='Don\'t shuffle training data')
        shuffle_train.set_defaults(shuffle_train=True)

        return parser

