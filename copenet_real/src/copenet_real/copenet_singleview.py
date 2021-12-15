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
from .models import model_copenet_singleview
from .dsets import aerialpeople, copenet_real
import cv2
import torchvision
from .smplx.smplx import SMPLX, lbs
import pickle as pk
from .utils.renderer import Renderer
from . import constants as CONSTANTS
from .utils.utils import transform_smpl, add_noise_input_cams,add_noise_input_smpltrans
from .utils.geometry import batch_rodrigues, perspective_projection, estimate_translation, rot6d_to_rotmat

import pytorch_lightning as pl

from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser

vp_model = load_model("/ps/scratch/common/vposer/V02_05", model_code=VPoser,remove_words_in_model_weights="vp_model.")[0]

smplx = None
smplx_test = None


def create_smplx(copenet_home,train_batch_size,val_batch_size):
    global smplx 
    smplx = SMPLX(os.path.join(copenet_home,"src/copenet/data/smplx/models/smplx"),
                         batch_size=train_batch_size,
                         create_transl=False)

    global smplx_test 
    smplx_test = SMPLX(os.path.join(copenet_home,"src/copenet/data/smplx/models/smplx"),
                         batch_size=train_batch_size,
                         create_transl=False)

class copenet_singleview(pl.LightningModule):

    def __init__(self, hparams):
        super(copenet_singleview, self).__init__()
        # not the best model...
        self.hparams = hparams
        self.model = model_copenet_singleview.getcopenet(os.path.join(self.hparams.copenet_home,"src/copenet/data/smpl_mean_params.npz"))

        create_smplx(self.hparams.copenet_home,self.hparams.batch_size,self.hparams.val_batch_size)
        
        smplx.to("cuda")
        smplx_test.to("cuda")
        vp_model.to("cuda")
        
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
        

    def get_loss(self,input_batch, pred_smpltrans, pred_rotmat, pred_betas, pred_output_cam, pred_joints_2d_cam):

        gt_smplpose_rotmat = input_batch['smplpose_rotmat'] # SMPL pose rotation matrices
        gt_smpltrans_rel = input_batch['smpltrans_rel0'] # SMPL trans parameters
        gt_smplorient_rel = input_batch['smplorient_rel0'] # SMPL orientation parameters
        gt_vertices = input_batch['smpl_vertices'].squeeze(1)
        gt_joints = input_batch['smpl_joints'].squeeze(1)
        gt_joints_2d_cam = input_batch['smpl_joints_2d0'].squeeze(1)
        
        loss_keypoints = (self.mseloss(pred_joints_2d_cam[:,:22], gt_joints_2d_cam[:,:22])).mean()

    
        loss = self.mseloss(pred_output_cam.joints[:,:22], gt_joints[:,:22])
        loss[:,[4,5,18,19]] *= self.hparams.limbs3d_loss_weight
        loss[:,[7,8,20,21]] *= self.hparams.limbs3d_loss_weight**2
        loss_keypoints_3d = loss.mean()

        loss_regr_shape = self.mseloss(pred_output_cam.vertices, gt_vertices).mean()

        loss_regr_trans = self.mseloss(pred_smpltrans, gt_smpltrans_rel).mean()
        
        loss_rootrot = self.mseloss(pred_rotmat[:,:1], gt_smplorient_rel).mean()

        loss_rotmat = self.mseloss(pred_rotmat[:,1:], gt_smplpose_rotmat)
        # one index less than actual limbs because the root joint is not there in rotmats
        loss_rotmat[:,[3,4,17,18]] *= self.hparams.limbstheta_loss_weight
        loss_rotmat[:,[6,7,19,20]] *= self.hparams.limbstheta_loss_weight**2
        loss_regr_pose =  loss_rotmat.mean()
        
        loss_regul_betas = torch.mul(pred_betas,pred_betas).mean()
        
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
            im = input_batch['im0'].float() # input image
            bb = input_batch['bb0']
            intr = input_batch['intr0']
            
        
            batch_size = im.shape[0]
            
            if is_test and self.hparams.testdata.lower() == "aircapdata":
                in_smpltrans = gt_smpltrans_rel
            else:
                in_smpltrans = torch.from_numpy(np.array([0,0,10])).float().expand(batch_size, -1).type_as(bb)
            # else:
            #     in_smpltrans, _ = add_noise_input_smpltrans(gt_smpltrans_rel,self.hparams.smpltrans_noise_sigma)

            # noisy input pose
            # gt_theta = torch.cat([gt_smplorient_rel,gt_smplpose_rotmat],dim=1)[:,:,:,:2] 
            # init_theta = gt_theta + self.hparams.theta_noise_sigma * torch.randn(batch_size,22,3,2).type_as(gt_smplorient_rel)
            # init_theta = init_theta.reshape(batch_size,22*6)

            # distance scaling
            distance_scaling = True
            if distance_scaling:
                trans_scale = 0.05
                in_smpltrans *= trans_scale
        
        pred_pose, pred_betas = self.model.forward(x = im,
                                                    bb = bb,
                                                    init_position = in_smpltrans,
                                                    iters = self.hparams.reg_iters)
                                                                        

        pred_smpltrans = pred_pose[:,:3]
        if distance_scaling:
            pred_smpltrans /= trans_scale
            in_smpltrans /= trans_scale
        pred_rotmat = rot6d_to_rotmat(pred_pose[:,3:]).view(batch_size, 22, 3, 3)

        # # Sanity check
        # #################################
        # pred_smpltrans = gt_smpltrans_rel
        # pred_rotmat = torch.cat([gt_smplorient_rel,gt_smplpose_rotmat],dim=1).view(batch_size,22,3,3)
        # #################################

        # import ipdb; ipdb.set_trace()
        if is_val or is_test:
            pred_output_cam = smplx_test.forward(betas=pred_betas, 
                                    body_pose=pred_rotmat[:,1:],
                                    global_orient=torch.eye(3,device=self.device).float().unsqueeze(0).repeat(batch_size,1,1).unsqueeze(1),
                                    transl = torch.zeros(batch_size,3).float().type_as(pred_betas),
                                    pose2rot=False)
            transf_mat = torch.cat([pred_rotmat[:,:1].squeeze(1),
                                pred_smpltrans.unsqueeze(2)],dim=2)
            pred_vertices_cam,pred_joints_cam,_,_ = transform_smpl(transf_mat,
                                                pred_output_cam.vertices.squeeze(1),
                                                pred_output_cam.joints.squeeze(1))
            if is_test:
                pred_output_cam_in = smplx_test.forward(betas=torch.zeros(batch_size,10).float().type_as(pred_betas), 
                                        body_pose=pred_rotmat[:,1:],
                                        global_orient=pred_rotmat[:,:1],
                                        transl = torch.zeros(batch_size,3).float().type_as(pred_betas),
                                        pose2rot=False)
                transf_mat_in = torch.cat([torch.eye(3,device=self.device).float().unsqueeze(0).repeat(batch_size,1,1),
                                    in_smpltrans.unsqueeze(2)],dim=2)
                pred_vertices_cam_in,_,_,_ = transform_smpl(transf_mat_in,
                                                    pred_output_cam_in.vertices.squeeze(1),
                                                    pred_output_cam_in.joints.squeeze(1))
        else:
            pred_output_cam = smplx.forward(betas=pred_betas, 
                                body_pose=pred_rotmat[:,1:],
                                global_orient=torch.eye(3,device=self.device).float().unsqueeze(0).repeat(batch_size,1,1).unsqueeze(1),
                                transl = torch.zeros(batch_size,3).float().type_as(pred_betas),
                                pose2rot=False)

            transf_mat = torch.cat([pred_rotmat[:,:1].squeeze(1),
                                pred_smpltrans.unsqueeze(2)],dim=2)

            pred_vertices_cam,pred_joints_cam,_,_ = transform_smpl(transf_mat,
                                                pred_output_cam.vertices.squeeze(1),
                                                pred_output_cam.joints.squeeze(1))
        
        pred_joints_2d_cam = perspective_projection(pred_joints_cam,
                                                   rotation=torch.eye(3).float().unsqueeze(0).repeat(batch_size,1,1).type_as(pred_betas),
                                                   translation=torch.zeros(batch_size, 3).type_as(pred_betas),
                                                   focal_length=self.focal_length,
                                                   camera_center=intr[:,:2,2].unsqueeze(0))
        
        
        if is_test:
            loss, losses = None, None
            # Pack output arguments for tensorboard logging
            pred_angles = tgm.rotation_matrix_to_angle_axis(torch.cat([pred_rotmat,torch.zeros(batch_size,22,3,1).float().type_as(pred_betas)],dim=3).view(-1,3,4)).view(batch_size,22,3)

            output = {'pred_vertices_cam': pred_vertices_cam.detach(),
                        "pred_vertices_cam_in": pred_vertices_cam_in.detach(),
                        "pred_angles": pred_angles.detach(),
                        'pred_smpltrans': pred_smpltrans.detach(),
                        'in_smpltrans': in_smpltrans.detach()}
        else:
            loss, losses = self.get_loss(input_batch,
                                pred_smpltrans,
                                pred_rotmat,
                                pred_betas,
                                pred_output_cam,
                                pred_joints_2d_cam)
            # Pack output arguments for tensorboard logging
            output = {'pred_vertices_cam': pred_vertices_cam.detach(),
                        'pred_smpltrans': pred_smpltrans.detach(),
                        'in_smpltrans': in_smpltrans.detach()}
        

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
        return {"val_loss":outputs[0]["val_loss"]}

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
        train_dset, _ = copenet_real.get_copenet_real_traintest(self.hparams.datapath)
        return DataLoader(train_dset, batch_size=self.hparams.batch_size,
                            num_workers=self.hparams.num_workers,
                            pin_memory=self.hparams.pin_memory,
                            shuffle=self.hparams.shuffle_train,
                            drop_last=True)

    def val_dataloader(self):
        # OPTIONAL
        _, val_dset = copenet_real.get_copenet_real_traintest(self.hparams.datapath)
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
            im = input_batch['im0'][::int(batch_size/skip_factor)] * torch.tensor([0.229, 0.224, 0.225], device=self.device).reshape(1,3,1,1)
            im = im + torch.tensor([0.485, 0.456, 0.406], device=self.device).reshape(1,3,1,1)
            images = []
            
            for i in range(0,batch_size,int(batch_size/skip_factor)):
                # blank_img = torch.zeros(3,1080,1920).float()
                # blank_img[:,input_batch["crop_info0"][i,0,0]:input_batch["crop_info0"][i,1,0],
                #     input_batch["crop_info0"][i,0,1]:input_batch["crop_info0"][i,1,1]] = torch.from_numpy(cv2.imread(input_batch['im0_path'][i])[:,:,::-1]/255.).float().permute(2,0,1)
                blank_img = torch.from_numpy(cv2.imread(input_batch['im0_path'][i])[:,:,::-1]/255.).float().permute(2,0,1)
                images.append(blank_img)
            images = torch.stack(images)
            pred_vertices_cam = output['pred_vertices_cam'][::int(batch_size/skip_factor)]
            # pred_vertices_cam = input_batch['smpl_vertices_rel'][::int(batch_size/skip_factor)].squeeze(1)
            
            images_pred_cam = self.renderer.visualize_tb(pred_vertices_cam,
                                                        torch.zeros(batch_size,3,device=self._device).float(),
                                                        torch.eye(3,device=self._device).float().unsqueeze(0).repeat(batch_size,1,1),
                                                        images)
            
            summ_pred_image = images_pred_cam[:,::img_downsize_factor,::img_downsize_factor]
            
            summ_in_image = torchvision.utils.make_grid(im)
            
            if is_test:
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
            train_dset, val_dset = copenet_real.get_copenet_real_traintest(self.hparams.datapath)
            train_dloader = DataLoader(train_dset, batch_size=self.hparams.val_batch_size,
                                        num_workers=self.hparams.num_workers,
                                        pin_memory=self.hparams.pin_memory,
                                        shuffle=True,
                                        drop_last=True)
            test_dloader = DataLoader(val_dset, batch_size=self.hparams.val_batch_size,
                                        num_workers=self.hparams.num_workers,
                                        pin_memory=self.hparams.pin_memory,
                                        shuffle=True,
                                        drop_last=True)

            return [test_dloader, train_dloader]


    def test_step(self, batch, batch_idx, dset_idx=0):
        # OPTIONAL
        output, losses, loss = self.fwd_pass_and_loss(batch,is_val=True,is_test=True)
        # if self.hparams.testdata.lower() == "aircapdata":
        train_summ_pred_image, train_summ_in_image = self.summaries(batch, output, losses, is_test=True)
        # cv2.imwrite(train_summ_pred_image.permute(1,2,0).data.numpy())
        
        return {"test_loss" : loss,
                "output" : output}

    def test_epoch_end(self, outputs):
        # OPTIONAL
        test_err_smpltrans = np.array([(x["output"]["pred_smpltrans"] - 
            x["output"]["gt_smpltrans"]).cpu().numpy() for x in outputs[0]]).reshape(-1,3)
        test_err_smplangles = np.array([(x["output"]["pred_angles"] - 
            x["output"]["gt_angles"]).cpu().numpy() for x in outputs[0]]).reshape(-1,22,3)

        mean_test_err_smpltrans = np.mean(np.sqrt(np.sum(test_err_smpltrans**2,1)))
        mean_test_err_smplangles = np.mean(np.sqrt(np.sum(test_err_smplangles**2,2)))

        train_err_smpltrans = np.array([(x["output"]["pred_smpltrans"] - 
            x["output"]["gt_smpltrans"]).cpu().numpy() for x in outputs[1]]).reshape(-1,3)
        train_err_smplangles = np.array([(x["output"]["pred_angles"] - 
            x["output"]["gt_angles"]).cpu().numpy() for x in outputs[1]]).reshape(-1,22,3)

        mean_train_err_smpltrans = np.mean(np.sqrt(np.sum(train_err_smpltrans**2,1)))
        mean_train_err_smplangles = np.mean(np.sqrt(np.sum(train_err_smplangles**2,2)))

        import ipdb; ipdb.set_trace()
        return {"outputs":outputs}


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
        gen.add_argument('--num_workers', type=int, default=8, help='Number of processes used for data loading')
        pin = gen.add_mutually_exclusive_group()
        pin.add_argument('--pin_memory', dest='pin_memory', action='store_true')
        pin.add_argument('--no_pin_memory', dest='pin_memory', action='store_false')
        gen.set_defaults(pin_memory=True)

        train = parser.add_argument_group('Training Options')
        train.add_argument('--datapath', type=str, default=None, help='Path to the dataset')
        train.add_argument('--model', type=str, default=None, required=True, help='model type')
        train.add_argument('--copenet_home', type=str, required=True, help='copenet repo home')
        train.add_argument('--log_dir', default='/is/cluster/nsaini/copenet_logs', help='Directory to store logs')
        train.add_argument('--testdata', type=str, default="aerialpeople", help='test dataset')
        train.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
        train.add_argument('--batch_size', type=int, default=30, help='Batch size')
        train.add_argument('--val_batch_size', type=int, default=30, help='Validation data batch size')
        train.add_argument('--summary_steps', type=int, default=100, help='Summary saving frequency')
        train.add_argument('--val_summary_steps', type=float, default=10, help='validation summary frequency')
        train.add_argument('--checkpoint_steps', type=int, default=10000, help='Checkpoint saving frequency')
        train.add_argument('--img_res', type=int, default=224, help='Rescale bounding boxes to size [img_res, img_res] before feeding them in the network') 
        train.add_argument('--shape_loss_weight', default=1, type=float, help='Weight of per-vertex loss') 
        train.add_argument('--keypoint2d_loss_weight', default=0.001, type=float, help='Weight of 2D keypoint loss')
        train.add_argument('--keypoint3d_loss_weight', default=1, type=float, help='Weight of 3D keypoint loss')
        train.add_argument('--limbs3d_loss_weight', default=3., type=float, help='Weight of limbs 3D keypoint loss')
        train.add_argument('--limbstheta_loss_weight', default=3., type=float, help='Weight of limbs rotation angle loss')
        train.add_argument('--cam_noise_sigma', default=[0,0], type=list, help='noise sigma to add to gt cams (trans and rot)')
        train.add_argument('--smpltrans_noise_sigma', default=0.5, type=float, help='noise sigma to add to smpltrans')
        train.add_argument('--theta_noise_sigma', default=0.2, type=float, help='noise sigma to add to smpl thetas')
        train.add_argument('--trans_loss_weight', default=1, type=float, help='Weight of SMPL translation loss') 
        train.add_argument('--rootrot_loss_weight', default=1, type=float, help='Weight of SMPL root rotation loss') 
        train.add_argument('--pose_loss_weight', default=1, type=float, help='Weight of SMPL pose loss') 
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


