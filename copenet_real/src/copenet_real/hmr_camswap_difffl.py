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
from .models import model_hmr
from .dsets import aerialpeople, copenet_real
import cv2
import torchvision
from .smplx.smplx import SMPLX, lbs
import pickle as pk
from .utils.renderer import Renderer
from . import constants as CONSTANTS
from .utils.utils import transform_smpl, add_noise_input_cams,add_noise_input_smpltrans
from .utils.geometry import batch_rodrigues, perspective_projection, estimate_translation, rot6d_to_rotmat

from .config import device

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

class hmr(pl.LightningModule):

    def __init__(self, hparams):
        super(hmr, self).__init__()
        # not the best model...
        self.hparams = hparams
        self.model = model_hmr.getcopenet(os.path.join(self.hparams.copenet_home,"src/copenet/data/smpl_mean_params.npz"))

        create_smplx(self.hparams.copenet_home,self.hparams.batch_size,self.hparams.val_batch_size)
        
        smplx.to(device)
        smplx_test.to(device)
        vp_model.to(device)
        smplx_hand_idx = pk.load(open(os.path.join(self.hparams.copenet_home,"src/copenet/data/smplx/MANO_SMPLX_vertex_ids.pkl"),'rb'))
        smplx_face_idx = np.load(os.path.join(self.hparams.copenet_home,"src/copenet/data/smplx/SMPL-X__FLAME_vertex_ids.npy"))
        self.register_buffer("body_only_mask",torch.ones(smplx.v_template.shape[0],1))
        self.body_only_mask[smplx_hand_idx['left_hand'],:] = 0
        self.body_only_mask[smplx_hand_idx['right_hand'],:] = 0
        self.body_only_mask[smplx_face_idx,:] = 0
        
        self.mseloss = nn.MSELoss(reduction='none')


        # self.focal_length0 = [1537,1517]
        # self.focal_length1 = [1361,1378]
        self.focal_length0 = [5000,5000]
        self.focal_length1 = [5000,5000]
        self.renderer0 = Renderer(focal_length=self.focal_length0, 
                            img_res=[224,224],
                            center=[112,112],
                            faces=smplx.faces)
        self.renderer1 = Renderer(focal_length=self.focal_length1, 
                            img_res=[224,224],
                            center=[112,112],
                            faces=smplx.faces)

    def forward(self, **kwargs):
        return self.model(**kwargs)
        
    def get_loss(self,input_batch, pred_cam_t, pred_rotmat, pred_betas, pred_output_cam, pred_joints_2d_cam):
        
        gt_joints_2d_cam = input_batch['smpl_joints_2d_crop0'][:,0]
        

        batch_size = gt_joints_2d_cam.shape[0]
        
        loss_keypoints = (self.mseloss(pred_joints_2d_cam[:,:22], 
                            gt_joints_2d_cam[:,:22,:2])*gt_joints_2d_cam[:,:22,2:])

        loss_keypoints[:,[4,5,18,19]] *= self.hparams.limbs2d_loss_weight
        loss_keypoints[:,[7,8,20,21]] *= self.hparams.limbs2d_loss_weight**2
        loss_keypoints = loss_keypoints.mean()
        
        pred_pose_aa = torch.cat([pred_rotmat[:,1:],torch.zeros(batch_size,21,3,1).type_as(pred_rotmat)],dim=3).view([-1,3,4])
        pred_pose_aa = tgm.rotation_matrix_to_angle_axis(pred_pose_aa).reshape([batch_size,21*3])
        q_z = vp_model.encode(pred_pose_aa)
        q_z_sample = q_z.rsample()
        
        loss_regul_vposer = torch.mul(q_z_sample,q_z_sample).mean()
        
        loss_regul_betas = torch.mul(pred_betas,pred_betas).mean()
        
        # Compute total loss
        loss = self.hparams.keypoint2d_loss_weight * loss_keypoints + \
                    self.hparams.beta_loss_weight * loss_regul_betas + \
                    self.hparams.vposer_loss_weight * loss_regul_vposer + \
                    ((torch.exp(-pred_cam_t[:,2])) ** 2 ).mean()

        loss *= 60

        losses = {'loss': loss.detach().item(),
                  'loss_regul_vposer': loss_regul_vposer.detach().item(),
                  'loss_keypoints': loss_keypoints.detach().item(),
                  'loss_regul_betas': loss_regul_betas.detach().item()}

        return loss, losses

    def fwd_pass_and_loss(self,input_batch,is_test=False, is_val=False):
        
        with torch.no_grad():
            # Get data from the batch
            im = input_batch['im0'].float() # input image

            batch_size = im.shape[0]

        cam0_idcs = input_batch["cam"]==0
        cam1_idcs = input_batch["cam"]==1
        cam_idcs = torch.cat([cam0_idcs.unsqueeze(0),cam1_idcs.unsqueeze(0)]).permute(1,0).reshape(2*batch_size)

        pred_pose, pred_betas, pred_camera = self.model.forward(x = im,
                                                        iters = self.hparams.reg_iters)                                                                
        # #####################
        # pred_pose = torch.cat([input_batch["smplorient_rel"],
        #                             input_batch["smplpose_rotmat"]],dim=1).view(batch_size,22,3,3)
        # #####################
        pred_rotmat = pred_pose

        if is_val or is_test:
            pred_output_cam = smplx_test.forward(betas=pred_betas, 
                                    body_pose=pred_pose[:,1:],
                                    global_orient=torch.eye(3).float().unsqueeze(0).unsqueeze(0).repeat(batch_size,1,1,1).type_as(pred_betas),
                                    transl = torch.zeros(batch_size,3).float().type_as(pred_betas),
                                    pose2rot=False)
        else:
            pred_output_cam = smplx.forward(betas=pred_betas, 
                                body_pose=pred_pose[:,1:],
                                global_orient=torch.eye(3).float().unsqueeze(0).unsqueeze(0).repeat(batch_size,1,1,1).type_as(pred_betas),
                                transl = torch.zeros(batch_size,3).float().type_as(pred_betas),
                                pose2rot=False)

        transf_mat = torch.cat([pred_pose[:,:1].squeeze(1),
                                torch.zeros(batch_size,3).float().unsqueeze(2).type_as(pred_betas)],dim=2)

        pred_vertices,pred_joints,_,_ = transform_smpl(transf_mat,
                                                pred_output_cam.vertices.squeeze(1),
                                                pred_output_cam.joints.squeeze(1))
        
        pred_cam_t0 = torch.stack([pred_camera[:,1],
                                  pred_camera[:,2],
                                  2*self.focal_length0[0]/(self.hparams.img_res * pred_camera[:,0] +1e-9)],dim=-1)
        pred_cam_t1 = torch.stack([pred_camera[:,1],
                                  pred_camera[:,2],
                                  2*self.focal_length1[0]/(self.hparams.img_res * pred_camera[:,0] +1e-9)],dim=-1)
        pred_cam_t = torch.cat([pred_cam_t0.unsqueeze(0),pred_cam_t1.unsqueeze(0)],dim=0).permute(1,0,2).reshape(2*batch_size,3)[cam_idcs]

        pred_joints_2d_cam0 = perspective_projection(pred_joints,
                                                   rotation=torch.eye(3).float().unsqueeze(0).repeat(batch_size,1,1).type_as(pred_betas),
                                                   translation=pred_cam_t,
                                                   focal_length=self.focal_length0,
                                                   camera_center=torch.zeros(batch_size, 2).type_as(pred_betas))  
        pred_joints_2d_cam1 = perspective_projection(pred_joints,
                                                   rotation=torch.eye(3).float().unsqueeze(0).repeat(batch_size,1,1).type_as(pred_betas),
                                                   translation=pred_cam_t,
                                                   focal_length=self.focal_length1,
                                                   camera_center=torch.zeros(batch_size, 2).type_as(pred_betas))  

        pred_joints_2d_cam = torch.cat([pred_joints_2d_cam0.unsqueeze(0),pred_joints_2d_cam1.unsqueeze(0)],dim=0).permute(1,0,2,3).reshape(2*batch_size,-1,2)[cam_idcs]
        
        # Pack output arguments for tensorboard logging
        if is_test:
            loss = None
            losses = None
            pred_angles = tgm.rotation_matrix_to_angle_axis(torch.cat([pred_rotmat,torch.zeros(batch_size,22,3,1).float().type_as(pred_betas)],dim=3).view(-1,3,4)).view(batch_size,22,3)
            
            bb0 = input_batch["bb0"]
            intr0 = copy.deepcopy(input_batch["intr0"])
            intr0[:,:2,2] = 0             # origin is the image center
            modif_intr0 = torch.eye(3).repeat([bb0.shape[0],1,1]).type_as(bb0)
            modif_intr0[:,0,0] = self.focal_length[0] / bb0[:,2]
            modif_intr0[:,1,1] = self.focal_length[1] / bb0[:,2]
            modif_intr0[:,:2,2] = bb0[:,:2] * input_batch["intr0"][:,:2,2]
        
            cam_trans0 = torch.bmm(torch.inverse(intr0),torch.bmm(modif_intr0,pred_cam_t.unsqueeze(2)))
            cam_trans_z0 = (pred_cam_t/((self.focal_length[0]/bb0[:,2])/self.focal_length[0]).unsqueeze(1))[:,2]
            pred_cam_trans0 = (cam_trans0.squeeze(2)*cam_trans_z0.unsqueeze(1)/cam_trans0[:,2])

            transf_mat = torch.cat([pred_pose[:,:1].squeeze(1),
                        pred_cam_trans0.unsqueeze(2)],dim=2)
            
            tr_pred_vertices,_,_,_ = transform_smpl(transf_mat,
                                                pred_output_cam.vertices.squeeze(1))
            
            output = {'pred_vertices_cam': pred_vertices.detach(),
                    'tr_pred_vertices_cam': tr_pred_vertices.detach(),
                    'pred_cam_t': pred_cam_t.detach(),
                    'pred_smpltrans': pred_cam_trans0.detach(),
                    "pred_angles": pred_angles.detach()}
        else:
            loss, losses = self.get_loss(input_batch,
                                pred_cam_t,
                                pred_pose,
                                pred_betas,
                                pred_output_cam,
                                pred_joints_2d_cam)
            output = {'pred_vertices_cam': pred_vertices.detach(),
                    'pred_cam_t': pred_cam_t.detach()}


        return output, losses, loss


    def training_step(self, batch, batch_idx):
        # REQUIRED
        if self.hparams.train_reg_only:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.fc1.parameters():
                param.requires_grad = True
            for param in self.model.fc2.parameters():
                param.requires_grad = True
            for param in self.model.decpose.parameters():
                param.requires_grad = True
            for param in self.model.decshape.parameters():
                param.requires_grad = True
            for param in self.model.deccam.parameters():
                param.requires_grad = True
        
        output, losses, loss = self.fwd_pass_and_loss(batch, is_val=False,is_test=False)

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
        train_dset, _ = copenet_real.get_copenet_real_traintest(self.hparams.datapath,shuffle_cams=True)
        return DataLoader(train_dset, batch_size=self.hparams.batch_size,
                            num_workers=self.hparams.num_workers,
                            pin_memory=self.hparams.pin_memory,
                            shuffle=self.hparams.shuffle_train,
                            drop_last=True)

    def val_dataloader(self):
        # OPTIONAL
        _, val_dset = copenet_real.get_copenet_real_traintest(self.hparams.datapath,shuffle_cams=True)
        return DataLoader(val_dset, batch_size=self.hparams.val_batch_size,
                            num_workers=self.hparams.num_workers,
                            pin_memory=self.hparams.pin_memory,
                            shuffle=self.hparams.shuffle_train,
                            drop_last=True)

    def summaries(self, input_batch,output, losses, is_test):
        batch_size = input_batch['im0'].shape[0]
        skip_factor = 4    # number of samples to be logged
        img_downsize_factor = 5
        cam0_idcs = input_batch["cam"][::int(batch_size/skip_factor)]==0
        cam1_idcs = input_batch["cam"][::int(batch_size/skip_factor)]==1

        with torch.no_grad():
            im = input_batch['im0'][::int(batch_size/skip_factor)] * torch.tensor([0.229, 0.224, 0.225], device=self.device).reshape(1,3,1,1)
            im = im + torch.tensor([0.485, 0.456, 0.406], device=self.device).reshape(1,3,1,1)
            # images = []
            # for i in range(0,batch_size,int(batch_size/skip_factor)):
            #     images.append(torch.from_numpy(cv2.imread(input_batch['im1_path'][i])[:,:,::-1]/255.).permute(2,0,1).float())
            pred_vertices_cam = output['pred_vertices_cam'][::int(batch_size/skip_factor)]
            pred_cam_t = output['pred_cam_t'][::int(batch_size/skip_factor)]
            # pred_vertices_cam = input_batch['smpl_vertices_rel'][::int(batch_size/skip_factor)].squeeze(1)
            
            if torch.sum(cam0_idcs) == 0:
                images_pred_cam = self.renderer1.visualize_tb(pred_vertices_cam[cam1_idcs],
                                                            pred_cam_t[cam1_idcs],
                                                            torch.eye(3,device=self._device).float().unsqueeze(0).repeat(batch_size,1,1),
                                                            im[cam1_idcs])
                
                summ_pred_image = images_pred_cam
                
                summ_in_image = torchvision.utils.make_grid(im[cam1_idcs])

            elif torch.sum(cam1_idcs) == 0:
                images_pred_cam = self.renderer0.visualize_tb(pred_vertices_cam[cam0_idcs],
                                                            pred_cam_t[cam0_idcs],
                                                            torch.eye(3,device=self._device).float().unsqueeze(0).repeat(batch_size,1,1),
                                                            im[cam0_idcs])
                
                summ_pred_image = images_pred_cam
                
                summ_in_image = torchvision.utils.make_grid(im[cam0_idcs])

            else:
                images_pred_cam0 = self.renderer0.visualize_tb(pred_vertices_cam[cam0_idcs],
                                                            pred_cam_t[cam0_idcs],
                                                            torch.eye(3,device=self._device).float().unsqueeze(0).repeat(batch_size,1,1),
                                                            im[cam0_idcs])
                images_pred_cam1 = self.renderer1.visualize_tb(pred_vertices_cam[cam1_idcs],
                                                            pred_cam_t[cam1_idcs],
                                                            torch.eye(3,device=self._device).float().unsqueeze(0).repeat(batch_size,1,1),
                                                            im[cam1_idcs])
                
                summ_pred_image = torch.cat((images_pred_cam0, images_pred_cam1),2)
                
                summ_in_image = torchvision.utils.make_grid(torch.cat((im[cam0_idcs], im[cam1_idcs]),0))
            # import ipdb; ipdb.set_trace()
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
            train_dset, val_dset = copenet_real.get_copenet_real_traintest(self.hparams.datapath,shuffle_cams=True)
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
        output, losses, loss = self.fwd_pass_and_loss(batch,is_test=True, is_val=True)
        
        # self.viz_3d(batch,output)
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
        train.add_argument('--testdata', type=str, default="aerialpeople", help='test dataset')
        train.add_argument('--log_dir', default='/is/cluster/nsaini/copenet_logs', help='Directory to store logs')
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

