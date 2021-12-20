# %% Imports
import torch
import torchvision
import numpy as np
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from tqdm import tqdm
import pickle as pkl
import os, sys; sys.path.append(os.path.dirname(os.path.abspath(__file__+"/..")))
os.environ["PYOPENGL_PLATFORM"] = 'egl'

from config import device
from copenet_real.hmr import hmr
from copenet_real.dsets import copenet_real

import sys
fname = sys.argv[1]
datapath = sys.argv[2]

# ckpt_path = "/is/ps3/nsaini/projects/copenet_real/copenet_logs/hmr/version_2_from_newlytrinedckpt/checkpoints/epoch=388.ckpt"
# datapath = "/home/nsaini/Datasets/copenet_data/"
# # check model type
# model_type = ckpt_path.split("/")[-4]

if device == "cuda":
    gpu = 1
else:
    gpu = 0

# # create trainer
trainer = Trainer(gpus=gpu)
# # create Network


net0 = hmr.load_from_checkpoint(checkpoint_path=os.path.join(fname,"epoch=388.ckpt"))
net1 = hmr.load_from_checkpoint(checkpoint_path=os.path.join(fname,"epoch=388.ckpt"))

# create dataset and dataloader
train_ds, test_ds = copenet_real.get_copenet_real_traintest(datapath)
train_ds_1, test_ds_1 = copenet_real.get_copenet_real_traintest(datapath,first_cam=1)

tst_dl = DataLoader(test_ds, batch_size=30,
                            num_workers=40,
                            pin_memory=True,
                            shuffle=False,
                            drop_last=True)
tst_dl1 = DataLoader(test_ds_1, batch_size=30,
                            num_workers=40,
                            pin_memory=True,
                            shuffle=False,
                            drop_last=True)
trn_dl = DataLoader(train_ds, batch_size=30,
                            num_workers=40,
                            pin_memory=True,
                            shuffle=False,
                            drop_last=True)
trn_dl1 = DataLoader(train_ds_1, batch_size=30,
                            num_workers=40,
                            pin_memory=True,
                            shuffle=False,
                            drop_last=True)


# %% draw sample and forward
res0 = trainer.test(net0,test_dataloaders=[tst_dl,trn_dl])
# pkl.dump(outputs,open(,"wb"))
res1 = trainer.test(net1,test_dataloaders=[tst_dl1,trn_dl1])


# %%
# fname = "/is/ps3/nsaini/projects/copenet_real/copenet_logs/hmr/version_2_from_newlytrinedckpt/checkpoints/epoch=388.pkl"
# fname = os.path.join(fname,"epoch=388.pkl")
# fname0 = fname + "0"
# fname1 = fname + "1"

# import sys
# # sys.path.append("/is/ps3/nsaini/projects/copenet_real/src")
# from copenet_real.utils.utils import transform_smpl
# import torchgeometry as tgm
# from copenet_real.utils.renderer import Renderer
# from copenet_real.utils.geometry import perspective_projection
# from copenet_real.smplx.smplx import SMPLX, lbs
# import pickle as pkl
# import os
# import torch
# import numpy as np

# smplx = SMPLX(os.path.join("/is/ps3/nsaini/projects/copenet","src/copenet/data/smplx/models/smplx"),
#                          batch_size=1,
#                          create_transl=False)

# img_res = [1920,1080]
# CX0 = 1018
# CY0 = 577
# CX1 = 978
# CY1 = 667
# real_focallen0 = [1537,1517]
# real_focallen1 = [1361,1378]

# real_renderer0 = Renderer(real_focallen0,img_res,[CX0,CY0],smplx.faces)
# real_renderer1 = Renderer(real_focallen1,img_res,[CX1,CY1],smplx.faces)

# res0 = pkl.load(open(fname0,"rb"))
# res1 = pkl.load(open(fname1,"rb"))


# test_extr0 = test_ds.extr0[8000:15000].to(device)
# test_extr1 = test_ds.extr1[8000:15000].to(device)
# intr0 = test_ds.intr0
# intr1 = test_ds.intr1
# images0 = np.array(test_ds.db["im0"])
# images1 = np.array(test_ds.db["im1"])

# dataset = "train"
# if dataset == "train":
#     res_idx = 1
# elif dataset == "test":
#     res_idx = 0
# else:
#     print("provide the dataset name")

# begin = 0
# end = len(res0[res_idx])


# batch_size = res0[res_idx][0]["output"]["tr_pred_vertices_cam"].shape[0]
# pred_vertices_cam0_test = torch.cat([i["output"]["tr_pred_vertices_cam"].to(device) for i in res0[res_idx][begin:end]])
# pred_vertices_cam1_test = torch.cat([i["output"]["tr_pred_vertices_cam"].to(device) for i in res1[res_idx][begin:end]])
# pred_j3d_cam0_test = torch.cat([i["output"]["tr_pred_j3d_cam"].to(device) for i in res0[res_idx][begin:end]])
# pred_j3d_cam1_test = torch.cat([i["output"]["tr_pred_j3d_cam"].to(device) for i in res1[res_idx][begin:end]])
# pred_j2d_cam0_test = perspective_projection(pred_j3d_cam0_test,
#         rotation=torch.eye(3).float().unsqueeze(0).repeat(pred_j3d_cam0_test.shape[0],1,1).type_as(pred_j3d_cam0_test),
#         translation=torch.zeros(3).float().unsqueeze(0).repeat(pred_j3d_cam0_test.shape[0],1).type_as(pred_j3d_cam0_test),
#         focal_length=real_focallen0,
#         camera_center=torch.tensor([CX0,CY0]).float().unsqueeze(0).repeat(pred_j3d_cam0_test.shape[0],1).type_as(pred_j3d_cam0_test))[:,:22]
# pred_j2d_cam1_test = perspective_projection(pred_j3d_cam1_test,
#         rotation=torch.eye(3).float().unsqueeze(0).repeat(pred_j3d_cam1_test.shape[0],1,1).type_as(pred_j3d_cam1_test),
#         translation=torch.zeros(3).float().unsqueeze(0).repeat(pred_j3d_cam1_test.shape[0],1).type_as(pred_j3d_cam1_test),
#         focal_length=real_focallen1,
#         camera_center=torch.tensor([CX1,CY1]).float().unsqueeze(0).repeat(pred_j3d_cam1_test.shape[0],1).type_as(pred_j3d_cam1_test))[:,:22]
# pred_betas0 = torch.cat([i["output"]["pred_betas"].to(device) for i in res0[res_idx][begin:end]])
# pred_betas1 = torch.cat([i["output"]["pred_betas"].to(device) for i in res1[res_idx][begin:end]])
# pred_smpltrans0_test = torch.cat([i["output"]["pred_smpltrans"].to(device) for i in res0[res_idx][begin:end]])
# pred_smpltrans1_test = torch.cat([i["output"]["pred_smpltrans"].to(device) for i in res1[res_idx][begin:end]])
# pred_angles0_test = torch.cat([i["output"]["pred_angles"].to(device) for i in res0[res_idx][begin:end]])
# pred_angles1_test = torch.cat([i["output"]["pred_angles"].to(device) for i in res1[res_idx][begin:end]])
# pred_rotmat0_test = tgm.angle_axis_to_rotation_matrix(pred_angles0_test.view(-1,3)).view(pred_angles0_test.shape[0],22,4,4)
# pred_rotmat1_test = tgm.angle_axis_to_rotation_matrix(pred_angles1_test.view(-1,3)).view(pred_angles1_test.shape[0],22,4,4)

# # import ipdb;ipdb.set_trace()
# ###########################
# robust_idcs = (pred_smpltrans0_test_hmr[:,2] < 25).detach().cpu().numpy()
# fig,ax = plt.subplots(3,1,sharex=True)
# ax[0].plot(pred_smpltrans0_test_hmr[robust_idcs,0].detach().cpu().numpy(),".",markersize=1,mec="indianred",mfc="indianred")
# ax[0].plot(pred_smpltrans0_test[robust_idcs,0].detach().cpu().numpy(),".",markersize=1,mec="chartreuse",mfc="chartreuse")
# ax[0].plot(pl_smpl_wrt_cam0[robust_idcs,0,3].detach().cpu().numpy(),".",markersize=1,mec="darkviolet",mfc="darkviolet")
# ax[0].yaxis.set_label_text("x(m)",{"fontsize":"large","fontweight":"bold"})
# ax[0].legend(["Baseline","AirPose","AirPose$^+$"],markerscale=10)
# ax[1].plot(pred_smpltrans0_test_hmr[robust_idcs,1].detach().cpu().numpy(),".",markersize=1,mec="seagreen",mfc="seagreen")
# ax[1].plot(pred_smpltrans0_test[robust_idcs,1].detach().cpu().numpy(),".",markersize=1,mec="fuchsia",mfc="fuchsia")
# ax[1].plot(pl_smpl_wrt_cam0[robust_idcs,1,3].detach().cpu().numpy(),".",markersize=1,mec="cyan",mfc="cyan")
# ax[1].yaxis.set_label_text("y(m)",{"fontsize":"large","fontweight":"bold"})
# ax[1].legend(["Baseline","AirPose","AirPose$^+$"],markerscale=10)
# ax[2].plot(pred_smpltrans0_test_hmr[robust_idcs,2].detach().cpu().numpy(),".",markersize=1,mec="dodgerblue",mfc="dodgerblue")
# ax[2].plot(pred_smpltrans0_test[robust_idcs,2].detach().cpu().numpy(),".",markersize=1,mec="mediumvioletred",mfc="mediumvioletred")
# ax[2].plot(pl_smpl_wrt_cam0[robust_idcs,2,3].detach().cpu().numpy(),".",markersize=1,mec="lawngreen",mfc="lawngreen")
# ax[2].yaxis.set_label_text("z(m)",{"fontsize":"large","fontweight":"bold"})
# ax[2].legend(["Baseline","AirPose","AirPose$^+$"],markerscale=10)
# ax[2].xaxis.set_label_text("frame number",{"fontsize":"large","fontweight":"bold"})
# plt.subplots_adjust(wspace=0, hspace=0.02)

# robust_idcs = (pred_smpltrans1_test_hmr[:,2] < 25).detach().cpu().numpy()
# fig,ax = plt.subplots(3,1,sharex=True)
# ax[0].plot(pred_smpltrans1_test_hmr[robust_idcs,0].detach().cpu().numpy(),".",markersize=1,mec="indianred",mfc="indianred")
# ax[0].plot(pred_smpltrans1_test[robust_idcs,0].detach().cpu().numpy(),".",markersize=1,mec="chartreuse",mfc="chartreuse")
# ax[0].plot(pl_smpl_wrt_cam1[robust_idcs,0,3].detach().cpu().numpy(),".",markersize=1,mec="darkviolet",mfc="darkviolet")
# ax[0].yaxis.set_label_text("x(m)",{"fontsize":"large","fontweight":"bold"})
# ax[0].legend(["Baseline","AirPose","AirPose$^+$"],markerscale=10)
# ax[1].plot(pred_smpltrans1_test_hmr[robust_idcs,1].detach().cpu().numpy(),".",markersize=1,mec="seagreen",mfc="seagreen")
# ax[1].plot(pred_smpltrans1_test[robust_idcs,1].detach().cpu().numpy(),".",markersize=1,mec="fuchsia",mfc="fuchsia")
# ax[1].plot(pl_smpl_wrt_cam1[robust_idcs,1,3].detach().cpu().numpy(),".",markersize=1,mec="cyan",mfc="cyan")
# ax[1].yaxis.set_label_text("y(m)",{"fontsize":"large","fontweight":"bold"})
# ax[1].legend(["Baseline","AirPose","AirPose$^+$"],markerscale=10)
# ax[2].plot(pred_smpltrans1_test_hmr[robust_idcs,2].detach().cpu().numpy(),".",markersize=1,mec="dodgerblue",mfc="dodgerblue")
# ax[2].plot(pred_smpltrans1_test[robust_idcs,2].detach().cpu().numpy(),".",markersize=1,mec="mediumvioletred",mfc="mediumvioletred")
# ax[2].plot(pl_smpl_wrt_cam1[robust_idcs,2,3].detach().cpu().numpy(),".",markersize=1,mec="lawngreen",mfc="lawngreen")
# ax[2].yaxis.set_label_text("z(m)",{"fontsize":"large","fontweight":"bold"})
# ax[2].legend(["Baseline","AirPose","AirPose$^+$"],markerscale=10)
# ax[2].xaxis.set_label_text("frame number",{"fontsize":"large","fontweight":"bold"})
# plt.subplots_adjust(wspace=0, hspace=0.02)
# ###########################
# plt.show()

# pred_vertices_cam0_test_wrt_origin,pred_j3d_cam0_test_wrt_origin,pred_orient0_wrt_origin, pred_trans0_wrt_origin \
#         = transform_smpl(torch.inverse(test_extr0[begin*batch_size:batch_size*(end-begin)]),pred_vertices_cam0_test,pred_j3d_cam0_test,pred_rotmat0_test[begin*batch_size:batch_size*(end-begin),1,:3,:3],pred_smpltrans0_test)
# pred_vertices_cam1_test_wrt_origin,pred_j3d_cam1_test_wrt_origin,pred_orient1_wrt_origin, pred_trans1_wrt_origin \
#         = transform_smpl(torch.inverse(test_extr1[begin*batch_size:batch_size*(end-begin)]),pred_vertices_cam1_test,pred_j3d_cam1_test,pred_rotmat1_test[begin*batch_size:batch_size*(end-begin),1,:3,:3],pred_smpltrans1_test)
    

# person_present = np.sum(test_ds.opose[:,:,:,2],axis=2)==0
# person_present = ~(person_present[0]*person_present[1])[:pred_betas0.shape[0]]
# err_idcs = np.load("/is/ps3/nsaini/projects/copenet_real/src/copenet_real/scripts/err_idcs.npy")
# torch.sqrt(torch.sum(((pred_j3d_cam0_test_wrt_origin-pred_trans0_wrt_origin.unsqueeze(1))[err_idcs] - 
#         (pred_j3d_cam1_test_wrt_origin-pred_trans1_wrt_origin.unsqueeze(1))[err_idcs])**2,dim=2)).mean()

# # %%
# # import cv2
# # import random
# # # samples = random.sample(range(begin*batch_size,batch_size*(end-begin)),5)
# # samples = [1520,2080,2690,3415,3832]
# # ims0 = torch.from_numpy(np.stack([cv2.imread(f)[:,:,::-1]/255. for f in images0[samples]])).permute(0,3,1,2)
# # ims1 = torch.from_numpy(np.stack([cv2.imread(f)[:,:,::-1]/255. for f in images1[samples]])).permute(0,3,1,2)

# # rend_ims0 = real_renderer0.visualize_tb(pred_vertices_cam0_test[samples],
# #                         torch.zeros(len(samples),3).float().to(device),
# #                         torch.eye(3).float().unsqueeze(0).repeat(len(samples),1,1).to(device),
# #                         ims0)
# # rend_ims1 = real_renderer1.visualize_tb(pred_vertices_cam1_test[samples],
# #                         torch.zeros(len(samples),3).float().to(device),
# #                         torch.eye(3).float().unsqueeze(0).repeat(len(samples),1,1).to(device),
# #                         ims1)

# # for_blender_cam0 = pred_vertices_cam0_test_wrt_origin[samples]
# # for_blender_cam1 = pred_vertices_cam1_test_wrt_origin[samples]

# # %%
# import meshcat
# import meshcat.geometry as g
# import meshcat.transformations as tf
# from tqdm import tqdm

# # Create a new visualizer
# vis = meshcat.Visualizer()

# temp_extr0 = test_extr0.detach().clone()
# temp_extr0[1:] = temp_extr0[0]
# pred_vertices_cam0_test_wrt_origin_temp,_,_, _ \
#         = transform_smpl(torch.inverse(temp_extr0[:6990]),pred_vertices_cam0_test)

# for i in tqdm(range(pred_vertices_cam0_test_wrt_origin_temp.shape[0])):
#     vis["mesh1"].set_object(g.TriangularMeshGeometry(pred_vertices_cam0_test_wrt_origin_temp[i].cpu().numpy(),smplx.faces),
#                         g.MeshLambertMaterial(
#                              color=0xff22dd,
#                              reflectivity=0.8))