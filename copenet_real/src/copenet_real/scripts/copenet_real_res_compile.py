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

from copenet_real.copenet_twoview import copenet_twoview
from copenet_real.dsets import copenet_real

import sys
fname = sys.argv[1]
datapath = sys.argv[2]

# # ckpt_path = "/is/ps3/nsaini/projects/copenet_real/copenet_logs/copenet_twoview/version_5_cont_limbwght/checkpoints/epoch=761.ckpt"
# ckpt_path = "/is/cluster/nsaini/copenet_logs/copenet_twoview_newcorrectedruns/copenet_twoview_newcorrectedruns/checkpoints/epoch-257.ckpt"
# datapath = "/home/nsaini/Datasets/copenet_data/"
# # check model type
# model_type = ckpt_path.split("/")[-4]

# # create trainer
# trainer = Trainer(gpus=1)
# # create Network


# net = copenet_twoview.load_from_checkpoint(checkpoint_path=ckpt_path)

# create dataset and dataloader
train_ds, test_ds = copenet_real.get_copenet_real_traintest(datapath)

tst_dl = DataLoader(test_ds, batch_size=30,
                            num_workers=40,
                            pin_memory=True,
                            shuffle=False,
                            drop_last=True)
trn_dl = DataLoader(train_ds, batch_size=30,
                            num_workers=40,
                            pin_memory=True,
                            shuffle=False,
                            drop_last=True)



# %% draw sample and forward
# res = trainer.test(net,test_dataloaders=[tst_dl,trn_dl])
# pkl.dump(outputs,open(,"wb"))


# %%
# fname = "/is/ps3/nsaini/projects/copenet_real/copenet_logs/copenet_twoview/version_5_cont_limbwght/checkpoints/epoch=761.pkl"
# fname = "/is/cluster/nsaini/copenet_logs/copenet_twoview_newcorrectedruns/copenet_twoview_newcorrectedruns/checkpoints/epoch-257.pkl"
fname = os.path.join(fname,"epoch-257.pkl")

import os
os.environ["PYOPENGL_PLATFORM"] = 'egl'
import sys
# sys.path.append("/is/ps3/nsaini/projects/copenet_real/src")
from copenet_real.utils.utils import transform_smpl
import torchgeometry as tgm
from copenet_real.utils.renderer import Renderer
from copenet_real.smplx.smplx import SMPLX, lbs
import pickle as pkl
import torch
import numpy as np
import matplotlib.pyplot as plt

smplx = SMPLX(os.path.join("/is/ps3/nsaini/projects/copenet","src/copenet/data/smplx/models/smplx"),
                         batch_size=1,
                         create_transl=False)

img_res = [1920,1080]
CX0 = 1018
CY0 = 577
CX1 = 978
CY1 = 667
real_focallen0 = [1537,1517]
real_focallen1 = [1361,1378]

real_renderer0 = Renderer(real_focallen0,img_res,[CX0,CY0],smplx.faces)
real_renderer1 = Renderer(real_focallen1,img_res,[CX1,CY1],smplx.faces)

res = pkl.load(open(fname,"rb"))


test_extr0 = test_ds.extr0[8000:15000].to(device)
test_extr1 = test_ds.extr1[8000:15000].to(device)
intr0 = test_ds.intr0
intr1 = test_ds.intr1
images0 = np.array(test_ds.db["im0"])
images1 = np.array(test_ds.db["im1"])

dataset = "train"
if dataset == "train":
    res_idx = 1
elif dataset == "test":
    res_idx = 0
else:
    print("provide the dataset name")

begin = 0
end = len(res[res_idx])


batch_size = res[res_idx][0]["output"]["pred_vertices_cam0"].shape[0]
pred_vertices_cam0_test = torch.cat([i["output"]["pred_vertices_cam0"].to(device) for i in res[res_idx][begin:end]])
pred_vertices_cam1_test = torch.cat([i["output"]["pred_vertices_cam1"].to(device) for i in res[res_idx][begin:end]])
pred_j2d_cam0_test = torch.cat([i["output"]["pred_j2d_cam0"].to(device) for i in res[res_idx][begin:end]])
pred_j2d_cam1_test = torch.cat([i["output"]["pred_j2d_cam1"].to(device) for i in res[res_idx][begin:end]])
pred_j3d_cam0_test = torch.cat([i["output"]["pred_j3d_cam0"].to(device) for i in res[res_idx][begin:end]])
pred_j3d_cam1_test = torch.cat([i["output"]["pred_j3d_cam1"].to(device) for i in res[res_idx][begin:end]])
pred_betas0 = torch.cat([i["output"]["pred_betas0"].to(device) for i in res[res_idx][begin:end]])
pred_betas1 = torch.cat([i["output"]["pred_betas1"].to(device) for i in res[res_idx][begin:end]])
pred_smpltrans0_test = torch.cat([i["output"]["pred_smpltrans0"].to(device) for i in res[res_idx][begin:end]])
pred_smpltrans1_test = torch.cat([i["output"]["pred_smpltrans1"].to(device) for i in res[res_idx][begin:end]])
pred_angles0_test = torch.cat([i["output"]["pred_angles0"].to(device) for i in res[res_idx][begin:end]])
pred_angles1_test = torch.cat([i["output"]["pred_angles1"].to(device) for i in res[res_idx][begin:end]])
pred_rotmat0_test = tgm.angle_axis_to_rotation_matrix(pred_angles0_test.view(-1,3)).view(pred_angles0_test.shape[0],22,4,4)
pred_rotmat1_test = tgm.angle_axis_to_rotation_matrix(pred_angles1_test.view(-1,3)).view(pred_angles1_test.shape[0],22,4,4)


pred_vertices_cam0_test_wrt_origin,pred_j3d_cam0_test_wrt_origin,pred_orient0_wrt_origin, pred_trans0_wrt_origin \
        = transform_smpl(torch.inverse(test_extr0[begin*batch_size:batch_size*(end-begin)]),pred_vertices_cam0_test,pred_j3d_cam0_test,pred_rotmat0_test[begin*batch_size:batch_size*(end-begin),1,:3,:3],pred_smpltrans0_test)
pred_vertices_cam1_test_wrt_origin,pred_j3d_cam1_test_wrt_origin,pred_orient1_wrt_origin, pred_trans1_wrt_origin \
        = transform_smpl(torch.inverse(test_extr1[begin*batch_size:batch_size*(end-begin)]),pred_vertices_cam1_test,pred_j3d_cam1_test,pred_rotmat1_test[begin*batch_size:batch_size*(end-begin),1,:3,:3],pred_smpltrans1_test)
    


person_present = np.sum(test_ds.opose[:,:,:,2],axis=2)==0
person_present = ~(person_present[0]*person_present[1])[:pred_betas0.shape[0]]
err_idcs = np.load("/is/ps3/nsaini/projects/copenet_real/src/copenet_real/scripts/err_idcs.npy")
torch.sqrt(torch.sum(((pred_j3d_cam0_test_wrt_origin-pred_trans0_wrt_origin.unsqueeze(1))[err_idcs] - 
        (pred_j3d_cam1_test_wrt_origin-pred_trans1_wrt_origin.unsqueeze(1))[err_idcs])**2,dim=2)).mean()

# %%
import cv2
import random
from tqdm import tqdm
# for idx in tqdm(range(0,7000,5)):
# idx = 995
    # samples = random.sample(range(begin*batch_size,batch_size*(end-begin)),5)
    # samples = [idx,idx+1,idx+2,idx+3,idx+4]
# samples = [1000,2100,3000,3800,5000]
# samples = [500,6500,2500,4200,6000]
samples = [200,5300,800,4700,6700]
ims0 = torch.from_numpy(np.stack([cv2.imread(f)[:,:,::-1]/255. for f in images0[samples]])).permute(0,3,1,2)
ims1 = torch.from_numpy(np.stack([cv2.imread(f)[:,:,::-1]/255. for f in images1[samples]])).permute(0,3,1,2)

rend_ims0 = real_renderer0.visualize_tb(pred_vertices_cam0_test[samples],
                            torch.zeros(len(samples),3).float().to(device),
                            torch.eye(3).float().unsqueeze(0).repeat(len(samples),1,1).to(device),
                            ims0)
rend_ims1 = real_renderer1.visualize_tb(pred_vertices_cam1_test[samples],
                            torch.zeros(len(samples),3).float().to(device),
                            torch.eye(3).float().unsqueeze(0).repeat(len(samples),1,1).to(device),
                            ims1)
    # cv2.imwrite("/home/nsaini/Desktop/test_res/{}_1.jpg".format(idx),rend_ims1.permute(1,2,0).cpu().numpy()[:,:,::-1]*255)
# #     for_blender_cam0 = pred_vertices_cam0_test_wrt_origin[samples]
# #     for_blender_cam1 = pred_vertices_cam1_test_wrt_origin[samples]


# %% Meshcat
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
from tqdm import tqdm

# Create a new visualizer
vis = meshcat.Visualizer()

temp_extr0 = test_extr0.detach().clone()
temp_extr0[1:] = temp_extr0[0]
pred_vertices_cam0_test_wrt_origin_temp,_,_, _ \
        = transform_smpl(torch.inverse(temp_extr0[:6990]),pred_vertices_cam0_test)

for i in tqdm(range(pred_vertices_cam0_test_wrt_origin_temp.shape[0])):
    vis["mesh1"].set_object(g.TriangularMeshGeometry(pred_vertices_cam0_test_wrt_origin_temp[i].cpu().numpy(),smplx.faces),
                        g.MeshLambertMaterial(
                             color=0xff22dd,
                             reflectivity=0.8))


# %% Benchtest data 
import rosbag
import glob
import numpy as np
import matplotlib.pyplot as plt

train_data_out_m1 = []
train_frId_m1 = []
train_data_out_m2 = []
train_frId_m2 = []
for bfile in sorted(glob.glob("/is/ps3/nsaini/projects/copenet_real/copenet_logs/copenet_twoview/version_5_cont_limbwght/checkpoints/epoch_761_benchtest_play_WO_images/train*")):
    bag = rosbag.Bag(bfile)
    for topic,msg,t in bag.read_messages(topics=["/machine_1/step3_pub"]):
        train_data_out_m1.append(np.array(msg.data).reshape(1,-1))
        train_frId_m1.append(int(msg.header.frame_id))
    for topic,msg,t in bag.read_messages(topics=["/machine_2/step3_pub"]):
        train_data_out_m2.append(np.array(msg.data).reshape(1,-1))
        train_frId_m2.append(int(msg.header.frame_id))

train_data_out_m1 = np.concatenate(train_data_out_m1)
train_frId_m1 = np.array(train_frId_m1)
train_data_out_m2 = np.concatenate(train_data_out_m2)
train_frId_m2 = np.array(train_frId_m2)


train_betas1 = train_data_out_m1[:,:10]
train_trans1 = train_data_out_m1[:,10:13]*20
train_pose1 = train_data_out_m1[:,13:]
train_betas2 = train_data_out_m2[:,:10]
train_trans2 = train_data_out_m2[:,10:13]*20
train_pose2 = train_data_out_m2[:,13:]

# plt.figure()
# plt.plot(train_frId_m1 ,train_trans1)
# plt.figure()
# plt.plot(train_frId_m2, train_trans2)


test_data_out_m1 = []
test_frId_m1 = []
test_data_out_m2 = []
test_frId_m2 = []
for bfile in sorted(glob.glob("/is/ps3/nsaini/projects/copenet_real/copenet_logs/copenet_twoview/version_5_cont_limbwght/checkpoints/epoch_761_benchtest_play_WO_images/test*.bag")):
    bag = rosbag.Bag(bfile)
    for topic,msg,t in bag.read_messages(topics=["/machine_1/step3_pub"]):
        test_data_out_m1.append(np.array(msg.data).reshape(1,-1))
        test_frId_m1.append(int(msg.header.frame_id))
    for topic,msg,t in bag.read_messages(topics=["/machine_2/step3_pub"]):
        test_data_out_m2.append(np.array(msg.data).reshape(1,-1))
        test_frId_m2.append(int(msg.header.frame_id))

test_data_out_m1 = np.concatenate(test_data_out_m1)
test_frId_m1 = np.array(test_frId_m1)
test_data_out_m2 = np.concatenate(test_data_out_m2)
test_frId_m2 = np.array(test_frId_m2)

test_betas1 = test_data_out_m1[:,:10]
test_trans1 = test_data_out_m1[:,10:13]*20
test_pose1 = test_data_out_m1[:,13:]
test_betas2 = test_data_out_m2[:,:10]
test_trans2 = test_data_out_m2[:,10:13]*20
test_pose2 = test_data_out_m2[:,13:]

# plt.figure()
# plt.plot(test_frId_m1,test_trans1)
# plt.figure()
# plt.plot(test_frId_m2,test_trans2)


begin = 0
end = len(res[0])

test_pred_betas0 = torch.cat([i["output"]["pred_betas0"] for i in res[0][begin:end]]).detach().cpu().numpy()
test_pred_betas1 = torch.cat([i["output"]["pred_betas1"] for i in res[0][begin:end]]).detach().cpu().numpy()
test_pred_pose0 = torch.cat([i["output"]["pred_pose0"] for i in res[0][begin:end]]).detach().cpu().numpy()
test_pred_pose1 = torch.cat([i["output"]["pred_pose1"] for i in res[0][begin:end]]).detach().cpu().numpy()

begin = 0
end = len(res[1])

train_pred_betas0 = torch.cat([i["output"]["pred_betas0"] for i in res[1][begin:end]]).detach().cpu().numpy()
train_pred_betas1 = torch.cat([i["output"]["pred_betas1"] for i in res[1][begin:end]]).detach().cpu().numpy()
train_pred_pose0 = torch.cat([i["output"]["pred_pose0"] for i in res[1][begin:end]]).detach().cpu().numpy()
train_pred_pose1 = torch.cat([i["output"]["pred_pose1"] for i in res[1][begin:end]]).detach().cpu().numpy()

test_frId_m1 = test_frId_m1[test_frId_m1<6990]
test_frId_m2 = test_frId_m2[test_frId_m2<6990]
train_frId_m1 = train_frId_m1[train_frId_m1<6990]
train_frId_m2 = train_frId_m2[train_frId_m2<6990]

print(np.abs((test_data_out_m1[:len(test_frId_m1),:10] - test_pred_betas0[test_frId_m1])).mean())
print(np.abs((test_data_out_m2[:len(test_frId_m2),:10] - test_pred_betas1[test_frId_m2])).mean())
print(np.abs((test_data_out_m1[:len(test_frId_m1),10:13]*20 - test_pred_pose0[test_frId_m1,:3])).mean())
print(np.abs((test_data_out_m2[:len(test_frId_m2),10:13]*20 - test_pred_pose1[test_frId_m2,:3])).mean())
print(np.abs((test_data_out_m1[:len(test_frId_m1),13:] - test_pred_pose0[test_frId_m1,3:])).mean())
print(np.abs((test_data_out_m2[:len(test_frId_m2),13:] - test_pred_pose1[test_frId_m2,3:])).mean())


print(np.abs((train_data_out_m1[:len(train_frId_m1),:10] - train_pred_betas0[train_frId_m1])).mean())
print(np.abs((train_data_out_m2[:len(train_frId_m2),:10] - train_pred_betas1[train_frId_m2])).mean())
print(np.abs((train_data_out_m1[:len(train_frId_m1),10:13]*20 - train_pred_pose0[train_frId_m1,:3])).mean())
print(np.abs((train_data_out_m2[:len(train_frId_m2),10:13]*20 - train_pred_pose1[train_frId_m2,:3])).mean())
print(np.abs((train_data_out_m1[:len(train_frId_m1),13:] - train_pred_pose0[train_frId_m1,3:])).mean())
print(np.abs((train_data_out_m2[:len(train_frId_m2),13:] - train_pred_pose1[train_frId_m2,3:])).mean())

plt.figure()
plt.plot(train_data_out_m1[:len(train_frId_m1),10:13]*20)
plt.plot(train_pred_pose0[train_frId_m1,:3])
plt.figure()
plt.plot(train_data_out_m2[:len(train_frId_m2),10:13]*20)
plt.plot(train_pred_pose1[train_frId_m2,:3])
plt.figure()
plt.plot(test_data_out_m1[:len(test_frId_m1),10:13]*20)
plt.plot(test_pred_pose0[test_frId_m1,:3])
plt.figure()
plt.plot(test_data_out_m2[:len(test_frId_m2),10:13]*20)
plt.plot(test_pred_pose1[test_frId_m2,:3])
