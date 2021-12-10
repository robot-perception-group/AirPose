# %% Imports
import torch
import torchvision
import numpy as np
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from tqdm import tqdm
import pickle as pkl
import os
os.environ["PYOPENGL_PLATFORM"] = 'egl'


from copenet.hmr import hmr
from copenet_real.dsets import aerialpeople


ckpt_path = "/is/ps3/nsaini/projects/copenet/copenet_logs/hmr/version_1_camswaps_correctcam/final.ckpt"
datapath = "/home/nsaini/Datasets/AerialPeople/agora_copenet_uniform_new_cropped/"
# check model type
model_type = ckpt_path.split("/")[-4]

# create trainer
trainer = Trainer(gpus=1)
# create Network


net0 = hmr.load_from_checkpoint(checkpoint_path=ckpt_path)
net1 = hmr.load_from_checkpoint(checkpoint_path=ckpt_path)

# create dataset and dataloader
train_ds, test_ds = aerialpeople.get_aerialpeople_seqsplit(datapath)
train_ds_1, test_ds_1 = aerialpeople.get_aerialpeople_seqsplit(datapath,first_cam=1)

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
# res0 = trainer.test(net0,test_dataloaders=[tst_dl,trn_dl])
# pkl.dump(outputs,open(,"wb"))
# res1 = trainer.test(net1,test_dataloaders=[tst_dl1,trn_dl1])


# %%
# fname = "/is/ps3/nsaini/projects/copenet/copenet_logs/hmr/version_1_camswaps_correctcam/final.pkl"
fname = "/is/ps3/nsaini/projects/copenet/copenet_logs/copenet_singleview/version_3_noinputtrans/checkpoints/epoch=176.pkl"
fname0 = fname + "0"
fname1 = fname + "1"

import sys
# sys.path.append("/is/ps3/nsaini/projects/copenet_real/src")
from copenet_real.utils.utils import transform_smpl
import torchgeometry as tgm
from copenet_real.utils.renderer import Renderer
from copenet_real.smplx.smplx import SMPLX, lbs
import pickle as pkl
import os
import torch
import numpy as np

smplx = SMPLX(os.path.join("/is/ps3/nsaini/projects/copenet","src/copenet/data/smplx/models/smplx"),
                         batch_size=1,
                         create_transl=False)

img_res = [1920,1080]
synth_focallen = [1475,1475]

synth_renderer = Renderer(synth_focallen,img_res,[img_res[0]/2,img_res[1]/2],smplx.faces)

res0 = pkl.load(open(fname0,"rb"))
res1 = pkl.load(open(fname1,"rb"))


test_extr0 = torch.from_numpy(np.stack([pkl.load(open(f,"rb"))["cam0"]["extr"] for f in test_ds.db])).float().to("cuda")
test_extr1 = torch.from_numpy(np.stack([pkl.load(open(f,"rb"))["cam1"]["extr"] for f in test_ds.db])).float().to("cuda")
intr0 = torch.from_numpy(np.stack([pkl.load(open(f,"rb"))["cam0"]["intr"] for f in test_ds.db])).float().to("cuda")
intr1 = torch.from_numpy(np.stack([pkl.load(open(f,"rb"))["cam1"]["intr"] for f in test_ds.db])).float().to("cuda")
images0 = np.array([os.path.join("/ps/project/datasets/AirCap_ICCV19/agora_copenet_uniform_new",pkl.load(open(f,"rb"))["im0"]) for f in test_ds.db])
images1 = np.array([os.path.join("/ps/project/datasets/AirCap_ICCV19/agora_copenet_uniform_new",pkl.load(open(f,"rb"))["im1"]) for f in test_ds_1.db])
gt_vertices = np.array([pkl.load(open(f,"rb"))["smpl_vertices_wrt_origin"] for f in test_ds.db])[:,0]
gt_smpltrans = torch.from_numpy(np.concatenate([pkl.load(open(f,"rb"))["smpltrans"] for f in test_ds.db])).float().to("cuda")
gt_j3d = torch.from_numpy(np.concatenate([pkl.load(open(f,"rb"))["smpl_joints_wrt_origin"] for f in test_ds.db])).float().to("cuda")


begin = 0
end = len(res0[0])


batch_size = res0[0][0]["output"]["pred_vertices_cam"].shape[0]
pred_vertices_cam0_test = torch.cat([i["output"]["pred_vertices_cam"].to("cuda") for i in res0[0][begin:end]])
pred_vertices_cam1_test = torch.cat([i["output"]["pred_vertices_cam"].to("cuda") for i in res1[0][begin:end]])
pred_j3d_cam0_test = torch.cat([i["output"]["pred_j3d_cam"].to("cuda") for i in res0[0][begin:end]])
pred_j3d_cam1_test = torch.cat([i["output"]["pred_j3d_cam"].to("cuda") for i in res1[0][begin:end]])
pred_smpltrans0_test = torch.cat([i["output"]["pred_smpltrans"].to("cuda") for i in res0[0][begin:end]])
pred_smpltrans1_test = torch.cat([i["output"]["pred_smpltrans"].to("cuda") for i in res1[0][begin:end]])
gt_smpltrans0_test = torch.cat([i["output"]["gt_smpltrans"].to("cuda") for i in res0[0][begin:end]])
gt_smpltrans1_test = torch.cat([i["output"]["gt_smpltrans"].to("cuda") for i in res1[0][begin:end]])
pred_betas0 = torch.cat([i["output"]["pred_betas"].to("cuda") for i in res0[0][begin:end]])
pred_betas1 = torch.cat([i["output"]["pred_betas"].to("cuda") for i in res1[0][begin:end]])
pred_angles0_test = torch.cat([i["output"]["pred_angles"].to("cuda") for i in res0[0][begin:end]])
pred_angles1_test = torch.cat([i["output"]["pred_angles"].to("cuda") for i in res1[0][begin:end]])
pred_rotmat0_test = tgm.angle_axis_to_rotation_matrix(pred_angles0_test.view(-1,3)).view(pred_angles0_test.shape[0],22,4,4)
pred_rotmat1_test = tgm.angle_axis_to_rotation_matrix(pred_angles1_test.view(-1,3)).view(pred_angles1_test.shape[0],22,4,4)



pred_vertices_cam0_test_wrt_origin,pred_j3d_cam0_test_wrt_origin,pred_orient0_wrt_origin, pred_trans0_wrt_origin \
        = transform_smpl(torch.inverse(test_extr0[begin*batch_size:batch_size*(end-begin)]),pred_vertices_cam0_test,pred_j3d_cam0_test,pred_rotmat0_test[begin*batch_size:batch_size*(end-begin),1,:3,:3],pred_smpltrans0_test)
pred_vertices_cam1_test_wrt_origin,pred_j3d_cam1_test_wrt_origin,pred_orient1_wrt_origin, pred_trans1_wrt_origin \
        = transform_smpl(torch.inverse(test_extr1[begin*batch_size:batch_size*(end-begin)]),pred_vertices_cam1_test,pred_j3d_cam1_test,pred_rotmat1_test[begin*batch_size:batch_size*(end-begin),1,:3,:3],pred_smpltrans1_test)

smplx = SMPLX(os.path.join("/is/ps3/nsaini/projects/copenet","src/copenet/data/smplx/models/smplx"),
                         batch_size=4,
                         create_transl=False).to("cuda").eval()

joints3d = [] 
for i in tqdm(range(pred_rotmat1_test.shape[0])):
        out = smplx.forward(body_pose=torch.cat([test_ds[i]["smplpose_rotmat"].unsqueeze(0).to("cuda"),
                                                test_ds[i]["smplpose_rotmat"].unsqueeze(0).to("cuda"),
                                                pred_rotmat0_test[i:i+1,1:22,:3,:3],
                                                pred_rotmat1_test[i:i+1,1:22,:3,:3]]),
                                global_orient= torch.cat([test_ds[i]["smplorient_rel0"].to("cuda").unsqueeze(0),
                                                test_ds[i]["smplorient_rel1"].to("cuda").unsqueeze(0),
                                                pred_rotmat0_test[i:i+1,0:1,:3,:3],
                                                pred_rotmat1_test[i:i+1,0:1,:3,:3]]),pose2rot=False)
        joints3d.append(out.joints.detach().cpu().numpy())

j3d = np.stack(joints3d)
print("mpjpe0: {}".format(np.mean(np.sqrt(np.sum((j3d[:,0] - j3d[:,2])**2,2))[:,:22])))
print("mpjpe1: {}".format(np.mean(np.sqrt(np.sum((j3d[:,1] - j3d[:,3])**2,2))[:,:22])))
print("mpe0: {}".format(np.mean(np.sqrt(np.sum((gt_smpltrans0_test.detach().cpu().numpy() - \
                        pred_smpltrans0_test.detach().cpu().numpy())**2,1)))))
print("mpe1: {}".format(np.mean(np.sqrt(np.sum((gt_smpltrans1_test.detach().cpu().numpy() - \
                        pred_smpltrans1_test.detach().cpu().numpy())**2,1)))))
# %%
import cv2
import random
samples = random.sample(range(begin*batch_size,batch_size*(end-begin)),5)
ims0 = torch.from_numpy(np.stack([cv2.imread(f)[:,:,::-1]/255. for f in images0[samples]])).permute(0,3,1,2)
ims1 = torch.from_numpy(np.stack([cv2.imread(f)[:,:,::-1]/255. for f in images1[samples]])).permute(0,3,1,2)

rend_ims0 = synth_renderer.visualize_tb(pred_vertices_cam0_test[samples],
                        torch.zeros(len(samples),3).float().to("cuda"),
                        torch.eye(3).float().unsqueeze(0).repeat(len(samples),1,1).to("cuda"),
                        ims0)
rend_ims1 = synth_renderer.visualize_tb(pred_vertices_cam1_test[samples],
                        torch.zeros(len(samples),3).float().to("cuda"),
                        torch.eye(3).float().unsqueeze(0).repeat(len(samples),1,1).to("cuda"),
                        ims1)


for_blender_cam0 = pred_vertices_cam0_test_wrt_origin[samples]
for_blender_cam1 = pred_vertices_cam1_test_wrt_origin[samples]
for_blender_gt = gt_vertices[samples]