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

from copenet_real.dsets import copenet_real as copenet_real_data
# from copenet.dsets import aerialpeople


ckpt_path = "/is/ps3/nsaini/projects/copenet_real/copenet_logs/copenet_twoview/version_5_cont_limbwght/checkpoints/epoch=563.ckpt"

# check model type
model_type = ckpt_path.split("/")[-4]

# create trainer
trainer = Trainer(gpus=1)
# create Network
if model_type == "copenet_twoview":
    from copenet_real.copenet_twoview import copenet_twoview
    net = copenet_twoview.load_from_checkpoint(checkpoint_path=ckpt_path)
elif model_type == "hmr":
    from copenet_real.hmr import hmr
    net0 = hmr.load_from_checkpoint(checkpoint_path=ckpt_path)
    net1 = hmr.load_from_checkpoint(checkpoint_path=ckpt_path)

# create dataset and dataloader
train_ds, test_ds = copenet_real_data.get_copenet_real_traintest("/home/nsaini/Datasets/copenet_data")
train_ds_1, test_ds_1 = copenet_real_data.get_copenet_real_traintest("/home/nsaini/Datasets/copenet_data",first_cam=1)

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

# draw sample and forward
if model_type == "copenet_twoview":
    trainer.test(net,test_dataloaders=[tst_dl,trn_dl])
    pkl.dump(outputs,open(,"wb"))

    res = pkl.load(open("/is/ps3/nsaini/projects/copenet_real/copenet_logs/copenet_twoview/version_5_cont_limbwght/checkpoints/epoch=563.pkl","rb"))

elif model_type == "hmr":
    res0 = trainer.test(net0,test_dataloaders=[tst_dl,trn_dl])
    res1 = trainer.test(net1,test_dataloaders=[tst_dl1,trn_dl1])




# %% processing

fname = "/ps/project/datasets/AirCap_ICCV19/cvpr21_mat/copenet_twoview_version_5_cont_limbwght_checkpoints_epoch=563.pkl"
dset = "/ps/project/datasets/AirCap_ICCV19/copenet_data/"

import sys
# sys.path.append("/is/ps3/nsaini/projects/copenet_real/src")
from copenet_real.utils.utils import transform_smpl
import torchgeometry as tgm
from copenet_real.utils.renderer import Renderer
from copenet_real.smplx.smplx import SMPLX, lbs
from copenet_real.dsets import copenet_real as copenet_real_data
from copenet_real.dsets import aerialpeople
import pickle as pkl
import os
import torch
import numpy as np

smplx = SMPLX(os.path.join("/is/ps3/nsaini/projects/copenet","src/copenet/data/smplx/models/smplx"),
                         batch_size=1,
                         create_transl=False)

img_res = [1920,1080]
synth_focallen = [1475,1475]
CX0 = 1018
CY0 = 577
CX1 = 978
CY1 = 667
real_focallen0 = [1537,1517]
real_focallen1 = [1361,1378]

synth_renderer = Renderer(synth_focallen,img_res,[img_res[0]/2,img_res[1]/2],smplx.faces)
real_renderer0 = Renderer(real_focallen0,img_res,[CX0,CY0],smplx.faces)
real_renderer1 = Renderer(real_focallen1,img_res,[CX1,CY1],smplx.faces)

res = pkl.load(open(fname,"rb"))

model_type = "_".join(fname.split("/")[-1].split("_")[:2])


if "agora" in dset:
    train_ds,test_ds = aerialpeople.get_aerialpeople_seqsplit("/ps/project/datasets/AirCap_ICCV19/agora_copenet_uniform_new/")
    train_ds,test_ds = aerialpeople.get_aerialpeople_seqsplit("/ps/project/datasets/AirCap_ICCV19/agora_copenet_uniform_new/",first_cam=1)
    test_extr0 = torch.from_numpy(np.stack([pkl.load(open(f,"rb"))["cam0"]["extr"] for f in test_ds.db])).float().to("cuda")
    test_extr1 = torch.from_numpy(np.stack([pkl.load(open(f,"rb"))["cam1"]["extr"] for f in test_ds.db])).float().to("cuda")
    intr0 = torch.from_numpy(np.stack([pkl.load(open(f,"rb"))["cam0"]["intr"] for f in test_ds.db])).float().to("cuda")
    intr1 = torch.from_numpy(np.stack([pkl.load(open(f,"rb"))["cam1"]["intr"] for f in test_ds.db])).float().to("cuda")
    images0 = np.array([os.path.join(test_ds.data_root,pkl.load(open(f,"rb"))["im0"]) for f in test_ds.db])
    images1 = np.array([os.path.join(test_ds.data_root,pkl.load(open(f,"rb"))["im1"]) for f in test_ds.db])
else:
    train_ds, test_ds = copenet_real_data.get_copenet_real_traintest(dset)
    train_ds_1, test_ds_1 = copenet_real_data.get_copenet_real_traintest(dset,first_cam=1)
    test_extr0 = test_ds.extr0[8000:15000].to("cuda")
    test_extr1 = test_ds.extr1[8000:15000].to("cuda")
    intr0 = test_ds.intr0
    intr1 = test_ds.intr1
    images0 = np.array(test_ds.db["im0"])
    images1 = np.array(test_ds.db["im1"])

begin = 0
end = 10

if model_type == "copenet_twoview":
    batch_size = res[0][0]["output"]["pred_vertices_cam0"].shape[0]
    pred_vertices_cam0_test = torch.cat([i["output"]["pred_vertices_cam0"].to("cuda") for i in res[0][begin:end]])
    pred_vertices_cam1_test = torch.cat([i["output"]["pred_vertices_cam1"].to("cuda") for i in res[0][begin:end]])
    pred_j2d_cam0_test = torch.cat([i["output"]["pred_j2d_cam0"].to("cuda") for i in res[0][begin:end]])
    pred_j2d_cam1_test = torch.cat([i["output"]["pred_j2d_cam1"].to("cuda") for i in res[0][begin:end]])
    pred_j3d_cam0_test = torch.cat([i["output"]["pred_j3d_cam0"].to("cuda") for i in res[0][begin:end]])
    pred_j3d_cam1_test = torch.cat([i["output"]["pred_j3d_cam1"].to("cuda") for i in res[0][begin:end]])
    pred_smpltrans0_test = torch.cat([i["output"]["pred_smpltrans0"].to("cuda") for i in res[0][begin:end]])
    pred_smpltrans1_test = torch.cat([i["output"]["pred_smpltrans1"].to("cuda") for i in res[0][begin:end]])
    pred_angles0_test = torch.cat([i["output"]["pred_angles0"].to("cuda") for i in res[0][begin:end]])
    pred_angles1_test = torch.cat([i["output"]["pred_angles1"].to("cuda") for i in res[0][begin:end]])
    pred_rotmat0_test = tgm.angle_axis_to_rotation_matrix(pred_angles0_test.view(-1,3)).view(pred_angles0_test.shape[0],22,4,4)
    pred_rotmat1_test = tgm.angle_axis_to_rotation_matrix(pred_angles1_test.view(-1,3)).view(pred_angles1_test.shape[0],22,4,4)
    

    pred_vertices_cam0_test_wrt_origin,pred_j3d_cam0_test_wrt_origin,pred_orient0_wrt_origin, pred_trans0_wrt_origin \
            = transform_smpl(torch.inverse(test_extr0[begin*batch_size:batch_size*(end-begin)]),pred_vertices_cam0_test,pred_j3d_cam0_test,pred_rotmat0_test[begin*batch_size:batch_size*(end-begin),1,:3,:3],pred_smpltrans0_test)
    pred_vertices_cam1_test_wrt_origin,pred_j3d_cam1_test_wrt_origin,pred_orient1_wrt_origin, pred_trans1_wrt_origin \
            = transform_smpl(torch.inverse(test_extr1[begin*batch_size:batch_size*(end-begin)]),pred_vertices_cam1_test,pred_j3d_cam1_test,pred_rotmat1_test[begin*batch_size:batch_size*(end-begin),1,:3,:3],pred_smpltrans1_test)
    
    import cv2
    samples = [0,1,2,3,4] 
    ims0 = torch.from_numpy(np.stack([cv2.imread(f)[:,:,::-1]/255. for f in images0[samples]])).permute(0,3,1,2)
    ims1 = torch.from_numpy(np.stack([cv2.imread(f)[:,:,::-1]/255. for f in images1[samples]])).permute(0,3,1,2)

    rend_ims0 = real_renderer0.visualize_tb(pred_vertices_cam0_test[samples],
                            torch.zeros(len(samples),3).float().to("cuda"),
                            torch.eye(3).float().unsqueeze(0).repeat(len(samples),1,1).to("cuda"),
                            ims0)

elif model_type == "hmr":
    


