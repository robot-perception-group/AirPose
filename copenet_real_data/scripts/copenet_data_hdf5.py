# create hdf5 dataset for aerialpeople
import h5py
import pickle as pk
import os
from tqdm import tqdm

pkl_main_dir = "/home/nsaini/Datasets/AerialPeople/agora_copenet_uniform_new_cropped/dataset/pkls"
h5_file = "/home/nsaini/Datasets/AerialPeople/agora_copenet_uniform_new_cropped/dataset/aerialpeople.hdf5"
pkl_dirs = os.listdir(pkl_main_dir)

f = h5py.File(h5_file, "w")

grp = f.create_group("train_test_split")
tr = pk.load(open("/home/nsaini/Datasets/AerialPeople/agora_copenet_uniform_new_cropped/dataset/train_pkls.pkl","rb"))
tst = pk.load(open("/home/nsaini/Datasets/AerialPeople/agora_copenet_uniform_new_cropped/dataset/test_pkls.pkl","rb"))
tr = ["/".join(i.split(".")[0].split("/")[-2:]).encode("ascii","ignore") for i in tr]
tst = ["/".join(i.split(".")[0].split("/")[-2:]).encode("ascii","ignore") for i in tst]
grp.create_dataset("train",(len(tr),1),'S10',tr)
grp.create_dataset("test",(len(tst),1),'S10',tst)

for i in tqdm(pkl_dirs):
    igrp = f.create_group(i)
    for j in os.listdir(os.path.join(pkl_main_dir,i)): 
        da = pk.load(open(os.path.join(pkl_main_dir,i,j),"rb"))
        jgrp = igrp.create_group(j)
        for k in da.keys():
            if k == "cam0" or k == "cam1":
                jgrp.create_dataset(k+"_extr",data=da[k]["extr"])
                jgrp.create_dataset(k+"_intr",data=da[k]["intr"])
            else:
                jgrp.create_dataset(k,data=da[k])

f.close()



#%% Create copenet real dataset for julia

from copenet_real.dsets import copenet_real
import torch
import h5py
import pickle as pkl
from pytorch3d import transforms

trn_range = range(0,7000)
tst_range = range(8000,15000)
train_ds, test_ds = copenet_real.get_copenet_real_traintest("/home/nsaini/Datasets/copenet_data",train_range=trn_range,test_range=tst_range)

f = h5py.File("/home/nsaini/Datasets/copenet_data/copenet_real.hdf5", "w")

joints2d_train_gt0 = torch.cat([train_ds.get_j2d_only(i)["smpl_joints_2d0"].unsqueeze(0) for i in range(len(train_ds))]).data.numpy()
joints2d_train_gt1 = torch.cat([train_ds.get_j2d_only(i)["smpl_joints_2d1"].unsqueeze(0) for i in range(len(train_ds))]).data.numpy()

joints2d_test_gt0 = torch.cat([test_ds.get_j2d_only(i)["smpl_joints_2d0"].unsqueeze(0) for i in range(len(test_ds))]).data.numpy()
joints2d_test_gt1 = torch.cat([test_ds.get_j2d_only(i)["smpl_joints_2d1"].unsqueeze(0) for i in range(len(test_ds))]).data.numpy()

fname = "/is/ps3/nsaini/projects/copenet_real/copenet_logs/copenet_twoview/version_5_cont_limbwght/checkpoints/epoch=761.pkl"
res = pkl.load(open(fname,"rb"))

smpl_angles0_train = torch.cat([i["output"]["pred_angles0"] for i in res[1]])
smpl_rotmat0_train = transforms.rotation_conversions.axis_angle_to_matrix(smpl_angles0_train)
smpl_wrt_cam0_train = torch.eye(4).float().unsqueeze(0).expand([6990,-1,-1]).clone()
smpl_wrt_cam0_train[:,:3,:3] = smpl_rotmat0_train[:,0]
smpl_wrt_cam0_train[:,:3,3] = torch.cat([i["output"]["pred_smpltrans0"] for i in res[1]])

smpl_angles1_train = torch.cat([i["output"]["pred_angles0"] for i in res[1]])
smpl_rotmat1_train = transforms.rotation_conversions.axis_angle_to_matrix(smpl_angles1_train)
smpl_wrt_cam1_train = torch.eye(4).float().unsqueeze(0).expand([6990,-1,-1]).clone()
smpl_wrt_cam1_train[:,:3,:3] = smpl_rotmat1_train[:,0]
smpl_wrt_cam1_train[:,:3,3] = torch.cat([i["output"]["pred_smpltrans1"] for i in res[1]])

smpl_angles0_test = torch.cat([i["output"]["pred_angles0"] for i in res[0]])
smpl_rotmat0_test = transforms.rotation_conversions.axis_angle_to_matrix(smpl_angles0_test)
smpl_wrt_cam0_test = torch.eye(4).float().unsqueeze(0).expand([6990,-1,-1]).clone()
smpl_wrt_cam0_test[:,:3,:3] = smpl_rotmat0_test[:,0]
smpl_wrt_cam0_test[:,:3,3] = torch.cat([i["output"]["pred_smpltrans0"] for i in res[1]])

smpl_angles1_test = torch.cat([i["output"]["pred_angles0"] for i in res[0]])
smpl_rotmat1_test = transforms.rotation_conversions.axis_angle_to_matrix(smpl_angles1_test)
smpl_wrt_cam1_test = torch.eye(4).float().unsqueeze(0).expand([6990,-1,-1]).clone()
smpl_wrt_cam1_test[:,:3,:3] = smpl_rotmat1_train[:,0]
smpl_wrt_cam1_test[:,:3,3] = torch.cat([i["output"]["pred_smpltrans1"] for i in res[0]])

f.create_dataset("joints2d_train_gt0",data=joints2d_train_gt0)
f.create_dataset("joints2d_train_gt1",data=joints2d_train_gt1)
f.create_dataset("joints2d_test_gt0",data=joints2d_test_gt0)
f.create_dataset("joints2d_test_gt1",data=joints2d_test_gt1)

f.create_dataset("smpl_wrt_cam0_train",data=smpl_wrt_cam0_train)
f.create_dataset("smpl_wrt_cam1_train",data=smpl_wrt_cam1_train)
f.create_dataset("smpl_wrt_cam0_test",data=smpl_wrt_cam0_test)
f.create_dataset("smpl_wrt_cam1_test",data=smpl_wrt_cam1_test)

tr0 = [train_ds.get_j2d_only(i)["im0_path"].encode("ascii","ignore") for i in range(len(train_ds))]
tr1 = [train_ds.get_j2d_only(i)["im1_path"].encode("ascii","ignore") for i in range(len(train_ds))]
tst0 = [test_ds.get_j2d_only(i)["im0_path"].encode("ascii","ignore") for i in range(len(test_ds))]
tst1 = [test_ds.get_j2d_only(i)["im1_path"].encode("ascii","ignore") for i in range(len(test_ds))]
f.create_dataset("im0_train",(len(tr0),1),'S10',tr0)
f.create_dataset("im1_train",(len(tr1),1),'S10',tr1)
f.create_dataset("im0_test",(len(tst0),1),'S10',tst0)
f.create_dataset("im1_test",(len(tst1),1),'S10',tst1)

f.close()