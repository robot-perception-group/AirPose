'''
json_file_path: path to settings.json
return camera extrinsics i.e. inverse of camera pose matrix
'''
import os
import os.path as osp
import numpy as np
import pickle as pk
import trimesh
from tqdm import tqdm
import torch
import torchgeometry as tgm
import csv
from smplx import SMPLX
from utils import npPerspProj
import json
import torchgeometry as tgm
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from plyfile import PlyData, PlyElement
import pandas as pd

# data_root = '/home/nsaini/Datasets/AerialPeople/agora_copenet_data'
dataset_root = '/ps/project/datasets/AirCap_ICCV19/agora_unreal_4views'
data_dir = osp.join(dataset_root,"data")
final_dataset_dir = osp.join(dataset_root,"dataset")
# data_root = '/is/cluster/work/nsaini/agora_copenet_data'
# gt_root = '/ps/project/common/renderpeople_initialfit/init_fit_good_results/fitting'
gt_root = "/home/nsaini/Datasets/AerialPeople/smplx_fittings"
# scan_data_root = '/ps/project/body/datasets/ClothingModels/renderpeople/data_render'
gender_file = '/home/nsaini/Datasets/AerialPeople/gt_scan_info/cam_ready_gt_info_train.csv'
# final_dataset_dir = '/home/nsaini/Datasets/AerialPeople/agora_copenet_test'
# final_dataset_dir = '/is/cluster/work/nsaini/agora_copenet'
# testset_file = "testset_rp_0302.txt"

# with open(testset_file,'r') as f:
#     tstset = f.readlines()

# tstset = [t.strip() for t in tstset]

gender_df = pd.read_csv(gender_file)
gender_dict = dict(zip(gender_df["Scan name"],gender_df["gender"]))

if os.path.exists(final_dataset_dir):
    print('Dataset directory exists. Continue to overwrite it!!')
    import ipdb; ipdb.set_trace()
else:
    os.mkdir(final_dataset_dir)


# def get_cams_from_airsim(json_file_path):
json_file_path = os.path.join(data_dir, 'settings.json')

print('Getting camera parameters')

# with open(os.path.join(data_root, 'camera_pose.pkl'), 'rb') as f:
#     settings = pk.load(f, encoding='latin1')

with open(json_file_path, 'r') as f:
    intr_settings = json.load(f)['Vehicles']['machine_1']['Cameras']

cams = {}
for i, cam in enumerate(intr_settings):
    im_width = intr_settings[cam]['CaptureSettings'][0]['Width']
    im_height = intr_settings[cam]['CaptureSettings'][0]['Height']
    fov_rad = intr_settings[cam]['CaptureSettings'][0]['FOV_Degrees'] * np.pi / 180
    # fov_rad = 90*np.pi/180
    f = im_width / (2 * np.tan(fov_rad / 2))
    cx = im_width / 2
    cy = im_height / 2
    intr = np.eye(3)
    intr[0, 0] = f
    intr[1, 1] = f
    intr[0, 2] = cx
    intr[1, 2] = cy
    # cams['cam' + str(i)] = {'intr': intr, 'extr': np.linalg.inv(settings['Camera' + str(i)])}

# with open(os.path.join(final_dataset_dir, 'cameras.pkl'), 'wb') as f:
#     pk.dump(cams, f)

# print('\nCameras written')

# %%
import os
import numpy as np
import pickle as pk
import trimesh
from tqdm import tqdm
import torch
import torchgeometry as tgm
import csv
from smplx import SMPLX
from utils import npPerspProj, transform_smpl
from scipy.spatial.transform import Rotation as scirot
import shutil

from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser

num_pos = 20
num_cams = 4
img_res = [1920, 1080]
# horiz_dist = 8
# vert_dist = 8

smplx_path = '/ps/project/common/smplifyx/models/smplx'

weight_file = '/ps/project/common/smplifyx/vposer_weights/0010_0700R04V04_cvpr19/weights_npy/vposerDecoderWeights.npz'

smplx_model_male = SMPLX(os.path.join(smplx_path, 'SMPLX_MALE.npz'))
smplx_model_female = SMPLX(os.path.join(smplx_path, 'SMPLX_FEMALE.npz'))
smplx_model_neutral = SMPLX(os.path.join(smplx_path, 'SMPLX_NEUTRAL.npz'))

# with open(gender_file) as f:
#     gender_csv = csv.reader(f, delimiter=';')
#     gender_dict = dict()
#     for row in gender_csv:
#         if row[1] == 'Single Model':
#             gender_dict['rp_' + row[0].lower().split(' ')[0]] = row[2]


not_added = []

expr_dir = './vposer_v1_0'  # 'TRAINED_MODEL_DIRECTORY'  in this directory the trained model along with the model code exist
vp, ps = load_model("/ps/scratch/common/vposer/V02_05", model_code=VPoser,remove_words_in_model_weights="vp_model.")
vp = vp.to('cuda')

# for env in tqdm(os.listdir(data_root)):
#     if os.path.exists(os.path.join(final_dataset_dir, env)):
#         shutil.rmtree(os.path.join(final_dataset_dir, env))

#     os.mkdir(os.path.join(final_dataset_dir, env))
#     os.mkdir(os.path.join(final_dataset_dir, env,'data'))
#     # os.mkdir(os.path.join(final_dataset_dir, env,'test'))


if os.path.exists(os.path.join(final_dataset_dir,"pkls")):
    print('Pkls directory exists. Continue to overwrite it!!')
    import ipdb; ipdb.set_trace()
else:
    os.mkdir(os.path.join(final_dataset_dir,"pkls"))

for subject in tqdm(os.listdir(os.path.join(data_dir))):
    if os.path.isdir(os.path.join(data_dir,subject)):
        sub = '_'.join(subject.split('_')[:4])
        os.mkdir(os.path.join(final_dataset_dir,"pkls",sub))
        files = os.listdir(os.path.join(data_dir, subject))
        use_this = True

        # get gt fittings data
        gt_file = os.path.join(gt_root, sub + '_0_0.pkl')

        if os.path.exists(gt_file):
            with open(gt_file, 'rb') as f:
                gt = pk.load(f)

            smpltrans = gt['transl']
            smplshape = gt['betas'][0:1, :10]
            smplorient = gt['global_orient']
            smplorient_rotmat = tgm.angle_axis_to_rotation_matrix(torch.from_numpy(smplorient).float())[:,:3,:3]
            try:
                if "gender" in gt.keys():
                    smplgender = gt["gender"]
                else:
                    smplgender = gender_dict[sub + "_30k"]
            except:
                import ipdb; ipdb.set_trace()
            
            smplpose = torch.from_numpy(gt['body_pose'].reshape(1,21,3)).float()
            smplpose_flattened = smplpose.view(-1,63)

            if smplgender.upper() == 'MALE':
                smpl_wrt_scan = smplx_model_male.forward(betas=torch.from_numpy(smplshape),
                                                                global_orient=torch.from_numpy(gt['global_orient']),
                                                                transl=torch.from_numpy(gt['transl']),
                                                                body_pose=smplpose_flattened)
                pelvis = smplx_model_male.forward(betas=torch.from_numpy(smplshape)).joints[0,0]
                p_minus_rp = pelvis - torch.matmul(smplorient_rotmat,pelvis)
            elif smplgender.upper() == 'FEMALE':
                smpl_wrt_scan = smplx_model_female.forward(betas=torch.from_numpy(smplshape),
                                                                global_orient=torch.from_numpy(gt['global_orient']),
                                                                transl=torch.from_numpy(gt['transl']),
                                                                body_pose=smplpose_flattened)
                pelvis = smplx_model_female.forward(betas=torch.from_numpy(smplshape)).joints[0,0]
                p_minus_rp = pelvis - torch.matmul(smplorient_rotmat,pelvis)
            else:
                smpl_wrt_scan = smplx_model_neutral.forward(betas=torch.from_numpy(smplshape),
                                                                global_orient=torch.from_numpy(gt['global_orient']),
                                                                transl=torch.from_numpy(gt['transl']),
                                                                body_pose=smplpose_flattened)
                pelvis = smplx_model_neutral.forward(betas=torch.from_numpy(smplshape)).joints[0,0]
                p_minus_rp = pelvis - torch.matmul(smplorient_rotmat,pelvis)

        
        else:
            tqdm.write('gt fitting does not exist')
            use_this = False
            dsample['reason'] = 'gt fitting'
            not_added.append(dsample)
            # import ipdb; ipdb.set_trace()
            continue

        # get gt position in unreal
        try:
            gt_data = np.load(os.path.join(data_dir, subject, 'pose_single.npy'),allow_pickle=True,encoding='latin1')
            gt_position = gt_data.item()['person']
            gt_cam = [gt_data.item()['MyCamera'+str(c)] for c in range(num_cams)]
        except:
            tqdm.write('unable to load pose')
            use_this = False
            dsample['reason'] = 'unreal pose'
            not_added.append(dsample)
            # import ipdb;ipdb.set_trace()
            continue

        # get images
        for pos in range(num_pos):
            dsample = dict()

            # generate GT pose of the subject in unreal (gt_position is the pose matrix)
            # import ipdb; ipdb.set_trace()
            smpl_wrt_origin = transform_smpl(torch.from_numpy(gt_position[pos:pos + 1]).float(),
                                                smpl_wrt_scan.vertices, smpl_wrt_scan.joints,smplorient_rotmat)
            
            dsample['smpl_vertices_wrt_origin'] = smpl_wrt_origin[0].data.cpu().numpy()
            dsample['smpl_joints_wrt_origin'] = smpl_wrt_origin[1].data.cpu().numpy()
            dsample['scanpose'] = gt_position[pos:pos + 1].astype(np.float32)
            tfm = gt_position[pos:pos+1][0]
            dsample['smpltrans'] = (np.matmul(tfm[:3,:3],(smpltrans[0]+p_minus_rp[0].data.numpy())) + tfm[:3,3])[np.newaxis]
            dsample['smplshape'] = smplshape
            dsample['smplpose'] = smplpose.data.cpu().numpy()
            dsample['smplorient'] = smplorient
            dsample['smplgender'] = smplgender
            dsample['smplorient_rotmat_wrt_origin'] = smpl_wrt_origin[2].data.cpu().numpy()
            # import ipdb; ipdb.set_trace()
            # get cameras
            for c in range(num_cams):
                dsample['cam'+str(c)] = {'extr':np.linalg.inv(gt_cam[c][pos]),'intr':intr}

            for cam in range(num_cams):
                if os.path.exists(
                        os.path.join(dataset_root,"data", subject, 'MyCamera' + str(cam) + '_' + str(pos) + '.png')):
                        
                    dsample['im' + str(cam)] = os.path.join("data", subject,
                                                            'MyCamera' + str(cam) + '_' + str(pos) + '.png')
                    
                    # get 3D joints
                    j3d = smpl_wrt_origin[1].data.cpu().numpy()
                    # project on the camera
                    j2d = npPerspProj(dsample['cam' + str(cam)]['intr'], j3d,
                                        np.linalg.inv(dsample['cam' + str(cam)]['extr']))[:23, :]
                    dsample['j2d' + str(cam)] = j2d
                    mincorner = np.array([max(min(j2d[:, 0]) - 20, 0), max(min(j2d[:, 1]) - 20, 0)]).astype(int)
                    maxcorner = np.array([min(max(j2d[:, 0]) + 20, im_width), min(max(j2d[:, 1]) + 20, im_height)]).astype(int)

                    
                    # if the subject is not in the frame ignore the sample
                    if min(j2d[:,0]) < 0 or max(j2d[:,0]) > im_width or min(j2d[:,1]) < 0 or max(j2d[:,1]) > im_height:
                    # if mincorner[0] > 2448 or mincorner[1] > 2048 or maxcorner[0] < 0 or maxcorner[1] < 0:
                        use_this = False
                        dsample['reason'] = 'subject not in frame'

                    dsample['bb' + str(cam)] = [mincorner, maxcorner]
                else:
                    tqdm.write('image does not exist')
                    use_this = False
                    # import ipdb; ipdb.set_trace()
                    continue
            
            

            if use_this:
                with open(os.path.join(final_dataset_dir,"pkls", sub ,sub + '_' + str(pos) + '.pkl'), 'wb') as f:
                    pk.dump(dsample, f)
                    
            else:
                not_added.append(dsample)


with open(os.path.join(final_dataset_dir, 'not_added.pkl'), 'wb') as f:
    pk.dump(not_added, f)
tqdm.write('\nData written')

# %% Visualization

import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
# return cam_extr
from renderer import Renderer
from smplx import SMPLX
import cv2
from smplx import lbs


# 4841
with open(train_data[4841], 'rb') as f:
    dsample = pk.load(f)
    j3do = torch.from_numpy(dsample['smpl_joints_wrt_origin'])
    vertso = torch.from_numpy(dsample['smpl_vertices_wrt_origin'])
# import ipdb; ipdb.set_trace()
renderer = Renderer(dsample['cam0']['intr'][0, 0],
                    [int(dsample['cam0']['intr'][0, 2] * 2), int(dsample['cam0']['intr'][1, 2] * 2)],
                    smplx_model_neutral.faces)

from utils import _npcircle
from utils import npPerspProj
import torchvision

if dsample['smplgender'].upper() == 'MALE':
    smplx_model = smplx_model_male
if dsample['smplgender'].upper() == 'FEMALE':
    smplx_model = smplx_model_female
if dsample['smplgender'].upper() == 'Neutral':
    smplx_model = smplx_model_neutral

from utils import get_weak_persp_cam_full_img
img = []
rend_img = []
for i in range(2):
    verts, j3d, smpl_orient_rotmat = transform_smpl(torch.from_numpy(dsample['cam' + str(i)]['extr']).unsqueeze(0).float(), vertso, j3do, torch.from_numpy(dsample['smplorient_rotmat_wrt_origin']))
    campose = np.linalg.inv(dsample['cam' + str(i)]['extr'])
    img.append(cv2.imread(dsample['im' + str(i)]))
    # rend_img.append(renderer(verts,campose[:3,3],campose[:3,:3],img[i]/255)[0])
    # intr = torch.from_numpy(dsample['cam'+str(i)]['intr']).float()
    # wcam = torch.from_numpy(get_weak_persp_cam_full_img(intr,dsample['j2d'+str(i)][0],j3d[0][0].data.cpu().numpy()))
    # smpl_trans_cam_z = 2*intr[0,0]/(intr[1,2]*2 * wcam[0] +1e-9)
    # smpl_trans_cam_x = smpl_trans_cam_z * wcam[1]
    # smpl_trans_cam_y = smpl_trans_cam_z * wcam[2]
    # smpl_trans_cam = torch.stack([smpl_trans_cam_x,smpl_trans_cam_y,smpl_trans_cam_z],dim=0).unsqueeze(0)
    # smplpose_rotmat = lbs.batch_rodrigues(dsample['smplpose'].reshape(-1,3))
    # smplverts_wrt_cam = smplx_model.forward(betas=torch.from_numpy(dsample['smplshape']),
    #                                                                 global_orient=smplorient_rotmat,
    #                                                                 transl=torch.zeros(1,3).float(),
    #                                                                 body_pose=smplpose_rotmat)
    # import ipdb; ipdb.set_trace()
    rend_img.append(renderer(verts[0].data.cpu().numpy(), torch.zeros(3), torch.eye(3), img[i] / 255)[0])
    j2d = npPerspProj(dsample['cam' + str(i)]['intr'], j3d.data.cpu().numpy(), np.eye(4))[:23, :]
    for j in j2d:
        _npcircle(img[i], j[0], j[1], 5, [255, 0, 0])

imgs = torchvision.utils.make_grid(torch.from_numpy(np.array(img)).permute(0, 3, 1, 2), nrow=2).permute(1, 2, 0)
rend_imgs = torchvision.utils.make_grid(torch.from_numpy(np.array(rend_img)).permute(0, 3, 1, 2), nrow=2).permute(1, 2, 0)

# grid_cam = 0
# campose = np.linalg.inv(dsample['cam'+str(grid_cam)]['extr'])
# out_img,cam = renderer(verts,campose[:3,3],campose[:3,:3],img[grid_cam]/255)
# tempoints = torch.zeros(30,3)
# tempoints[:10,0] = torch.arange(10)
# tempoints[10:20,1] = torch.arange(10)
# tempoints[20:30,2] = torch.arange(10)
# axis2d = npPerspProj(dsample['cam'+str(grid_cam)]['intr'],tempoints,campose)[:30,:]
# for j in axis2d[:10]:
#     _npcircle(out_img,j[0],j[1],8,[0,255,255])
# for j in axis2d[10:20]:
#     _npcircle(out_img,j[0],j[1],8,[255,0,255])
# for j in axis2d[20:30]:
#     _npcircle(out_img,j[0],j[1],8,[255,255,0])

# orig2d = npPerspProj(dsample['cam'+str(grid_cam)]['intr'],torch.zeros(1,3),campose)[:21,:]
# for j in orig2d:
#     _npcircle(out_img,j[0],j[1],8,[255,255,255])

plt.imshow(rend_imgs.data.numpy()[:, :, ::-1])

# import torchgeometry as tgm
# airsim_cam_correk = np.array([[0,0,1,0],[1,0,0,0],[0,1,0,0],[0,0,0,1]])
# torch.matmul(tgm.angle_axis_to_rotation_matrix(tgm.quaternion_to_angle_axis(torch.tensor(quat).unsqueeze(0))),torch.from_numpy(airsim_cam_correk).float())

# %%
