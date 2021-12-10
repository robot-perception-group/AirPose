# %%

import torch
import torchvision
from copenet_real.dsets import copenet_real
from copenet_real.utils.renderer import Renderer
from copenet.smplx.smplx import SMPLX 
from copenet.utils.geometry import perspective_projection, rot6d_to_rotmat
from copenet_real.utils.utils import transform_smpl
import torchgeometry as tgm
import pickle as pkl
from tqdm import tqdm
from torch import autograd
import cv2
import numpy as np

device = torch.device("cuda")

# %% bad grad check code
from graphviz import Digraph
import torch
from torch.autograd import Variable, Function

def iter_graph(root, callback):
    queue = [root]
    seen = set()
    while queue:
        fn = queue.pop()
        if fn in seen:
            continue
        seen.add(fn)
        for next_fn, _ in fn.next_functions:
            if next_fn is not None:
                queue.append(next_fn)
        callback(fn)

def register_hooks(var):
    fn_dict = {}
    def hook_cb(fn):
        def register_grad(grad_input, grad_output):
            fn_dict[fn] = grad_input
        fn.register_hook(register_grad)
    iter_graph(var.grad_fn, hook_cb)
    
    def is_bad_grad(grad_output):
        if grad_output is None:
            return False
        return torch.isnan(grad_output).any() or (grad_output.abs() >= 1e6).any()

    def make_dot():
        node_attr = dict(style='filled',
                        shape='box',
                        align='left',
                        fontsize='12',
                        ranksep='0.1',
                        height='0.2')
        dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))

        def size_to_str(size):
            return '('+(', ').join(map(str, size))+')'

        def build_graph(fn):
            if hasattr(fn, 'variable'):  # if GradAccumulator
                u = fn.variable
                node_name = 'Variable\n ' + size_to_str(u.size())
                dot.node(str(id(u)), node_name, fillcolor='lightblue')
            else:
                assert fn in fn_dict, fn
                fillcolor = 'white'
                if any(is_bad_grad(gi) for gi in fn_dict[fn]):
                    fillcolor = 'red'
                dot.node(str(id(fn)), str(type(fn).__name__), fillcolor=fillcolor)
            for next_fn, _ in fn.next_functions:
                if next_fn is not None:
                    next_id = id(getattr(next_fn, 'variable', next_fn))
                    dot.edge(str(next_id), str(id(fn)))
        iter_graph(var.grad_fn, build_graph)

        return dot

    return make_dot

# %% get the datset
trn_range = range(0,7000)
tst_range = range(8000,15000)
train_ds, test_ds = copenet_real.get_copenet_real_traintest("/home/nsaini/Datasets/copenet_data",train_range=trn_range,test_range=tst_range)
smplx_model = SMPLX("/is/ps3/nsaini/projects/copenet/src/copenet/data/smplx/models/smplx",
                batch_size=1,
                create_transl=False)
smplx_model.to(device)
smplx_model.eval()

from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser

vp_model = load_model("/ps/scratch/common/vposer/V02_05", model_code=VPoser,remove_words_in_model_weights="vp_model.")[0]
vp_model.to(device)
# vp_model.eval()

# def geman mcclure
def gmcclure(x,sigma):
    return x**2/1000
# gmcclure = torch.nn.MSELoss(reduction="none")

# %% Hypterparamerters
sigma2d = 40
sigma_camr = 1
sigma_camt = 1
w_beta = 10
w_cam = 100
w_vposer = 10
intr0 = torch.from_numpy(train_ds.intr0).float().to(device)
intr1 = torch.from_numpy(train_ds.intr1).float().to(device)

renderer0 = Renderer(focal_length=[intr0[0,0],intr0[1,1]], 
                            img_res=[1920,1080],
                            center=intr0[:2,2],
                            faces=smplx_model.faces)
renderer1 = Renderer(focal_length=[intr1[0,0],intr1[1,1]], 
                    img_res=[1920,1080],
                    center=intr1[:2,2],
                    faces=smplx_model.faces)

# %% all the vairables
camera_extr = torch.eye(4,device=device).float().unsqueeze(0).expand([2,train_ds.__len__(),-1,-1]).clone()
camera_extr_opt = torch.eye(4,device=device).float().unsqueeze(0).expand([2,train_ds.__len__(),-1,-1]).clone()
smplxtheta = (torch.eye(3,device=device)+torch.randn(3,3,device=device)*0.1).float().unsqueeze(0).expand([train_ds.__len__(),21,-1,-1]).clone()
smplxphi = torch.eye(3,device=device).float().unsqueeze(0).expand([train_ds.__len__(),-1,-1]).clone()
smplxtau = torch.zeros([train_ds.__len__(),3],device=device).float().clone()
smplxbeta = torch.zeros([train_ds.__len__(),10],device=device).float().clone()

# load camera extrinsics
c0 = pkl.load(open("/home/nsaini/Datasets/copenet_data/machine_1/markerposes_corrected_all.pkl","rb"))
c1 = pkl.load(open("/home/nsaini/Datasets/copenet_data/machine_2/markerposes_corrected_all.pkl","rb"))

k0 = sorted(c0.keys())
for k in trn_range:
    camera_extr[0,k,:,:] = tgm.angle_axis_to_rotation_matrix(torch.from_numpy(c0[k0[k]]["0"]["rvec"]).unsqueeze(0)).to(device)
    camera_extr[0,k,:3,3] = torch.from_numpy(c0[k0[k]]["0"]["tvec"]).to(device)
k1 = sorted(c1.keys())
for k in trn_range:
    camera_extr[1,k,:,:] = tgm.angle_axis_to_rotation_matrix(torch.from_numpy(c1[k1[k]]["0"]["rvec"]).unsqueeze(0)).to(device)
    camera_extr[1,k,:3,3] = torch.from_numpy(c1[k1[k]]["0"]["tvec"]).to(device)


# viz
from psbody.mesh.meshviewer import MeshViewer
from psbody.mesh.mesh import Mesh
mv = MeshViewer()
# import meshcat
# import meshcat.geometry as g
# import meshcat.transformations as tf
# vis = meshcat.Visualizer()

# main loop
first_loop = True
with autograd.detect_anomaly():
    for i in tqdm(range(350,train_ds.__len__())):
        if first_loop:
            pl_camera_extr_rot = camera_extr[:,i,:3,:2].detach().clone()
            pl_camera_extr_rot.requires_grad = True
            pl_camera_extr_trans = camera_extr[:,i,:3,3].detach().clone()
            pl_camera_extr_trans.requires_grad = True
            pl_smplxtheta = smplxtheta[i,:,:3,:2].detach().clone()
            pl_smplxtheta.requires_grad = True
            pl_smplxphi = smplxphi[i,:3,:2].detach().clone()
            pl_smplxphi.requires_grad = True
            pl_smplxtau = smplxtau[i].detach().clone()
            pl_smplxtau.requires_grad = True
            pl_smplxbeta = smplxbeta[i].detach().clone()
            pl_smplxbeta.requires_grad = True
            n_iters = 1000
        else:
            pl_camera_extr_rot = camera_extr[:,i-1,:3,:2].detach().clone()
            pl_camera_extr_rot.requires_grad = True
            pl_camera_extr_trans = camera_extr[:,i-1,:3,3].detach().clone()
            pl_camera_extr_trans.requires_grad = True
            pl_smplxtheta = smplxtheta[i-1,:,:3,:2].detach().clone()
            pl_smplxtheta.requires_grad = True
            pl_smplxphi = smplxphi[i-1,:3,:2].detach().clone()
            pl_smplxphi.requires_grad = True
            pl_smplxtau = smplxtau[i-1].detach().clone()
            pl_smplxtau.requires_grad = True
            pl_smplxbeta = smplxbeta[i-1].detach().clone()
            pl_smplxbeta.requires_grad = True
            n_iters = 100

        # create optimizer
        optim1 = torch.optim.Adam([pl_smplxphi,
                                pl_smplxtau],lr=0.01)
        optim2 = torch.optim.Adam([pl_camera_extr_rot,
                                pl_camera_extr_trans,
                                pl_smplxtheta,
                                pl_smplxphi,
                                pl_smplxtau,
                                pl_smplxbeta],lr=0.01)

        j2d = train_ds.get_j2d_only(i)
        joints2d_gt0 = j2d["smpl_joints_2d0"].to(device)
        joints2d_gt1 = j2d["smpl_joints_2d1"].to(device)

        # image0 = cv2.imread(j2d["im0_path"])/255.
        # image1 = cv2.imread(j2d["im1_path"])/255.

        for j in tqdm(range(n_iters)):

            pl_smplxtheta_9d = rot6d_to_rotmat(pl_smplxtheta.reshape(-1,6))
            # forward SMPLX
            smplx_out = smplx_model.forward(betas=pl_smplxbeta.unsqueeze(0), 
                                        body_pose=pl_smplxtheta_9d.unsqueeze(0),
                                        global_orient=torch.eye(3,device=device).float().unsqueeze(0).unsqueeze(0),
                                        transl = torch.zeros(1,3,device=device).float(),
                                        pose2rot=False)
            pl_smplxphi_9d = rot6d_to_rotmat(pl_smplxphi.view(1,6)).squeeze(0)
            
            # joints3d = torch.matmul(pl_smplxphi_9d[:3,:3],smplx_out.joints[0].permute(1,0)).permute(1,0) +\
            #                 pl_smplxtau.unsqueeze(0)
            transf_mat = torch.cat([pl_smplxphi_9d[:3,:3],
                                pl_smplxtau.unsqueeze(1)],dim=1).unsqueeze(0)
            verts,joints3d,_,_ = transform_smpl(transf_mat,
                                    smplx_out.vertices.squeeze(1),
                                    smplx_out.joints.squeeze(1))
            

            pl_camera_extr_rot_9d = rot6d_to_rotmat(pl_camera_extr_rot.reshape(2,6))
            pl_camera_extr = torch.cat([pl_camera_extr_rot_9d,pl_camera_extr_trans.unsqueeze(2)],dim=2)

            joints2d0 = perspective_projection(joints3d[:,:24],
                                                    rotation=pl_camera_extr[0,:3,:3].unsqueeze(0),
                                                    translation=pl_camera_extr[0,:3,3].unsqueeze(0),
                                                    focal_length=[intr0[0,0],intr0[1,1]],
                                                    camera_center=intr0[:2,2]).squeeze(0)
            
            joints2d1 = perspective_projection(joints3d[:,:24],
                                                    rotation=pl_camera_extr[1,:3,:3].unsqueeze(0),
                                                    translation=pl_camera_extr[1,:3,3].unsqueeze(0),
                                                    focal_length=[intr1[0,0],intr1[1,1]],
                                                    camera_center=intr1[:2,2]).squeeze(0)

            if first_loop:
                sigma = 1000
            else:
                sigma= sigma2d
            
            loss_2d = (joints2d_gt0[0,:,2:]*gmcclure(joints2d0-joints2d_gt0[0,:,:2],sigma)).mean() + \
                        (joints2d_gt0[1,:,2:]*gmcclure(joints2d0-joints2d_gt0[1,:,:2],sigma)).mean() + \
                        (joints2d_gt1[0,:,2:]*gmcclure(joints2d1-joints2d_gt1[0,:,:2],sigma)).mean() + \
                        (joints2d_gt1[1,:,2:]*gmcclure(joints2d1-joints2d_gt1[1,:,:2],sigma)).mean()

            
            loss_cam = gmcclure(pl_camera_extr[:,:3,:3]-camera_extr[:,i,:3,:3],sigma_camr).mean() + \
                        gmcclure(pl_camera_extr[:,:3,3]-camera_extr[:,i,:3,3],sigma_camt).mean()

            smplxtheta_aa = torch.cat([pl_smplxtheta_9d,torch.zeros([21,3,1],device=device)],dim=2)
            smplxtheta_aa = tgm.rotation_matrix_to_angle_axis(smplxtheta_aa).reshape([1,21*3])
            q_z0 = vp_model.encode(smplxtheta_aa)
            q_z_sample0 = q_z0.rsample()

            loss_vposer = torch.mul(q_z_sample0,q_z_sample0).mean()

            loss_beta = torch.mul(smplxbeta,smplxbeta).mean()

            loss = loss_2d + w_beta*loss_beta + w_cam*loss_cam + w_vposer*loss_vposer


            # viz #############################
            # vis0 = renderer0(verts[0].detach().clone().cpu(),
            #                                 pl_camera_extr[0,:3,3].detach().clone().cpu(),
            #                                 pl_camera_extr[0,:3,:3].detach().clone().cpu(),
            #                                 image0)
            # vis1 = renderer0(verts[0].detach().clone().cpu(),
            #                                 pl_camera_extr[1,:3,3].detach().clone().cpu(),
            #                                 pl_camera_extr[1,:3,:3].detach().clone().cpu(),
            #                                 image1)
            
            # cv2.imshow("im",np.concatenate([vis0[::3,::3],vis1[::3,::3]],axis=0))
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
            msh = Mesh(v=verts[0].detach().cpu().numpy(),f=smplx_model.faces)
            mv.static_meshes = [msh]
            # vis["mesh1"].set_object(g.TriangularMeshGeometry(verts[0].detach().cpu().numpy(),smplx_model.faces))
            ###################################

            # get_dot = register_hooks(loss)
            # loss.backward()
            # dot = get_dot()
            print(loss)
            if j < 100:
                optim1.zero_grad()
                loss.backward()
                optim1.step()
            else:
                optim2.zero_grad()
                loss.backward()
                optim2.step()

        

    # camera_extr_opt[:,i] = pl_camera_extr.detach().clone()
    # smplxtheta[i] = pl_smplxtheta.detach().clone()
    # smplxphitau[i] = pl_smplxphitau.detach().clone()
    # smplxbeta[i] = pl_smplxbeta.detach().clone()