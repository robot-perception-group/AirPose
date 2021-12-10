# %%
import torch
import torchvision
from copenet_real.dsets import copenet_real
from copenet_real.utils.renderer import Renderer
from human_body_prior.tools.model_loader import load_model
from human_body_prior.models.vposer_model import VPoser
from human_body_prior.body_model.body_model import BodyModel
from copenet.smplx.smplx import SMPLX 
from copenet.utils.geometry import perspective_projection, rot6d_to_rotmat
from copenet_real.utils.utils import transform_smpl
from pytorch3d import transforms
import pickle as pkl
from tqdm import tqdm
from torch import autograd
import cv2
import numpy as np
import copy

device = torch.device("cuda")
smpl2op_jmap = torch.tensor([15,12,17,19,21,16,18,20,2,5,8,1,4,7])
cmap = np.random.rand(14,3)
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

def kp_viz(im,j2d):
    im_cp = copy.deepcopy(im)
    for i,j in enumerate(j2d[smpl2op_jmap]):
        cv2.circle(im_cp,(j[0],j[1]),10,cmap[i],-1)
    return im_cp

# %% get the datset

trn_range = range(0,7000)
tst_range = range(8000,15000)
train_ds, test_ds = copenet_real.get_copenet_real_traintest("/home/nsaini/Datasets/copenet_data",train_range=trn_range,test_range=tst_range)
smplx_model = BodyModel(bm_path="/is/ps3/nsaini/projects/copenet/src/copenet/data/smplx/models/smplx/SMPLX_NEUTRAL.npz")
smplx_model.to(device)
smplx_model.eval()

vp_model = load_model("/ps/scratch/common/vposer/V02_05", model_code=VPoser,remove_words_in_model_weights="vp_model.")[0]
vp_model.to(device)
vp_model.eval()

# def geman mcclure
def gmcclure(a,b,sigma):
    x=a-b
    return x**2/(x**2 + sigma**2)

# %% Hypterparamerters

for dataset in tqdm(["train"]):

    for begin in tqdm([0]):

        if begin == 6000:
            end = 6990
        else:
            end = begin + 2000
        lseq = end-begin

        sigma2d = 40
        
        if dataset == "train":
            viz_dir = "train_data"
            ds = train_ds
            res_id = 1
        else:
            viz_dir = "test_data"
            ds = test_ds
            res_id = 0
        w_beta = 100
        w_vposer = 100
        w_temporal = 100
        intr0 = torch.from_numpy(ds.intr0).float().to(device)
        intr1 = torch.from_numpy(ds.intr1).float().to(device)

        renderer0 = Renderer(focal_length=[intr0[0,0],intr0[1,1]], 
                                    img_res=[1920,1080],
                                    center=intr0[:2,2],
                                    faces=smplx_model.f.data.cpu())
        renderer1 = Renderer(focal_length=[intr1[0,0],intr1[1,1]], 
                            img_res=[1920,1080],
                            center=intr1[:2,2],
                            faces=smplx_model.f.data.cpu())

        # %% all the vairables
        smplxbeta = torch.zeros([10],device=device).float().clone()

        # fix parameters
        cam0_extr = torch.eye(4,device=device).float()

        # viz

        # from psbody.mesh.meshviewer import MeshViewer
        # from psbody.mesh.mesh import Mesh
        # mv = MeshViewer()

        # import meshcat
        # import meshcat.geometry as g
        # import meshcat.transformations as tf
        # vis = meshcat.Visualizer()

        # Get data and initializations
        fname = "/is/ps3/nsaini/projects/copenet_real/copenet_logs/copenet_twoview/version_5_cont_limbwght/checkpoints/epoch=761.pkl"
        res = pkl.load(open(fname,"rb"))

        smpl_angles0 = torch.cat([i["output"]["pred_angles0"].to("cuda") for i in res[res_id]])[begin:end]
        smpl_rootangle0 = smpl_angles0[:,0]
        smpl_z_init = vp_model.encode(smpl_angles0[:,1:]).mean

        smpl_rotmat0 = transforms.rotation_conversions.axis_angle_to_matrix(smpl_angles0)
        smpl_wrt_cam0 = torch.eye(4,device=device).float().unsqueeze(0).expand([lseq,-1,-1]).clone()
        smpl_wrt_cam0[:,:3,:3] = smpl_rotmat0[:,0]
        smpl_wrt_cam0[:,:3,3] = torch.cat([i["output"]["pred_smpltrans0"].to("cuda") for i in res[res_id]])[begin:end]

        smpl_wrt_cam1 = torch.eye(4,device=device).float().unsqueeze(0).expand([lseq,-1,-1]).clone()
        smpl_rootangle1 = torch.cat([i["output"]["pred_angles1"][:,0].to("cuda") for i in res[res_id]])[begin:end]
        smpl_wrt_cam1[:,:3,:3] = transforms.rotation_conversions.axis_angle_to_matrix(smpl_rootangle1)
        smpl_wrt_cam1[:,:3,3] = torch.cat([i["output"]["pred_smpltrans1"].to("cuda") for i in res[res_id]])[begin:end]
        cam1_wrt_smpl = torch.inverse(smpl_wrt_cam1)
        cam1_wrt_cam0 = torch.matmul(smpl_wrt_cam0,cam1_wrt_smpl)
        cam1_extr = torch.inverse(cam1_wrt_cam0)
        
        joints2d_gt0 = torch.cat([torch.from_numpy(ds.opose_smpl_fmt[0]).unsqueeze(1),torch.from_numpy(ds.apose_smpl_fmt[0]).unsqueeze(1)],dim=1).float().to(device)[begin:end]
        joints2d_gt1 = torch.cat([torch.from_numpy(ds.opose_smpl_fmt[1]).unsqueeze(1),torch.from_numpy(ds.apose_smpl_fmt[1]).unsqueeze(1)],dim=1).float().to(device)[begin:end]
        # joints2d_gt0 = torch.cat([ds.get_j2d_only(i)["smpl_joints_2d0"].unsqueeze(0) for i in range(len(ds))])[begin:end].to(device)
        # joints2d_gt1 = torch.cat([ds.get_j2d_only(i)["smpl_joints_2d1"].unsqueeze(0) for i in range(len(ds))])[begin:end].to(device)

        j_regressor = torch.from_numpy(pkl.load(open("/ps/project/common/expose_release/data/SMPLX_to_J14.pkl","rb"),encoding="latin1")).float().to(device)

        # main loop
        with autograd.detect_anomaly():
            pl_smplxtheta = smpl_z_init.detach().clone()
            pl_smplxtheta.requires_grad = True
            pl_smplxphi0 = transforms.matrix_to_rotation_6d(smpl_rotmat0[:,0,:3,:3]).detach().clone()
            pl_smplxphi0.requires_grad = True
            pl_smplxtau0 = smpl_wrt_cam0[:,:3,3].detach().clone()
            pl_smplxtau0.requires_grad = True
            pl_camextrrot1 = transforms.matrix_to_rotation_6d(cam1_wrt_cam0[:,:3,:3] ).detach().clone()
            pl_camextrrot1.requires_grad = True
            pl_camextrtau1 = cam1_wrt_cam0[:,:3,3].detach().clone()
            pl_camextrtau1.requires_grad = True
            pl_smplxbeta = smplxbeta.detach().clone()
            pl_smplxbeta.requires_grad = True
            n_iters = 300

            # create optimizer
            optim1 = torch.optim.Adam([pl_smplxphi0,
                                    pl_smplxtau0,
                                    pl_camextrrot1,
                                    pl_camextrtau1,
                                    pl_smplxbeta],
                                    lr=0.01)
            optim2 = torch.optim.Adam([pl_smplxtheta,
                                    pl_smplxphi0,
                                    pl_smplxtau0,
                                    pl_camextrrot1,
                                    pl_camextrtau1,
                                    pl_smplxbeta],
                                    lr=0.01)
            optim = optim1
            
            image0 = cv2.imread(ds.get_j2d_only(begin)["im0_path"])/255.
            image1 = cv2.imread(ds.get_j2d_only(begin)["im1_path"])/255.
            global verts0
            global verts1

            for j in tqdm(range(n_iters)):
                
                if j == 200:
                    optim = optim2
                
                pl_smplxtheta_3d = vp_model.decode(pl_smplxtheta)["pose_body"].reshape(-1,63)
                
                # forward SMPLX
                smplx_out = smplx_model.forward(betas=pl_smplxbeta.unsqueeze(0).expand([lseq,-1]), 
                                            pose_body=pl_smplxtheta_3d,
                                            root_orient=torch.zeros(lseq,3,device=device).float(),
                                            trans = torch.zeros(lseq,3,device=device).float())
                pl_smplxphi0_9d = transforms.rotation_6d_to_matrix(pl_smplxphi0).squeeze(0)
                
                transf_mat0 = torch.cat([pl_smplxphi0_9d[:,:3,:3],
                                    pl_smplxtau0.unsqueeze(2)],dim=2)
                verts0,joints3d0,_,_ = transform_smpl(transf_mat0,
                                        smplx_out.v,
                                        smplx_out.Jtr)

                # joints3d0 = torch.matmul(j_regressor,verts0)
                # joints3d1 = torch.matmul(j_regressor,verts1)
                joints2d0 = perspective_projection(joints3d0[:,:24],
                                                        rotation=cam0_extr[:3,:3].unsqueeze(0).expand([lseq,-1,-1]),
                                                        translation=cam0_extr[:3,3].expand([lseq,-1]),
                                                        focal_length=[intr0[0,0],intr0[1,1]],
                                                        camera_center=intr0[:2,2]).squeeze(0)
                cam_extr_rot1_9d = transforms.rotation_6d_to_matrix(pl_camextrrot1)
                joints2d1 = perspective_projection(joints3d0[:,:24],
                                                        rotation=cam_extr_rot1_9d,
                                                        translation=pl_camextrtau1,
                                                        focal_length=[intr1[0,0],intr1[1,1]],
                                                        camera_center=intr1[:2,2]).squeeze(0)

                sigma = sigma2d
                mse_loss = torch.nn.MSELoss(reduction="none")
                
                joints2d_gt0[:,:,[1,2],2:] /= 2   # less weight for hips
                joints2d_gt1[:,:,[1,2],2:] /= 2
                
                loss_2d = (joints2d_gt0[:,0,:,2:]*mse_loss(joints2d0,joints2d_gt0[:,0,:,:2])).mean() + \
                            (joints2d_gt1[:,0,:,2:]*mse_loss(joints2d1,joints2d_gt1[:,0,:,:2])).mean() + \
                            (joints2d_gt0[:,1,:,2:]*mse_loss(joints2d0,joints2d_gt0[:,1,:,:2])).mean() + \
                            (joints2d_gt1[:,1,:,2:]*mse_loss(joints2d1,joints2d_gt1[:,1,:,:2])).mean()

                

                loss_vposer = torch.mul(pl_smplxtheta,pl_smplxtheta).mean()

                loss_beta = torch.mul(smplxbeta,smplxbeta).mean()
                
                loss_temporal = mse_loss(pl_smplxtheta_3d[1:],pl_smplxtheta_3d[:-1]).mean() + \
                                mse_loss(pl_smplxphi0[1:],pl_smplxphi0[:-1]).mean() + \
                                mse_loss(pl_smplxtau0[1:],pl_smplxtau0[:-1]).mean() + \
                                mse_loss(pl_camextrrot1[1:],pl_camextrrot1[:-1]).mean() + \
                                mse_loss(pl_camextrtau1[1:],pl_camextrtau1[:-1]).mean()

                loss = loss_2d + w_beta*loss_beta + w_vposer*loss_vposer + w_temporal*loss_temporal
                

                # if j % 100 == 0:
                # # viz #############################
                #     vis0 = renderer0(verts0[0].detach().clone().cpu(),
                #                                     cam0_extr[:3,3].detach().clone().cpu(),
                #                                     cam0_extr[:3,:3].detach().clone().cpu(),
                #                                     image0)
                #     vis1 = renderer1(verts1[0].detach().clone().cpu(),
                #                                     cam1_extr[:3,3].detach().clone().cpu(),
                #                                     cam1_extr[:3,:3].detach().clone().cpu(),
                #                                     image1)
                #     # vis0 = kp_viz(vis0,joints2d0[0])
                #     # vis1 = kp_viz(vis1,joints2d1[0])
                #     cv2.imshow("im",np.concatenate([vis0[::3,::3],vis1[::3,::3]],axis=0))
                #     if cv2.waitKey(1) & 0xFF == ord('q'):
                #         break
                    
                    # msh = Mesh(v=verts[0].detach().cpu().numpy(),f=smplx_model.faces)
                    # mv.static_meshes = [msh]
                
                    # for m in [0,end-1]:
                    #     vis["mesh"+str(m)].set_object(g.TriangularMeshGeometry(verts[m].detach().cpu().numpy(),smplx_model.faces))
                ###################################
                # import ipdb;ipdb.set_trace()
                # get_dot = register_hooks(loss)
                # loss.backward()
                # dot = get_dot()
                
                # print(loss)
                optim.zero_grad()
                loss.backward()
                optim.step()

            import ipdb;ipdb.set_trace()

            import meshcat
            import meshcat.geometry as g
            import meshcat.transformations as tf
            vis = meshcat.Visualizer()
            import time
            for m in range(verts0.shape[0]):
                vis["mesh"].set_object(g.TriangularMeshGeometry(verts0[m].detach().cpu().numpy(),smplx_model.f.detach().cpu().numpy()))
                time.sleep(0.05)

            # camera_extr_opt[:,i] = pl_camera_extr.detach().clone()
            # smplxtheta[i] = pl_smplxtheta.detach().clone()
            # smplxphitau[i] = pl_smplxphitau.detach().clone()
            # smplxbeta[i] = pl_smplxbeta.detach().clone()
            
            # for v_id in tqdm(range(verts0.shape[0])):
            #     image0 = cv2.imread(ds.get_j2d_only(begin+v_id)["im0_path"])/255.
            #     image1 = cv2.imread(ds.get_j2d_only(begin+v_id)["im1_path"])/255.
            #     vis0 = renderer0(verts0[v_id].detach().clone().cpu(),
            #                                 cam0_extr[:3,3].detach().clone().cpu(),
            #                                 cam0_extr[:3,:3].detach().clone().cpu(),
            #                                 image0)
            #     vis1 = renderer1(verts1[v_id].detach().clone().cpu(),
            #                                     cam1_extr[:3,3].detach().clone().cpu(),
            #                                     cam1_extr[:3,:3].detach().clone().cpu(),
            #                                     image1)
            #     # vis0 = kp_viz(vis0,joints2d0[0])
            #     # vis1 = kp_viz(vis1,joints2d1[0])
            #     cv2.imwrite("/is/ps3/nsaini/projects/copenet_real_data/scripts/fit_viz/"+viz_dir+"/{:06d}".format(begin+v_id)+".jpg",np.concatenate([vis0[::3,::3],vis1[::3,::3]],axis=0)*255)

            # np.savez("/is/ps3/nsaini/projects/copenet_real_data/scripts/fit_viz/"+viz_dir+"_"+str(begin)+"_"+str(end)+".npz",
            #             thetas=pl_smplxtheta_3d.detach().cpu().numpy(),
            #             betas=pl_smplxbeta.detach().cpu().numpy(),
            #             root_rot0=pl_smplxphi0_9d.detach().cpu().numpy(),
            #             root_rot1=pl_smplxphi1_9d.detach().cpu().numpy(),
            #             trans0=pl_smplxtau0.detach().cpu().numpy(),
            #             trans1=pl_smplxtau1.detach().cpu().numpy())