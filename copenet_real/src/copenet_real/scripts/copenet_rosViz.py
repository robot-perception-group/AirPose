import rospy
import sys
import torch
import numpy as np
# from airpose.msg import AirposeNetworkResult
from std_msgs.msg import Float32MultiArray
import torch.nn.functional as F
from smplx import SMPLX
import meshcat
import meshcat.geometry as g


topic = sys.argv[1]

##############################################

def transform_smpl(trans_mat,smplvertices=None,smpljoints=None, orientation=None, smpltrans=None):
    verts =  torch.bmm(trans_mat[:,:3,:3],smplvertices.permute(0,2,1)).permute(0,2,1) +\
                    trans_mat[:,:3,3].unsqueeze(1)
    if smpljoints is not None:
        joints = torch.bmm(trans_mat[:,:3,:3],smpljoints.permute(0,2,1)).permute(0,2,1) +\
                         trans_mat[:,:3,3].unsqueeze(1)
    else:
        joints = None
    
    if smpltrans is not None:
        trans = torch.bmm(trans_mat[:,:3,:3],smpltrans.unsqueeze(2)).squeeze(2) +\
                         trans_mat[:,:3,3]
    else:
        trans = None

    if orientation is not None:
        orient = torch.bmm(trans_mat[:,:3,:3],orientation)
    else:
        orient = None    
    return verts, joints, orient, trans

def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x = x.reshape(-1,3,2)
    a1 = x[:, :, 0]
    a2 = x[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)

#################################################

rospy.init_node("AirPoseViz", anonymous=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

smplx = SMPLX("/is/ps3/nsaini/projects/copenet_real/src/copenet_real/data/smplx/models/smplx",
                         batch_size=1,
                         create_transl=False).to(device)
smplx.eval()

# Create a new visualizer
vis = meshcat.Visualizer()

meshname = topic.split("/")[1]
colorlist = [0x440154,
 0x482475,
 0x414487,
 0x355f8d,
 0x2a788e,
 0x21918c,
 0x22a884,
 0x44bf70,
 0x7ad151,
 0xbddf26]

clr = colorlist[int(meshname.split("_")[-1])]

def callback(data):
    betas = torch.from_numpy(np.array(data.data[:10])).to(device).float().unsqueeze(0)
    trans = torch.from_numpy(np.array(data.data[10:13])).to(device).float().unsqueeze(0)
    pose = rot6d_to_rotmat(torch.from_numpy(np.array(data.data[13:])).to(device).float()).unsqueeze(0)

    smplx_out = smplx.forward(betas=betas, 
                                body_pose=pose[:,1:],
                                global_orient=torch.eye(3,device=device).float().unsqueeze(0).unsqueeze(1),
                                transl = torch.zeros(1,3).float().type_as(betas),
                                pose2rot=False)
    transf_mat0 = torch.cat([pose[:,:1].squeeze(1),
                                trans.unsqueeze(2)],dim=2)
    verts,joints,_,_ = transform_smpl(transf_mat0,
                                                smplx_out.vertices.squeeze(1),
                                                smplx_out.joints.squeeze(1))

    vis[meshname].set_object(g.TriangularMeshGeometry(verts[0].detach().cpu().numpy(),smplx.faces),
                        g.MeshLambertMaterial(
                             color=clr,
                             reflectivity=0.8))


rospy.Subscriber(topic, Float32MultiArray, callback)

rospy.spin()