import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchgeometry as tgm
import cv2

class CRotDecoder(nn.Module):
    ''' Decoder for transforming a latent representation to rotation matrices
        Author: Unknown
        Implements the decoding method described in:
        "On the Continuity of Rotation Representations in Neural Networks"
    '''

    def __init__(self, num_angles, dtype=torch.float32,
                 **kwargs):
        super(CRotDecoder, self).__init__()
        self.num_angles = num_angles
        self.dtype = dtype

    def get_dim_size(self):
        return self.num_angles * 6

    def get_mean(self):
        mean = torch.tensor(
            [1.0, 0.0, 0.0, -1.0, 0.0, 0.0],
            dtype=self.dtype).unsqueeze(dim=0).expand(
                self.num_angles, -1).contiguous().view(-1)
        return mean

    def forward(self, module_input):
        batch_size = module_input.shape[0]
        reshaped_input = module_input.view(-1, 3, 2)

        # Normalize the first vector
        b1 = F.normalize(reshaped_input[:, :, 0], dim=1)

        dot_prod = torch.sum(b1 * reshaped_input[:, :, 1],
                             dim=1, keepdim=True)
        # Compute the second vector by finding the orthogonal complement to it
        b2 = F.normalize(reshaped_input[:, :, 1] -
                         dot_prod * b1, dim=-1)
        # Finish building the basis by taking the cross product
        b3 = torch.cross(b1, b2, dim=1)
        rot_mats = torch.stack([b1, b2, b3], dim=-1)

        return rot_mats.view(batch_size, 3, 3)


def rot2cont_rep(rot_mat):
    '''
    :param rotation matrix rep: Nx3x3 tensor
    :return: Nx6 mat
    '''
    return rot_mat[:,:,:2].reshape(-1,6)


def aa2cont_rep(aa):
    '''
    :param aa: axis angle rep: Nx3 tensor
    :return:
    '''
    rot_mat = tgm.angle_axis_to_rotation_matrix(aa)
    return rot_mat[:,:3,:2].reshape(-1,6)


def OrthoProj(scale,points3d,cam_rottrans,half_im_size):
    '''
    :param scale: orthographics projection scale vector
    :param points3d: Nx3 tensor
    :param cam_rot: camera pose matrix with translation appended to rotation (3x4)
    '''
    extr_rot = torch.t(cam_rottrans[:3,:3])
    extr_trans = -torch.matmul(extr_rot,cam_rottrans[:3,3].reshape(3,1))
    points3d_local = torch.matmul(points3d.reshape(-1,3),torch.t(extr_rot)) + extr_trans.reshape(1,3)
    points2d = points3d_local[:,:2]/scale
    
    return points2d + half_im_size

def batchOrthoProj(scale,points3d,cam_rottrans,half_im_size):
    '''
    :param scale: orthographics projection scale vector
    :param points3d: Nx3 tensor
    :param cam_rot: camera pose matrix with translation appended to rotation (3x4)
    '''
    batch_size = scale.shape[0]
    extr_rot = cam_rottrans[:,:3,:3].permute(0,2,1)
    extr_trans = -torch.bmm(extr_rot,cam_rottrans[:,:3,3].reshape(-1,3,1))
    points3d_local = torch.bmm(points3d.reshape(batch_size,-1,3),cam_rottrans[:,:3,:3]) + extr_trans.reshape(-1,1,3)
    points2d = points3d_local[:,:,:2]/scale.reshape(batch_size,-1,1)
    
    
    return points2d + half_im_size
    
    
def npPerspProj(intr,points3d,cam_rottrans):
    '''
    :param scale: orthographics projection scale vector
    :param points3d: Nx3 tensor
    :param cam_rot: camera pose matrix with translation appended to rotation (3x4)
    '''
    extr_rot = np.transpose(cam_rottrans[:3,:3])
    extr_trans = -np.matmul(extr_rot,cam_rottrans[:3,3].reshape(3,1))
    points3d_local = np.matmul(points3d.reshape(-1,3),cam_rottrans[:3,:3]) + extr_trans.reshape(1,3)
    points2d = np.transpose(np.matmul(intr,np.transpose(points3d_local)))
    
    return points2d[:,:2]/points2d[:,2:], extr_rot, extr_trans
    
    
def batchPerspProj(intr,points3d,cam_rottrans):
    '''
    :param scale: orthographics projection scale vector
    :param points3d: Nx3 tensor
    :param cam_rot: camera pose matrix with translation appended to rotation (3x4)
    '''
    batch_size = intr.shape[0]
    extr_rot = cam_rottrans[:,:3,:3].permute(0,2,1)
    extr_trans = -torch.bmm(extr_rot,cam_rottrans[:,:3,3].reshape(-1,3,1))
    points3d_local = torch.bmm(points3d.reshape(batch_size,-1,3),cam_rottrans[:,:3,:3]) + extr_trans.reshape(-1,1,3)
    points2d = torch.bmm(intr.float(),points3d_local.permute(0,2,1)).permute(0,2,1)
    
    return points2d[:,:,:2]/points2d[:,:,2:]


def get_mean_params():
    mean_pose = np.array([[0., 0., 0., -0.22387259, 0.0174436,
                           0.09247071, -0.23784273, -0.04646965, -0.07860077, 0.27820579,
                           0.01414277, 0.01381316, 0.43278152, -0.06290711, -0.09606631,
                           0.50428283, 0.00345129, 0.0609754, 0.02297339, -0.03170039,
                           0.00579749, 0.00695809, 0.13169473, -0.05443741, -0.05891175,
                           -0.17524343, 0.13545137, 0.0134158, -0.00365581, 0.00887857,
                           -0.20932178, 0.16004365, 0.10919978, -0.03871734, 0.0823698,
                           -0.20413892, -0.0056038, -0.00751232, -0.00347825, -0.02369,
                           -0.12479898, -0.27360466, -0.04594801, 0.19914683, 0.23728603,
                           0.06672108, -0.04049612, 0.03286229, 0.05357843, -0.29137463,
                           -0.69688406, 0.05585425, 0.28579422, 0.65245777, 0.12222859,
                           -0.91159104, 0.23825037, -0.03660429, 0.92367181, -0.25544496,
                           -0.06566227, -0.1044708, 0.05014435, -0.03878127, 0.09087035,
                           -0.07071638, -0.14365816, -0.05897377, -0.18009904, -0.08745479,
                           0.10929292, 0.20091476]])

    mean_shape = np.array([[0.20560974, 0.33556296, -0.35068284, 0.35612895, 0.41754073,
                            0.03088791, 0.30475675, 0.23613405, 0.20912663, 0.31212645]])
    return mean_pose, mean_shape

def _npcircle(image, cx, cy, radius, color, transparency=0.0):
    """Draw a circle on an image using only numpy methods."""
    radius = int(radius)
    cx = int(cx)
    cy = int(cy)
    if cx < radius:
        cx = radius
    if cx > image.shape[1]-radius:
        cx = image.shape[1]-radius
    if cy > image.shape[0]-radius:
        cy = image.shape[0]-radius
    if cy < radius:
        cy = radius
    y, x = np.ogrid[-radius: radius, -radius: radius]
    index = x ** 2 + y ** 2 <= radius ** 2
    # import pdb;pdb.set_trace()
    image[cy - radius:cy + radius, cx - radius:cx + radius][index] = (
        image[cy - radius:cy + radius, cx - radius:cx + radius][index].astype('float32') * transparency +
        np.array(color).astype('float32') * (1.0 - transparency)).astype('uint8')


def get_weak_persp_cam_full_img_gt(intr,person_position_wrt_cam):
    fx = intr[0,0]
    fy = intr[1,1]
    cx = intr[0,2]
    cy = intr[1,2]
    # person_rootx = person_root_2d[0]
    # person_rooty = person_root_2d[1]
    # box_cx = (bb[1][0] - bb[0][0])//2
    # box_cy = (bb[1][1] - bb[0][1])//2
    z_dist = person_position_wrt_cam[2]
    # if person behind the camera
    if z_dist < 0:
        z_dist = -z_dist
    # sx = (person_rootx - cx)/cx     # cx is Imgwidth/2
    # sy = (person_rooty - cy)/cy     # cy is Imgheight/2
    sx = person_position_wrt_cam[0]/z_dist
    sy = person_position_wrt_cam[1]/z_dist
    sz = fy/(z_dist*cy)
    # sz = (2*fy)/(z_dist*1000)

    return np.array([sz,sx,sy])

def weakcam2trans(batch_intr,batch_weakcam):
    fy = batch_intr[:,1,1]
    cy = batch_intr[:,1,2]
    z = fy/(batch_weakcam[:,0]*cy)
    x = batch_weakcam[:,1]*z
    y = batch_weakcam[:,2]*z

    return torch.stack([x,y,z],dim=1)

def get_weak_persp_cam_full_img_input(intr,bb):
    fx = intr[0,0]
    fy = intr[1,1]
    cx = intr[0,2]
    cy = intr[1,2]
    bb_center = [(bb[0][0]+bb[1][0])/2 - cx, (bb[0][1]+bb[1][1])/2 - cy]
    bb_height = bb[1][1] - bb[0][1]

    sz = bb_height/(2*cy)
    sx = bb_center[0]/cx
    sy = bb_center[1]/cy

    return np.array([sz,sx,sy])



def resize_with_pad(img,size=224):
    '''
    size: (Int) output would be size x size
    '''
    if img.shape[0] > img.shape[1]:
        biggr_dim = img.shape[0]
    else:
        biggr_dim = img.shape[1]
    scale = size/biggr_dim
    out_img = cv2.resize(img,(int(scale*img.shape[1]),int(scale*img.shape[0])))
    pad_top = (size - out_img.shape[0])//2
    pad_bottom = size - out_img.shape[0] - pad_top
    pad_left = (size - out_img.shape[1])//2
    pad_right = size - out_img.shape[1] - pad_left
    out_img = cv2.copyMakeBorder(out_img,
                                    pad_top,
                                    pad_bottom,
                                    pad_left,
                                    pad_right,
                                    cv2.BORDER_CONSTANT)

    return out_img, scale, [pad_left,pad_top]

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


def add_noise_input_cams(extr,noise_sigma):
    batch_size = extr.shape[0]
    device = extr.device
    
    cam = torch.cat([extr[:,:3,3] + noise_sigma[0]*torch.randn(batch_size,3).to(device),
                        extr[:,:3,:2].reshape(-1,6) + noise_sigma[1]*torch.randn(batch_size,6).to(device)],axis=1)


    gt_cam = torch.cat([extr[:,:3,3],
                        extr[:,:3,:2].reshape(-1,6)],axis=1)

    return gt_cam, cam


def add_noise_input_smpltrans(gt_smpltrans,noise_sigma):
    batch_size = gt_smpltrans.shape[0]
    device = gt_smpltrans.device
    
    in_smpltrans0 = gt_smpltrans + noise_sigma*torch.randn(batch_size,3).to(device)
    in_smpltrans1 = gt_smpltrans + noise_sigma*torch.randn(batch_size,3).to(device)
    return in_smpltrans0, in_smpltrans1    


    
