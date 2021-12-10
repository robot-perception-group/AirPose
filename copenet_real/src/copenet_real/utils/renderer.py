import os
# os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import torch
from torchvision.utils import make_grid
import numpy as np
import pyrender
import trimesh

class Renderer:
    """
    Renderer used for visualizing the SMPL model
    Code adapted from https://github.com/vchoutas/smplify-x
    """
    def __init__(self, focal_length=[1475,1475], img_res=[224,224], center = [112,112], faces=None):
        self.renderer = pyrender.OffscreenRenderer(viewport_width=img_res[0],
                                       viewport_height=img_res[1],
                                       point_size=1.0)
        self.focal_length = focal_length
        self.camera_center = center
        self.faces = faces

    def visualize_tb(self, vertices, camera_translation,camera_rotation, images,nrow=5,color=(0.3, 0.3, 0.8, 1.0)):
        
        vertices = vertices.cpu().numpy()
        camera_translation = camera_translation.cpu().numpy()
        camera_rotation = camera_rotation.cpu().numpy()
        images = images.cpu()
        images_np = np.transpose(images.numpy(), (0,2,3,1))
        rend_imgs = []
        for i in range(vertices.shape[0]):
            rend_img = torch.from_numpy(np.transpose(self.__call__(vertices[i], camera_translation[i], camera_rotation[i],images_np[i],color=color), (2,0,1))).float()
            # rend_imgs.append(images[i])
            rend_imgs.append(rend_img)
        rend_imgs = make_grid(rend_imgs, nrow,padding=0)
        return rend_imgs

    def __call__(self, vertices, camera_translation, camera_rotation, image, color=(0.3, 0.3, 0.8, 1.0)):
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.2,
            alphaMode='OPAQUE',
            baseColorFactor=color)

        mesh = trimesh.Trimesh(vertices, self.faces)
        # perp = np.cross(np.array([0,0,1]),camera_translation)
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1,0,0])
        # import ipdb; ipdb.set_trace()
        camera_pose = np.eye(4)
        camera_pose[:3,:3] = camera_rotation
        camera_pose[:3, 3] = camera_translation
        # mesh.apply_transform(np.linalg.inv(camera_pose))
        mesh.apply_transform(camera_pose)
        mesh.apply_transform(rot)
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(ambient_light=(0.5, 0.5, 0.5))
        scene.add(mesh, 'mesh')

        # camera_pose = np.eye(4)
        # camera_pose[:3,:3] = camera_rotation
        # camera_pose = np.matmul(rot,camera_pose)
        # camera_pose[:3, 3] = camera_translation
        
        self.camera = pyrender.IntrinsicsCamera(fx=self.focal_length[0], fy=self.focal_length[1],
                                           cx=self.camera_center[0], cy=self.camera_center[1])
        scene.add(self.camera, pose=np.eye(4))

        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1)
        light_pose = np.eye(4)

        light_pose[:3, 3] = np.array([0, -1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([0, 1, 1])
        scene.add(light, pose=light_pose)

        light_pose[:3, 3] = np.array([1, 1, 2])
        scene.add(light, pose=light_pose)


        color, rend_depth = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        valid_mask = (rend_depth > 0)[:,:,None]
        output_img = (color[:, :, :3] * valid_mask +
                  (1 - valid_mask) * image)
        return output_img
