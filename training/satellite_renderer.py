import math
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch_utils import persistence
import nvdiffrast.torch as dr
from uni_rep.camera.perspective_camera import PerspectiveCamera
from uni_rep.render.neural_render import NeuralRender
from uni_rep.rep_3d.dmtet import DMTetGeometry
from training.sample_camera_distribution import sample_camera



class SatRenderer(torch.nn.Module):
    def __init__(
            self,
            device='cuda'
            
    ):  #
        super().__init__()

        self.DISTANCE = 5.0
        self.SCALING_FACTORS_RANGE = [0.26, 0.36]
        self.INTENSITIES_RANGE = [0.5, 2.0]
        self.device = device
        self.img_resolution = 1024
        self.data_camera_mode = 'shapenet_car'

        # Camera defination, we follow the defination from Blender (check the render_shapenet_data/rener_shapenet.py for more details)
        fovy = np.arctan(32 / 2 / 35) * 2
        fovyangle = fovy / np.pi * 180.0
        self.dmtet_camera = PerspectiveCamera(fovy=fovyangle, device=self.device)

        # Renderer we used.
        self.dmtet_renderer = NeuralRender(device, camera_model=self.dmtet_camera)

        # Geometry
        render_type='neural_render'
        self.dmtet_geometry = DMTetGeometry(
            grid_res=64, scale=0.26, renderer=self.dmtet_renderer, render_type=render_type,
            device=self.device)

    def sample_random_elev_azimuth(self, x_min, y_min, x_max, y_max, distance):
        """
        This function samples x and y coordinates on a plane, and converts them to elevation and azimuth angles.
        
        It was found that x_min = y_min = -1.287 and x_max = y_max = 1.287 result in the best angles, where elevation ranges roughly from 70 to 90, and azimuth goes from 0 to 360.
        """
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)

        if x == 0 and y == 0:
            elevation = 90.0
            azimuth = 0.0
        elif x == 0:
            elevation = math.atan(distance / math.sqrt(x * x + y * y)) * 180.0 / math.pi
            azimuth = 0.0
        else:
            elevation = math.atan(distance / math.sqrt(x * x + y * y)) * 180.0 / math.pi
            azimuth = math.atan(y / x) * 180.0 / math.pi
            if x < 0:
                if y > 0:
                    azimuth += 180
                else:
                    azimuth -= 180

        return (elevation, azimuth)

    def sample_rendering_params(self, batch_size):
        distance = self.DISTANCE
        # elevation = self.attack_cfg.RENDERER.ELEVATION
        # azimuth = self.attack_cfg.RENDERER.AZIMUTH
        sf_range = self.SCALING_FACTORS_RANGE
        int_range = self.INTENSITIES_RANGE
        els_azs = [self.sample_random_elev_azimuth(-1.287, -1.287, 1.287, 1.287, self.DISTANCE) for _ in range(batch_size)]
        distances = [distance for _ in range(batch_size)]
        # elevations = [elevation for _ in range(batch_size)]
        # azimuths = [azimuth for _ in range(batch_size)]
        elevations = [els_azs_[0] for els_azs_ in els_azs]
        azimuths = [els_azs_[1] for els_azs_ in els_azs] # No need to rotate the camera if the vehicles is rotated
        lights_directions = torch.rand(batch_size, 3, device=self.device) * 2 - 1
        lights_directions[:, 1] = -1
        scaling_factors = torch.rand(batch_size, 1, device=self.device) * (sf_range[1] - sf_range[0]) + sf_range[0]
        intensities = torch.rand(batch_size, 1, device=self.device) * (int_range[1] - int_range[0]) + int_range[0]
        
        return (distances, elevations, azimuths, lights_directions, scaling_factors, intensities)
    
    def generate_random_camera(self, batch_size, n_views=2):
        '''
        Sample a random camera from the camera distribution during training
        :param batch_size: batch size for the generator
        :param n_views: number of views for each shape within a batch
        :return:
        '''
        sample_r = None
        world2cam_matrix, forward_vector, camera_origin, rotation_angle, elevation_angle = sample_camera(
            self.data_camera_mode, batch_size * n_views, self.device)
        mv_batch = world2cam_matrix
        campos = camera_origin
        return campos.reshape(batch_size, n_views, 3), mv_batch.reshape(batch_size, n_views, 4, 4), \
               rotation_angle, elevation_angle, sample_r

    def render_mesh(self, mesh_v, mesh_f, cam_mv):
        '''
        Function to render a generated mesh with nvdiffrast
        :param mesh_v: List of vertices for the mesh
        :param mesh_f: List of faces for the mesh
        :param cam_mv:  4x4 rotation matrix
        :return:
        '''
        return_value_list = []
        for i_mesh in range(len(mesh_v)):
            return_value = self.dmtet_geometry.render_mesh(
                mesh_v[i_mesh],
                mesh_f[i_mesh].int(),
                cam_mv[i_mesh],
                resolution=self.img_resolution,
                hierarchical_mask=False
            )
            return_value_list.append(return_value)

        return_keys = return_value_list[0].keys()
        return_value = dict()
        for k in return_keys:
            value = [v[k] for v in return_value_list]
            return_value[k] = value

        mask_list, hard_mask_list = torch.cat(return_value['mask'], dim=0), torch.cat(return_value['hard_mask'], dim=0)
        return mask_list, hard_mask_list, return_value
    

    
    def generate(self, mesh_v, mesh_f, ws_tex):
        #params = self.sample_rendering_params(1)
        #cam_mv = params[1:3]

        with torch.no_grad():
            campos, cam_mv, rotation_angle, elevation_angle, sample_r = self.generate_random_camera(
                        ws_tex.shape[0], n_views=2)
            
        cam_mv = cam_mv[:, 0, :, :].unsqueeze(1)  

        cam_mv = torch.tensor([[[ 0.7743, -0.6151, -0.1489, -0.7443],
         [ 0.0000,  0.2352, -0.9719, -4.8597],
         [ 0.6329,  0.7525,  0.1821,  0.9105],
         [ 0.0000,  0.0000,  0.0000,  1.0000]],

        [[-0.7290,  0.6677,  0.1508,  0.7540],
         [ 0.0000,  0.2203, -0.9754, -4.8772],
         [-0.6846, -0.7111, -0.1606, -0.8030],
         [ 0.0000,  0.0000,  0.0000,  1.0000]],

        [[-0.7067, -0.7027, -0.0823, -0.4115],
         [ 0.0000,  0.1163, -0.9932, -4.9661],
         [ 0.7075, -0.7019, -0.0822, -0.4110],
         [ 0.0000,  0.0000,  0.0000,  1.0000]],

        [[ 0.9207, -0.3860, -0.0577, -0.2887],
         [ 0.0000,  0.1480, -0.9890, -4.9450],
         [ 0.3903,  0.9106,  0.1362,  0.6811],
         [ 0.0000,  0.0000,  0.0000,  1.0000]]], device = self.device).unsqueeze(1)   


        antilias_mask, hard_mask, return_value = self.render_mesh(mesh_v, mesh_f, cam_mv)


        

        return antilias_mask
    
    def orthographiccamera(self, R, T, distances):
        znear = 1.0
        zfar = 100.0
        max_y = 1.0
        min_y = -1.0
        max_x = 1.0
        min_x = -1.0