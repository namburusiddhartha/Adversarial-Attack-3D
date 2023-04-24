import os
import glob
import math
import torch
import torchvision
import pickle
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from random import shuffle
from functools import reduce
from torchvision import transforms
#from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import join_meshes_as_scene
from pytorch3d.transforms import euler_angles_to_matrix
from torch.utils.data.distributed import DistributedSampler
from pytorch3d.renderer import RasterizationSettings, MeshRasterizer, MeshRenderer, SoftSilhouetteShader, FoVOrthographicCameras, look_at_view_transform
from torchvision.transforms import ToTensor

from torch import nn
from torchvision.transforms import Resize

from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras,
    DirectionalLights, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    BlendParams
)
from pytorch3d.structures import join_meshes_as_batch, Meshes

import pdb

#from attacker.UniformTexturesAttacker import UniformTexturesAttacker

class differentiablerenderer():
    def __init__(self) -> None:
        self.DISTANCE = 5.0
        self.SCALING_FACTORS_RANGE = [0.26, 0.36]
        self.INTENSITIES_RANGE = [0.5, 2.0]
        self.DEVICE = "cpu"
        self.device = "cpu"
        pass

    def load_meshes(self, meshes_dir, device = 'cpu'):
        meshes = []
        obj_paths = glob.glob(meshes_dir + "/*.obj")
        for obj_path in tqdm(obj_paths):
            mesh = load_objs_as_meshes([obj_path], device=device)[0]
            meshes.append(mesh)
        # mesh = load_objs_as_meshes([obj_paths[0]], device=device)[0]
        # print(mesh.shape)
        return meshes
    
    def extract_screen_coordinates(self, meshes, distances, elevations, azimuths, scaling_factors):
        assert len(meshes) == len(distances) == len(elevations) == len(azimuths)
        image_size = 384
        
        # Initialize the cameras
        scaling_factors = scaling_factors.repeat(1, 3)
        R, T = look_at_view_transform(dist=distances, elev=elevations, azim=azimuths)
        cameras = FoVOrthographicCameras(
            device=self.device,
            R=R,
            T=T,
            scale_xyz=scaling_factors
        )
        batch_size = len(meshes)
        locations_batch = []
        for i in range(batch_size):
            # Extract locations for the given image
            meshes_ = meshes[i]
            verts = torch.stack([mesh.verts_padded() for mesh in meshes_]).squeeze(1)
            center_coords = torch.mean(verts, dim=1)
            locations = cameras.transform_points_screen(center_coords, image_size=(image_size, image_size))[i, :, :2]
            
            # Remove locations which fall outside the image
            valid_list = []
            for idx, location in enumerate(locations):
                if 0 <= location[0] <= image_size and 0 <= location[1] <= image_size:
                    valid_list.append(idx)
            locations = locations[[valid_list]]
            
            # Append to the batch
            locations_batch.append(locations)
        return locations_batch
    

    def randomly_move_and_rotate_mesh(self, mesh, scaling_factor):
        # Apply random rotation
        mesh_rotation = euler_angles_to_matrix(torch.tensor([0, random.uniform(0, 2 * math.pi), 0]), convention="XYZ").to(self.device)
        mesh_rotation = torch.matmul(mesh_rotation, mesh.verts_packed().data.T).T - mesh.verts_packed()
        mesh.offset_verts_(vert_offsets_packed=mesh_rotation)
        
        # Apply random translation (forcing the center of the vehicle to stay in the image)
        mesh_dx = random.uniform(-1, 1)
        mesh_dz = random.uniform(-1, 1)
        
        # Compute the offset
        offset = np.array([-mesh_dx, -mesh_dz]) # To be in the (x, y) format on the image
        
        mesh_dx /= scaling_factor
        mesh_dz /= scaling_factor
        
        # Center the mesh before applying translation
        mesh_dx -= torch.mean(mesh.verts_padded(), dim=1)[0][0].item()
        mesh_dz -= torch.mean(mesh.verts_padded(), dim=1)[0][2].item()
        
        # Apply the translation
        mesh_translation = torch.tensor([mesh_dx, 0, mesh_dz], device=self.device) * torch.ones(size=mesh.verts_padded().shape[1:], device=self.device)
        mesh.offset_verts_(vert_offsets_packed=mesh_translation)
        
        return (mesh.clone(), offset)
    
    def randomly_move_and_rotate_meshes(self, meshes, scaling_factor, distance, elevation, azimuth, intensity):
        invalid_image = True
        
        # Create the silhouette renderer
        R, T = look_at_view_transform(dist=distance, elev=elevation, azim=azimuth)
        cameras = FoVOrthographicCameras(
            device=self.device, 
            R=R,
            T=T, 
            scale_xyz=((scaling_factor, scaling_factor, scaling_factor),)
        ) 
        sigma = 1e-4
        raster_settings_silhouette = RasterizationSettings(
            image_size=384, 
            blur_radius=np.log(1. / 1e-4 - 1.)*sigma, 
            faces_per_pixel=50, 
        )
        silhouette_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings_silhouette
            ),
            shader=SoftSilhouetteShader()
        )
        
        while invalid_image:
            offsets = []
            silhouettes = []
            
            for i in range(len(meshes)):
                meshes[i], offset = self.randomly_move_and_rotate_mesh(meshes[i], scaling_factor)
                silhouette = silhouette_renderer(meshes[i], cameras=cameras)
                silhouette = (silhouette[..., 3] > 0.5).float()
                silhouettes.append(silhouette)
                offsets.append(offset)
            
            # Check whether any of the meshes intersect
            if torch.any(reduce(lambda x, y: x + y, silhouettes) > 1.0):
                invalid_image = True
            else:
                invalid_image = False
        
        return (meshes, offsets)
    
    def randomly_place_meshes(self, meshes, scaling_factors):
        offsets = [None for _ in range(len(meshes))]
        for i in range(len(meshes)):
            meshes[i], offsets[i] = self.randomly_move_and_rotate_mesh(meshes[i], scaling_factors[i])
        return meshes, offsets
    
    # TO DO: move this function to utils
    def randomly_place_meshes_multi(self, meshes_list, scaling_factors, distances, elevations, azimuths, intensities):
        assert len(meshes_list) == len(scaling_factors)
        meshes = []
        offsets = []
        
        for i in range(len(meshes_list)):
            meshes_ = meshes_list[i]
            scaling_factor = scaling_factors[i]
            distance = distances[i]
            elevation = elevations[i]
            azimuth = azimuths[i]
            intensity = intensities[i]
            
            meshes_, offsets_ = self.randomly_move_and_rotate_meshes(meshes_, scaling_factor, distance, elevation, azimuth, intensity)
            
            meshes.append(meshes_)
            offsets.append(offsets_)
        
        locations_batch = self.extract_screen_coordinates(meshes, distances, elevations, azimuths, scaling_factors)
        
        return meshes, locations_batch


        
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
        lights_directions = torch.rand(batch_size, 3, device=self.DEVICE) * 2 - 1
        lights_directions[:, 1] = -1
        scaling_factors = torch.rand(batch_size, 1, device=self.DEVICE) * (sf_range[1] - sf_range[0]) + sf_range[0]
        intensities = torch.rand(batch_size, 1, device=self.DEVICE) * (int_range[1] - int_range[0]) + int_range[0]
        
        return (distances, elevations, azimuths, lights_directions, scaling_factors, intensities)
    
    def construct_annotations_files(self, locations_batch):
        annotations_batch = []
        
        for locations in locations_batch:
            annotations = {
                "van_rv": np.empty(shape=(0, 2)),
                "truck": np.empty(shape=(0, 2)),
                "bus": np.empty(shape=(0, 2)),
                "trailer_small": np.empty(shape=(0, 2)),
                "specialized": np.empty(shape=(0, 2)),
                "trailer_large": np.empty(shape=(0, 2)),
                "unknown": np.empty(shape=(0, 2)),
                "small": locations.cpu().numpy()
            }
            annotations_batch.append(annotations)
    
        return annotations_batch
    
    def render_batch(self, meshes, background_images, elevations, azimuths, light_directions, distances,
                     scaling_factors, intensities, image_size=384, blur_radius=0.0, faces_per_pixel=1, ambient_color=((0.05, 0.05, 0.05),)):
        # Image needs to be upscaled and then average pooled to make the car less sharp-edged
        transform = Resize((image_size, image_size))
        background_images = transform(background_images).permute(0, 2, 3, 1)
        scaling_factors = scaling_factors.repeat(1, 3)
        intensities = intensities.repeat(1, 3)
        
        R, T = look_at_view_transform(dist=distances, elev=elevations, azim=azimuths)
        cameras = FoVOrthographicCameras(
            device=self.device,
            R=R,
            T=T,
            scale_xyz=scaling_factors
        )
        
        raster_settings = RasterizationSettings(
            image_size=image_size, 
            blur_radius=blur_radius, 
            faces_per_pixel=faces_per_pixel, 
        )
        
        lights = DirectionalLights(
            device=self.device, 
            direction=light_directions, 
            ambient_color=ambient_color, 
            diffuse_color=intensities
        )
        
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras, 
                raster_settings=raster_settings,
            ),
            shader=SoftPhongShader(
                device=self.device, 
                cameras=cameras,
                lights=lights,
                blend_params=BlendParams(background_color=background_images)
            )
        )
        
        if isinstance(meshes, Meshes):
            pass
        elif isinstance(meshes, list):
            meshes = join_meshes_as_batch(meshes)
        else:
            raise Exception("Incorrect data type for the 'meshes' variable.")
        
        images = renderer(meshes, lights=lights, cameras=cameras)
        images = images[..., :3]
        images = images.permute(0, 3, 1, 2)
        
        return images
    
    def render(self):
        M = self.load_meshes(meshes_dir = "/home/snamburu/attack/GAN-vehicles/")
        distances, elevations, azimuths, lights_directions, scaling_factors, intensities = self.sample_rendering_params(32)
        image_path = "/home/snamburu/attack/sid/empty/009_00011_altered_00234.jpg"
        self.TS = ToTensor()
        images_batch = self.TS(Image.open(image_path)).unsqueeze(0)
        print(images_batch.shape)
        M = M[:32]
        n_vehicles_list = [1]*32 
        meshes_batch_list = [random.choices(M, k=n_vehicles) for n_vehicles in n_vehicles_list]
        meshes, locations_batch = self.randomly_place_meshes_multi(
            meshes_batch_list, 
            scaling_factors,
            distances,
            elevations,
            azimuths,
            intensities,
        )
        annotations = self.construct_annotations_files(locations_batch)

        meshes_joined = []
        for i in range(32):
            meshes_joined.append(join_meshes_as_scene(meshes[i]).to(self.device))
            
            # Render the images
        synthetic_images = self.render_batch(
            meshes_joined, 
            images_batch, 
            elevations, 
            azimuths,
            lights_directions,
            scaling_factors=scaling_factors,
            intensities=intensities,
            distances=distances,
            image_size=384
        )

        print(synthetic_images.shape)
        iter_counter = 0
        for syn_img in synthetic_images:
            save_image(syn_img, f"./results_mik/image_{str(iter_counter).zfill(5)}.png", quality=100)
            iter_counter += 1

        return synthetic_images



if __name__ == '__main__':
    print("Differentiable renderer")
    dif = differentiablerenderer()
    dif.render()



