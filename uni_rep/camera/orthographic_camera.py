import torch
from uni_rep.camera import Camera
import numpy as np
import random


def projection(znear, zfar, max_x, min_x, max_y, min_y, scaling_factors, device):
    K = torch.zeros((1, 4, 4), dtype=torch.float32, device=device)
    ones = torch.ones((1), dtype=torch.float32, device=device)

    z_sign = +1.0
    K[0, 0, 0] = (2.0 / (max_x - min_x)) * scaling_factors[0]
    K[0, 1, 1] = (2.0 / (max_y - min_y)) * scaling_factors[1]
    K[0, 0, 3] = -(max_x + min_x) / (max_x - min_x)
    K[0, 1, 3] = -(max_y + min_y) / (max_y - min_y)
    K[0, 3, 3] = ones

    K[0, 2, 2] = z_sign * (1.0 / (zfar - znear)) * scaling_factors[2]
    K[0, 2, 3] = -znear / (zfar - znear)

    return K


class OrthographicCamera(Camera):
    def __init__(self, scaling_factors=[0.26, 0.26, 0.26], device='cuda'):
        super(OrthographicCamera, self).__init__()
        self.device = device
        znear = 1.0
        zfar = 100.0
        max_y = 1.0
        min_y = -1.0
        max_x = 1.0
        min_x = -1.0
        scale = random.uniform(0.2, 0.3)
        scaling_factors = [scale, scale, scale]
        self.proj_mtx = projection(znear, zfar, max_x, min_x, max_y, min_y, scaling_factors=scaling_factors, device=device)

    def project(self, points_bxnx4):
        out = torch.matmul(
            points_bxnx4,
            torch.transpose(self.proj_mtx, 1, 2))
        return out
    
    def get_matrix(self):
        return torch.transpose(self.proj_mtx, 1, 2)
