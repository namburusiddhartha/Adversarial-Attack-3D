import copy
import os

import numpy as np
import torch
import dnnlib
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import grid_sample_gradfix
from metrics import metric_main
from training.inference_utils import save_visualization, save_visualization_for_interpolation, \
    save_textured_mesh_for_inference, save_geo_for_inference
from training.attack_utils import attack_train_execute, save_attack, save_model, save_data


import detectron2.utils.comm as comm
from detectron2.config import get_cfg
#from yacs.config import CfgNode as CN
from torch.utils.data import DataLoader
from detectron2.engine import default_setup
from detectron2.modeling import build_model
from detectron2.data import transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from model_zoo.RetinaNetPoint import RetinaNetPoint

from training.attack_dataset import AttackDataset, collate_fn


def clean_training_set_kwargs_for_metrics(training_set_kwargs):
    if 'add_camera_cond' in training_set_kwargs:
        training_set_kwargs['add_camera_cond'] = True
    return training_set_kwargs

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


# ----------------------------------------------------------------------------
def attack(
        run_dir='.',  # Output directory.
        training_set_kwargs={},  # Options for training set.
        G_kwargs={},  # Options for generator network.
        D_kwargs={},  # Options for discriminator network.
        detector_attack_config = None,
        metrics=[],  # Metrics to evaluate during training.
        random_seed=0,  # Global random seed.
        num_gpus=1,  # Number of GPUs participating in the training.
        rank=0,  # Rank of the current process in [0, num_gpus[.
        resume_pretrain=None,
        batch_size=4,
        attack_train=False,
        attack_type = 0,
        attack_logdir = "./tmp",
        attack_background_data = "./tmp",
        **dummy_kawargs
):
    from torch_utils.ops import upfirdn2d
    from torch_utils.ops import bias_act
    from torch_utils.ops import filtered_lrelu
    upfirdn2d._init()
    bias_act._init()
    filtered_lrelu._init()

    print(dummy_kawargs)

    device = torch.device('cuda', rank)
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True  # Improves training speed.
    torch.backends.cuda.matmul.allow_tf32 = True  # Improves numerical accuracy.
    torch.backends.cudnn.allow_tf32 = True  # Improves numerical accuracy.
    conv2d_gradfix.enabled = True  # Improves training speed.
    grid_sample_gradfix.enabled = True  # Avoids errors with the augmentation pipe.


    common_kwargs = dict(
        c_dim=0, img_resolution=training_set_kwargs['resolution'] if 'resolution' in training_set_kwargs else 1024, img_channels=3)
    G_kwargs['device'] = device

    #Load the Get3d model
    G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(True).to(
        device)  # subclass of torch.nn.Module
    # D = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).train().requires_grad_(False).to(
    #     device)  # subclass of torch.nn.Module
    G_ema = copy.deepcopy(G).eval()  # deepcopy can make sure they are correct.
    if resume_pretrain is not None and (rank == 0):
        print('==> resume from pretrained path %s' % (resume_pretrain))
        model_state_dict = torch.load(resume_pretrain, map_location=device)
        G.load_state_dict(model_state_dict['G'], strict=True)
        G_ema.load_state_dict(model_state_dict['G_ema'], strict=True)
        # D.load_state_dict(model_state_dict['D'], strict=True)
    grid_size = (batch_size//2, batch_size//2)
    n_shape = grid_size[0] * grid_size[1]
    grid_z = torch.randn([n_shape, G.z_dim], device=device).split(1)  # random code for geometry
    grid_tex_z = torch.randn([n_shape, G.z_dim], device=device).split(1)  # random code for texture
    grid_c = torch.ones(n_shape, device=device).split(1)

    #Load the attacker model
    cfg = get_cfg()
    cfg.merge_from_file(detector_attack_config)
    cfg.freeze()
    args = None
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code
    detector_attack_model = build_model(cfg)
    DetectionCheckpointer(detector_attack_model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
                cfg.MODEL.WEIGHTS, resume=True
            )
    freeze_model(detector_attack_model)

    if attack_train:
        augmentations = T.AugmentationList([
            T.RandomContrast(0.2, 2.0),
            T.RandomBrightness(0.2, 2.0),
            T.RandomSaturation(0.2, 2.0),
            T.RandomLighting(1.0),
            T.RandomFlip(horizontal=True),
            T.RandomFlip(horizontal=False, vertical=True),
            T.RandomRotation([0.0, 360.0], expand=False)
        ])
    else:
        augmentations = None


    # Build the dataset of background images
    # if attack_train:
    #     data_dir = "/home/snamburu/Storage/SatDet-Real-GoogleMaps-384px-0.125m/train/"
    # else:
    #     data_dir = "/home/snamburu/Storage/SatDet-Real-GoogleMaps-384px-0.125m/validation/"
    attack_set = AttackDataset(attack_background_data, augmentations=augmentations, device=device)
    attack_loader = DataLoader(attack_set, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

    print('==> Attacking train')

    if attack_train:
        attack_train_execute(
        G_ema, grid_z, grid_c, training_set_kwargs['resolution'], attack_type, detector_attack_model, attack_loader, attack_logdir, grid_tex_z=grid_tex_z, device=device
        )

    # save_attack(
    #     G_ema, grid_z, grid_c, grid_tex_z=grid_tex_z
    # )

    #save_model(
    #   G_ema, grid_z, grid_c, grid_tex_z=grid_tex_z
    #)

    #save_data(G_ema, training_set_kwargs['resolution'], detector_attack_model, attack_loader, device=device)
