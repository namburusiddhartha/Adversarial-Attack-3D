'''
Utily functions for the Attack
'''
import torch
import numpy as np
import math
import random
import os
import PIL.Image
from training.utils.utils_3d import save_obj, savemeshtes2
import imageio
import cv2
from tqdm import tqdm
import torchvision
import pickle

from torchvision.utils import save_image
from training.sample_camera_distribution import sample_camera, create_camera_from_angle_and_translation

from training.attack_losses import ScoreLoss, TVLoss, NPSLoss, DiversionLoss, DiscriminatorLoss

from detectron2.utils.events import EventStorage
from detectron2.structures import Boxes
from detectron2.structures.instances import Instances
from torch.utils.tensorboard import SummaryWriter

def blur_objects(gaussian_blurrer, rendered_images, background_images):
    # Find the difference between the images
    diff_images = rendered_images - background_images
    
    # Blur the difference
    diff_images = gaussian_blurrer(diff_images)
    
    # Add the blurred difference images to the background images
    blurred_images = background_images + diff_images
    
    # Convert to float
    blurred_images = blurred_images.float()
    
    return blurred_images

def init_adv_textures(adv_textures, requires_grad=True):
    """
    Initialize an adversarial texture map. If config file contains path to a previously saved map -- it is loaded.
    
    outputs:
        - adv_textures (torch.Tensor): a tensor of shape (H, W, 3) which represents a unified adversarial texture map
    """
        
    if requires_grad:
        adv_textures.requires_grad_(True)
    
    return adv_textures

def init_optimizer_scheduler(optimized_params):
        optimizer = torch.optim.Adam(optimized_params, lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
        
        return optimizer, scheduler

def setup_loss_fns():
    loss_fns = {}
    loss_fns_list = ["native"]
    
    # Check that the loss terms are valid
    loss_fns_set = set(loss_fns_list)
    pure_set = set(("yolov5", "native"))
    attack_set = set(("scoreloss", "diversion"))
    loss2pure = len(loss_fns_set.intersection(pure_set))
    loss2attack = len(loss_fns_set.intersection(attack_set))
    valid_intersection = not ((loss2pure > 0) and (loss2attack > 0)) # Check that loss terms do not contain elements from both pure and attack sets
    assert valid_intersection, f"Loss terms combination is invalid! Cannot use loss terms from the pure set ({pure_set}) and the attack set ({attack_set}) at the same time!"
    
    # Construct the loss dictionary
    for loss_keyword in loss_fns_list:
        if loss_keyword == 'scoreloss':
            coefficient = 1.0
            loss = ScoreLoss(coefficient=coefficient)
        elif loss_keyword == 'TV':
            coefficient = 2000.0
            loss = TVLoss(coefficient=coefficient)
        # elif loss_keyword == 'NPS':
        #     coefficient = 4.0
        #     #printable_colors = #torch.load(self.attack_cfg.ATTACKER.OPTIMIZATION.LOSS.FILES.NPS_COLORS)
        #     loss = NPSLoss(printable_colors, coefficient=coefficient, device=dev)
        elif loss_keyword == 'diversion':
            coefficient = 1.0
            loss = DiversionLoss(coefficient=coefficient)
        elif loss_keyword == 'discriminator':
            coefficient = 1.0
            loss = DiscriminatorLoss(discriminator=self.discriminator, coefficient=coefficient)
        elif loss_keyword == 'native':
            loss = None
        else:
            raise NotImplementedError
        loss_fns[loss_keyword] = loss
    return loss_fns

def sample_random_elev_azimuth(x_min, y_min, x_max, y_max, distance):
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

def generate_matrix(angle1, angle2, device):
    angle1 = angle1 * math.pi / 180.0 
    angle2 = angle2 * math.pi / 180.0 
    r_matrix = torch.tensor([[math.cos(angle1), 0.0, math.sin(angle1)], [0.0, 1.0, 0.0], [-math.sin(angle1), 0.0, math.cos(angle1)]], device=device)
    r_matrix2 = torch.tensor([[math.cos(angle2), -math.sin(angle2), 0.0], [math.sin(angle2), math.cos(angle2), 0.0], [0.0, 0.0, 1.0]], device=device)
    return r_matrix @ r_matrix2

def sample_cam_attack(batch_size, device):
    n_camera = batch_size
    els_azs = [sample_random_elev_azimuth(-1.287, -1.287, 1.287, 1.287, 5.0) for _ in range(batch_size)]
    elevations = [els_azs_[0] for els_azs_ in els_azs]
    azimuths = [els_azs_[1] for els_azs_ in els_azs]
    camera_radius = 1  # align with what ww did in blender
    camera_r = torch.zeros(n_camera, 1, device=device) + camera_radius
    camera_phi = torch.zeros(n_camera, 1, device=device) #+ 180.0 * math.pi / 180.0 #(90.0 - 15.0) / 90.0 * 0.5 * math.pi
    camera_theta = torch.zeros(n_camera, 1, device=device) #+ 360.0 * math.pi / 180.0 #torch.range(0, n_camera - 1, device=device).unsqueeze(dim=-1) / n_camera * math.pi * 2.0
    camera_t = torch.zeros(n_camera, 3, device=device)
    azi_rotation = torch.zeros(n_camera, 3, 3, device=device)


    

    for ind in range(batch_size):
        azi_rotation[ind, :, :] = generate_matrix(azimuths[ind], 90.0 - abs(elevations[ind]), device)
        camera_phi[ind, :] +=  181.0 * math.pi / 180.0 #(abs(els_azs[ind][0]) + 180.0) * math.pi / 180.0
        camera_theta[ind, :] += 360.0 * math.pi / 180.0 #els_azs[ind][1] * math.pi / 180.0
        dx = random.uniform(-2, 2)
        dz = random.uniform(-2, 2)
        camera_t[ind, 0] += dx
        camera_t[ind, 2] += dz
        #(180.0 + 90.0 - abs(elevations[ind]))


        
    world2cam_matrix, camera_origin, _, _, _ = create_camera_from_angle_and_translation(
            camera_phi, camera_theta, camera_r, sample_t=camera_t, sample_R=azi_rotation, device=device)

    cam_mv = world2cam_matrix.unsqueeze(1)

    return cam_mv

def construct_model_inputs(synthetic_images_batch, batch_size, img_res, image_id):
    assert synthetic_images_batch.shape[0] == batch_size

    empty_gt = True
    
    model_inputs = []
    for i in range(batch_size):
        model_input = {}
        model_input['image_id'] = image_id
        model_input['height'] = img_res
        model_input['width'] = img_res
        model_input['image'] = synthetic_images_batch[i] * 255.

        # Construct the instances field
        if empty_gt:
            gt_classes = torch.empty(0)
            gt_boxes = torch.empty(size=(0, 4))
        else:
            gt_classes = torch.tensor([0] * len(labels_batch[i]))
            gt_boxes = torch.empty(size=(labels_batch[i].shape[0], 4))
            gt_boxes[:, :2] = labels_batch[i] - 24 // 2
            gt_boxes[:, 2:] = labels_batch[i] + 24 // 2
            gt_boxes.data.clamp_(0, img_res)
        gt_boxes = Boxes(gt_boxes)
        instances = Instances(
            (img_res, img_res),
            gt_boxes=gt_boxes,
            gt_classes=gt_classes
        )
        model_input['instances'] = instances

        # Append to the list
        model_inputs.append(model_input)

        # Modify image ID
        image_id += 1
    
    
    return model_inputs

def losses_forward(results, model_inputs, loss_fns, adv_textures):
    losses_dict = {}
    for loss_fn_keyword in loss_fns.keys():
        if loss_fn_keyword == 'scoreloss':
            loss = loss_fns[loss_fn_keyword](results)
        elif loss_fn_keyword == 'TV':
            loss = loss_fns[loss_fn_keyword](adv_textures.unsqueeze(0).permute(0, 3, 1, 2))
        elif loss_fn_keyword == 'NPS':
            loss = loss_fns[loss_fn_keyword](adv_textures.unsqueeze(0))
        elif loss_fn_keyword == 'diversion':
            gt_points_batch = [model_input['instances'].gt_boxes.get_centers().to(self.attack_cfg.DEVICE) for model_input in model_inputs]
            pred_points_batch = [result.pred_boxes.get_centers() for result in results]
            loss = loss_fns[loss_fn_keyword](pred_points_batch, gt_points_batch)
        elif loss_fn_keyword == 'discriminator':
            images_batch = torch.stack([model_input['image'] for model_input in model_inputs]) / 255.
            images_batch = None#self.normalizer(images_batch)
            loss = loss_fns[loss_fn_keyword](images_batch)
        elif loss_fn_keyword == 'native':
            loss = sum(results.values())
        losses_dict[loss_fn_keyword] = loss
    return losses_dict

def log_info(iter_counter, loss_dict, total_loss, results, writer, scheduler):
    for k, v in loss_dict.items():
        writer.add_scalar(k, v.item(), iter_counter)
        writer.add_scalar("Total loss", total_loss, iter_counter)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], iter_counter)
        #if self.forward_mode == 'attack':
        #writer.add_scalar("Maximum score", max([torch.max(result.scores) for result in results if result.scores.shape[0] > 0]).item(), iter_counter)

def update_LR(scheduler, lr_decay_rate, epoch):
    if (epoch + 1) % lr_decay_rate == 0:
        scheduler.step()

def save_latent(adv_latent_space, out_file):
    np.save(out_file, adv_latent_space.detach().cpu().numpy())


def save_attack(
        G_ema, grid_z, grid_c,
        grid_tex_z=None
):
    '''
    Save visualization during training
    :param G_ema: GET3D generator
    :param grid_z: latent code for geometry latent code
    :param grid_c: None
    :param run_dir: path to save images
    :param cur_nimg: current k images during training
    :param grid_size: size of the image
    :param cur_tick: current kicks for training
    :param image_snapshot_ticks: current snapshot ticks
    :param save_gif_name: the name to save if we want to export gif
    :param save_all:  whether we want to save gif or not
    :param grid_tex_z: the latent code for texture geenration
    :return:
    '''
    with torch.no_grad():
        G_ema.update_w_avg()
        if grid_tex_z is None:
            grid_tex_z = grid_z

        n_camera = 4
        device = "cuda"
        camera_radius = 1  # align with what ww did in blender
        camera_r = torch.zeros(n_camera, 1, device=device) + camera_radius
        camera_phi = torch.zeros(n_camera, 1, device=device) + 180.0 * math.pi / 180.0 #(90.0 - 15.0) / 90.0 * 0.5 * math.pi
        camera_theta = torch.zeros(n_camera, 1, device=device) + 360.0 * math.pi / 180.0 #torch.range(0, n_camera - 1, device=device).unsqueeze(dim=-1) / n_camera * math.pi * 2.0
        camera_t = torch.zeros(n_camera, 3, device=device)

        print(camera_phi, camera_theta)


        world2cam_matrix, camera_origin, _, _, _ = create_camera_from_angle_and_translation(
            camera_phi, camera_theta, camera_r, sample_t=camera_t, device=device)

        cam_mv = world2cam_matrix.unsqueeze(1)

        Bimg = torchvision.io.read_image("/home/snamburu/attack/sid/empty/009_00011_altered_00234.jpg")
        TF = torchvision.transforms.Resize(1024)
        Bimg = TF(Bimg)

        
        Bimg = Bimg.unsqueeze(0).cuda() / 255.0

        blur_sigma = 2
        gaussian_blurrer = torchvision.transforms.GaussianBlur(math.ceil(blur_sigma) * 6 - 1, sigma=blur_sigma)


        

        iter_counter = 0

        for i_camera, camera in enumerate(cam_mv):
            print(camera.shape)
            camera = camera.unsqueeze(0)
            images_list = []
            mesh_v_list = []
            mesh_f_list = []
            for z, geo_z, c in zip(grid_tex_z, grid_z, grid_c):
                img, mask, sdf, deformation, v_deformed, mesh_v, mesh_f, gen_camera, img_wo_light, tex_hard_mask = G_ema.generate_3d(
                    z=z, geo_z=geo_z, c=c, noise_mode='const',
                    generate_no_light=True, truncation_psi=0.7, camera=camera)
                rgb_img = img[:, :3]
                
                
                #mesh_v_list.extend([v.data.cpu().numpy() for v in mesh_v])
                #mesh_f_list.extend([f.data.cpu().numpy() for f in mesh_f])
                #images = np.concatenate(images_list, axis=0)
                Bmask = tex_hard_mask.permute(0, 3, 1, 2).repeat(1, 3, 1, 1)
                Bimg_right = rgb_img * Bmask + Bimg * (1 - Bmask)

                blurred = blur_objects(gaussian_blurrer, Bimg_right, Bimg)

                save_img = torch.cat([blurred, mask.permute(0, 3, 1, 2).expand(-1, 3, -1, -1)], dim=-1).detach()
                images_list.append(save_img)
            
            
            for syn_img in images_list:
                #syn_img = syn_img[0:3, :, : ]
                print(syn_img.shape)
                save_image(syn_img, f"./results_sample/image_{str(iter_counter).zfill(5)}.png", quality=100)
                iter_counter += 1



def attack_train_execute(
        G_ema, grid_z, grid_c, img_res, attack_type, detector_attack_model, attack_loader, attack_logdir,
        grid_tex_z=None, device=None
):
    '''
    Save visualization during training
    :param G_ema: GET3D generator
    :param grid_z: latent code for geometry latent code
    :param grid_c: None
    :param run_dir: path to save images
    :param cur_nimg: current k images during training
    :param grid_size: size of the image
    :param cur_tick: current kicks for training
    :param image_snapshot_ticks: current snapshot ticks
    :param save_gif_name: the name to save if we want to export gif
    :param save_all:  whether we want to save gif or not
    :param grid_tex_z: the latent code for texture geenration
    :return:
    '''

    #if attack_type == 0:
    adv_grid_tex_z = init_adv_textures(grid_tex_z[0])
    #adv_grid_tex_z = grid_tex_z[0]#init_adv_textures(grid_tex_z[0])
    #adv_grid_tex_z = torch.from_numpy(np.load("/home/snamburu/get3d/GET3D_modified/attack_output/adv_textures_final.npy")).to('cuda')
    adv_grid_z = torch.from_numpy(np.load("/home/snamburu/get3d/GET3D_modified/attack_output/adv_model_final.npy")).to('cuda')
    #G_ema = init_adv_textures(params)

    #adv_grid_z = init_adv_textures(grid_z[0])

    optimizer, scheduler = init_optimizer_scheduler([adv_grid_tex_z])

    lr_decay_rate = 1

    loss_fns = setup_loss_fns()

    iter_counter = 0
    image_id = 0

    with torch.no_grad():
        G_ema.update_w_avg()
        if grid_tex_z is None:
            grid_tex_z = grid_z
    
    writer = SummaryWriter(log_dir=attack_logdir)

    blur_sigma = 1.2
    gaussian_blurrer = torchvision.transforms.GaussianBlur(math.ceil(blur_sigma) * 6 - 1, sigma=blur_sigma)

    print("Running the adversarial attack")

    with EventStorage(iter_counter) as storage:
        for epoch in range(5):
            progress_bar = tqdm(attack_loader, desc=f"Epoch #{epoch + 1}")
            for background_images_batch, labels_batch in progress_bar:
                batch_size = background_images_batch.shape[0]

                cam_mv = sample_cam_attack(4, device=device)
                background_images_batch.to(device)


                ind = 0
                images_list = []
                #adv_grid_tex_z = torch.randn([4, 512], device=device).split(1)  # random code for geometry
                #adv_grid_z = torch.randn([4, 512], device=device).split(1)  # random code for texture
                
                for c in zip(grid_c):
                    camera = cam_mv[ind].unsqueeze(0)
                    z = adv_grid_tex_z
                    geo_z = adv_grid_z 
                    img, mask, sdf, deformation, v_deformed, mesh_v, mesh_f, gen_camera, img_wo_light, tex_hard_mask = G_ema.generate_3d(
                        z=z, geo_z=geo_z, c=c, noise_mode='const',
                        generate_no_light=True, truncation_psi=0.7, camera=camera)
                    rgb_img = img[:, :3]
                    
                    
                    #mesh_v_list.extend([v.data.cpu().numpy() for v in mesh_v])
                    #mesh_f_list.extend([f.data.cpu().numpy() for f in mesh_f])
                    #images = np.concatenate(images_list, axis=0)
                    Bmask = tex_hard_mask.permute(0, 3, 1, 2).repeat(1, 3, 1, 1)
                    Bimg_right = rgb_img * Bmask + background_images_batch[ind] * (1 - Bmask)

                    blurred = blur_objects(gaussian_blurrer, Bimg_right, background_images_batch[ind])

                    images_list.append(blurred)
                    ind += 1

                #Debug
                if iter_counter==0:
                    img_counter = 0
                    for syn_img in images_list:
                        save_image(syn_img, f"./results_sample/imagetrain_{str(img_counter).zfill(5)}.png", quality=100)
                        img_counter += 1

                images_input = torch.cat(images_list)

                model_inputs = construct_model_inputs(images_input, 4, img_res, image_id)

                results = detector_attack_model(model_inputs)

                loss_dict = losses_forward(results, model_inputs, loss_fns, adv_grid_z)

                total_loss = sum(loss_dict.values())
                #print(total_loss)
                total_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # Clamp the texture latent space to the valid image range
                #adv_grid_tex_z.data.clamp_(-4, 4)
                #adv_grid_z.data.clamp_(-4, 4)

                log_info(iter_counter, loss_dict, total_loss, results, writer, scheduler)

                iter_counter += 1

                #if(iter_counter % 100 == 0):
                #    break

            update_LR(scheduler, lr_decay_rate, epoch)
            save_latent(adv_grid_tex_z, os.path.join(attack_logdir, f"adv_textures_{epoch}.npy"))
            save_latent(adv_grid_z, os.path.join(attack_logdir, f"adv_model_{epoch}.npy"))
        
        # Save the final adversarial texture map
        save_latent(adv_grid_tex_z, os.path.join(attack_logdir, f"adv_textures_final.npy"))
        save_latent(adv_grid_z, os.path.join(attack_logdir, f"adv_model_final.npy"))

def save_model(
        G_ema, grid_z, grid_c,
        grid_tex_z=None
):
    #grid_z = torch.zeros([1, 512], device="cuda") #torch.from_numpy(np.load("/home/snamburu/get3d/GET3D_modified/attack_output_exp2/adv_model_2.npy")).to('cuda')
    #grid_tex_z = torch.zeros([100, 512], device="cuda") #torch.from_numpy(np.load("/home/snamburu/get3d/GET3D_modified/attack_output_exp2/adv_textures_2.npy")).to('cuda')
    grid_z = torch.from_numpy(np.load("/home/snamburu/get3d/GET3D_modified/attack_output/adv_model_final.npy")).to('cuda')
    grid_tex_z = torch.from_numpy(np.load("/home/snamburu/get3d/GET3D_modified/attack_output/adv_textures_final.npy")).to('cuda')

    # grid_c = torch.ones(100, device="cuda").split(1)
    # new_vectors = torch.ones(100, 512)
    # x = torch.linspace(-1, 1, 10)
    # y = torch.linspace(-1, 1, 10)
    # grid_x, grid_y = torch.meshgrid(x, y)
    # z = torch.ones(1, 512)
    # z = z.repeat(100, 1)
    # z[:,0] = grid_x.flatten()
    # z[:,1] = grid_y.flatten()
    # z = z.to("cuda")
    # grid_tex_z = z
    with torch.no_grad():
        G_ema.update_w_avg()
        camera_list = G_ema.synthesis.generate_rotate_camera_list(n_batch=grid_z[0].shape[0])
        camera_img_list = []
        #if not save_all:
        camera_list = [camera_list[4]]  # we only save one camera for this
        if grid_tex_z is None:
            grid_tex_z = grid_z
        for i_camera, camera in enumerate(camera_list):
            images_list = []
            #mesh_v_list = []
            #mesh_f_list = []
            for c in zip(grid_c):
                geo_z = grid_z
                z = grid_tex_z
                img, mask, sdf, deformation, v_deformed, mesh_v, mesh_f, gen_camera, img_wo_light, tex_hard_mask = G_ema.generate_3d(
                    z=z, geo_z=geo_z, c=c, noise_mode='const',
                    generate_no_light=True, truncation_psi=0.7, camera=camera)
                rgb_img = img[:, :3]
                save_img = torch.cat([rgb_img, mask.permute(0, 3, 1, 2).expand(-1, 3, -1, -1)], dim=-1).detach()
                images_list.append(save_img)
                # mesh_v_list.extend([v.data.cpu().numpy() for v in mesh_v])
                # mesh_f_list.extend([f.data.cpu().numpy() for f in mesh_f])


            "Code to save mesh"
            mesh_dir = "./attack_output/"
            generated_mesh = G_ema.generate_3d_mesh(
                geo_z=geo_z, tex_z=z, c=None, truncation_psi=0.7,
                use_style_mixing=False)
            for mesh_v, mesh_f, all_uvs, all_mesh_tex_idx, tex_map in zip(*generated_mesh):
                savemeshtes2(
                    mesh_v.data.cpu().numpy(),
                    all_uvs.data.cpu().numpy(),
                    mesh_f.data.cpu().numpy(),
                    all_mesh_tex_idx.data.cpu().numpy(),
                    os.path.join(mesh_dir, '%07d.obj' % (1))
                )
                lo, hi = (-1, 1)
                img = np.asarray(tex_map.permute(1, 2, 0).data.cpu().numpy(), dtype=np.float32)
                img = (img - lo) * (255 / (hi - lo))
                img = img.clip(0, 255)
                mask = np.sum(img.astype(float), axis=-1, keepdims=True)
                mask = (mask <= 3.0).astype(float)
                kernel = np.ones((3, 3), 'uint8')
                dilate_img = cv2.dilate(img, kernel, iterations=1)
                img = img * (1 - mask) + dilate_img * mask
                img = img.clip(0, 255).astype(np.uint8)
                PIL.Image.fromarray(np.ascontiguousarray(img[::-1, :, :]), 'RGB').save(
                    os.path.join(mesh_dir, '%07d.png' % (1)))
                #save_mesh_idx += 1

        iter_counter = 4
        for syn_img in images_list:
            #syn_img = syn_img[0:3, :, : ]
            print(syn_img.shape)
            save_image(syn_img, f"./attack_output/image_{str(iter_counter).zfill(5)}.png", quality=100)
            iter_counter += 1


def save_data(
        G_ema, img_res, detector_attack_model, attack_loader, device=None
):
    '''
    Save visualization during training
    :param G_ema: GET3D generator
    :param grid_z: latent code for geometry latent code
    :param grid_c: None
    :param run_dir: path to save images
    :param cur_nimg: current k images during training
    :param grid_size: size of the image
    :param cur_tick: current kicks for training
    :param image_snapshot_ticks: current snapshot ticks
    :param save_gif_name: the name to save if we want to export gif
    :param save_all:  whether we want to save gif or not
    :param grid_tex_z: the latent code for texture geenration
    :return:
    '''
    log_dir = "./test/"

    #adv_grid_z = init_adv_textures(grid_z[0])


    iter_counter = 0
    image_id = 0

    grid_z = torch.from_numpy(np.load("/home/snamburu/get3d/GET3D_modified/attack_output_shape/adv_model_0.npy")).to('cuda')
    grid_tex_z = torch.from_numpy(np.load("/home/snamburu/get3d/GET3D_modified/attack_output_shape/adv_textures_0.npy")).to('cuda')

    with torch.no_grad():
        G_ema.update_w_avg()
    
    writer = SummaryWriter(log_dir=log_dir)

    blur_sigma = 2
    gaussian_blurrer = torchvision.transforms.GaussianBlur(math.ceil(blur_sigma) * 6 - 1, sigma=blur_sigma)

    print("Running the adversarial attack")
    img_counter = 0

    with EventStorage(iter_counter) as storage:
        progress_bar = tqdm(attack_loader, desc=f"data")
        for background_images_batch, labels_batch in progress_bar:
            batch_size = background_images_batch.shape[0]

            cam_mv = sample_cam_attack(16, device=device)
            background_images_batch.to(device)


            ind = 0
            images_list = []
            #grid_z = torch.randn([16, 512], device=device).split(1)  # random code for geometry
            #grid_tex_z = torch.randn([16, 512], device=device).split(1)  # random code for texture
            grid_c = torch.ones(16, device=device).split(1)
            annotations = []
            
            #for z, geo_z, c in zip(grid_tex_z, grid_z, grid_c):
            for c in zip(grid_c):
                z = grid_tex_z
                geo_z = grid_z
                camera = cam_mv[ind].unsqueeze(0)
                img, mask, sdf, deformation, v_deformed, mesh_v, mesh_f, gen_camera, img_wo_light, tex_hard_mask = G_ema.generate_3d(
                    z=z, geo_z=geo_z, c=c, noise_mode='const',
                    generate_no_light=True, truncation_psi=0.7, camera=camera)
                rgb_img = img[:, :3]
                Xmm = torch.nonzero(mask.squeeze(0).squeeze(-1)).float()
                centermass = torch.mean(Xmm, dim = 0).cpu().numpy()

                annotations.append({
                "van_rv": np.empty(shape=(0, 2)),
                "truck": np.empty(shape=(0, 2)),
                "bus": np.empty(shape=(0, 2)),
                "trailer_small": np.empty(shape=(0, 2)),
                "specialized": np.empty(shape=(0, 2)),
                "trailer_large": np.empty(shape=(0, 2)),
                "unknown": np.empty(shape=(0, 2)),
                "small": np.expand_dims(np.flip(centermass, axis=0), 0)
                 })

                
                
                
                #mesh_v_list.extend([v.data.cpu().numpy() for v in mesh_v])
                #mesh_f_list.extend([f.data.cpu().numpy() for f in mesh_f])
                #images = np.concatenate(images_list, axis=0)
                Bmask = tex_hard_mask.permute(0, 3, 1, 2).repeat(1, 3, 1, 1)
                Bimg_right = rgb_img * Bmask + background_images_batch[ind] * (1 - Bmask)

                blurred = blur_objects(gaussian_blurrer, Bimg_right, background_images_batch[ind])

                images_list.append(blurred)
                ind += 1

            #Debug

            for i, syn_img in enumerate(images_list):
                save_image(syn_img, f"./val_data_attack_shape/images/imagetrain_{str(img_counter).zfill(5)}.png", quality=100)
                with open(f"./val_data_attack_shape/annotations/imagetrain_{str(img_counter).zfill(5)}.pkl", 'wb') as handle:
                    pickle.dump(annotations[i], handle, protocol=pickle.HIGHEST_PROTOCOL)

                img_counter += 1

            if img_counter >= 5000:
                break

            

