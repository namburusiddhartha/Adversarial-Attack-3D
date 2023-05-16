import torch
import torch.nn as nn

class ScoreLoss(nn.Module):
    def __init__(self, coefficient=1.0):
        super(ScoreLoss, self).__init__()
        
        self.coefficient = coefficient
        
    def forward(self, outputs):
        """
        Input variables 'outputs' is expected to have the structure of detectron2 inference prediction format.
        """
        loss = 0
        num_el_counter = 0
        
        for output in outputs:
            if output.scores.shape[0] > 0:
                loss = loss + torch.max(output.scores)
                num_el_counter += 1
        
        # Normalize and weigh the loss
        loss = self.coefficient * loss / num_el_counter
        
        return loss
    
class TVLoss(nn.Module):
    """
    Module providing the functionality necessary to calculate the total variation (TV) of an adversarial patch.
    
    inputs:
        - adv_patch: adversarial patch of shape (B, C, H, W), where B is the batch size, C is the number of channels,
        H and W are the height and width of the patch respectively.
    """
    def __init__(self, coefficient=1.0):
        super(TVLoss, self).__init__()
        
        self.coefficient = coefficient
        
    def forward(self, adv_textures):
        tv = 0
        for i in range(adv_textures.size(0)):
            tvcomp1 = torch.norm(adv_textures[i, :, :, 1:] - adv_textures[i, :, :, :-1], p=2)
            tvcomp2 = torch.norm(adv_textures[i, :, 1:, :] - adv_textures[i, :, :-1, :], p=2)
 
            tv += tvcomp1 + tvcomp2
        return self.coefficient * tv / torch.numel(adv_textures)
    
class NPSLoss(nn.Module):
    """
    Non-printability score as defined in "Physical Adversarial Attacks on an Aerial Imagery Object Detector".
    """
    def __init__(self, printable_colors, coefficient=1.0, device='cuda'):
        super(NPSLoss, self).__init__()
        self.printable_colors = printable_colors.to(device)
        self.coefficient = coefficient
        self.device = device
        
    def forward(self, adv_patch):
        nps = 0
        adv_patch_pixels = adv_patch.view(-1, 3) # Flatten the adversarial patch
        distances_matrix = torch.cdist(adv_patch_pixels, self.printable_colors)
        closest_distances = torch.min(distances_matrix, dim=1)[0]
        nps = torch.mean(closest_distances)
        return self.coefficient * nps
    
class DiversionLoss(nn.Module):
    """
    Loss that enforces diversion of the prediction from the GT locations.
    """
    def __init__(self, coefficient=1.0, sigma=4.0):
        super(DiversionLoss, self).__init__()
        self.coefficient = coefficient
        self.sigma = sigma
        
    def forward(self, pred_points_batch, gt_points_batch):
        distances_batch = [torch.cdist(gt_points, pred_points) for pred_points, gt_points in zip(pred_points_batch, gt_points_batch)]
        mean_distances = [torch.mean(distances, dim=1)[0] for distances in distances_batch if distances.shape[0] > 0]
        if len(mean_distances) > 0:
            loss_batch = [torch.exp(-mean_distance * mean_distance / (2 * self.sigma * self.sigma)) for mean_distance in mean_distances]
            loss = sum(loss_batch) / len(loss_batch)
        else:
            loss = 0
        return self.coefficient * loss
    
class DiscriminatorLoss(nn.Module):
    """
    Loss that returns the probability of the input image being classified as a synthetic image.
    """
    def __init__(self, discriminator, coefficient=1.0):
        super(DiscriminatorLoss, self).__init__()
        self.coefficient = coefficient
        self.discriminator = discriminator
        
    def forward(self, images_batch):
        self.discriminator.eval()
        with torch.no_grad():
            preds = self.discriminator(images_batch)
        return torch.mean(preds)