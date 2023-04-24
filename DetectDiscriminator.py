import torch
from torch import Tensor
from typing import List, Dict
from detectron2.layers import cat
from torch.nn import functional as F
from fvcore.nn import sigmoid_focal_loss_jit
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.utils.events import get_event_storage
from detectron2.modeling.meta_arch.retinanet import RetinaNet
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser
from detectron2.modeling import build_model
from PIL import Image

import cv2
from detectron2.engine import DefaultPredictor

from RetinaNetPoint import RetinaNetPoint

from torchvision.transforms import ToTensor


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    # default_setup(
    #     cfg, args
    # )  # if you don't like any of the default setup, write your own setup code
    return cfg

# def _dense_box_regression_loss(
#     anchors: List[Union[Boxes, torch.Tensor]],
#     box2box_transform: Box2BoxTransform,
#     pred_anchor_deltas: List[torch.Tensor],
#     gt_boxes: List[torch.Tensor],
#     fg_mask: torch.Tensor,
#     box_reg_loss_type="smooth_l1",
#     smooth_l1_beta=0.0,
# ):
#     """
#     Compute loss for dense multi-level box regression.
#     Loss is accumulated over ``fg_mask``.
#     Args:
#         anchors: #lvl anchor boxes, each is (HixWixA, 4)
#         pred_anchor_deltas: #lvl predictions, each is (N, HixWixA, 4)
#         gt_boxes: N ground truth boxes, each has shape (R, 4) (R = sum(Hi * Wi * A))
#         fg_mask: the foreground boolean mask of shape (N, R) to compute loss on
#         box_reg_loss_type (str): Loss type to use. Supported losses: "smooth_l1", "giou",
#             "diou", "ciou".
#         smooth_l1_beta (float): beta parameter for the smooth L1 regression loss. Default to
#             use L1 loss. Only used when `box_reg_loss_type` is "smooth_l1"
#     """
#     if isinstance(anchors[0], Boxes):
#         anchors = type(anchors[0]).cat(anchors).tensor  # (R, 4)
#     else:
#         anchors = cat(anchors)
#     if box_reg_loss_type == "smooth_l1":
#         gt_anchor_deltas = [box2box_transform.get_deltas(anchors, k) for k in gt_boxes]
#         gt_anchor_deltas = torch.stack(gt_anchor_deltas)  # (N, R, 4)
#         loss_box_reg = smooth_l1_loss(
#             cat(pred_anchor_deltas, dim=1)[fg_mask],
#             gt_anchor_deltas[fg_mask],
#             beta=smooth_l1_beta,
#             reduction="sum",
#         )
#     elif box_reg_loss_type == "giou":
#         pred_boxes = [
#             box2box_transform.apply_deltas(k, anchors) for k in cat(pred_anchor_deltas, dim=1)
#         ]
#         loss_box_reg = giou_loss(
#             torch.stack(pred_boxes)[fg_mask], torch.stack(gt_boxes)[fg_mask], reduction="sum"
#         )
#     elif box_reg_loss_type == "diou":
#         pred_boxes = [
#             box2box_transform.apply_deltas(k, anchors) for k in cat(pred_anchor_deltas, dim=1)
#         ]
#         loss_box_reg = diou_loss(
#             torch.stack(pred_boxes)[fg_mask], torch.stack(gt_boxes)[fg_mask], reduction="sum"
#         )
#     elif box_reg_loss_type == "ciou":
#         pred_boxes = [
#             box2box_transform.apply_deltas(k, anchors) for k in cat(pred_anchor_deltas, dim=1)
#         ]
#         loss_box_reg = ciou_loss(
#             torch.stack(pred_boxes)[fg_mask], torch.stack(gt_boxes)[fg_mask], reduction="sum"
#         )
#     elif box_reg_loss_type == "smooth_l1_point":
#         gt_anchor_deltas = [box2box_transform.get_deltas(anchors, k) for k in gt_boxes]
#         gt_anchor_deltas = torch.stack(gt_anchor_deltas)  # (N, R, 4)
#         pred_anchor_deltas = cat(pred_anchor_deltas, dim=1)
        
#         # Retain only centers, remove box sizes
#         pred_anchor_deltas = pred_anchor_deltas[..., :2]
#         gt_anchor_deltas = gt_anchor_deltas[..., :2]

#         # Compute the loss
#         loss_box_reg = smooth_l1_loss(
#             pred_anchor_deltas[fg_mask],
#             gt_anchor_deltas[fg_mask],
#             beta=smooth_l1_beta,
#             reduction="sum",
#         )
#     else:
#         raise ValueError(f"Invalid dense box regression loss type '{box_reg_loss_type}'")
#     return loss_box_reg



class RetinaNetPoint(RetinaNet):
    def __init__(self, cfg):
        super().__init__(get_cfg)
        
    def losses(self, anchors, pred_logits, gt_labels, pred_anchor_deltas, gt_boxes):
        """
        Args:
            anchors (list[Boxes]): a list of #feature level Boxes
            gt_labels, gt_boxes: see output of :meth:`RetinaNet.label_anchors`.
                Their shapes are (N, R) and (N, R, 4), respectively, where R is
                the total number of anchors across levels, i.e. sum(Hi x Wi x Ai)
            pred_logits, pred_anchor_deltas: both are list[Tensor]. Each element in the
                list corresponds to one level and has shape (N, Hi * Wi * Ai, K or 4).
                Where K is the number of classes used in `pred_logits`.
        Returns:
            dict[str, Tensor]:
                mapping from a named loss to a scalar tensor storing the loss.
                Used during training only. The dict keys are: "loss_cls" and "loss_box_reg"
        """
        num_images = len(gt_labels)
        gt_labels = torch.stack(gt_labels)  # (N, R)

        valid_mask = gt_labels >= 0
        pos_mask = (gt_labels >= 0) & (gt_labels != self.num_classes)
        num_pos_anchors = pos_mask.sum().item()
        get_event_storage().put_scalar("num_pos_anchors", num_pos_anchors / num_images)
        normalizer = self._ema_update("loss_normalizer", max(num_pos_anchors, 1), 100)

        # classification and regression loss
        gt_labels_target = F.one_hot(gt_labels[valid_mask], num_classes=self.num_classes + 1)[
            :, :-1
        ]  # no loss for the last (background) class
        loss_cls = sigmoid_focal_loss_jit(
            cat(pred_logits, dim=1)[valid_mask],
            gt_labels_target.to(pred_logits[0].dtype),
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        )

        loss_point_reg = 0 # _dense_box_regression_loss(
        #     anchors,
        #     self.box2box_transform,
        #     pred_anchor_deltas,
        #     gt_boxes,
        #     pos_mask,
        #     box_reg_loss_type=self.box_reg_loss_type,
        #     smooth_l1_beta=self.smooth_l1_beta,
        # )
        
        return {
            "loss_cls": loss_cls / normalizer,
            "loss_box_reg": loss_point_reg / normalizer,
        }
    
    def forward_attack(self, batched_inputs: List[Dict[str, Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * image: Tensor, image in (C, H, W) format.
                * instances: Instances
                Other information that's included in the original dicts, such as:
                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            In training, dict[str, Tensor]: mapping from a named loss to a tensor storing the
            loss. Used during training only. In inference, the standard output format, described
            in :doc:`/tutorials/models`.
        """
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        features = [features[f] for f in self.head_in_features]
        predictions = self.head(features)
        
        pred_logits, pred_anchor_deltas = self._transpose_dense_predictions(
            predictions, [self.num_classes, 4]
        )
        anchors = self.anchor_generator(features)

        results: List[Instances] = []
        for img_idx, image_size in enumerate(images.image_sizes):
            scores_per_image = [x[img_idx].sigmoid() for x in pred_logits]
            deltas_per_image = [x[img_idx] for x in pred_anchor_deltas]
            results_per_image = self.inference_single_image(
                anchors, scores_per_image, deltas_per_image, image_size
            )
            results.append(results_per_image[0])
        
        return results
    
if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    print("Discriminator detector")
    cfg = setup(args)
    model = build_model(cfg)
    predictor = DefaultPredictor(cfg)
    image_path = "/home/snamburu/attack/satellite-vehicle-point-detection/results_mik/image_00026.png"
    images_batch = cv2.imread(image_path)
    model.eval()
    with torch.no_grad():
        out = predictor(images_batch)
    print(out)

    #dif.render()