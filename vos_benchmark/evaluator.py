import numpy as np
from collections import defaultdict
import cv2
from skimage.morphology import disk

from .utils import _seg2bmap


def get_iou(intersection, pixel_sum):
    # handle edge cases without resorting to epsilon
    if intersection == pixel_sum:
        # both mask and gt have zero pixels in them
        assert intersection == 0
        return 1

    return intersection / (pixel_sum - intersection)


class Evaluator:

    def __init__(self, boundary=0.008, name=None):
        # boundary: used in computing boundary F-score
        self.boundary = boundary
        self.name = name
        self.objects_in_gt = set()
        self.objects_in_masks = set()

        self.object_iou = defaultdict(list)
        self.boundary_f = defaultdict(list)

    def feed_frame(self, mask: np.ndarray, gt: np.ndarray):
        """
        Compute and accumulate metrics for a single frame (mask/gt pair)
        """

        # get all objects in the ground-truth
        gt_objects = np.unique(gt)
        gt_objects = gt_objects[gt_objects != 0].tolist()

        # get all objects in the predicted mask
        mask_objects = np.unique(mask)
        mask_objects = mask_objects[mask_objects != 0].tolist()

        self.objects_in_gt.update(set(gt_objects))
        self.objects_in_masks.update(set(mask_objects))

        all_objects = self.objects_in_gt.union(self.objects_in_masks)

        # boundary disk for boundary F-score. It is the same for all objects.
        bound_pix = np.ceil(self.boundary * np.linalg.norm(mask.shape))
        boundary_disk = disk(bound_pix)

        for obj_idx in all_objects:
            obj_mask = (mask == obj_idx)
            obj_gt = (gt == obj_idx)

            # object iou
            self.object_iou[obj_idx].append(
                get_iou((obj_mask * obj_gt).sum(),
                        obj_mask.sum() + obj_gt.sum()))
            """
            # boundary f-score
            This part is copied from davis2017-evaluation
            """
            mask_boundary = _seg2bmap(obj_mask)
            gt_boundary = _seg2bmap(obj_gt)
            mask_dilated = cv2.dilate(mask_boundary.astype(np.uint8), boundary_disk)
            gt_dilated = cv2.dilate(gt_boundary.astype(np.uint8), boundary_disk)

            # Get the intersection
            gt_match = gt_boundary * mask_dilated
            fg_match = mask_boundary * gt_dilated

            # Area of the intersection
            n_fg = np.sum(mask_boundary)
            n_gt = np.sum(gt_boundary)

            # Compute precision and recall
            if n_fg == 0 and n_gt > 0:
                precision = 1
                recall = 0
            elif n_fg > 0 and n_gt == 0:
                precision = 0
                recall = 1
            elif n_fg == 0 and n_gt == 0:
                precision = 1
                recall = 1
            else:
                precision = np.sum(fg_match) / float(n_fg)
                recall = np.sum(gt_match) / float(n_gt)

            # Compute F measure
            if precision + recall == 0:
                F = 0
            else:
                F = 2 * precision * recall / (precision + recall)
            self.boundary_f[obj_idx].append(F)

    def conclude(self):
        all_iou = {}
        all_boundary_f = {}

        for object_id in self.objects_in_gt:
            all_iou[object_id] = np.mean(self.object_iou[object_id]) * 100
            all_boundary_f[object_id] = np.mean(self.boundary_f[object_id]) * 100

        return all_iou, all_boundary_f
