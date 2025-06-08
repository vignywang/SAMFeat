#
# Created  on 2020/8/31
#
from collections import defaultdict

import cv2
import torch
import torch.nn.functional as f
from torch.optim.lr_scheduler import _LRScheduler
from PIL import Image
import numpy as np


class PolynomialLR(_LRScheduler):
    def __init__(self, optimizer, step_size, iter_max, power, last_epoch=-1):
        self.step_size = step_size
        self.iter_max = iter_max
        self.power = power
        super(PolynomialLR, self).__init__(optimizer, last_epoch)

    def polynomial_decay(self, lr):
        return lr * (1 - float(self.last_epoch) / self.iter_max) ** self.power

    def get_lr(self):
        if (
            (self.last_epoch == 0)
            or (self.last_epoch % self.step_size != 0)
            or (self.last_epoch > self.iter_max)
        ):
            return [group["lr"] for group in self.optimizer.param_groups]
        return [self.polynomial_decay(lr) for lr in self.base_lrs]


def resize_labels(labels, size):
    """
    Downsample labels for 0.5x and 0.75x logits by nearest interpolation.
    Other nearest methods result in misaligned labels.
    -> F.interpolate(labels, shape, mode='nearest')
    -> cv2.resize(labels, shape, interpolation=cv2.INTER_NEAREST)
    """
    new_labels = []
    for label in labels:
        label = label.float().numpy()
        label = Image.fromarray(label).resize(size, resample=Image.NEAREST)
        new_labels.append(np.asarray(label))
    new_labels = torch.LongTensor(new_labels)
    return new_labels


def resize_labels_torch(label, size, mode='nearest'):
    """
    similar like resize_labels, but direct do it for torch tensors,
    size: require shape of [h,w]
    label: require shape of [bt,h,w]
    """
    label = f.interpolate(label.unsqueeze(dim=1), size, mode=mode).squeeze(dim=1)
    return label


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist


def scores(label_trues, label_preds, n_class):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    valid = hist.sum(axis=1) > 0  # added
    mean_iu = np.nanmean(iu[valid])
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    cls_iu = dict(zip(range(n_class), iu))

    return {
        "Pixel Accuracy": acc,
        "Mean Accuracy": acc_cls,
        "Frequency Weighted IoU": fwavacc,
        "Mean IoU": mean_iu,
        # "Class IoU": cls_iu,
    }


class DepthEvaluator(object):

    def __init__(self):
        self.errors = defaultdict(list)
        self.min_depth = 1e-3
        self.max_depth = 70

    def reset(self):
        self.errors = defaultdict(list)

    def val(self):
        error = {}
        for k, v in self.errors.items():
            error[k] = np.stack(v).mean()
        return error

    def eval(self, pred_depth, gt_depth):
        gt_depth = np.clip(gt_depth, a_min=None, a_max=self.max_depth)

        # mask = np.logical_and(gt_depth > self.min_depth, gt_depth <= self.max_depth)
        mask = gt_depth > self.min_depth

        scalor = np.median(gt_depth[mask]) / np.median(pred_depth[mask])
        pred_depth[mask] *= scalor

        pred_depth[pred_depth < self.min_depth] = self.min_depth
        pred_depth[pred_depth > self.max_depth] = self.max_depth

        errors = self.compute_errors(gt_depth[mask], pred_depth[mask])
        for k, v in errors.items():
            self.errors[k].append(v)

    @staticmethod
    def compute_errors(gt, pred):
        thresh = np.maximum((gt / pred), (pred / gt))
        a1 = (thresh < 1.25).mean()
        a2 = (thresh < 1.25 ** 2).mean()
        a3 = (thresh < 1.25 ** 3).mean()

        rmse = (gt - pred) ** 2
        rmse = np.sqrt(rmse.mean())

        rmse_log = (np.log(gt) - np.log(pred)) ** 2
        rmse_log = np.sqrt(rmse_log.mean())

        abs_rel = np.abs((gt - pred) / gt)
        abs_rel = abs_rel.mean()

        sq_rel = (gt - pred) ** 2 / gt
        sq_rel = sq_rel.mean()

        return {
            'abs_rel': abs_rel,
            'sq_rel': sq_rel,
            'rmse': rmse,
            'rmse_log': rmse_log,
            'a1': a1,
            'a2': a2,
            'a3': a3,
        }






