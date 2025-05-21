#
# Created  on 2019/8/14
#
import os
from glob import glob

import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f


def generate_testing_file(folder, prefix="model"):
    models = glob(os.path.join(folder, prefix + "_*.pt"))
    models = sorted(models)
    return models


def compute_batched_dist(x, y, hamming=False):
    # x:[bt,256,n], y:[bt,256,n]
    cos_similarity = torch.matmul(x.transpose(1, 2), y)  # [bt,n,n]
    if hamming is False:
        square_norm_x = (torch.norm(x, dim=1, keepdim=True).transpose(1, 2)) ** 2  # [bt,n,1]
        square_norm_y = (torch.norm(y, dim=1, keepdim=True)) ** 2  # [bt,1,n]
        dist = torch.sqrt((square_norm_x + square_norm_y - 2 * cos_similarity + 1e-4))
        return dist
    else:
        dist = 0.5 * (256 - cos_similarity)
    return dist


def compute_cos_similarity_general(x, y):
    x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
    y_norm = torch.norm(y, p=2, dim=1, keepdim=True)
    x = x.div(x_norm + 1e-4)
    y = y.div(y_norm + 1e-4)
    cos_similarity = torch.matmul(x.transpose(1, 2), y)  # [bt,h*w,h*w]
    return cos_similarity


def compute_cos_similarity_binary(x, y, k=256):
    x = x.div(np.sqrt(k))
    y = y.div(np.sqrt(k))
    cos_similarity = torch.matmul(x.transpose(1, 2), y)
    return cos_similarity


def spatial_nms(prob, kernel_size=9):
    """

    """
    padding = int(kernel_size // 2)
    pooled = f.max_pool2d(prob, kernel_size=kernel_size, stride=1, padding=padding)
    prob = torch.where(torch.eq(prob, pooled), prob, torch.zeros_like(prob))
    return prob


def draw_image_keypoints(image, points, color=(0, 255, 0), show=False):
    """

    """
    n, _ = points.shape
    cv_keypoints = []
    for i in range(n):
        keypt = cv.KeyPoint()
        keypt.pt = (points[i, 1], points[i, 0])
        cv_keypoints.append(keypt)
    image = cv.drawKeypoints(image.astype(np.uint8), cv_keypoints, None, color=color)
    if show:
        cv.imshow("image&keypoints", image)
        cv.waitKey()
    return image


class JointLoss(object):

    def __init__(self, w_data=1.0, w_grad=0.5):
        self.w_data = w_data
        self.w_grad = w_grad

    @staticmethod
    def gradient_loss(log_prediction_d, log_gt, mask, scale):
        assert scale in [1, 2, 4, 8]
        log_prediction_d = log_prediction_d[:, ::scale, ::scale]
        log_gt = log_gt[:, ::scale, ::scale]
        mask = mask[:, ::scale, ::scale]

        N = torch.sum(mask, dim=[1, 2]) + 1.
        log_d_diff = log_prediction_d - log_gt
        log_d_diff = torch.mul(log_d_diff, mask)

        v_gradient = torch.abs(log_d_diff[:, 0:-2, :] - log_d_diff[:, 2:, :])
        v_mask = torch.mul(mask[:, 0:-2, :], mask[:, 2:, :])
        v_gradient = torch.mul(v_gradient, v_mask)

        h_gradient = torch.abs(log_d_diff[:, :, 0:-2] - log_d_diff[:, :, 2:])
        h_mask = torch.mul(mask[:, :, 0:-2], mask[:, :, 2:])
        h_gradient = torch.mul(h_gradient, h_mask)

        gradient_loss = torch.sum(h_gradient, dim=[1, 2]) + torch.sum(v_gradient, dim=[1, 2])
        gradient_loss = gradient_loss / N
        gradient_loss = torch.mean(gradient_loss)

        return gradient_loss

    @staticmethod
    def data_loss(log_prediction_d, log_gt, mask):
        N = torch.sum(mask, dim=[1, 2]) + 1.
        log_d_diff = log_prediction_d - log_gt
        log_d_diff = torch.mul(log_d_diff, mask)
        s1 = torch.sum(torch.pow(log_d_diff, 2), dim=[1, 2]) / N
        s2 = torch.pow(torch.sum(log_d_diff, dim=[1, 2]), 2) / (N * N)
        data_loss = s1 - s2
        data_loss = torch.mean(data_loss)

        return data_loss

    def __call__(self, log_pred, depth_gt, mask):
        log_gt = torch.log(torch.clamp(depth_gt, 1e-5))

        data_loss = self.w_data * self.data_loss(log_pred, log_gt, mask)
        gradient_loss1 = self.w_grad * self.gradient_loss(log_pred, log_gt, mask, 1)
        gradient_loss2 = self.w_grad * self.gradient_loss(log_pred, log_gt, mask, 2)
        gradient_loss4 = self.w_grad * self.gradient_loss(log_pred, log_gt, mask, 4)
        gradient_loss8 = self.w_grad * self.gradient_loss(log_pred, log_gt, mask, 8)

        gradient_loss = gradient_loss1 + gradient_loss2 + gradient_loss4 + gradient_loss8
        total_loss = data_loss + gradient_loss

        return total_loss, data_loss, gradient_loss


class JointLossSmooth(JointLoss):

    def __init__(self, w_data=1.0, w_grad=1.0):
        super(JointLossSmooth, self).__init__(w_data, w_grad)

    @staticmethod
    def gradient_loss(log_prediction_d, img):
        depth_pred = torch.exp(log_prediction_d)
        grad_depth_x = torch.abs(depth_pred[:, :, :-1] - depth_pred[:, :, 1:])
        grad_depth_y = torch.abs(depth_pred[:, :-1, :] - depth_pred[:, 1:, :])

        grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1)
        grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1)

        grad_depth_x *= torch.exp(-grad_img_x)
        grad_depth_y *= torch.exp(-grad_img_y)

        return grad_depth_x.mean() + grad_depth_y.mean()

    def __call__(self, log_pred, depth_gt, mask, image):
        log_gt = torch.log(torch.clamp(depth_gt, 1e-5))

        data_loss = self.w_data * self.data_loss(log_pred, log_gt, mask)
        gradient_loss = self.w_grad * self.gradient_loss(log_pred, image)

        total_loss = data_loss + gradient_loss

        return total_loss, data_loss, gradient_loss


class SoftCrossEntropyVectorLoss(object):
    """
    L1 loss with mask
    """

    def __init__(self):
        self.softmax = torch.nn.Softmax(dim=2)
        self.log_softmax = torch.nn.LogSoftmax(dim=2)

    def __call__(self, teacher_logit, pred_logit, valid_mask):
        teacher_prob = self.softmax(teacher_logit)
        log_pred_prob = self.log_softmax(pred_logit)
        loss = torch.sum(-teacher_prob * log_pred_prob, dim=2) * valid_mask

        total_valid = torch.sum(valid_mask)
        loss = torch.sum(loss) / total_valid

        return loss


class MaskL2Loss(object):

    def __init__(self):
        self.log_softmax = torch.nn.LogSoftmax(dim=1)
        self.softmax = torch.nn.Softmax(dim=1)

    def __call__(self, teacher_logit, pred_logit, valid_mask):
        '''
        valid_mask: [bt,h,w]
        '''
        _, _, h, w = teacher_logit.shape
        valid_mask = f.interpolate(valid_mask.unsqueeze(dim=1), size=(h, w), mode='bilinear', align_corners=True)[:, 0,
                     :, :]
        dense_loss = torch.mean((teacher_logit - pred_logit) ** 2, dim=1) * valid_mask  # [bt,h,w]
        valid_sum = torch.sum(valid_mask)
        loss = torch.sum(dense_loss) / valid_sum

        return loss


class DescriptorHingeLoss(object):
    """
    According the Paper of SuperPoint
    """

    def __init__(self, device, lambda_d=250, m_p=1, m_n=0.2, ):
        self.device = device
        self.lambda_d = torch.tensor(lambda_d, device=self.device)
        self.m_p = torch.tensor(m_p, device=self.device)
        self.m_n = torch.tensor(m_n, device=self.device)
        self.one = torch.tensor(1, device=self.device)

    def __call__(self, desp_0, desp_1, desp_mask, valid_mask):
        batch_size, dim, h, w = desp_0.shape
        desp_0 = torch.reshape(desp_0, (batch_size, dim, -1)).transpose(dim0=2, dim1=1)  # [bt, h*w, dim]
        desp_1 = torch.reshape(desp_1, (batch_size, dim, -1))

        cos_similarity = torch.matmul(desp_0, desp_1)
        positive_term = f.relu(self.m_p - cos_similarity) * self.lambda_d
        negative_term = f.relu(cos_similarity - self.m_n)

        positive_mask = desp_mask
        negative_mask = self.one - desp_mask

        loss = positive_mask * positive_term + negative_mask * negative_term

        # 考虑warp带来的某些区域没有图像,则其对应的描述子应当无效
        valid_mask = torch.unsqueeze(valid_mask, dim=1)  # [bt, 1, h*w]
        total_num = torch.sum(valid_mask, dim=(1, 2)) * h * w
        loss = torch.sum(valid_mask * loss, dim=(1, 2)) / total_num
        loss = torch.mean(loss)

        return loss


def prob_2_entropy(prob):
    """ convert probabilistic prediction maps to weighted self-information maps
    """
    n, c, h, w = prob.size()
    return -torch.mul(prob, torch.log2(prob + 1e-30)) / np.log2(c)


class DescriptorPairwiseLoss(object):

    def __init__(self, device):
        self.device = device

    def __call__(self, desp_0, desp_1, positive_mask, positive_valid_mask, negative_mask, negative_valid_mask):
        cos_similarity = torch.matmul(desp_0, desp_1.transpose(1, 2))

        # compute positive pair loss
        positive_loss_total = positive_mask * (1. - cos_similarity) * positive_valid_mask
        positive_loss_num = torch.sum(positive_valid_mask, dim=(1, 2))
        positive_loss = torch.sum(positive_loss_total, dim=(1, 2)) / positive_loss_num

        # compute negative pair loss
        negative_loss_total = negative_mask * torch.relu(cos_similarity - 0.2) * negative_valid_mask
        negative_loss_num = torch.sum(negative_valid_mask, dim=(1, 2))
        negative_loss = torch.sum(negative_loss_total, dim=(1, 2)) / negative_loss_num

        loss = positive_loss + negative_loss
        loss = torch.mean(loss)

        return loss


class DescriptorTripletLoss(object):

    def __init__(self, device):
        self.device = device

    def __call__(self, desp_0, desp_1, matched_idx, matched_valid, not_search_mask, debug_use=False):

        bt, dim, h, w = desp_0.shape
        desp_0 = torch.reshape(desp_0, (bt, dim, h * w)).transpose(1, 2)  # [bt,h*w,dim]
        desp_1 = torch.reshape(desp_1, (bt, dim, h * w))

        matched_idx = torch.unsqueeze(matched_idx, dim=1).repeat(1, dim, 1)  # [bt,dim,h*w]
        desp_1 = torch.gather(desp_1, dim=2, index=matched_idx)  # [bt,dim,h*w]

        cos_similarity = torch.matmul(desp_0, desp_1)
        dist = torch.sqrt(2. * (1. - cos_similarity).clamp(1e-5))  # [bt,h*w,h*w]

        positive_pair = torch.diagonal(dist, dim1=1, dim2=2)  # [bt,h*w]
        dist = dist + 10 * not_search_mask

        hardest_negative_pair, hardest_negative_idx = torch.min(dist, dim=2)  # [bt,h*w]

        zeros = torch.zeros_like(positive_pair)
        loss_total, _ = torch.max(torch.stack((zeros, 1.0 + positive_pair - hardest_negative_pair), dim=2), dim=2)
        loss_total *= matched_valid

        valid_num = torch.sum(matched_valid, dim=1)
        loss = torch.mean(torch.sum(loss_total, dim=1) / (valid_num + 1.))

        if debug_use:
            positive_dist = torch.mean(torch.sum(positive_pair * matched_valid, dim=1) / valid_num)
            negative_dist = torch.mean(torch.sum(hardest_negative_pair * matched_valid, dim=1) / valid_num)
            return loss, positive_dist, negative_dist
        else:
            return loss

class AttentionMapWithSamCatLoss(object):
    def __init__(self, device):
        self.device = device

    def __call__(self, attention_map, cat_0, cat_1):
        category_map = torch.cat((cat_0, cat_1), dim=0).to(self.device)
        new_attention_map = nn.functional.interpolate(attention_map, size=(400, 400), mode='bilinear', align_corners=False)
        # Convert category information map to category map
        batch_size, _, height, width = attention_map.size()

        # Initialize feature map with zeros
        back_feature_map = torch.zeros(batch_size, 1, 400, 400, device=self.device)

        # Select coordinates where category_map[:, :, 2] == 0
        back_zero_category_coords = category_map[:, :, 2] == 0

        # Update feature map with ones at the selected coordinates
        back_scaled_x_coords = category_map[:, :, 0][back_zero_category_coords]
        back_scaled_y_coords = category_map[:, :, 1][back_zero_category_coords]

        # Ensure indices are of the appropriate data type
        back_scaled_x_coords = back_scaled_x_coords.to(torch.long)
        back_scaled_y_coords = back_scaled_y_coords.to(torch.long)

        # Update feature map with ones at the scaled coordinates
        back_feature_map[:, :, back_scaled_y_coords, back_scaled_x_coords] = 1.0

        # Multiply attention map with the category map
        back_product = new_attention_map * back_feature_map

        background_mean =torch.mean(back_product)


        # Foreground
        # Todo 400 att map
        foreground_feature_map = torch.zeros(batch_size, 1, 400, 400, device=self.device)
        foreground_zero_category_coords = (category_map[:, :, 2] != 0) & (category_map[:, :, 2] != -1)

        # Update feature map with ones at the selected coordinates
        foreground_scaled_x_coords = category_map[:, :, 0][foreground_zero_category_coords]
        foreground_scaled_y_coords = category_map[:, :, 1][foreground_zero_category_coords]

        # Ensure indices are of the appropriate data type
        foreground_scaled_x_coords = foreground_scaled_x_coords.to(torch.long)
        foreground_scaled_y_coords = foreground_scaled_y_coords.to(torch.long)

        # Update feature map with ones at the scaled coordinates
        foreground_feature_map[:, :, foreground_scaled_y_coords, foreground_scaled_x_coords] = 1.0

        # Multiply attention map with the category map
        foreground_product = new_attention_map * foreground_feature_map

        # Take the mean of the product
        foreground_mean = torch.mean(foreground_product)

        return background_mean / foreground_mean

class AttentionWeightedTripletLossWithSam(object):

    def __init__(self, device, T, margin):
        self.device = device
        self.T = T
        self.margin = margin

    def select_flagged_tensors(self, input_tensor, flags_tensor):
        _, F2 = torch.chunk(flags_tensor, 2, dim=0)
        indices = torch.nonzero(F2.squeeze(), as_tuple=False).squeeze()

        selected_tensor = input_tensor[indices]

        return selected_tensor

    def _compute_dist(self, X, Y):
        """

        """
        XTX = torch.pow(X, 2).sum(dim=2)  # [bt,n]
        YTY = torch.pow(Y, 2).sum(dim=2)  # [bt,m]
        XTY = torch.bmm(X, Y.transpose(1, 2))

        dist2 = XTX.unsqueeze(dim=2) - 2 * XTY + YTY.unsqueeze(dim=1)  # [bt,n,m]
        dist = torch.sqrt(torch.clamp(dist2, 1e-5))
        return dist

    def InfoNCE(self, distance_matrix, positive_pairs_mask, negative_pairs_mask):
        # Extract the positive and negative pairs from the distance matrix
        temperature = self.T
        positive_pairs = torch.max(self.margin * positive_pairs_mask, distance_matrix * positive_pairs_mask) #/ torch.sum(positive_pairs_mask)
        negative_pairs = distance_matrix * negative_pairs_mask #/ torch.sum(negative_pairs_mask)
        sim_pos = torch.exp(-positive_pairs / temperature)
        sim_neg = torch.exp(-negative_pairs / temperature)

        # Calculate the denominator for normalization (sum of exp(sim_neg) for each row)
        denominator = sim_neg.sum(dim=1)

        # Calculate the normalized probabilities for each positive pair
        probs = sim_pos.sum(dim=1) / (sim_pos.sum(dim=1) + denominator)
        # Calculate InfoNCE loss
        loss = -torch.log(probs).mean()
        """
        P = torch.mean(positive_pairs, dim=1)
        N = torch.mean(negative_pairs, dim=1)

        PN = torch.bmm(P.unsqueeze(2), N.unsqueeze(1))
        PP = torch.bmm(P.unsqueeze(2), P.unsqueeze(1))

        denominator = torch.exp(PN) + torch.sum(torch.exp(PP), dim=1, keepdim=True)

        # Calculate the loss
        loss = -torch.log(torch.exp(PN) / denominator).mean()
        """
        return loss
    def compute_loss(self, distance_matrix, category_tensor):
        # Step 1: Create a binary mask for valid points (category index != -1)
        valid_points = category_tensor[:, :, -1] != -1

        # Step 2: Form the similarity matrix based on the categorical relationship
        category_index = category_tensor[:, :, -1]
        category_matrix = (category_index.unsqueeze(1) == category_index.unsqueeze(2)).float()

        # Step 3: Create the positive pairs mask, considering only valid points
        positive_pairs_mask = category_matrix * valid_points.unsqueeze(2) * valid_points.unsqueeze(1)

        # Step 4: Create the negative pairs mask, considering only valid points
        negative_pairs_mask = (1 - category_matrix) * valid_points.unsqueeze(2) * valid_points.unsqueeze(1)

        # Step 5: Extract the positive and negative pairs from the distance matrix
        #positive_pairs = distance_matrix * positive_pairs_mask
        #negative_pairs = distance_matrix * negative_pairs_mask

        # Step 7: Calculate contrastive loss
        #loss = torch.sum(positive_pairs) / (torch.sum(positive_pairs) + torch.sum(negative_pairs))
        #loss = torch.relu(torch.sum(positive_pairs)/torch.sum(positive_pairs_mask) - torch.sum(negative_pairs)/torch.sum(negative_pairs_mask))
        loss = self.InfoNCE(distance_matrix, positive_pairs_mask, negative_pairs_mask)
        return loss

    def __call__(self, desp_0, desp_1, w_0, w_1, valid_mask, not_search_mask, cat_0, cat_1, flags):
        """

        """
        desp_0 = desp_0 * w_0
        desp_1 = desp_1 * w_1
        dist_0 = self._compute_dist(desp_0, desp_0) + 10 * not_search_mask
        desp_1 = self.select_flagged_tensors(desp_1, flags)
        dist_1 = self._compute_dist(desp_1, desp_1) + 10 * self.select_flagged_tensors(not_search_mask, flags)
        l0 = self.compute_loss(dist_0, cat_0)
        l1 = self.compute_loss(dist_1, self.select_flagged_tensors(cat_1, flags))
        loss = (l0 + l1) / 2
        return loss

class AttentionWeightedTripletLoss(object):

    def __init__(self, device, T):
        self.device = device
        self.T = T

    def _compute_dist(self, X, Y):
        """

        """
        XTX = torch.pow(X, 2).sum(dim=2)  # [bt,n]
        YTY = torch.pow(Y, 2).sum(dim=2)  # [bt,m]
        XTY = torch.bmm(X, Y.transpose(1, 2))

        dist2 = XTX.unsqueeze(dim=2) - 2 * XTY + YTY.unsqueeze(dim=1)  # [bt,n,m]
        dist = torch.sqrt(torch.clamp(dist2, 1e-5))
        return dist

    def __call__(self, desp_0, desp_1, w_0, w_1, valid_mask, not_search_mask):
        """

        """
        desp_0 = desp_0 * w_0
        desp_1 = desp_1 * w_1
        dist = self._compute_dist(desp_0, desp_1)
        positive_pair = torch.diagonal(dist, dim1=1, dim2=2)  # [bt,n]
        dist = dist + 10 * not_search_mask
        hardest_negative_pair, hardest_negative_idx = torch.min(dist, dim=2)  # [bt,n]
        loss_total = torch.relu(1 + positive_pair - hardest_negative_pair)
        weight = w_0.squeeze() / self.T
        weight = torch.exp(weight)
        weight = weight * valid_mask
        weight = weight / torch.clamp((torch.sum(weight, dim=1, keepdim=True)), 1e-5)
        loss = torch.mean(torch.sum(loss_total * weight, dim=1))
        return loss


class DescriptorTripletAugmentationLoss(object):

    def __init__(self, device):
        self.device = device

    def __call__(self, desp1, desp2, desp3, desp4, valid_mask12, valid_mask13, valid_mask42, not_search_mask12,
                 not_search_mask13, not_search_mask42):
        loss12 = self.compute_loss(desp1, desp2, valid_mask12, not_search_mask12)
        loss13 = self.compute_loss(desp1, desp3, valid_mask13, not_search_mask13)
        loss42 = self.compute_loss(desp4, desp2, valid_mask42, not_search_mask42)

        loss = (loss12 + loss13 + loss42) / 3.
        return loss, loss12, loss13, loss42

    @staticmethod
    def compute_loss(desp_0, desp_1, valid_mask, not_search_mask):
        desp_1 = desp_1.transpose(1, 2)  # [bt,dim,n]

        cos_similarity = torch.matmul(desp_0, desp_1)  # [bt,n,n]
        dist = torch.sqrt(2. * (1. - cos_similarity) + 1e-4)

        positive_pair = torch.diagonal(dist, dim1=1, dim2=2)  # [bt,n]
        dist = dist + 10 * not_search_mask

        hardest_negative_pair, hardest_negative_idx = torch.min(dist, dim=2)  # [bt,n]

        loss_total = torch.relu(1. + positive_pair - hardest_negative_pair)
        loss_total *= valid_mask

        valid_num = torch.sum(valid_mask, dim=1)
        loss = torch.mean(torch.sum(loss_total, dim=1) / (valid_num + 1.))

        return loss


class DescriptorRankedListLoss(object):
    """
    """

    def __init__(self, margin, alpha, t, device):
        self.device = device
        self.positive_threshold = alpha - margin
        self.negative_threshold = alpha
        self.t = t

    def __call__(self, desp_0, desp_1, matched_valid, not_search_mask):
        desp_1 = torch.transpose(desp_1, 1, 2)

        cos_similarity = torch.matmul(desp_0, desp_1)
        dist = torch.sqrt(2. * (1. - cos_similarity) + 1e-4)

        positive_pair = torch.diagonal(dist, dim1=1, dim2=2)
        positive_mask = ((positive_pair >= self.positive_threshold).to(torch.float) * matched_valid).detach()
        positive_loss = self.compute_masked_positive_loss(positive_pair, positive_mask)

        # can't write as dist += 10*not_search_mask which will cost backprop error
        dist = dist + 10 * not_search_mask
        negative_pair = dist
        negative_mask = ((dist <= self.negative_threshold).to(torch.float)).detach()
        negative_loss = self.negative_threshold - negative_pair
        negative_loss = self.compute_masked_negative_loss(negative_loss, negative_mask)

        loss = positive_loss + negative_loss

        return loss, positive_loss, negative_loss

    @staticmethod
    def compute_masked_positive_loss(loss, mask):
        """

        """
        sum_loss = torch.sum(loss * mask)
        sum_mask = torch.sum(mask)
        loss = sum_loss / (sum_mask + 1e-5)
        return loss

    @staticmethod
    def compute_masked_negative_loss(loss, mask):
        sum_point_loss = torch.sum(loss * mask, dim=-1)
        sum_point_mask = torch.sum(mask, dim=-1)
        point_loss = sum_point_loss / (sum_point_mask + 1e-5)

        loss = torch.mean(point_loss)
        return loss


class DescriptorValidator(object):
    """

    """

    def __init__(self):
        pass

    def __call__(self, desp_0, desp_1, matched_valid, not_search_mask):
        desp_1 = torch.transpose(desp_1, 1, 2)

        cos_similarity = torch.matmul(desp_0, desp_1)
        dist = torch.sqrt(2. * (1. - cos_similarity) + 1e-4)

        positive_dist = torch.diagonal(dist, dim1=1, dim2=2)
        dist = dist + 10 * not_search_mask

        negative_dist, _ = torch.min(dist, dim=2)

        correct_match = (positive_dist < negative_dist).to(torch.float)
        correct_sum = torch.sum(correct_match * matched_valid).item()
        valid_sum = torch.sum(matched_valid).item()

        return correct_sum, valid_sum


class DescriptorPreciseTripletLoss(object):

    def __init__(self, device):
        self.device = device

    def __call__(self, desp_0, desp_1, matched_valid, not_search_mask):
        bt, dim, h, w = desp_0.shape
        desp_0 = torch.reshape(desp_0, (bt, dim, h * w)).transpose(1, 2)  # [bt,h*w,dim]
        desp_1 = torch.reshape(desp_1, (bt, dim, h * w))

        cos_similarity = torch.matmul(desp_0, desp_1)
        dist = torch.sqrt(2. * (1. - cos_similarity) + 1e-4)  # [bt,h*w,h*w]

        positive_pair = torch.diagonal(dist, dim1=1, dim2=2)  # [bt,h*w]
        dist = dist + 10 * not_search_mask

        hardest_negative_pair, hardest_negative_idx = torch.min(dist, dim=2)  # [bt,h*w]

        zeros = torch.zeros_like(positive_pair)
        loss_total, _ = torch.max(torch.stack((zeros, 1. + positive_pair - hardest_negative_pair), dim=2), dim=2)
        loss_total *= matched_valid

        valid_num = torch.sum(matched_valid, dim=1)
        loss = torch.mean(torch.sum(loss_total, dim=1) / (valid_num + 1.))

        return loss


class BinaryDescriptorPairwiseLoss(object):

    def __init__(self, device, lambda_d=250, m_p=1, m_n=0.2, ):
        self.device = device
        self.lambda_d = torch.tensor(lambda_d, device=self.device)
        self.m_p = torch.tensor(m_p, device=self.device)
        self.m_n = torch.tensor(m_n, device=self.device)
        self.one = torch.tensor(1, device=self.device)

    def __call__(self, desp_0, desp_1, feature_0, feature_1, desp_mask, valid_mask):
        batch_size, dim, h, w = desp_0.shape
        desp_0 = torch.reshape(desp_0, (batch_size, dim, -1))
        desp_1 = torch.reshape(desp_1, (batch_size, dim, -1))
        feature_0 = torch.reshape(feature_0, (batch_size, dim, -1))
        feature_1 = torch.reshape(feature_1, (batch_size, dim, -1))

        cos_similarity = compute_cos_similarity_binary(desp_0, desp_1)
        positive = -f.logsigmoid(cos_similarity) * self.lambda_d
        negative = -f.logsigmoid(1. - cos_similarity)

        positive_mask = desp_mask
        negative_mask = self.one - desp_mask

        pairwise_loss = positive_mask * positive + negative_mask * negative

        # 考虑warp带来的某些区域没有图像,则其对应的描述子应当无效
        total_num = torch.sum(valid_mask, dim=1) * h * w
        pairwise_loss = torch.sum(valid_mask.unsqueeze(dim=1) * pairwise_loss, dim=(1, 2)) / total_num
        pairwise_loss = torch.mean(pairwise_loss)

        sqrt_1_k = 1. / np.sqrt(dim)
        desp_1_valid_num = torch.sum(valid_mask, dim=1)

        ones_1_k = sqrt_1_k * torch.ones_like(feature_0)
        quantization_loss_0 = torch.norm((torch.abs(feature_0) - ones_1_k), p=1, dim=1)
        quantization_loss_1 = torch.norm((torch.abs(feature_1) - ones_1_k), p=1, dim=1)
        quantization_loss_0 = torch.mean(quantization_loss_0, dim=1)
        quantization_loss_1 = torch.sum(quantization_loss_1 * valid_mask, dim=1) / desp_1_valid_num
        quantization_loss = torch.mean(torch.cat((quantization_loss_0, quantization_loss_1)))

        return pairwise_loss, quantization_loss


class BinaryDescriptorTripletLoss(object):

    def __init__(self):
        self.gamma = 10
        self.threshold = 1.0
        self.quantization_weight = 1.0
        self.dim = 256.

    def __call__(self, desp_0, desp_1, matched_idx, matched_valid, not_search_mask, valid_mask):
        bt, dim, h, w = desp_0.shape
        desp_0 = torch.reshape(desp_0, (bt, dim, h * w))  # [bt,dim,h*w]
        desp_1 = torch.reshape(desp_1, (bt, dim, h * w))

        matched_idx = torch.unsqueeze(matched_idx, dim=1).repeat(1, dim, 1)  # [bt,dim,h*w]
        desp_1 = torch.gather(desp_1, dim=2, index=matched_idx)  # [bt,dim,h*w]

        cos_similarity = torch.matmul(desp_0.transpose(1, 2), desp_1)  # [bt,h*w,h*w]

        dist = torch.sqrt(2. * (1. - cos_similarity).clamp(1e-5))  # [bt,h*w,h*w]
        positive_pair = torch.diagonal(dist, dim1=1, dim2=2)  # [bt,h*w]
        dist = dist + 10 * not_search_mask
        hardest_negative_pair, hardest_negative_idx = torch.min(dist, dim=2)  # [bt,h*w]
        triplet_metric = f.relu(1. + positive_pair - hardest_negative_pair)

        triplet_loss = triplet_metric * matched_valid
        match_valid_num = torch.sum(matched_valid, dim=1)
        triplet_loss = torch.mean(torch.sum(triplet_loss, dim=1) / match_valid_num)

        sqrt_1_k = 1. / np.sqrt(dim)
        desp_1_valid_num = torch.sum(valid_mask, dim=1)

        ones_1_k = sqrt_1_k * torch.ones_like(desp_0)
        quantization_loss_0 = torch.norm((torch.abs(desp_0) - ones_1_k), p=1, dim=1)
        quantization_loss_1 = torch.norm((torch.abs(desp_1) - ones_1_k), p=1, dim=1)
        quantization_loss_0 = torch.mean(quantization_loss_0, dim=1)
        quantization_loss_1 = torch.sum(quantization_loss_1 * valid_mask, dim=1) / desp_1_valid_num
        quantization_loss = torch.mean(torch.cat((quantization_loss_0, quantization_loss_1)))

        return triplet_loss, quantization_loss


class BinaryDescriptorTripletDirectLoss(object):

    def __init__(self):
        self.gamma = 10
        self.threshold = 1.0
        self.quantization_weight = 1.0
        self.dim = 256.

    def __call__(self, desp_0, desp_1, feature_0, feature_1, matched_idx, matched_valid, not_search_mask, valid_mask):
        bt, dim, h, w = desp_0.shape
        norm_factor = np.sqrt(dim)
        desp_0 = torch.reshape(desp_0, (bt, dim, h * w))  # [bt,dim,h*w]
        desp_1 = torch.reshape(desp_1, (bt, dim, h * w))
        feature_0 = torch.reshape(feature_0, (bt, dim, h * w))
        feature_1 = torch.reshape(feature_1, (bt, dim, h * w))

        matched_idx = torch.unsqueeze(matched_idx, dim=1).repeat(1, dim, 1)  # [bt,dim,h*w]
        desp_1 = torch.gather(desp_1, dim=2, index=matched_idx)  # [bt,dim,h*w]
        feature_1 = torch.gather(feature_1, dim=2, index=matched_idx)

        desp_0 = desp_0 / norm_factor
        desp_1 = desp_1 / norm_factor
        cos_similarity = torch.matmul(desp_0.transpose(1, 2), desp_1)

        positive_pair = torch.diagonal(cos_similarity, dim1=1, dim2=2)  # [bt,h*w]
        minus_cos_sim = 1.0 - cos_similarity + not_search_mask * 10
        hardest_negative_pair, hardest_negative_idx = torch.min(minus_cos_sim, dim=2)  # [bt,h*w]

        triplet_metric = -f.logsigmoid(positive_pair) - f.logsigmoid(hardest_negative_pair)

        triplet_loss = triplet_metric * matched_valid
        match_valid_num = torch.sum(matched_valid, dim=1)
        triplet_loss = torch.mean(torch.sum(triplet_loss, dim=1) / match_valid_num)

        sqrt_1_k = 1. / np.sqrt(dim)
        desp_1_valid_num = torch.sum(valid_mask, dim=1)

        ones_1_k = sqrt_1_k * torch.ones_like(feature_0)
        quantization_loss_0 = torch.norm((torch.abs(feature_0) - ones_1_k), p=1, dim=1)
        quantization_loss_1 = torch.norm((torch.abs(feature_1) - ones_1_k), p=1, dim=1)
        quantization_loss_0 = torch.mean(quantization_loss_0, dim=1)
        quantization_loss_1 = torch.sum(quantization_loss_1 * valid_mask, dim=1) / desp_1_valid_num
        quantization_loss = torch.mean(torch.cat((quantization_loss_0, quantization_loss_1)))

        return triplet_loss, quantization_loss


class BinaryDescriptorTripletTanhLoss(object):

    def __init__(self):
        pass

    def __call__(self, desp_0, desp_1, matched_idx, matched_valid, not_search_mask, valid_mask):
        bt, dim, h, w = desp_0.shape
        desp_0 = torch.reshape(desp_0, (bt, dim, h * w))  # [bt,dim,h*w]
        desp_1 = torch.reshape(desp_1, (bt, dim, h * w))

        matched_idx = torch.unsqueeze(matched_idx, dim=1).repeat(1, dim, 1)  # [bt,dim,h*w]
        desp_1 = torch.gather(desp_1, dim=2, index=matched_idx)  # [bt,dim,h*w]

        cos_similarity = compute_cos_similarity_general(desp_0, desp_1)

        positive_pair = torch.diagonal(cos_similarity, dim1=1, dim2=2)  # [bt,h*w]
        minus_cos_sim = 1.0 - cos_similarity + not_search_mask * 10
        hardest_negative_pair, hardest_negative_idx = torch.min(minus_cos_sim, dim=2)  # [bt,h*w]

        triplet_metric = -f.logsigmoid(positive_pair) - f.logsigmoid(hardest_negative_pair)

        triplet_loss = triplet_metric * matched_valid
        match_valid_num = torch.sum(matched_valid, dim=1)
        triplet_loss = torch.mean(torch.sum(triplet_loss, dim=1) / match_valid_num)

        return triplet_loss


class BinaryDescriptorTripletTanhSigmoidLoss(object):

    def __init__(self, logger):
        self.logger = logger
        logger.info("Initialize the Alpha Sigmoid Tanh Triplet loss.")

    def __call__(self, desp_0, desp_1, matched_idx, matched_valid, not_search_mask, valid_mask):
        bt, dim, h, w = desp_0.shape
        desp_0 = torch.reshape(desp_0, (bt, dim, h * w))  # [bt,dim,h*w]
        desp_1 = torch.reshape(desp_1, (bt, dim, h * w))
        sigmoid_params = 10. / dim

        matched_idx = torch.unsqueeze(matched_idx, dim=1).repeat(1, dim, 1)  # [bt,dim,h*w]
        desp_1 = torch.gather(desp_1, dim=2, index=matched_idx)  # [bt,dim,h*w]

        inner_product = torch.matmul(desp_0.transpose(1, 2), desp_1) / dim

        positive_pair = torch.diagonal(inner_product, dim1=1, dim2=2)  # [bt,h*w]

        minus_cos_sim = 1. - inner_product + not_search_mask * 10.
        hardest_negative_pair, _ = torch.min(minus_cos_sim, dim=2)  # [bt,h*w]
        triplet_metric = -f.logsigmoid(positive_pair) - f.logsigmoid(hardest_negative_pair)

        # masked_cos_sim = inner_product - not_search_mask*10.
        # hardest_negative_pair, _ = torch.max(masked_cos_sim, dim=2)
        # triplet_metric = -f.logsigmoid(positive_pair)-f.logsigmoid(-hardest_negative_pair)

        triplet_loss = triplet_metric * matched_valid
        match_valid_num = torch.sum(matched_valid, dim=1)
        triplet_loss = torch.mean(torch.sum(triplet_loss, dim=1) / match_valid_num)

        return triplet_loss

class SegmentationDistillationBCELoss(torch.nn.Module):
    def __init__(self, temperature=1.0):
        super(SegmentationDistillationBCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, student_soft_prediction, teacher_soft_target):
        # Apply temperature scaling
        student_soft_prediction /= self.temperature
        teacher_soft_target /= self.temperature

        # Apply sigmoid to convert logits into probabilities
        student_soft_prediction = torch.sigmoid(student_soft_prediction)
        teacher_soft_target = torch.sigmoid(teacher_soft_target)

        # Calculate BCE loss
        distillation_loss = f.binary_cross_entropy(student_soft_prediction, teacher_soft_target, reduction='mean')
        return distillation_loss

class SegmentationDistillationLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(SegmentationDistillationLoss, self).__init__()
        self.temperature = temperature

    def forward(self, student_logits, teacher_logits):
        B,C,H,W = student_logits.shape
        teacher_logits = torch.nn.functional.interpolate(teacher_logits, size=(H, W), mode='bilinear',
                                                                   align_corners=False)
        student_logits = student_logits.view(-1, student_logits.shape[-1])
        teacher_logits = teacher_logits.view(-1, teacher_logits.shape[-1])
        soft_teacher = f.softmax(teacher_logits / self.temperature, dim=1)
        soft_student = f.log_softmax(student_logits / self.temperature, dim=1)
        distillation_loss = nn.KLDivLoss(reduction='batchmean')(soft_student, soft_teacher)
        return distillation_loss

class AffinityAttentionLoss(nn.Module):
    def __init__(self):
        super(AffinityAttentionLoss, self).__init__()

    def spatial_similarity(self, fm):
        fm = fm.view(fm.size(0), fm.size(1), -1)
        norm_fm = fm / (torch.sqrt(torch.sum(torch.pow(fm, 2), 1)).unsqueeze(1).expand(fm.shape) + 0.0000001)
        s = norm_fm.transpose(1, 2).bmm(norm_fm)
        s = s.unsqueeze(1)
        return s

    def select_flagged_tensors(self, input_tensor, flags_tensor):
        indices = torch.nonzero(flags_tensor.squeeze(), as_tuple=False).squeeze()

        # Select the input tensor using the flagged indices
        selected_tensor = input_tensor[indices]

        return selected_tensor

    def forward(self, student_features, teacher_features, flags, attmap):
        # Compute student and teacher affinities
        student_features = self.select_flagged_tensors(student_features, flags)
        teacher_features = self.select_flagged_tensors(teacher_features, flags)
        attmap = self.select_flagged_tensors(attmap, flags)
        B, C, H, W = student_features.shape
        #student_features = student_features[:B//2,:,:,:]
        #teacher_features = teacher_features[:B//2,:,:,:]

        teacher_features = torch.nn.functional.interpolate(teacher_features, size=(H, W), mode='bilinear',
                                                         align_corners=False)
        attmap = torch.nn.functional.interpolate(attmap, size=(H, W), mode='bilinear',
                                                           align_corners=False)
        #student_features = torch.nn.functional.interpolate(student_features, size=(H // 2, W // 2), mode='bilinear',
        #                                                   align_corners=False)
        student_features = student_features * attmap
        teacher_features = teacher_features * attmap
        student_sim = self.spatial_similarity(student_features)
        teacher_sim = self.spatial_similarity(teacher_features)

        loss = nn.L1Loss()(student_sim, teacher_sim)

        return loss

class SegmentationDistillation_L1Loss(nn.Module):
    def __init__(self):
        super(SegmentationDistillation_L1Loss, self).__init__()
        self.criterion = nn.L1Loss()
    def forward(self, student_logits, teacher_logits):
        B, C, H, W = student_logits.shape
        teacher_logits = torch.nn.functional.interpolate(teacher_logits, size=(H, W), mode='bilinear',
                                                         align_corners=False)
        student_logits = student_logits[:B // 2, :, :, :]
        teacher_logits = teacher_logits[:B // 2, :, :, :]
        distillation_loss = self.criterion(student_logits, teacher_logits)
        return distillation_loss

class EdgeDistillation_Loss(nn.Module):
    def __init__(self):
        super(EdgeDistillation_Loss, self).__init__()
        self.criterion = nn.L1Loss()

    def select_flagged_tensors(self, input_tensor, flags_tensor):
        indices = torch.nonzero(flags_tensor.squeeze(), as_tuple=False).squeeze()

        # Select the input tensor using the flagged indices
        selected_tensor = input_tensor[indices]

        return selected_tensor

    def forward(self, student_features, teacher_features, flags):
        student_features = self.select_flagged_tensors(student_features, flags)
        teacher_features = self.select_flagged_tensors(teacher_features.unsqueeze(dim=1), flags)
        teacher_features = teacher_features / 255
        B, C, H, W = student_features.shape
        teacher_features = torch.nn.functional.interpolate(teacher_features, size=(H, W), mode='bilinear',align_corners=False)
        #distillation_loss = self.criterion(student_features, teacher_features)
        distillation_loss = torch.nn.functional.mse_loss(student_features, teacher_features)
        #distillation_loss = self.dice_loss(student_features, teacher_features)
        return distillation_loss


class AffinityDistillationLoss(nn.Module):
    def __init__(self):
        super(AffinityDistillationLoss, self).__init__()

    def spatial_similarity(self, fm):
        fm = fm.view(fm.size(0), fm.size(1), -1)
        norm_fm = fm / (torch.sqrt(torch.sum(torch.pow(fm, 2), 1)).unsqueeze(1).expand(fm.shape) + 0.0000001)
        s = norm_fm.transpose(1, 2).bmm(norm_fm)
        s = s.unsqueeze(1)
        return s

    def select_flagged_tensors(self, input_tensor, flags_tensor):
        indices = torch.nonzero(flags_tensor.squeeze(), as_tuple=False).squeeze()

        # Select the input tensor using the flagged indices
        selected_tensor = input_tensor[indices]

        return selected_tensor

    def forward(self, student_features, teacher_features, flags):
        # Compute student and teacher affinities
        student_features = self.select_flagged_tensors(student_features, flags)
        teacher_features = self.select_flagged_tensors(teacher_features, flags)
        B, C, H, W = student_features.shape
        #student_features = student_features[:B//2,:,:,:]
        #teacher_features = teacher_features[:B//2,:,:,:]

        teacher_features = torch.nn.functional.interpolate(teacher_features, size=(H, W), mode='bilinear',
                                                         align_corners=False)
        #student_features = torch.nn.functional.interpolate(student_features, size=(H // 2, W // 2), mode='bilinear',
        #                                                   align_corners=False)
        student_sim = self.spatial_similarity(student_features)
        teacher_sim = self.spatial_similarity(teacher_features)

        loss = nn.L1Loss()(student_sim, teacher_sim)

        return loss

class PointHeatmapWeightedBCELoss(object):

    def __init__(self, weight=200):
        self.weight = weight

    def __call__(self, heatmap_pred, heatmap_gt, heatmap_valid):
        """只适用one-hot的情况
        """
        sigmoid_0 = torch.sigmoid(heatmap_pred)  # 预测为关键点的概率
        sigmoid_1 = 1. - sigmoid_0  # 预测为非关键点的概率

        module_factor_0 = self.weight * heatmap_gt
        module_factor_1 = 1. * (1 - heatmap_gt)

        unmasked_heatmap_loss = -(
                    module_factor_0 * torch.log(sigmoid_0 + 1e-4) + module_factor_1 * torch.log(sigmoid_1 + 1e-4))
        loss = self._compute_masked_loss(unmasked_heatmap_loss, heatmap_valid)

        return loss

    @staticmethod
    def _compute_masked_loss(unmasked_loss, mask):
        assert len(unmasked_loss.shape) == len(mask.shape)
        if len(unmasked_loss.shape) == 3:
            dim = (1, 2)
        else:
            dim = (2, 3)
        valid_num = torch.sum(mask, dim=dim)
        masked_loss = torch.sum(unmasked_loss * mask, dim=dim)
        masked_loss = masked_loss / (valid_num + 1)
        loss = torch.mean(masked_loss)
        return loss


class PointHeatmapSpatialFocalWeightedBCELoss(object):
    """
    引入真值点与实际预测点之间的距离作为对应真值点的加权项，其主要思想是距离真值点越近的预测点（只是接近而非重合），这些点属于错分且程度激烈，
    应当增大权重
    """

    def __init__(self, device, positive_weight=200, fn_scale=1.0, gamma=2, kernel_size=5, sigma=1.5):
        """
        初始化
        Args:
            device: 指定loss在cpu/gpu上计算
            positive_weight: positive_weight控制正样本的权重
            fn_scale: 用于负样本假错权重的放大系数
            gamma: gamma为focal的指数
            kernel_size: kernel_size为高斯核的大小
            sigma: sigma为高斯分布权重的标准差
        """
        self.device = device
        self.gamma = gamma
        self.positive_weight = positive_weight
        self.fn_scale = fn_scale

        # 生成固定权重的二维高斯卷积核
        center = np.array((kernel_size // 2, kernel_size // 2), dtype=np.float32)
        grid = np.zeros((kernel_size, kernel_size, 2), dtype=np.float32)
        for i in range(kernel_size):
            for j in range(kernel_size):
                grid[i, j] = (i, j)  # y,x的顺序

        kernel = np.exp(-np.linalg.norm(grid - center, axis=2, keepdims=False) / (2 * sigma ** 2))
        kernel[int(kernel_size // 2), int(kernel_size // 2)] = 0  # 中心点的权重置为0

        # 进行归一化
        kernel = kernel / np.sum(kernel)
        self.kernel = torch.from_numpy(kernel).unsqueeze(dim=0).unsqueeze(dim=0).to(
            self.device)  # [1,1,ksize,ksize]的高斯卷积核
        self.padding = int(kernel_size // 2)

    def __call__(self, heatmap_pred, heatmap_gt, heatmap_valid, **kwargs):
        heatmap_pred = torch.sigmoid(heatmap_pred)  # 将logit转换为概率

        fp_heatmap = (heatmap_pred * (1. - heatmap_gt)).unsqueeze(dim=1)  # 所有非关键点被判断为关键点的概率->假正率
        fn_heatmap = ((1. - heatmap_pred) * heatmap_gt).unsqueeze(dim=1)  # 所有关键点被判断为非关键点的概率->假错率

        fp_weight = f.conv2d(fp_heatmap, self.kernel, padding=self.padding).squeeze()
        fn_weight = f.conv2d(fn_heatmap, self.kernel, padding=self.padding).squeeze()

        loss_positive = -self.positive_weight * (1. + fp_weight) ** self.gamma * torch.log(
            heatmap_pred + 1e-5) * heatmap_gt
        loss_negative = -(1. + self.fn_scale * fn_weight) ** self.gamma * torch.log(1 - heatmap_pred + 1e-5) * (
                    1 - heatmap_gt)

        loss_unmasked = loss_positive + loss_negative
        loss = self._compute_masked_loss(loss_unmasked, heatmap_valid)

        return loss

    @staticmethod
    def _compute_masked_loss(unmasked_loss, mask):
        valid_num = torch.sum(mask, dim=(1, 2))
        masked_loss = torch.sum(unmasked_loss * mask, dim=(1, 2))
        masked_loss = masked_loss / (valid_num + 1)
        loss = torch.mean(masked_loss)
        return loss


class PointHeatmapFocalLoss(object):

    def __init__(self, gamma=2, p_ratio=0.75):
        self.gamma = gamma
        self.p_ratio = p_ratio
        self.unmasked_loss = torch.nn.BCEWithLogitsLoss(reduction='none')

    def __call__(self, heatmap_pred, heatmap_gt, mask):
        """只适用one-hot的情况"""
        sigmoid_0 = torch.sigmoid(heatmap_pred)  # 预测为关键点的概率
        sigmoid_1 = 1. - sigmoid_0  # 预测为非关键点的概率

        module_factor_0 = torch.pow(1. - sigmoid_0, self.gamma) * self.p_ratio * heatmap_gt
        module_factor_1 = torch.pow(1. - sigmoid_1, self.gamma) * (1. - self.p_ratio) * (1 - heatmap_gt)
        # module_factor = module_factor_0 + module_factor_1

        unmasked_loss = -(module_factor_0 * torch.log(sigmoid_0 + 1e-4) + module_factor_1 * torch.log(sigmoid_1 + 1e-4))
        # unmasked_loss = module_factor * self.unmasked_loss(heatmap_pred, heatmap_gt)

        valid_num = torch.sum(mask, dim=(1, 2))
        masked_loss = torch.sum(unmasked_loss * mask, dim=(1, 2))
        # masked_loss = masked_loss / (valid_num + 1)
        loss = torch.mean(masked_loss)
        return loss


class HeatmapAlignLoss(object):
    """
    用于计算两幅heatmap在其相关单应变换下的对齐loss
    """

    def __init__(self):
        pass

    def __call__(self, heatmap_s, heatmap_t, homography, mask, **kwargs):
        """
        将heatmap_s经单应变换插值到heatmap_t',计算heatmap_t'与heatmap_t在预先计算的mask中的有效区域的差异作为对齐loss
        Args:
            heatmap_s: [bt,1,h,w],s视角下预测的热力图
            heatmap_t: [bt,1,h,w],t视角下预测的热力图
            homography: s->t的单应变换
            mask: 经上述变换后的有效区域的mask
        Returns:
            loss: 对齐差异作为loss返回
        """
        heatmap_s = torch.sigmoid(heatmap_s)
        heatmap_t = torch.sigmoid(heatmap_t)

        project_heatmap_t = interpolation(heatmap_s, homography)
        unmasked_loss = torch.abs(project_heatmap_t - heatmap_t).squeeze()
        loss = self._compute_masked_loss(unmasked_loss, mask)
        return loss

    @staticmethod
    def _compute_masked_loss(unmasked_loss, mask):
        valid_num = torch.sum(mask, dim=(1, 2))
        masked_loss = torch.sum(unmasked_loss * mask, dim=(1, 2))
        masked_loss = masked_loss / (valid_num + 1)
        loss = torch.mean(masked_loss)
        return loss


class HeatmapWeightedAlignLoss(HeatmapAlignLoss):
    """
    带权重的对齐loss计算子，对于关键点位置处一个自定义区域内的差异值会被权重放大，而非关键点处的权重则都为1
    """

    def __init__(self, weight=200):
        super(HeatmapWeightedAlignLoss, self).__init__()
        self.weight = weight

    def __call__(self, heatmap_s, heatmap_t, homography, mask, **kwargs):
        heatmap_gt_t = kwargs["heatmap_gt_t"]

        heatmap_s = torch.sigmoid(heatmap_s)
        heatmap_t = torch.sigmoid(heatmap_t)

        module_factor_keypoint = self.weight * heatmap_gt_t
        module_factor_others = 1. * (1 - heatmap_gt_t)
        module_factor = module_factor_keypoint + module_factor_others

        project_heatmap_t = interpolation(heatmap_s, homography)
        unmasked_loss = module_factor * torch.abs(project_heatmap_t - heatmap_t).squeeze()
        loss = self._compute_masked_loss(unmasked_loss, mask)
        return loss


class Matcher(object):

    def __init__(self, dtype='float'):
        if dtype == 'float':
            self.compute_desp_dist = self._compute_desp_dist
        elif dtype == 'binary':
            self.compute_desp_dist = self._compute_desp_dist_binary
        else:
            assert False

    def __call__(self, point_0, desp_0, point_1, desp_1):
        dist_0_1 = self.compute_desp_dist(desp_0, desp_1)  # [n,m]
        dist_1_0 = dist_0_1.transpose((1, 0))  # [m,n]
        nearest_idx_0_1 = np.argmin(dist_0_1, axis=1)  # [n]
        nearest_idx_1_0 = np.argmin(dist_1_0, axis=1)  # [m]
        matched_src = []
        matched_tgt = []
        for i, idx_0_1 in enumerate(nearest_idx_0_1):
            if i == nearest_idx_1_0[idx_0_1]:
                matched_src.append(point_0[i])
                matched_tgt.append(point_1[idx_0_1])
        if len(matched_src) <= 4:
            print("There exist too little matches")
            # assert False
            return None
        if len(matched_src) != 0:
            matched_src = np.stack(matched_src, axis=0)
            matched_tgt = np.stack(matched_tgt, axis=0)
        return matched_src, matched_tgt