#
# Created  on 2020/8/28
#
import time

import torch
import cv2 as cv
from tensorboardX import SummaryWriter

from evaluation_hpatch.utils.utils import Matcher
from utils.evaluation_tools import *
from data_utils import get_dataset


class BaseTrainer(object):

    def __init__(self, **config):
        self.config = config
        self.logger = config['logger']

        if torch.cuda.is_available():
            self.logger.info('gpu is available, set device to cuda !')
            self.device = torch.device('cuda:0')
            self.gpu_count = 1
        else:
            self.logger.info('gpu is not available, set device to cpu !')
            self.device = torch.device('cpu')
        self.multi_gpus = False
        self.drop_last = False
        if torch.cuda.device_count() > 1:
            self.gpu_count = torch.cuda.device_count()
            self.config['train']['batch_size'] *= self.gpu_count
            self.multi_gpus = True
            self.drop_last = True
            self.logger.info("Multi gpus is available, let's use %d GPUS" % torch.cuda.device_count())

        # 初始化summary writer
        self.summary_writer = SummaryWriter(self.config['ckpt_path'])
        self._initialize_dataset()
        self._initialize_model()
        self._initialize_optimizer()
        self._initialize_scheduler()
        self._initialize_loss()
        self._initialize_matcher()

        if self.config['train']['Distillation_loss'] == 'KL':
            self._initialize_sam_feature_KL_loss()
        elif self.config['train']['Distillation_loss'] == 'L1':
            self._initialize_sam_feature_L1_loss()
        elif self.config['train']['Distillation_loss'] == 'AF':
            self._initialize_sam_feature_AF_loss()

        if self.config['train']['NCE_cat_loss']:
            self._initialize_sam_category_loss()

        if self.config['train']['Edge_loss'] == 'MSE':
            self._initialize_sam_edge_loss()

        self.logger.info("Initialize cat func: _cat_c1c2c3c4")
        self.cat = self._cat_c1c2c3c4

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def _inference_func(self, *args, **kwargs):
        raise NotImplementedError

    def _initialize_dataset(self, *args, **kwargs):
        raise NotImplementedError

    def _initialize_model(self, *args, **kwargs):
        self.model = None
        raise NotImplementedError

    def _initialize_sam_feature_KL_loss(self, *args, **kwargs):
        raise NotImplementedError

    def _initialize_sam_feature_L1_loss(self, *args, **kwargs):
        raise NotImplementedError

    def _initialize_sam_edge_loss(self, *args, **kwargs):
        raise NotImplementedError

    def _initialize_sam_feature_AF_loss(self, *args, **kwargs):
        raise NotImplementedError

    def _initialize_sam_category_loss(self, *args, **kwargs):
        raise NotImplementedError

    def _initialize_optimizer(self, *args, **kwargs):
        self.optimizer = None
        raise NotImplementedError

    def _initialize_scheduler(self, *args, **kwargs):
        self.scheduler = None
        raise NotImplementedError

    def _train_func(self, *args, **kwargs):
        raise NotImplementedError

    def _initialize_loss(self, *args, **kwargs):
        raise NotImplementedError

    def _train_one_epoch(self, *args, **kwargs):
        raise NotImplementedError

    def train(self):
        start_time = time.time()

        # start training
        for i in range(self.config['train']['epoch_num']):

            # train
            self._train_one_epoch(i)

            # validation
            if i >= int(self.config['train']['epoch_num'] * self.config['train']['validate_after']):
                self._validate_one_epoch(i)

            if self.config['train']['adjust_lr']:
                # adjust learning rate
                self.scheduler.step(i)

        end_time = time.time()
        self.logger.info("The whole training process takes %.3f h" % ((end_time - start_time)/3600))

    def _initialize_matcher(self):
        # 初始化匹配算子
        self.logger.info("Initialize matcher of Nearest Neighbor.")
        self.general_matcher = Matcher('float')

    def _load_model_params(self, ckpt_file, previous_model):
        if ckpt_file is None:
            print("Please input correct checkpoint file dir!")
            return False

        self.logger.info("Load pretrained model %s " % ckpt_file)
        if not self.multi_gpus:
            model_dict = previous_model.state_dict()
            pretrain_dict = torch.load(ckpt_file, map_location=self.device)
            pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
            model_dict.update(pretrain_dict)
            previous_model.load_state_dict(model_dict)
        else:
            model_dict = previous_model.module.state_dict()
            pretrain_dict = torch.load(ckpt_file, map_location=self.device)
            pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
            model_dict.update(pretrain_dict)
            previous_model.module.load_state_dict(model_dict)
        return previous_model



    @staticmethod
    def _compute_total_metric(illum_metric, view_metric):
        illum_acc, illum_sum, illum_num = illum_metric.average()
        view_acc, view_sum, view_num = view_metric.average()
        return illum_acc, view_acc, (illum_sum+view_sum)/(illum_num+view_num+1e-4)

    @staticmethod
    def _compute_match_outlier_distribution(illum_metric, view_metric):
        illum_distribution = illum_metric.average_outlier()
        view_distribution = view_metric.average_outlier()
        return illum_distribution, view_distribution

    @staticmethod
    def _cat_c1c2c3c4(c1, c2, c3, c4, dim):
        return torch.cat((c1, c2, c3, c4), dim=dim)

    @staticmethod
    def _cat_c2c3c4(c1, c2, c3, c4, dim):
        return torch.cat((c2, c3, c4), dim=dim)

    @staticmethod
    def _cat_c1c2c4(c1, c2, c3, c4, dim):
        return torch.cat((c1, c2, c4), dim=dim)

    @staticmethod
    def _cat_c1c3c4(c1, c2, c3, c4, dim):
        return torch.cat((c1, c3, c4), dim=dim)

    @staticmethod
    def _cat_c1c4(c1, c2, c3, c4, dim):
        return torch.cat((c1, c4), dim=dim)

    @staticmethod
    def _cat_c2c4(c1, c2, c3, c4, dim):
        return torch.cat((c2, c4), dim=dim)

    @staticmethod
    def _cat_c3c4(c1, c2, c3, c4, dim):
        return torch.cat((c3, c4), dim=dim)

    @staticmethod
    def _cat_c4(c1, c2, c3, c4, dim):
        return c4

    @staticmethod
    def _convert_pt2cv(point_list):
        cv_point_list = []

        for i in range(len(point_list)):
            cv_point = cv.KeyPoint()
            cv_point.pt = tuple(point_list[i][::-1])
            cv_point_list.append(cv_point)

        return cv_point_list

    @staticmethod
    def _convert_pt2cv_np(point):
        cv_point_list = []
        for i in range(point.shape[0]):
            cv_point = cv.KeyPoint()
            cv_point.pt = tuple(point[i, ::-1])
            cv_point_list.append(cv_point)

        return cv_point_list

    @staticmethod
    def _convert_cv2pt(cv_point):
        point_list = []
        for i, cv_pt in enumerate(cv_point):
            pt = np.array((cv_pt.pt[1], cv_pt.pt[0]))  # y,x的顺序
            point_list.append(pt)
        point = np.stack(point_list, axis=0)
        return point

    @staticmethod
    def _compute_masked_loss(unmasked_loss, mask):
        total_num = torch.sum(mask, dim=(1, 2))
        loss = torch.sum(mask*unmasked_loss, dim=(1, 2)) / total_num
        loss = torch.mean(loss)
        return loss

    @staticmethod
    def _convert_match2cv(first_point_list, second_point_list, sample_ratio=1.0):
        cv_first_point = []
        cv_second_point = []
        cv_matched_list = []

        assert len(first_point_list) == len(second_point_list)

        inc = 1
        if sample_ratio < 1:
            inc = int(1.0 / sample_ratio)

        count = 0
        if len(first_point_list) > 0:
            for j in range(0, len(first_point_list), inc):
                cv_point = cv.KeyPoint()
                cv_point.pt = tuple(first_point_list[j][::-1])
                cv_first_point.append(cv_point)

                cv_point = cv.KeyPoint()
                cv_point.pt = tuple(second_point_list[j][::-1])
                cv_second_point.append(cv_point)

                cv_match = cv.DMatch()
                cv_match.queryIdx = count
                cv_match.trainIdx = count
                cv_matched_list.append(cv_match)

                count += 1

        return cv_first_point, cv_second_point, cv_matched_list

    @staticmethod
    def _generate_predict_point(prob, detection_threshold, scale=None, top_k=0):
        point_idx = np.where(prob > detection_threshold)

        if len(point_idx[0]) == 0 or len(point_idx[1]) == 0:
            point = np.empty((0, 2))
            return point, 0

        prob = prob[point_idx]
        sorted_idx = np.argsort(prob)[::-1]
        if sorted_idx.shape[0] >= top_k:
            sorted_idx = sorted_idx[:top_k]

        point = np.stack(point_idx, axis=1)  # [n,2]
        top_k_point = []
        for idx in sorted_idx:
            top_k_point.append(point[idx])

        point = np.stack(top_k_point, axis=0)
        point_num = point.shape[0]

        if scale is not None:
            point = point*scale
        return point, point_num

    @staticmethod
    def _cvpoint2numpy(point_cv):
        """将opencv格式的特征点转换成numpy数组"""
        point_list = []
        for pt_cv in point_cv:
            point = np.array((pt_cv.pt[1], pt_cv.pt[0]))
            point_list.append(point)
        point_np = np.stack(point_list, axis=0)
        return point_np










