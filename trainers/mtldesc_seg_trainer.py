# 
# Created  on 2019/9/18
#
import os
import time

import torch
import torch.nn.functional as f
from torch.utils.data import DataLoader

from nets import get_model
from data_utils import get_dataset
from trainers.base_trainer import BaseTrainer
from utils.utils import spatial_nms
from utils.utils import AttentionWeightedTripletLoss, AttentionWeightedTripletLossWithSam, AttentionMapWithSamCatLoss
from utils.utils import PointHeatmapWeightedBCELoss
from utils.utils import SegmentationDistillationLoss#, SegmentationDistillationBCELoss
import sys


class MTLDesc_seg_Trainer(BaseTrainer):

    def __init__(self, **config):
        super(MTLDesc_seg_Trainer, self).__init__(**config)

    def _initialize_dataset(self):
        self.logger.info('Initialize {}'.format(self.config['train']['dataset']))
        self.train_dataset = get_dataset(self.config['train']['dataset'])(**self.config['train'])

        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config['train']['batch_size'],
            shuffle=True,
            num_workers=self.config['train']['num_workers'],
            drop_last=True,
        )
        self.epoch_length = len(self.train_dataset) // self.config['train']['batch_size']

    def _initialize_model(self):
        self.logger.info("Initialize network arch {}".format(self.config['model']['backbone']))
        model = get_model(self.config['model']['backbone'])(self.config['train']['desc_dim'])

        if self.multi_gpus:
            model = torch.nn.DataParallel(model)
        self.model = model.to(self.device)

    def _initialize_sam(self):
        self.logger.info("Initialize SAM network arch {}".format(self.config['sam']['model_name']))
        model = sam_model_registry[self.config['sam']['sam_build']](checkpoint=self.config['sam']['cpt_pth'])
        if self.multi_gpus:
            model = torch.nn.DataParallel(model)
        self.sam = model.to(self.device)

    def _initialize_loss(self):
        self.logger.info("Initialize the PointHeatmapWeightedBCELoss.")
        self.point_loss = PointHeatmapWeightedBCELoss(weight=self.config['train']['point_loss_weight'])
        self.logger.info("Initialize the DescriptorGeneralTripletLoss.")
        self.descriptor_loss=AttentionWeightedTripletLoss(self.device,T=self.config['train']['T'])
        self.logger.info("Initialize the DescriptorInfoNCELoss.")
        self.descriptor_loss_with_SamInfo = AttentionWeightedTripletLossWithSam(self.device, T=self.config['train']['T2'], margin=self.config['train']['margin'])
        self.Attention_Cat_loss = AttentionMapWithSamCatLoss(self.device)

    def _initialize_sam_loss(self):
        self.logger.info("Initialize the CE Loss for sam distillation.")
        self.sam_loss = SegmentationDistillationLoss()

    def _initialize_optimizer(self):
        self.logger.info("Initialize Adam optimizer with weight_decay: {:.5f}.".format(self.config['train']['weight_decay']))
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=self.config['train']['lr'],
            weight_decay=self.config['train']['weight_decay'])

    def _initialize_scheduler(self):
        if self.config['train']['lr_mod']=='LambdaLR':
            self.logger.info("Initialize lr_scheduler of LambdaLR: (%d, %d)" % (self.config['train']['maintain_epoch'], self.config['train']['decay_epoch']))
            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch  - self.config['train']['maintain_epoch']) / float(self.config['train']['decay_epoch'] + 1)
                return lr_l
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_rule)
        else:
            milestones = [20, 30]
            self.logger.info("Initialize lr_scheduler of MultiStepLR: (%d, %d)" % (milestones[0], milestones[1]))
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.1)

    def _train_one_epoch(self, epoch_idx):
        self.model.train()

        self.logger.info("-----------------------------------------------------")
        self.logger.info("Training epoch %2d begin:" % epoch_idx)

        self._train_func(epoch_idx)

        self.logger.info("Training epoch %2d done." % epoch_idx)
        self.logger.info("-----------------------------------------------------")

    def _train_func(self, epoch_idx):
        self.model.train()
        stime = time.time()
        total_loss = 0
        for i, data in enumerate(self.train_dataloader):

            image = data["image"].to(self.device)
            heatmap_gt = data['heatmap'].to(self.device)
            point_mask = data['point_mask'].to(self.device)
            desp_point = data["desp_point"].to(self.device)

            warped_image = data["warped_image"].to(self.device)
            warped_heatmap_gt = data['warped_heatmap'].to(self.device)
            warped_point_mask = data['warped_point_mask'].to(self.device)
            warped_desp_point = data["warped_desp_point"].to(self.device)

            # SAM Here
            image_segmentation_mask = data['single_seg'] #self.sam(image, False)
            warped_image_segmentation_mask = data['single_seg_warped'] #self.sam(warped_image, False)
            seg_pair = torch.cat((image_segmentation_mask, warped_image_segmentation_mask), dim=0)

            # remove nan
            mask = torch.isnan(seg_pair)
            mean_value = 0#torch.nanmean(seg_pair)
            seg_pair[mask] = mean_value

            valid_mask = data["valid_mask"].to(self.device)
            not_search_mask = data["not_search_mask"].to(self.device)
            image_pair = torch.cat((image, warped_image), dim=0)
            heatmap_pred_pair, feature, weight_map, seg_map = self.model(image_pair)
            desp_point_pair = torch.cat((desp_point, warped_desp_point), dim=0)
            feature_pair = f.grid_sample(feature, desp_point_pair, mode="bilinear", padding_mode="border")
            weight_pair = f.grid_sample(weight_map, desp_point_pair, mode="bilinear", padding_mode="border").squeeze(
                dim=1)
            feature_pair = feature_pair[:, :, :, 0].transpose(1, 2)
            desp_pair = feature_pair / torch.norm(feature_pair, p=2, dim=2, keepdim=True)  # L2 Normalization
            weight_0,weight_1=torch.chunk(weight_pair, 2, dim=0)
            desp_0, desp_1 = torch.chunk(desp_pair, 2, dim=0)
            desp_loss = self.descriptor_loss(desp_0, desp_1,weight_0,weight_1,valid_mask, not_search_mask)

            cat_0, cat_1 = data['desp_cat'], data['desp_cat_warped']
            desp_sam_loss = self.descriptor_loss_with_SamInfo(desp_0, desp_1,weight_0,weight_1,valid_mask, not_search_mask, cat_0, cat_1)

            heatmap_gt_pair = torch.cat((heatmap_gt, warped_heatmap_gt), dim=0)
            point_mask_pair = torch.cat((point_mask, warped_point_mask), dim=0)
            point_loss = self.point_loss(heatmap_pred_pair[:, 0, :, :], heatmap_gt_pair, point_mask_pair)

            # Seg dis loss
            dis_loss = self.sam_loss(seg_map, seg_pair)

            # Attention_Cat_loss here
            #Attention_Cat_loss = self.Attention_Cat_loss(weight_map, cat_0, cat_1)

            loss = desp_loss + point_loss + dis_loss + desp_sam_loss
            total_loss += loss
            if torch.isnan(loss):
                self.logger.error('loss is nan!')

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            if i % self.config['train']['log_freq'] == 0:

                point_loss_val = point_loss.item()
                desp_loss_val = desp_loss.item()
                dis_loss_val = dis_loss.item()
                loss_val = loss.item()

                self.logger.info(
                    "[Epoch:%2d][Step:%5d:%5d]: loss = %.4f, point_loss = %.4f, desp_loss = %.4f, dis_loss = %.4f, NCE_loss = %.4f"
                    " one step cost %.4fs. " % (
                        epoch_idx, i, self.epoch_length,
                        loss_val,
                        point_loss_val,
                        desp_loss_val,
                        dis_loss_val,
                        desp_sam_loss,
                        (time.time() - stime) / self.config['train']['log_freq'],
                    ))
                stime = time.time()
        self.logger.info("Total_loss:" + str(total_loss.detach().cpu().numpy()))
        # save the model
        if self.multi_gpus:
            torch.save(
                self.model.module.state_dict(), os.path.join(self.config['ckpt_path'], 'model_%02d.pt' % epoch_idx))
        else:
            torch.save(
                self.model.state_dict(), os.path.join(self.config['ckpt_path'], 'model_%02d.pt' % epoch_idx))
    def _inference_func(self, image_pair):
        """
        image_pair: [2,1,h,w]
        """
        self.model.eval()
        _, _, height, width = image_pair.shape
        heatmap_pair, feature_pair, weightmap_pair = self.model(image_pair)
        c1, c2 = torch.chunk(feature_pair, 2, dim=0)
        w1, w2 = torch.chunk(weightmap_pair, 2, dim=0)
        heatmap_pair = torch.sigmoid(heatmap_pair)
        prob_pair = spatial_nms(heatmap_pair)

        prob_pair = prob_pair.detach().cpu().numpy()
        first_prob = prob_pair[0, 0]
        second_prob = prob_pair[1, 0]

        first_point, first_point_num = self._generate_predict_point(
            first_prob,
            detection_threshold=self.config['test']['detection_threshold'],
            top_k=self.config['test']['top_k'])  # [n,2]

        second_point, second_point_num = self._generate_predict_point(
            second_prob,
            detection_threshold=self.config['test']['detection_threshold'],
            top_k=self.config['test']['top_k'])  # [n,2]

        if first_point_num <= 4 or second_point_num <= 4:
            print("skip this pair because there's little point!")
            return None

        select_first_desp = self._generate_combined_descriptor_fast(first_point, c1,w1, height, width)
        select_second_desp = self._generate_combined_descriptor_fast(second_point, c2,w2, height, width)

        return first_point, first_point_num, second_point, second_point_num, select_first_desp, select_second_desp

    def _generate_combined_descriptor_fast(self, point, feature,weight, height, width):
        point = torch.from_numpy(point[:, ::-1].copy()).to(torch.float).to(self.device)

        point = point * 2. / torch.tensor((width - 1, height - 1), dtype=torch.float, device=self.device) - 1
        point = point.unsqueeze(dim=0).unsqueeze(dim=2)  # [1,n,1,2]

        feature = f.grid_sample(feature, point, mode="bilinear")[:, :, :, 0].transpose(1, 2)[0]
        weight = f.grid_sample(weight, point, mode="bilinear")[:, :, :, 0].transpose(1, 2)[0]
        desp_pair = feature / torch.norm(feature, p=2, dim=1, keepdim=True)
        desp = desp_pair * weight.expand_as(desp_pair)
        desp = desp.detach().cpu().numpy()

        return desp

