#
# Created  on 2020/6/29
#
import os
from glob import glob

import matplotlib.pyplot as plt

import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset

from data_utils.dataset_tools import HomographyAugmentation
from data_utils.dataset_tools import ImgAugTransform
from data_utils.dataset_tools import space_to_depth
import torch.multiprocessing as mp

def display_image(image, edge_map):
    # Display the image
    plt.imshow(image)
    plt.axis('off')
    plt.show()

    plt.imshow(edge_map, cmap='gray')
    plt.axis('off')
    plt.show()

class MegaDepthTrainDataset(Dataset):
    """
    Combination of MegaDetph and COCO
    """
    def __init__(self, **config):
        self.data_list = self._format_file_list(
            config['mega_image_dir'],
            config['mega_keypoint_dir'],
            config['mega_despoint_dir'],
        )
        self.sydesp_type=config['sydesp_type']
        self.height = config['height']
        self.width = config['width']

        self.homography = HomographyAugmentation()
        self.photometric = ImgAugTransform()
        self.fix_grid = self._generate_fixed_grid()

        self.offline_root = config['end_data']
        self.cat_info = config['sam_cat_info']
        #self.corner_info = config['corner_info']

        mp.set_start_method('spawn', force=True)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_info = self.data_list[idx]
        if data_info['type'] == 'synthesis':
            return self._get_synthesis_data(data_info)
        elif data_info['type'] == 'real':
            return self._get_real_data(data_info)
        else:
            assert False

    def _remove_leading_zeros(self, num_str):
        # Remove leading zeros from the number string
        num_str = num_str.lstrip('0')

        # If the number string is empty, return '0'
        if not num_str:
            return '0'

        # If the first character is not '0', return the number string
        if num_str[0] != '0':
            return num_str

        # Otherwise, find the index of the first non-zero character
        for i in range(len(num_str)):
            if num_str[i] != '0':
                return num_str[i:]

        # If all characters are zero, return '0'
        return '0'

    def _get_real_data(self, data_info):
        name = data_info['name']
        image_dir = data_info['image']
        info_dir = data_info['info']
        label_dir = data_info['label']

        # Add in offline end data
        #off_line_name = self._remove_leading_zeros(name)
        sam_dict = np.load(os.path.join(self.offline_root, name) + '.npy', allow_pickle=True).item()
        sam_cat_info = np.load(os.path.join(self.cat_info, name) + '.npy', allow_pickle=True).item()
        #sam_corner_info = np.load(os.path.join(self.corner_info, name) + '.npy', allow_pickle=True).item()

        image12 = cv.imread(image_dir)[:, :, ::-1].copy()  # 交换BGR为RGB
        image1, image2 = np.split(image12, 2, axis=1)
        h, w, _ = image1.shape

        if torch.rand([]).item() < 0.5:
            image1 = self.photometric(image1)
            image2 = self.photometric(image2)

        info = np.load(info_dir)
        desp_point1 = info["desp_point1"]
        desp_point2 = info["desp_point2"]
        valid_mask = info["valid_mask"]
        not_search_mask = info["not_search_mask"]

        label = np.load(label_dir)
        points1 = label["points_0"]
        points2 = label["points_1"]

        # 2.1 得到第一副图点构成的热图
        heatmap1 = self._convert_points_to_heatmap(points1)
        #corner1 = torch.from_numpy(sam_corner_info['corner'] // 255)
        #heatmap1 = torch.logical_or(heatmap1, corner1).float()
        point_mask1 = torch.ones_like(heatmap1)


        # 2.2 得到第二副图点构成的热图
        heatmap2 = self._convert_points_to_heatmap(points2)
        #corner2 = torch.from_numpy(sam_corner_info['corner_warped'] // 255)
        #heatmap2 = torch.logical_or(heatmap2, corner2).float()
        point_mask2 = torch.ones_like(heatmap2)


        # debug use
        # desp_point1 = ((desp_point1 + 1) * np.array((self.width - 1, self.height - 1)) / 2.)[:, 0, :]
        # desp_point2 = ((desp_point2 + 1) * np.array((self.width - 1, self.height - 1)) / 2.)[:, 0, :]
        # image_point1 = draw_image_keypoints(image1[:, :, ::-1], desp_point1[valid_mask][:, ::-1], show=False)
        # image_point2 = draw_image_keypoints(image2[:, :, ::-1], desp_point2[valid_mask][:, ::-1], show=False)
        # image_point1 = draw_image_keypoints(image1[:, :, ::-1], points1, show=False)
        # image_point2 = draw_image_keypoints(image2[:, :, ::-1], points2, show=False)
        # cat_all = np.concatenate((image_point1, image_point2), axis=0)
        # cv.imwrite("/home/yuyang/tmp/debug_%05d.jpg" % idx, cat_all)
        # cv.imshow("cat_all", cat_all)
        # cv.waitKey()
        #display_image(heatmap2, image2)
        image1 = (torch.from_numpy(image1).to(torch.float) * 2. / 255. - 1.).permute((2, 0, 1)).contiguous()
        image2 = (torch.from_numpy(image2).to(torch.float) * 2. / 255. - 1.).permute((2, 0, 1)).contiguous()

        desp_point1 = torch.from_numpy(desp_point1)
        desp_point2 = torch.from_numpy(desp_point2)

        valid_mask = torch.from_numpy(valid_mask).to(torch.float)
        not_search_mask = torch.from_numpy(not_search_mask).to(torch.float)

        return {
            "image": image1,
            "point_mask": point_mask1,
            "heatmap": heatmap1,
            "warped_image": image2,
            "warped_point_mask": point_mask2,
            "warped_heatmap": heatmap2,
            "desp_point": desp_point1,
            "warped_desp_point": desp_point2,
            "valid_mask": valid_mask,
            "not_search_mask": not_search_mask,
            "image_name": torch.as_tensor(int(name)),
            "single_seg": sam_dict['single_seg'].cuda(),
            "single_seg_warped": sam_dict['single_seg_warped'].cuda(),
            "desp_cat": sam_cat_info['desp_cat'],
            "desp_cat_warped": sam_cat_info['desp_cat_warped'],
        }

    def _get_synthesis_data(self, data_info):
        name = data_info['name']

        #off_line_name = self._remove_leading_zeros(name)
        sam_dict_warped = np.load(os.path.join(self.offline_root, name) + '_warped.npy', allow_pickle=True).item()
        sam_cat_info = np.load(os.path.join(self.cat_info, name) + '_warped.npy', allow_pickle=True).item()
        #sam_corner_info = np.load(os.path.join(self.corner_info, name) + '_warped.npy', allow_pickle=True).item()


        image12 = cv.imread(data_info['image'])[:, :, ::-1].copy()  # 交换BGR为RGB
        image1, image2 = np.split(image12, 2, axis=1)
        point = np.load(data_info['label'])
        info = np.load(data_info['info'])
        if torch.rand([]).item() < 0.5:
            image = cv.resize(image1, dsize=(self.width, self.height), interpolation=cv.INTER_LINEAR)
            point = point["points_0"]
            desp_point_load = info["raw_desp_point1"]
        else:
            image = cv.resize(image2, dsize=(self.width, self.height), interpolation=cv.INTER_LINEAR)
            point = point["points_1"]
            desp_point_load = info["raw_desp_point2"]
        point_mask = np.ones_like(image).astype(np.float32)[:, :, 0].copy()

        # 1、由随机采样的单应变换得到第二副图像及其对应的关键点位置、原始掩膜和该单应变换
        if torch.rand([]).item() < 0.5:
            warped_image, warped_point_mask, warped_point, homography = \
                image.copy(), point_mask.copy(), point.copy(), np.eye(3)
        else:
            warped_image, warped_point_mask, warped_point, homography = self.homography(image, point, return_homo=True)
            warped_point_mask = warped_point_mask[:, :, 0].copy()

        if torch.rand([]).item() < 0.5:
            image = self.photometric(image)
            warped_image = self.photometric(warped_image)

        # 2.1 得到第一副图点构成的热图
        heatmap = self._convert_points_to_heatmap(point)
        #corner1 = torch.from_numpy(sam_corner_info['corner'] // 255)
        #heatmap = torch.logical_or(heatmap, corner1).float()

        # 2.2 得到第二副图点构成的热图
        warped_heatmap = self._convert_points_to_heatmap(warped_point)
        #corner2 = torch.from_numpy(sam_corner_info['corner_warped'] // 255)
        #warped_heatmap = torch.logical_or(warped_heatmap, corner2).float()
        #display_image(warped_heatmap, image2)
        # 3、采样训练描述子要用的点
        if self.sydesp_type =='random':
            desp_point = self._random_sample_point()
        else:
            desp_point = desp_point_load

        shape = image.shape

        warped_desp_point, valid_mask, not_search_mask = self._generate_warped_point(
            desp_point, homography, shape[0], shape[1])

        # debug use
        # image_point = draw_image_keypoints(image, desp_point, show=False)
        # warped_image_point = draw_image_keypoints(warped_image, warped_desp_point, show=False)
        # cat_all = np.concatenate((image, warped_image), axis=1)
        # cat_all = np.concatenate((image_point, warped_image_point), axis=1)
        # cv.imwrite("/home/yuyang/tmp/coco_tmp/%d.jpg" % idx, cat_all)
        # cv.imshow("cat_all", cat_all)
        # cv.waitKey()

        image = image.astype(np.float32) * 2. / 255. - 1.
        warped_image = warped_image.astype(np.float32) * 2. / 255. - 1.

        image = torch.from_numpy(image).permute((2, 0, 1))
        warped_image = torch.from_numpy(warped_image).permute((2, 0, 1))

        point_mask = torch.from_numpy(point_mask)
        warped_point_mask = torch.from_numpy(warped_point_mask)

        desp_point = torch.from_numpy(self._scale_point_for_sample(desp_point))
        warped_desp_point = torch.from_numpy(self._scale_point_for_sample(warped_desp_point))

        valid_mask = torch.from_numpy(valid_mask)
        not_search_mask = torch.from_numpy(not_search_mask)

        return {
            "image": image,  # [1,h,w]
            "point_mask": point_mask,  # [h,w]
            "heatmap": heatmap,  # [h,w]
            "warped_image": warped_image,  # [1,h,w]
            "warped_point_mask": warped_point_mask,  # [h,w]
            "warped_heatmap": warped_heatmap,  # [h,w]
            "desp_point": desp_point,  # [n,1,2]
            "warped_desp_point": warped_desp_point,  # [n,1,2]
            "valid_mask": valid_mask,  # [n]
            "not_search_mask": not_search_mask,  # [n,n]
            "image_name": torch.as_tensor(-int(name)),
            "single_seg": sam_dict_warped['single_seg'].cuda(),
            "single_seg_warped": sam_dict_warped['single_seg_warped'].cuda(),
            "desp_cat": sam_cat_info['desp_cat'],
            "desp_cat_warped": sam_cat_info['desp_cat_warped'],
        }

    @ staticmethod
    def _generate_warped_point(point, homography, height, width, threshold=16):
        """
        根据投影变换得到变换后的坐标点，有效关系及不参与负样本搜索的矩阵
        Args:
            point: [n,2] 与warped_point一一对应
            homography: 点对之间的变换关系

        Returns:
            not_search_mask: [n,n] type为float32的mask,不搜索的位置为1
        """
        # 得到投影点的坐标
        point = np.concatenate((point[:, ::-1], np.ones((point.shape[0], 1))), axis=1)[:, :, np.newaxis]  # [n,3,1]
        project_point = np.matmul(homography, point)[:, :, 0]
        project_point = project_point[:, :2] / project_point[:, 2:3]
        project_point = project_point[:, ::-1]  # 调换为y,x的顺序

        # 投影点在图像范围内的点为有效点，反之则为无效点
        boarder_0 = np.array((0, 0), dtype=np.float32)
        boarder_1 = np.array((height-1, width-1), dtype=np.float32)
        valid_mask = (project_point >= boarder_0) & (project_point <= boarder_1)
        valid_mask = np.all(valid_mask, axis=1)
        invalid_mask = ~valid_mask

        # 根据无效点及投影点之间的距离关系确定不搜索的负样本矩阵

        dist = np.linalg.norm(project_point[:, np.newaxis, :] - project_point[np.newaxis, :, :], axis=2)
        not_search_mask = ((dist <= threshold) | invalid_mask[np.newaxis, :]).astype(np.float32)
        return project_point.astype(np.float32), valid_mask.astype(np.float32), not_search_mask

    def _scale_point_for_sample(self, point):
        """
        将点归一化到[-1,1]的区间范围内，并调换顺序为x,y，方便采样
        Args:
            point: [n,2] y,x的顺序，原始范围为[0,height-1], [0,width-1]
        Returns:
            point: [n,1,2] x,y的顺序，范围为[-1,1]
        """
        org_size = np.array((self.height-1, self.width-1), dtype=np.float32)
        point = ((point * 2. / org_size - 1.)[:, ::-1])[:, np.newaxis, :].copy()
        return point

    def _random_sample_point(self):
        """
        根据预设的输入图像大小，随机均匀采样坐标点
        """
        grid = self.fix_grid.copy()
        # 随机选择指定数目个格子

        point_list = []
        for i in range(grid.shape[0]):
            y_start, x_start, y_end, x_end = grid[i]
            rand_y = np.random.randint(y_start, y_end)
            rand_x = np.random.randint(x_start, x_end)
            point_list.append(np.array((rand_y, rand_x), dtype=np.float32))
        point = np.stack(point_list, axis=0)

        return point

    def _generate_fixed_grid(self, option=None):
        """
        预先采样固定间隔的225个图像格子
        """
        if option == None:
            y_num = 20
            x_num = 20
        else:
            y_num = option[0]
            x_num = option[1]

        grid_y = np.linspace(0, self.height-1, y_num+1, dtype=np.int)
        grid_x = np.linspace(0, self.width-1, x_num+1, dtype=np.int)

        grid_y_start = grid_y[:y_num].copy()
        grid_y_end = grid_y[1:y_num+1].copy()
        grid_x_start = grid_x[:x_num].copy()
        grid_x_end = grid_x[1:x_num+1].copy()

        grid_start = np.stack((np.tile(grid_y_start[:, np.newaxis], (1, x_num)),
                               np.tile(grid_x_start[np.newaxis, :], (y_num, 1))), axis=2).reshape((-1, 2))
        grid_end = np.stack((np.tile(grid_y_end[:, np.newaxis], (1, x_num)),
                             np.tile(grid_x_end[np.newaxis, :], (y_num, 1))), axis=2).reshape((-1, 2))
        grid = np.concatenate((grid_start, grid_end), axis=1)

        return grid

    def _convert_points_to_heatmap(self, points):
        """
        将原始点位置经下采样后得到heatmap与incmap，heatmap上对应下采样整型点位置处的值为1，其余为0；incmap与heatmap一一对应，
        在关键点位置处存放整型点到亚像素角点的偏移量，以及训练时用来屏蔽非关键点inc量的incmap_valid
        Args:
            points: [n,2]

        Returns:
            heatmap: [h,w] 关键点位置为1，其余为0
            incmap: [2,h,w] 关键点位置存放实际偏移，其余非关键点处的偏移量为0
            incmap_valid: [h,w] 关键点位置为1，其余为0，用于训练时屏蔽对非关键点偏移量的训练，只关注关键点的偏移量

        """
        height = self.height
        width = self.width

        # localmap = self.localmap.clone()
        # padded_heatmap = torch.zeros(
        #     (height+self.g_paddings*2, width+self.g_paddings*2), dtype=torch.float)
        heatmap = torch.zeros((height, width), dtype=torch.float)

        num_pt = points.shape[0]
        if num_pt > 0:
            for i in range(num_pt):
                pt = points[i]
                pt_y_float, pt_x_float = pt

                pt_y_int = round(pt_y_float)
                pt_x_int = round(pt_x_float)

                pt_y = int(pt_y_int)  # 对真值点位置进行下采样,这里有量化误差
                pt_x = int(pt_x_int)

                # 排除掉经下采样后在边界外的点
                if pt_y < 0 or pt_y > height - 1:
                    continue
                if pt_x < 0 or pt_x > width - 1:
                    continue

                # 关键点位置在heatmap上置1，并在incmap上记录该点离亚像素点的偏移量
                heatmap[pt_y, pt_x] = 1.0

        return heatmap

    def convert_points_to_label(self, points):

        height = self.height
        width = self.width
        n_height = int(height / 8)
        n_width = int(width / 8)
        assert n_height * 8 == height and n_width * 8 == width

        num_pt = points.shape[0]
        label = torch.zeros((height * width))
        if num_pt > 0:
            points_h, points_w = torch.split(points, 1, dim=1)
            points_idx = points_w + points_h * width
            label = label.scatter_(dim=0, index=points_idx[:, 0], value=1.0).reshape((height, width))
        else:
            label = label.reshape((height, width))

        dense_label = space_to_depth(label)
        dense_label = torch.cat((dense_label, 0.5 * torch.ones((1, n_height, n_width))), dim=0)  # [65, 30, 40]
        sparse_label = torch.argmax(dense_label, dim=0)  # [30,40]

        return sparse_label
    @staticmethod
    def _format_file_list(mega_image_dir, mega_keypoint_dir,mega_despoint_dir):
        data_list = []

        # format megadepth related list
        mega_image_list = glob(os.path.join(mega_image_dir, '*.jpg'))
        mega_image_list = sorted(mega_image_list)
        data_type = 'real'
        for img in mega_image_list:
            img_name = img.split('/')[-1].split('.')[0]
            info = os.path.join(mega_despoint_dir, img_name + '.npz')
            label = os.path.join(mega_keypoint_dir, img_name + '.npz')
            data_list.append(
                {
                    'type': data_type,
                    'image': img,
                    'info': info,
                    'label': label,
                    'name': img_name,
                }
            )

        # format coco related list
        data_type = 'synthesis'
        for img in mega_image_list:
            img_name = img.split('/')[-1].split('.')[0]
            label = os.path.join(mega_keypoint_dir, img_name + '.npz')
            info = os.path.join(mega_despoint_dir, img_name + '.npz')
            data_list.append(
                {
                    'type': data_type,
                    'image': img,
                    'info': info,
                    'label': label,
                    'name': img_name,
                }
            )

        return data_list


class MegaDepthTrainDataset_NoSAM(Dataset):
    """
    Combination of MegaDetph and COCO
    """
    def __init__(self, **config):
        self.data_list = self._format_file_list(
            config['mega_image_dir'],
            config['mega_keypoint_dir'],
            config['mega_despoint_dir'],
        )
        self.sydesp_type=config['sydesp_type']
        self.height = config['height']
        self.width = config['width']

        self.homography = HomographyAugmentation()
        self.photometric = ImgAugTransform()
        self.fix_grid = self._generate_fixed_grid()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_info = self.data_list[idx]
        if data_info['type'] == 'synthesis':
            return self._get_synthesis_data(data_info)
        elif data_info['type'] == 'real':
            return self._get_real_data(data_info)
        else:
            assert False

    def _get_real_data(self, data_info):
        image_dir = data_info['image']
        info_dir = data_info['info']
        label_dir = data_info['label']

        image12 = cv.imread(image_dir)[:, :, ::-1].copy()  # 交换BGR为RGB
        image1, image2 = np.split(image12, 2, axis=1)
        h, w, _ = image1.shape

        if torch.rand([]).item() < 0.5:
            image1 = self.photometric(image1)
            image2 = self.photometric(image2)

        info = np.load(info_dir)
        desp_point1 = info["desp_point1"]
        desp_point2 = info["desp_point2"]
        valid_mask = info["valid_mask"]
        not_search_mask = info["not_search_mask"]

        label = np.load(label_dir)
        points1 = label["points_0"]
        points2 = label["points_1"]

        # 2.1 得到第一副图点构成的热图
        heatmap1 = self._convert_points_to_heatmap(points1)
        point_mask1 = torch.ones_like(heatmap1)

        # 2.2 得到第二副图点构成的热图
        heatmap2 = self._convert_points_to_heatmap(points2)
        point_mask2 = torch.ones_like(heatmap2)

        # debug use
        # desp_point1 = ((desp_point1 + 1) * np.array((self.width - 1, self.height - 1)) / 2.)[:, 0, :]
        # desp_point2 = ((desp_point2 + 1) * np.array((self.width - 1, self.height - 1)) / 2.)[:, 0, :]
        # image_point1 = draw_image_keypoints(image1[:, :, ::-1], desp_point1[valid_mask][:, ::-1], show=False)
        # image_point2 = draw_image_keypoints(image2[:, :, ::-1], desp_point2[valid_mask][:, ::-1], show=False)
        # image_point1 = draw_image_keypoints(image1[:, :, ::-1], points1, show=False)
        # image_point2 = draw_image_keypoints(image2[:, :, ::-1], points2, show=False)
        # cat_all = np.concatenate((image_point1, image_point2), axis=0)
        # cv.imwrite("/home/yuyang/tmp/debug_%05d.jpg" % idx, cat_all)
        # cv.imshow("cat_all", cat_all)
        # cv.waitKey()

        image1 = (torch.from_numpy(image1).to(torch.float) * 2. / 255. - 1.).permute((2, 0, 1)).contiguous()
        image2 = (torch.from_numpy(image2).to(torch.float) * 2. / 255. - 1.).permute((2, 0, 1)).contiguous()

        desp_point1 = torch.from_numpy(desp_point1)
        desp_point2 = torch.from_numpy(desp_point2)

        valid_mask = torch.from_numpy(valid_mask).to(torch.float)
        not_search_mask = torch.from_numpy(not_search_mask).to(torch.float)

        return {
            "image": image1,
            "point_mask": point_mask1,
            "heatmap": heatmap1,
            "warped_image": image2,
            "warped_point_mask": point_mask2,
            "warped_heatmap": heatmap2,
            "desp_point": desp_point1,
            "warped_desp_point": desp_point2,
            "valid_mask": valid_mask,
            "not_search_mask": not_search_mask,
        }

    def _get_synthesis_data(self, data_info):
        image12 = cv.imread(data_info['image'])[:, :, ::-1].copy()  # 交换BGR为RGB
        image1, image2 = np.split(image12, 2, axis=1)
        point = np.load(data_info['label'])
        info = np.load(data_info['info'])
        if torch.rand([]).item() < 0.5:
            image = cv.resize(image1, dsize=(self.width, self.height), interpolation=cv.INTER_LINEAR)
            point = point["points_0"]
            desp_point_load = info["raw_desp_point1"]
        else:
            image = cv.resize(image2, dsize=(self.width, self.height), interpolation=cv.INTER_LINEAR)
            point = point["points_1"]
            desp_point_load = info["raw_desp_point2"]
        point_mask = np.ones_like(image).astype(np.float32)[:, :, 0].copy()

        # 1、由随机采样的单应变换得到第二副图像及其对应的关键点位置、原始掩膜和该单应变换
        if torch.rand([]).item() < 0.5:
            warped_image, warped_point_mask, warped_point, homography = \
                image.copy(), point_mask.copy(), point.copy(), np.eye(3)
        else:
            warped_image, warped_point_mask, warped_point, homography = self.homography(image, point, return_homo=True)
            warped_point_mask = warped_point_mask[:, :, 0].copy()

        if torch.rand([]).item() < 0.5:
            image = self.photometric(image)
            warped_image = self.photometric(warped_image)

        # 2.1 得到第一副图点构成的热图
        heatmap = self._convert_points_to_heatmap(point)

        # 2.2 得到第二副图点构成的热图
        warped_heatmap = self._convert_points_to_heatmap(warped_point)

        # 3、采样训练描述子要用的点
        if self.sydesp_type =='random':
            desp_point = self._random_sample_point()
        else:
            desp_point = desp_point_load

        shape = image.shape

        warped_desp_point, valid_mask, not_search_mask = self._generate_warped_point(
            desp_point, homography, shape[0], shape[1])

        # debug use
        # image_point = draw_image_keypoints(image, desp_point, show=False)
        # warped_image_point = draw_image_keypoints(warped_image, warped_desp_point, show=False)
        # cat_all = np.concatenate((image, warped_image), axis=1)
        # cat_all = np.concatenate((image_point, warped_image_point), axis=1)
        # cv.imwrite("/home/yuyang/tmp/coco_tmp/%d.jpg" % idx, cat_all)
        # cv.imshow("cat_all", cat_all)
        # cv.waitKey()

        image = image.astype(np.float32) * 2. / 255. - 1.
        warped_image = warped_image.astype(np.float32) * 2. / 255. - 1.

        image = torch.from_numpy(image).permute((2, 0, 1))
        warped_image = torch.from_numpy(warped_image).permute((2, 0, 1))

        point_mask = torch.from_numpy(point_mask)
        warped_point_mask = torch.from_numpy(warped_point_mask)

        desp_point = torch.from_numpy(self._scale_point_for_sample(desp_point))
        warped_desp_point = torch.from_numpy(self._scale_point_for_sample(warped_desp_point))

        valid_mask = torch.from_numpy(valid_mask)
        not_search_mask = torch.from_numpy(not_search_mask)

        return {
            "image": image,  # [1,h,w]
            "point_mask": point_mask,  # [h,w]
            "heatmap": heatmap,  # [h,w]
            "warped_image": warped_image,  # [1,h,w]
            "warped_point_mask": warped_point_mask,  # [h,w]
            "warped_heatmap": warped_heatmap,  # [h,w]
            "desp_point": desp_point,  # [n,1,2]
            "warped_desp_point": warped_desp_point,  # [n,1,2]
            "valid_mask": valid_mask,  # [n]
            "not_search_mask": not_search_mask,  # [n,n]
        }

    @ staticmethod
    def _generate_warped_point(point, homography, height, width, threshold=16):
        """
        根据投影变换得到变换后的坐标点，有效关系及不参与负样本搜索的矩阵
        Args:
            point: [n,2] 与warped_point一一对应
            homography: 点对之间的变换关系

        Returns:
            not_search_mask: [n,n] type为float32的mask,不搜索的位置为1
        """
        # 得到投影点的坐标
        point = np.concatenate((point[:, ::-1], np.ones((point.shape[0], 1))), axis=1)[:, :, np.newaxis]  # [n,3,1]
        project_point = np.matmul(homography, point)[:, :, 0]
        project_point = project_point[:, :2] / project_point[:, 2:3]
        project_point = project_point[:, ::-1]  # 调换为y,x的顺序

        # 投影点在图像范围内的点为有效点，反之则为无效点
        boarder_0 = np.array((0, 0), dtype=np.float32)
        boarder_1 = np.array((height-1, width-1), dtype=np.float32)
        valid_mask = (project_point >= boarder_0) & (project_point <= boarder_1)
        valid_mask = np.all(valid_mask, axis=1)
        invalid_mask = ~valid_mask

        # 根据无效点及投影点之间的距离关系确定不搜索的负样本矩阵

        dist = np.linalg.norm(project_point[:, np.newaxis, :] - project_point[np.newaxis, :, :], axis=2)
        not_search_mask = ((dist <= threshold) | invalid_mask[np.newaxis, :]).astype(np.float32)
        return project_point.astype(np.float32), valid_mask.astype(np.float32), not_search_mask

    def _scale_point_for_sample(self, point):
        """
        将点归一化到[-1,1]的区间范围内，并调换顺序为x,y，方便采样
        Args:
            point: [n,2] y,x的顺序，原始范围为[0,height-1], [0,width-1]
        Returns:
            point: [n,1,2] x,y的顺序，范围为[-1,1]
        """
        org_size = np.array((self.height-1, self.width-1), dtype=np.float32)
        point = ((point * 2. / org_size - 1.)[:, ::-1])[:, np.newaxis, :].copy()
        return point

    def _random_sample_point(self):
        """
        根据预设的输入图像大小，随机均匀采样坐标点
        """
        grid = self.fix_grid.copy()
        # 随机选择指定数目个格子

        point_list = []
        for i in range(grid.shape[0]):
            y_start, x_start, y_end, x_end = grid[i]
            rand_y = np.random.randint(y_start, y_end)
            rand_x = np.random.randint(x_start, x_end)
            point_list.append(np.array((rand_y, rand_x), dtype=np.float32))
        point = np.stack(point_list, axis=0)

        return point

    def _generate_fixed_grid(self, option=None):
        """
        预先采样固定间隔的225个图像格子
        """
        if option == None:
            y_num = 20
            x_num = 20
        else:
            y_num = option[0]
            x_num = option[1]

        grid_y = np.linspace(0, self.height-1, y_num+1, dtype=np.int)
        grid_x = np.linspace(0, self.width-1, x_num+1, dtype=np.int)

        grid_y_start = grid_y[:y_num].copy()
        grid_y_end = grid_y[1:y_num+1].copy()
        grid_x_start = grid_x[:x_num].copy()
        grid_x_end = grid_x[1:x_num+1].copy()

        grid_start = np.stack((np.tile(grid_y_start[:, np.newaxis], (1, x_num)),
                               np.tile(grid_x_start[np.newaxis, :], (y_num, 1))), axis=2).reshape((-1, 2))
        grid_end = np.stack((np.tile(grid_y_end[:, np.newaxis], (1, x_num)),
                             np.tile(grid_x_end[np.newaxis, :], (y_num, 1))), axis=2).reshape((-1, 2))
        grid = np.concatenate((grid_start, grid_end), axis=1)

        return grid

    def _convert_points_to_heatmap(self, points):
        """
        将原始点位置经下采样后得到heatmap与incmap，heatmap上对应下采样整型点位置处的值为1，其余为0；incmap与heatmap一一对应，
        在关键点位置处存放整型点到亚像素角点的偏移量，以及训练时用来屏蔽非关键点inc量的incmap_valid
        Args:
            points: [n,2]

        Returns:
            heatmap: [h,w] 关键点位置为1，其余为0
            incmap: [2,h,w] 关键点位置存放实际偏移，其余非关键点处的偏移量为0
            incmap_valid: [h,w] 关键点位置为1，其余为0，用于训练时屏蔽对非关键点偏移量的训练，只关注关键点的偏移量

        """
        height = self.height
        width = self.width

        # localmap = self.localmap.clone()
        # padded_heatmap = torch.zeros(
        #     (height+self.g_paddings*2, width+self.g_paddings*2), dtype=torch.float)
        heatmap = torch.zeros((height, width), dtype=torch.float)

        num_pt = points.shape[0]
        if num_pt > 0:
            for i in range(num_pt):
                pt = points[i]
                pt_y_float, pt_x_float = pt

                pt_y_int = round(pt_y_float)
                pt_x_int = round(pt_x_float)

                pt_y = int(pt_y_int)  # 对真值点位置进行下采样,这里有量化误差
                pt_x = int(pt_x_int)

                # 排除掉经下采样后在边界外的点
                if pt_y < 0 or pt_y > height - 1:
                    continue
                if pt_x < 0 or pt_x > width - 1:
                    continue

                # 关键点位置在heatmap上置1，并在incmap上记录该点离亚像素点的偏移量
                heatmap[pt_y, pt_x] = 1.0

        return heatmap

    def convert_points_to_label(self, points):

        height = self.height
        width = self.width
        n_height = int(height / 8)
        n_width = int(width / 8)
        assert n_height * 8 == height and n_width * 8 == width

        num_pt = points.shape[0]
        label = torch.zeros((height * width))
        if num_pt > 0:
            points_h, points_w = torch.split(points, 1, dim=1)
            points_idx = points_w + points_h * width
            label = label.scatter_(dim=0, index=points_idx[:, 0], value=1.0).reshape((height, width))
        else:
            label = label.reshape((height, width))

        dense_label = space_to_depth(label)
        dense_label = torch.cat((dense_label, 0.5 * torch.ones((1, n_height, n_width))), dim=0)  # [65, 30, 40]
        sparse_label = torch.argmax(dense_label, dim=0)  # [30,40]

        return sparse_label
    @staticmethod
    def _format_file_list(mega_image_dir, mega_keypoint_dir,mega_despoint_dir):
        data_list = []

        # format megadepth related list
        mega_image_list = glob(os.path.join(mega_image_dir, '*.jpg'))
        mega_image_list = sorted(mega_image_list)
        data_type = 'real'
        for img in mega_image_list:
            img_name = img.split('/')[-1].split('.')[0]
            info = os.path.join(mega_despoint_dir, img_name + '.npz')
            label = os.path.join(mega_keypoint_dir, img_name + '.npz')
            data_list.append(
                {
                    'type': data_type,
                    'image': img,
                    'info': info,
                    'label': label,
                }
            )

        # format coco related list
        data_type = 'synthesis'
        for img in mega_image_list:
            img_name = img.split('/')[-1].split('.')[0]
            label = os.path.join(mega_keypoint_dir, img_name + '.npz')
            info = os.path.join(mega_despoint_dir, img_name + '.npz')
            data_list.append(
                {
                    'type': data_type,
                    'image': img,
                    'info': info,
                    'label': label,
                }
            )

        return data_list

    @staticmethod
    def _format_file_list2(mega_image_dir, mega_keypoint_dir,mega_despoint_dir):
        data_list = []

        # format megadepth related list
        mega_image_list = glob(os.path.join(mega_image_dir, '*.jpg'))
        mega_image_list = sorted(mega_image_list)
        data_type = 'real'
        for img in mega_image_list:
            img_name = img.split('/')[-1].split('.')[0]
            info = os.path.join(mega_despoint_dir, img_name + '.npz')
            label = os.path.join(mega_keypoint_dir, img_name + '.npz')
            data_list.append(
                {
                    'type': data_type,
                    'image': img,
                    'info': info,
                    'label': label,
                }
            )

        # format coco related list
        data_type = 'synthesis'
        for img in mega_image_list:
            img_name = img.split('/')[-1].split('.')[0]
            label = os.path.join(mega_keypoint_dir, img_name + '.npz')
            info = os.path.join(mega_despoint_dir, img_name + '.npz')
            data_list.append(
                {
                    'type': data_type,
                    'image': img,
                    'info': info,
                    'label': label,
                }
            )

        return data_list


class MegaDepthTrainDataset_WithSamFeature(Dataset):
    """
    Combination of MegaDetph and COCO
    """
    def __init__(self, **config):
        self.data_list = self._format_file_list(
            config['mega_image_dir'],
            config['mega_keypoint_dir'],
            config['mega_despoint_dir'],
        )
        self.sydesp_type=config['sydesp_type']
        self.height = config['height']
        self.width = config['width']

        self.homography = HomographyAugmentation()
        self.photometric = ImgAugTransform()
        self.fix_grid = self._generate_fixed_grid()

        self.sam_feature_root = config['sam_feature']
        #mp.set_start_method('spawn', force=True)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_info = self.data_list[idx]
        if data_info['type'] == 'synthesis':
            return self._get_synthesis_data(data_info)
        elif data_info['type'] == 'real':
            return self._get_real_data(data_info)
        else:
            assert False

    def _get_real_data(self, data_info):
        image_dir = data_info['image']
        info_dir = data_info['info']
        label_dir = data_info['label']
        name = data_info['name']
        sam_dict = np.load(os.path.join(self.sam_feature_root, name) + '.npy', allow_pickle=True).item()

        image12 = cv.imread(image_dir)[:, :, ::-1].copy()  # 交换BGR为RGB
        image1, image2 = np.split(image12, 2, axis=1)
        h, w, _ = image1.shape

        if torch.rand([]).item() < 0.5:
            image1 = self.photometric(image1)
            image2 = self.photometric(image2)

        info = np.load(info_dir)
        desp_point1 = info["desp_point1"]
        desp_point2 = info["desp_point2"]
        valid_mask = info["valid_mask"]
        not_search_mask = info["not_search_mask"]

        label = np.load(label_dir)
        points1 = label["points_0"]
        points2 = label["points_1"]

        # 2.1 得到第一副图点构成的热图
        heatmap1 = self._convert_points_to_heatmap(points1)
        point_mask1 = torch.ones_like(heatmap1)

        # 2.2 得到第二副图点构成的热图
        heatmap2 = self._convert_points_to_heatmap(points2)
        point_mask2 = torch.ones_like(heatmap2)

        # debug use
        # desp_point1 = ((desp_point1 + 1) * np.array((self.width - 1, self.height - 1)) / 2.)[:, 0, :]
        # desp_point2 = ((desp_point2 + 1) * np.array((self.width - 1, self.height - 1)) / 2.)[:, 0, :]
        # image_point1 = draw_image_keypoints(image1[:, :, ::-1], desp_point1[valid_mask][:, ::-1], show=False)
        # image_point2 = draw_image_keypoints(image2[:, :, ::-1], desp_point2[valid_mask][:, ::-1], show=False)
        # image_point1 = draw_image_keypoints(image1[:, :, ::-1], points1, show=False)
        # image_point2 = draw_image_keypoints(image2[:, :, ::-1], points2, show=False)
        # cat_all = np.concatenate((image_point1, image_point2), axis=0)
        # cv.imwrite("/home/yuyang/tmp/debug_%05d.jpg" % idx, cat_all)
        # cv.imshow("cat_all", cat_all)
        # cv.waitKey()

        image1 = (torch.from_numpy(image1).to(torch.float) * 2. / 255. - 1.).permute((2, 0, 1)).contiguous()
        image2 = (torch.from_numpy(image2).to(torch.float) * 2. / 255. - 1.).permute((2, 0, 1)).contiguous()

        desp_point1 = torch.from_numpy(desp_point1)
        desp_point2 = torch.from_numpy(desp_point2)

        valid_mask = torch.from_numpy(valid_mask).to(torch.float)
        not_search_mask = torch.from_numpy(not_search_mask).to(torch.float)

        return {
            "image": image1,
            "point_mask": point_mask1,
            "heatmap": heatmap1,
            "warped_image": image2,
            "warped_point_mask": point_mask2,
            "warped_heatmap": heatmap2,
            "desp_point": desp_point1,
            "warped_desp_point": desp_point2,
            "valid_mask": valid_mask,
            "not_search_mask": not_search_mask,
            "left_sam_feature": sam_dict['SAM_feature'],
            "right_sam_feature": sam_dict['SAM_feature_warped'],
            "left_flag": torch.tensor(1, dtype=torch.float32),
            "right_flag": torch.tensor(1, dtype=torch.float32),
            "left_sam_edge": sam_dict["SAM_edge"],
            "right_sam_edge": sam_dict["SAM_edge_warped"],
        }

    def _get_synthesis_data(self, data_info):
        image12 = cv.imread(data_info['image'])[:, :, ::-1].copy()  # 交换BGR为RGB
        image1, image2 = np.split(image12, 2, axis=1)
        point = np.load(data_info['label'])
        info = np.load(data_info['info'])
        name = data_info['name']
        sam_dict = np.load(os.path.join(self.sam_feature_root, name) + '.npy', allow_pickle=True).item()

        if torch.rand([]).item() < 0.5:
            image = cv.resize(image1, dsize=(self.width, self.height), interpolation=cv.INTER_LINEAR)
            point = point["points_0"]
            desp_point_load = info["raw_desp_point1"]
            SAM_feature_Left = True
        else:
            image = cv.resize(image2, dsize=(self.width, self.height), interpolation=cv.INTER_LINEAR)
            point = point["points_1"]
            desp_point_load = info["raw_desp_point2"]
            SAM_feature_Left = False
        if SAM_feature_Left:
            SAM_Feature = sam_dict['SAM_feature']
            SAM_edge = sam_dict['SAM_edge']
        else:
            SAM_Feature = sam_dict['SAM_feature_warped']
            SAM_edge = sam_dict['SAM_edge_warped']
        point_mask = np.ones_like(image).astype(np.float32)[:, :, 0].copy()

        # 1、由随机采样的单应变换得到第二副图像及其对应的关键点位置、原始掩膜和该单应变换
        if torch.rand([]).item() < 0.5:
            warped_image, warped_point_mask, warped_point, homography = \
                image.copy(), point_mask.copy(), point.copy(), np.eye(3)
            right_flag = torch.tensor(1, dtype=torch.float32)
        else:
            warped_image, warped_point_mask, warped_point, homography = self.homography(image, point, return_homo=True)
            warped_point_mask = warped_point_mask[:, :, 0].copy()
            right_flag = torch.tensor(0, dtype=torch.float32)

        if torch.rand([]).item() < 0.5:
            image = self.photometric(image)
            warped_image = self.photometric(warped_image)

        # 2.1 得到第一副图点构成的热图
        heatmap = self._convert_points_to_heatmap(point)

        # 2.2 得到第二副图点构成的热图
        warped_heatmap = self._convert_points_to_heatmap(warped_point)

        # 3、采样训练描述子要用的点
        if self.sydesp_type =='random':
            desp_point = self._random_sample_point()
        else:
            desp_point = desp_point_load

        shape = image.shape

        warped_desp_point, valid_mask, not_search_mask = self._generate_warped_point(
            desp_point, homography, shape[0], shape[1])

        # debug use
        # image_point = draw_image_keypoints(image, desp_point, show=False)
        # warped_image_point = draw_image_keypoints(warped_image, warped_desp_point, show=False)
        # cat_all = np.concatenate((image, warped_image), axis=1)
        # cat_all = np.concatenate((image_point, warped_image_point), axis=1)
        # cv.imwrite("/home/yuyang/tmp/coco_tmp/%d.jpg" % idx, cat_all)
        # cv.imshow("cat_all", cat_all)
        # cv.waitKey()

        image = image.astype(np.float32) * 2. / 255. - 1.
        warped_image = warped_image.astype(np.float32) * 2. / 255. - 1.

        image = torch.from_numpy(image).permute((2, 0, 1))
        warped_image = torch.from_numpy(warped_image).permute((2, 0, 1))

        point_mask = torch.from_numpy(point_mask)
        warped_point_mask = torch.from_numpy(warped_point_mask)

        desp_point = torch.from_numpy(self._scale_point_for_sample(desp_point))
        warped_desp_point = torch.from_numpy(self._scale_point_for_sample(warped_desp_point))

        valid_mask = torch.from_numpy(valid_mask)
        not_search_mask = torch.from_numpy(not_search_mask)

        return {
            "image": image,  # [1,h,w]
            "point_mask": point_mask,  # [h,w]
            "heatmap": heatmap,  # [h,w]
            "warped_image": warped_image,  # [1,h,w]
            "warped_point_mask": warped_point_mask,  # [h,w]
            "warped_heatmap": warped_heatmap,  # [h,w]
            "desp_point": desp_point,  # [n,1,2]
            "warped_desp_point": warped_desp_point,  # [n,1,2]
            "valid_mask": valid_mask,  # [n]
            "not_search_mask": not_search_mask,  # [n,n]
            "left_sam_feature": SAM_Feature,
            "right_sam_feature": SAM_Feature,
            "left_flag": torch.tensor(1, dtype=torch.float32),
            "right_flag": right_flag,
            "left_sam_edge": SAM_edge,
            "right_sam_edge": SAM_edge,
        }

    @ staticmethod
    def _generate_warped_point(point, homography, height, width, threshold=16):
        """
        根据投影变换得到变换后的坐标点，有效关系及不参与负样本搜索的矩阵
        Args:
            point: [n,2] 与warped_point一一对应
            homography: 点对之间的变换关系

        Returns:
            not_search_mask: [n,n] type为float32的mask,不搜索的位置为1
        """
        # 得到投影点的坐标
        point = np.concatenate((point[:, ::-1], np.ones((point.shape[0], 1))), axis=1)[:, :, np.newaxis]  # [n,3,1]
        project_point = np.matmul(homography, point)[:, :, 0]
        project_point = project_point[:, :2] / project_point[:, 2:3]
        project_point = project_point[:, ::-1]  # 调换为y,x的顺序

        # 投影点在图像范围内的点为有效点，反之则为无效点
        boarder_0 = np.array((0, 0), dtype=np.float32)
        boarder_1 = np.array((height-1, width-1), dtype=np.float32)
        valid_mask = (project_point >= boarder_0) & (project_point <= boarder_1)
        valid_mask = np.all(valid_mask, axis=1)
        invalid_mask = ~valid_mask

        # 根据无效点及投影点之间的距离关系确定不搜索的负样本矩阵

        dist = np.linalg.norm(project_point[:, np.newaxis, :] - project_point[np.newaxis, :, :], axis=2)
        not_search_mask = ((dist <= threshold) | invalid_mask[np.newaxis, :]).astype(np.float32)
        return project_point.astype(np.float32), valid_mask.astype(np.float32), not_search_mask

    def _scale_point_for_sample(self, point):
        """
        将点归一化到[-1,1]的区间范围内，并调换顺序为x,y，方便采样
        Args:
            point: [n,2] y,x的顺序，原始范围为[0,height-1], [0,width-1]
        Returns:
            point: [n,1,2] x,y的顺序，范围为[-1,1]
        """
        org_size = np.array((self.height-1, self.width-1), dtype=np.float32)
        point = ((point * 2. / org_size - 1.)[:, ::-1])[:, np.newaxis, :].copy()
        return point

    def _random_sample_point(self):
        """
        根据预设的输入图像大小，随机均匀采样坐标点
        """
        grid = self.fix_grid.copy()
        # 随机选择指定数目个格子

        point_list = []
        for i in range(grid.shape[0]):
            y_start, x_start, y_end, x_end = grid[i]
            rand_y = np.random.randint(y_start, y_end)
            rand_x = np.random.randint(x_start, x_end)
            point_list.append(np.array((rand_y, rand_x), dtype=np.float32))
        point = np.stack(point_list, axis=0)

        return point

    def _generate_fixed_grid(self, option=None):
        """
        预先采样固定间隔的225个图像格子
        """
        if option == None:
            y_num = 20
            x_num = 20
        else:
            y_num = option[0]
            x_num = option[1]

        grid_y = np.linspace(0, self.height-1, y_num+1, dtype=np.int)
        grid_x = np.linspace(0, self.width-1, x_num+1, dtype=np.int)

        grid_y_start = grid_y[:y_num].copy()
        grid_y_end = grid_y[1:y_num+1].copy()
        grid_x_start = grid_x[:x_num].copy()
        grid_x_end = grid_x[1:x_num+1].copy()

        grid_start = np.stack((np.tile(grid_y_start[:, np.newaxis], (1, x_num)),
                               np.tile(grid_x_start[np.newaxis, :], (y_num, 1))), axis=2).reshape((-1, 2))
        grid_end = np.stack((np.tile(grid_y_end[:, np.newaxis], (1, x_num)),
                             np.tile(grid_x_end[np.newaxis, :], (y_num, 1))), axis=2).reshape((-1, 2))
        grid = np.concatenate((grid_start, grid_end), axis=1)

        return grid

    def _convert_points_to_heatmap(self, points):
        """
        将原始点位置经下采样后得到heatmap与incmap，heatmap上对应下采样整型点位置处的值为1，其余为0；incmap与heatmap一一对应，
        在关键点位置处存放整型点到亚像素角点的偏移量，以及训练时用来屏蔽非关键点inc量的incmap_valid
        Args:
            points: [n,2]

        Returns:
            heatmap: [h,w] 关键点位置为1，其余为0
            incmap: [2,h,w] 关键点位置存放实际偏移，其余非关键点处的偏移量为0
            incmap_valid: [h,w] 关键点位置为1，其余为0，用于训练时屏蔽对非关键点偏移量的训练，只关注关键点的偏移量

        """
        height = self.height
        width = self.width

        # localmap = self.localmap.clone()
        # padded_heatmap = torch.zeros(
        #     (height+self.g_paddings*2, width+self.g_paddings*2), dtype=torch.float)
        heatmap = torch.zeros((height, width), dtype=torch.float)

        num_pt = points.shape[0]
        if num_pt > 0:
            for i in range(num_pt):
                pt = points[i]
                pt_y_float, pt_x_float = pt

                pt_y_int = round(pt_y_float)
                pt_x_int = round(pt_x_float)

                pt_y = int(pt_y_int)  # 对真值点位置进行下采样,这里有量化误差
                pt_x = int(pt_x_int)

                # 排除掉经下采样后在边界外的点
                if pt_y < 0 or pt_y > height - 1:
                    continue
                if pt_x < 0 or pt_x > width - 1:
                    continue

                # 关键点位置在heatmap上置1，并在incmap上记录该点离亚像素点的偏移量
                heatmap[pt_y, pt_x] = 1.0

        return heatmap

    def convert_points_to_label(self, points):

        height = self.height
        width = self.width
        n_height = int(height / 8)
        n_width = int(width / 8)
        assert n_height * 8 == height and n_width * 8 == width

        num_pt = points.shape[0]
        label = torch.zeros((height * width))
        if num_pt > 0:
            points_h, points_w = torch.split(points, 1, dim=1)
            points_idx = points_w + points_h * width
            label = label.scatter_(dim=0, index=points_idx[:, 0], value=1.0).reshape((height, width))
        else:
            label = label.reshape((height, width))

        dense_label = space_to_depth(label)
        dense_label = torch.cat((dense_label, 0.5 * torch.ones((1, n_height, n_width))), dim=0)  # [65, 30, 40]
        sparse_label = torch.argmax(dense_label, dim=0)  # [30,40]

        return sparse_label
    @staticmethod
    def _format_file_list(mega_image_dir, mega_keypoint_dir,mega_despoint_dir):
        data_list = []

        # format megadepth related list
        mega_image_list = glob(os.path.join(mega_image_dir, '*.jpg'))
        mega_image_list = sorted(mega_image_list)
        data_type = 'real'
        for img in mega_image_list:
            img_name = img.split('/')[-1].split('.')[0]
            info = os.path.join(mega_despoint_dir, img_name + '.npz')
            label = os.path.join(mega_keypoint_dir, img_name + '.npz')
            data_list.append(
                {
                    'type': data_type,
                    'image': img,
                    'info': info,
                    'label': label,
                    'name': img_name,
                }
            )

        # format coco related list
        data_type = 'synthesis'
        for img in mega_image_list:
            img_name = img.split('/')[-1].split('.')[0]
            label = os.path.join(mega_keypoint_dir, img_name + '.npz')
            info = os.path.join(mega_despoint_dir, img_name + '.npz')
            data_list.append(
                {
                    'type': data_type,
                    'image': img,
                    'info': info,
                    'label': label,
                    'name': img_name,
                }
            )

        return data_list

    @staticmethod
    def _format_file_list2(mega_image_dir, mega_keypoint_dir,mega_despoint_dir):
        data_list = []

        # format megadepth related list
        mega_image_list = glob(os.path.join(mega_image_dir, '*.jpg'))
        mega_image_list = sorted(mega_image_list)
        data_type = 'real'
        for img in mega_image_list:
            img_name = img.split('/')[-1].split('.')[0]
            info = os.path.join(mega_despoint_dir, img_name + '.npz')
            label = os.path.join(mega_keypoint_dir, img_name + '.npz')
            data_list.append(
                {
                    'type': data_type,
                    'image': img,
                    'info': info,
                    'label': label,
                    'name': img_name,
                }
            )

        # format coco related list
        data_type = 'synthesis'
        for img in mega_image_list:
            img_name = img.split('/')[-1].split('.')[0]
            label = os.path.join(mega_keypoint_dir, img_name + '.npz')
            info = os.path.join(mega_despoint_dir, img_name + '.npz')
            data_list.append(
                {
                    'type': data_type,
                    'image': img,
                    'info': info,
                    'label': label,
                    'name': img_name,
                }
            )

        return data_list

class MegaDepthTrainDataset_WithSamFeatureAndCat(Dataset):
    """
    Combination of MegaDetph and COCO
    """
    def __init__(self, **config):
        self.data_list = self._format_file_list(
            config['mega_image_dir'],
            config['mega_keypoint_dir'],
            config['mega_despoint_dir'],
        )
        self.sydesp_type=config['sydesp_type']
        self.height = config['height']
        self.width = config['width']

        self.homography = HomographyAugmentation()
        self.photometric = ImgAugTransform()
        self.fix_grid = self._generate_fixed_grid()

        self.sam_feature_root = config['sam_feature']
        #mp.set_start_method('spawn', force=True)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_info = self.data_list[idx]
        if data_info['type'] == 'synthesis':
            return self._get_synthesis_data(data_info)
        elif data_info['type'] == 'real':
            return self._get_real_data(data_info)
        else:
            assert False

    def _get_real_data(self, data_info):
        image_dir = data_info['image']
        info_dir = data_info['info']
        label_dir = data_info['label']
        name = data_info['name']
        sam_dict = np.load(os.path.join(self.sam_feature_root, name) + '.npy', allow_pickle=True).item()

        image12 = cv.imread(image_dir)[:, :, ::-1].copy()  # 交换BGR为RGB
        image1, image2 = np.split(image12, 2, axis=1)
        h, w, _ = image1.shape

        if torch.rand([]).item() < 0.5:
            image1 = self.photometric(image1)
            image2 = self.photometric(image2)

        info = np.load(info_dir)
        desp_point1 = info["desp_point1"]
        desp_point2 = info["desp_point2"]
        valid_mask = info["valid_mask"]
        not_search_mask = info["not_search_mask"]

        label = np.load(label_dir)
        points1 = label["points_0"]
        points2 = label["points_1"]

        # 2.1 得到第一副图点构成的热图
        heatmap1 = self._convert_points_to_heatmap(points1)
        point_mask1 = torch.ones_like(heatmap1)

        # 2.2 得到第二副图点构成的热图
        heatmap2 = self._convert_points_to_heatmap(points2)
        point_mask2 = torch.ones_like(heatmap2)

        # debug use
        # desp_point1 = ((desp_point1 + 1) * np.array((self.width - 1, self.height - 1)) / 2.)[:, 0, :]
        # desp_point2 = ((desp_point2 + 1) * np.array((self.width - 1, self.height - 1)) / 2.)[:, 0, :]
        # image_point1 = draw_image_keypoints(image1[:, :, ::-1], desp_point1[valid_mask][:, ::-1], show=False)
        # image_point2 = draw_image_keypoints(image2[:, :, ::-1], desp_point2[valid_mask][:, ::-1], show=False)
        # image_point1 = draw_image_keypoints(image1[:, :, ::-1], points1, show=False)
        # image_point2 = draw_image_keypoints(image2[:, :, ::-1], points2, show=False)
        # cat_all = np.concatenate((image_point1, image_point2), axis=0)
        # cv.imwrite("/home/yuyang/tmp/debug_%05d.jpg" % idx, cat_all)
        # cv.imshow("cat_all", cat_all)
        # cv.waitKey()

        image1 = (torch.from_numpy(image1).to(torch.float) * 2. / 255. - 1.).permute((2, 0, 1)).contiguous()
        image2 = (torch.from_numpy(image2).to(torch.float) * 2. / 255. - 1.).permute((2, 0, 1)).contiguous()

        desp_point1 = torch.from_numpy(desp_point1)
        desp_point2 = torch.from_numpy(desp_point2)

        valid_mask = torch.from_numpy(valid_mask).to(torch.float)
        not_search_mask = torch.from_numpy(not_search_mask).to(torch.float)

        return {
            "image": image1,
            "point_mask": point_mask1,
            "heatmap": heatmap1,
            "warped_image": image2,
            "warped_point_mask": point_mask2,
            "warped_heatmap": heatmap2,
            "desp_point": desp_point1,
            "warped_desp_point": desp_point2,
            "valid_mask": valid_mask,
            "not_search_mask": not_search_mask,
            "left_sam_feature": sam_dict['SAM_feature'],
            "right_sam_feature": sam_dict['SAM_feature_warped'],
            "left_flag": torch.tensor(1, dtype=torch.float32),
            "right_flag": torch.tensor(1, dtype=torch.float32),
            "left_sam_edge": sam_dict["SAM_edge"],
            "right_sam_edge": sam_dict["SAM_edge_warped"],
            "left_cat": sam_dict['SAM_out_cat'],
            "right_cat": sam_dict['SAM_out_warped_cat'],
        }

    def _get_synthesis_data(self, data_info):
        image12 = cv.imread(data_info['image'])[:, :, ::-1].copy()  # 交换BGR为RGB
        image1, image2 = np.split(image12, 2, axis=1)
        point = np.load(data_info['label'])
        info = np.load(data_info['info'])
        name = data_info['name']
        sam_dict = np.load(os.path.join(self.sam_feature_root, name) + '.npy', allow_pickle=True).item()

        if torch.rand([]).item() < 0.5:
            image = cv.resize(image1, dsize=(self.width, self.height), interpolation=cv.INTER_LINEAR)
            point = point["points_0"]
            desp_point_load = info["raw_desp_point1"]
            SAM_feature_Left = True
        else:
            image = cv.resize(image2, dsize=(self.width, self.height), interpolation=cv.INTER_LINEAR)
            point = point["points_1"]
            desp_point_load = info["raw_desp_point2"]
            SAM_feature_Left = False
        if SAM_feature_Left:
            SAM_Feature = sam_dict['SAM_feature']
            SAM_edge = sam_dict['SAM_edge']
            SAM_cat = sam_dict['SAM_out_cat']
        else:
            SAM_Feature = sam_dict['SAM_feature_warped']
            SAM_edge = sam_dict['SAM_edge_warped']
            SAM_cat = sam_dict['SAM_out_warped_cat']
        point_mask = np.ones_like(image).astype(np.float32)[:, :, 0].copy()

        # 1、由随机采样的单应变换得到第二副图像及其对应的关键点位置、原始掩膜和该单应变换
        if torch.rand([]).item() < 0.5:
            warped_image, warped_point_mask, warped_point, homography = \
                image.copy(), point_mask.copy(), point.copy(), np.eye(3)
            right_flag = torch.tensor(1, dtype=torch.float32)
        else:
            warped_image, warped_point_mask, warped_point, homography = self.homography(image, point, return_homo=True)
            warped_point_mask = warped_point_mask[:, :, 0].copy()
            right_flag = torch.tensor(0, dtype=torch.float32)

        if torch.rand([]).item() < 0.5:
            image = self.photometric(image)
            warped_image = self.photometric(warped_image)

        # 2.1 得到第一副图点构成的热图
        heatmap = self._convert_points_to_heatmap(point)

        # 2.2 得到第二副图点构成的热图
        warped_heatmap = self._convert_points_to_heatmap(warped_point)

        # 3、采样训练描述子要用的点
        if self.sydesp_type =='random':
            desp_point = self._random_sample_point()
        else:
            desp_point = desp_point_load

        shape = image.shape

        warped_desp_point, valid_mask, not_search_mask = self._generate_warped_point(
            desp_point, homography, shape[0], shape[1])

        # debug use
        # image_point = draw_image_keypoints(image, desp_point, show=False)
        # warped_image_point = draw_image_keypoints(warped_image, warped_desp_point, show=False)
        # cat_all = np.concatenate((image, warped_image), axis=1)
        # cat_all = np.concatenate((image_point, warped_image_point), axis=1)
        # cv.imwrite("/home/yuyang/tmp/coco_tmp/%d.jpg" % idx, cat_all)
        # cv.imshow("cat_all", cat_all)
        # cv.waitKey()

        image = image.astype(np.float32) * 2. / 255. - 1.
        warped_image = warped_image.astype(np.float32) * 2. / 255. - 1.

        image = torch.from_numpy(image).permute((2, 0, 1))
        warped_image = torch.from_numpy(warped_image).permute((2, 0, 1))

        point_mask = torch.from_numpy(point_mask)
        warped_point_mask = torch.from_numpy(warped_point_mask)

        desp_point = torch.from_numpy(self._scale_point_for_sample(desp_point))
        warped_desp_point = torch.from_numpy(self._scale_point_for_sample(warped_desp_point))

        valid_mask = torch.from_numpy(valid_mask)
        not_search_mask = torch.from_numpy(not_search_mask)

        return {
            "image": image,  # [1,h,w]
            "point_mask": point_mask,  # [h,w]
            "heatmap": heatmap,  # [h,w]
            "warped_image": warped_image,  # [1,h,w]
            "warped_point_mask": warped_point_mask,  # [h,w]
            "warped_heatmap": warped_heatmap,  # [h,w]
            "desp_point": desp_point,  # [n,1,2]
            "warped_desp_point": warped_desp_point,  # [n,1,2]
            "valid_mask": valid_mask,  # [n]
            "not_search_mask": not_search_mask,  # [n,n]
            "left_sam_feature": SAM_Feature,
            "right_sam_feature": SAM_Feature,
            "left_flag": torch.tensor(1, dtype=torch.float32),
            "right_flag": right_flag,
            "left_sam_edge": SAM_edge,
            "right_sam_edge": SAM_edge,
            "left_cat": SAM_cat,
            "right_cat": SAM_cat
        }

    @ staticmethod
    def _generate_warped_point(point, homography, height, width, threshold=16):
        """
        根据投影变换得到变换后的坐标点，有效关系及不参与负样本搜索的矩阵
        Args:
            point: [n,2] 与warped_point一一对应
            homography: 点对之间的变换关系

        Returns:
            not_search_mask: [n,n] type为float32的mask,不搜索的位置为1
        """
        # 得到投影点的坐标
        point = np.concatenate((point[:, ::-1], np.ones((point.shape[0], 1))), axis=1)[:, :, np.newaxis]  # [n,3,1]
        project_point = np.matmul(homography, point)[:, :, 0]
        project_point = project_point[:, :2] / project_point[:, 2:3]
        project_point = project_point[:, ::-1]  # 调换为y,x的顺序

        # 投影点在图像范围内的点为有效点，反之则为无效点
        boarder_0 = np.array((0, 0), dtype=np.float32)
        boarder_1 = np.array((height-1, width-1), dtype=np.float32)
        valid_mask = (project_point >= boarder_0) & (project_point <= boarder_1)
        valid_mask = np.all(valid_mask, axis=1)
        invalid_mask = ~valid_mask

        # 根据无效点及投影点之间的距离关系确定不搜索的负样本矩阵

        dist = np.linalg.norm(project_point[:, np.newaxis, :] - project_point[np.newaxis, :, :], axis=2)
        not_search_mask = ((dist <= threshold) | invalid_mask[np.newaxis, :]).astype(np.float32)
        return project_point.astype(np.float32), valid_mask.astype(np.float32), not_search_mask

    def _scale_point_for_sample(self, point):
        """
        将点归一化到[-1,1]的区间范围内，并调换顺序为x,y，方便采样
        Args:
            point: [n,2] y,x的顺序，原始范围为[0,height-1], [0,width-1]
        Returns:
            point: [n,1,2] x,y的顺序，范围为[-1,1]
        """
        org_size = np.array((self.height-1, self.width-1), dtype=np.float32)
        point = ((point * 2. / org_size - 1.)[:, ::-1])[:, np.newaxis, :].copy()
        return point

    def _random_sample_point(self):
        """
        根据预设的输入图像大小，随机均匀采样坐标点
        """
        grid = self.fix_grid.copy()
        # 随机选择指定数目个格子

        point_list = []
        for i in range(grid.shape[0]):
            y_start, x_start, y_end, x_end = grid[i]
            rand_y = np.random.randint(y_start, y_end)
            rand_x = np.random.randint(x_start, x_end)
            point_list.append(np.array((rand_y, rand_x), dtype=np.float32))
        point = np.stack(point_list, axis=0)

        return point

    def _generate_fixed_grid(self, option=None):
        """
        预先采样固定间隔的225个图像格子
        """
        if option == None:
            y_num = 20
            x_num = 20
        else:
            y_num = option[0]
            x_num = option[1]

        grid_y = np.linspace(0, self.height-1, y_num+1, dtype=np.int)
        grid_x = np.linspace(0, self.width-1, x_num+1, dtype=np.int)

        grid_y_start = grid_y[:y_num].copy()
        grid_y_end = grid_y[1:y_num+1].copy()
        grid_x_start = grid_x[:x_num].copy()
        grid_x_end = grid_x[1:x_num+1].copy()

        grid_start = np.stack((np.tile(grid_y_start[:, np.newaxis], (1, x_num)),
                               np.tile(grid_x_start[np.newaxis, :], (y_num, 1))), axis=2).reshape((-1, 2))
        grid_end = np.stack((np.tile(grid_y_end[:, np.newaxis], (1, x_num)),
                             np.tile(grid_x_end[np.newaxis, :], (y_num, 1))), axis=2).reshape((-1, 2))
        grid = np.concatenate((grid_start, grid_end), axis=1)

        return grid

    def _convert_points_to_heatmap(self, points):
        """
        将原始点位置经下采样后得到heatmap与incmap，heatmap上对应下采样整型点位置处的值为1，其余为0；incmap与heatmap一一对应，
        在关键点位置处存放整型点到亚像素角点的偏移量，以及训练时用来屏蔽非关键点inc量的incmap_valid
        Args:
            points: [n,2]

        Returns:
            heatmap: [h,w] 关键点位置为1，其余为0
            incmap: [2,h,w] 关键点位置存放实际偏移，其余非关键点处的偏移量为0
            incmap_valid: [h,w] 关键点位置为1，其余为0，用于训练时屏蔽对非关键点偏移量的训练，只关注关键点的偏移量

        """
        height = self.height
        width = self.width

        # localmap = self.localmap.clone()
        # padded_heatmap = torch.zeros(
        #     (height+self.g_paddings*2, width+self.g_paddings*2), dtype=torch.float)
        heatmap = torch.zeros((height, width), dtype=torch.float)

        num_pt = points.shape[0]
        if num_pt > 0:
            for i in range(num_pt):
                pt = points[i]
                pt_y_float, pt_x_float = pt

                pt_y_int = round(pt_y_float)
                pt_x_int = round(pt_x_float)

                pt_y = int(pt_y_int)  # 对真值点位置进行下采样,这里有量化误差
                pt_x = int(pt_x_int)

                # 排除掉经下采样后在边界外的点
                if pt_y < 0 or pt_y > height - 1:
                    continue
                if pt_x < 0 or pt_x > width - 1:
                    continue

                # 关键点位置在heatmap上置1，并在incmap上记录该点离亚像素点的偏移量
                heatmap[pt_y, pt_x] = 1.0

        return heatmap

    def convert_points_to_label(self, points):

        height = self.height
        width = self.width
        n_height = int(height / 8)
        n_width = int(width / 8)
        assert n_height * 8 == height and n_width * 8 == width

        num_pt = points.shape[0]
        label = torch.zeros((height * width))
        if num_pt > 0:
            points_h, points_w = torch.split(points, 1, dim=1)
            points_idx = points_w + points_h * width
            label = label.scatter_(dim=0, index=points_idx[:, 0], value=1.0).reshape((height, width))
        else:
            label = label.reshape((height, width))

        dense_label = space_to_depth(label)
        dense_label = torch.cat((dense_label, 0.5 * torch.ones((1, n_height, n_width))), dim=0)  # [65, 30, 40]
        sparse_label = torch.argmax(dense_label, dim=0)  # [30,40]

        return sparse_label
    @staticmethod
    def _format_file_list(mega_image_dir, mega_keypoint_dir,mega_despoint_dir):
        data_list = []

        # format megadepth related list
        mega_image_list = glob(os.path.join(mega_image_dir, '*.jpg'))
        mega_image_list = sorted(mega_image_list)
        data_type = 'real'
        for img in mega_image_list:
            img_name = img.split('/')[-1].split('.')[0]
            info = os.path.join(mega_despoint_dir, img_name + '.npz')
            label = os.path.join(mega_keypoint_dir, img_name + '.npz')
            data_list.append(
                {
                    'type': data_type,
                    'image': img,
                    'info': info,
                    'label': label,
                    'name': img_name,
                }
            )

        # format coco related list
        data_type = 'synthesis'
        for img in mega_image_list:
            img_name = img.split('/')[-1].split('.')[0]
            label = os.path.join(mega_keypoint_dir, img_name + '.npz')
            info = os.path.join(mega_despoint_dir, img_name + '.npz')
            data_list.append(
                {
                    'type': data_type,
                    'image': img,
                    'info': info,
                    'label': label,
                    'name': img_name,
                }
            )

        return data_list

    @staticmethod
    def _format_file_list2(mega_image_dir, mega_keypoint_dir,mega_despoint_dir):
        data_list = []

        # format megadepth related list
        mega_image_list = glob(os.path.join(mega_image_dir, '*.jpg'))
        mega_image_list = sorted(mega_image_list)
        data_type = 'real'
        for img in mega_image_list:
            img_name = img.split('/')[-1].split('.')[0]
            info = os.path.join(mega_despoint_dir, img_name + '.npz')
            label = os.path.join(mega_keypoint_dir, img_name + '.npz')
            data_list.append(
                {
                    'type': data_type,
                    'image': img,
                    'info': info,
                    'label': label,
                    'name': img_name,
                }
            )

        # format coco related list
        data_type = 'synthesis'
        for img in mega_image_list:
            img_name = img.split('/')[-1].split('.')[0]
            label = os.path.join(mega_keypoint_dir, img_name + '.npz')
            info = os.path.join(mega_despoint_dir, img_name + '.npz')
            data_list.append(
                {
                    'type': data_type,
                    'image': img,
                    'info': info,
                    'label': label,
                    'name': img_name,
                }
            )

        return data_list



class My_MegaDepth(Dataset):
    def __init__(self, **config):
        self.data_list = self._format_file_list(
            config['mega_image_dir'],
            config['mega_keypoint_dir'],
            config['mega_despoint_dir'],
        )
        self.sydesp_type=config['sydesp_type']
        self.height = config['height']
        self.width = config['width']

        self.homography = HomographyAugmentation()
        self.photometric = ImgAugTransform()
        self.fix_grid = self._generate_fixed_grid()

        mp.set_start_method('spawn', force=True)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_info = self.data_list[idx]
        if data_info['type'] == 'synthesis':
            return self._get_synthesis_data(data_info)
        elif data_info['type'] == 'real':
            return self._get_real_data(data_info)
        else:
            assert False

    def _remove_leading_zeros(self, num_str):
        # Remove leading zeros from the number string
        num_str = num_str.lstrip('0')

        # If the number string is empty, return '0'
        if not num_str:
            return '0'

        # If the first character is not '0', return the number string
        if num_str[0] != '0':
            return num_str

        # Otherwise, find the index of the first non-zero character
        for i in range(len(num_str)):
            if num_str[i] != '0':
                return num_str[i:]

        # If all characters are zero, return '0'
        return '0'

    def _get_real_data(self, data_info):
        name = data_info['name']
        image_dir = data_info['image']
        info_dir = data_info['info']
        label_dir = data_info['label']


        image12 = cv.imread(image_dir)[:, :, ::-1].copy()  # 交换BGR为RGB
        image1, image2 = np.split(image12, 2, axis=1)
        h, w, _ = image1.shape

        if torch.rand([]).item() < 0.5:
            image1 = self.photometric(image1)
            image2 = self.photometric(image2)

        info = np.load(info_dir)
        desp_point1 = info["desp_point1"]
        desp_point2 = info["desp_point2"]
        valid_mask = info["valid_mask"]
        not_search_mask = info["not_search_mask"]

        label = np.load(label_dir)
        points1 = label["points_0"]
        points2 = label["points_1"]

        # 2.1 得到第一副图点构成的热图
        heatmap1 = self._convert_points_to_heatmap(points1)
        #corner1 = torch.from_numpy(sam_corner_info['corner'] // 255)
        #heatmap1 = torch.logical_or(heatmap1, corner1).float()
        point_mask1 = torch.ones_like(heatmap1)


        # 2.2 得到第二副图点构成的热图
        heatmap2 = self._convert_points_to_heatmap(points2)
        #corner2 = torch.from_numpy(sam_corner_info['corner_warped'] // 255)
        #heatmap2 = torch.logical_or(heatmap2, corner2).float()
        point_mask2 = torch.ones_like(heatmap2)


        # debug use
        # desp_point1 = ((desp_point1 + 1) * np.array((self.width - 1, self.height - 1)) / 2.)[:, 0, :]
        # desp_point2 = ((desp_point2 + 1) * np.array((self.width - 1, self.height - 1)) / 2.)[:, 0, :]
        # image_point1 = draw_image_keypoints(image1[:, :, ::-1], desp_point1[valid_mask][:, ::-1], show=False)
        # image_point2 = draw_image_keypoints(image2[:, :, ::-1], desp_point2[valid_mask][:, ::-1], show=False)
        # image_point1 = draw_image_keypoints(image1[:, :, ::-1], points1, show=False)
        # image_point2 = draw_image_keypoints(image2[:, :, ::-1], points2, show=False)
        # cat_all = np.concatenate((image_point1, image_point2), axis=0)
        # cv.imwrite("/home/yuyang/tmp/debug_%05d.jpg" % idx, cat_all)
        # cv.imshow("cat_all", cat_all)
        # cv.waitKey()
        #display_image(heatmap2, image2)
        image1 = (torch.from_numpy(image1).to(torch.float) * 2. / 255. - 1.).permute((2, 0, 1)).contiguous()
        image2 = (torch.from_numpy(image2).to(torch.float) * 2. / 255. - 1.).permute((2, 0, 1)).contiguous()

        desp_point1 = torch.from_numpy(desp_point1)
        desp_point2 = torch.from_numpy(desp_point2)

        valid_mask = torch.from_numpy(valid_mask).to(torch.float)
        not_search_mask = torch.from_numpy(not_search_mask).to(torch.float)

        return {
            "image": image1,
            "point_mask": point_mask1,
            "heatmap": heatmap1,
            "warped_image": image2,
            "warped_point_mask": point_mask2,
            "warped_heatmap": heatmap2,
            "desp_point": desp_point1,
            "warped_desp_point": desp_point2,
            "valid_mask": valid_mask,
            "not_search_mask": not_search_mask,
        }

    def _get_synthesis_data(self, data_info):
        root = data_info['data_root']
        name = data_info['name']
        if torch.rand([]).item() < 0.5:
            # left
            data_folder = os.path.join(root, "left")
        else:
            data_folder = os.path.join(root, "right")

        if torch.rand([]).item() < 0.5:
            # original copy
            file_path = os.path.join(data_folder, name + ".npy")
            data = np.load(file_path, allow_pickle=True).item()
        else:
            # Homo
            ran = np.random.randint(0, 5)
            file_path = os.path.join(data_folder, name + "_" + str(ran) + ".npy")
            data = np.load(file_path, allow_pickle=True).item()
        if torch.rand([]).item() < 0.5:
            temp1 = self.photometric(((data["image"] + 1) * 255 / 2).squeeze().permute(1, 2, 0).to(torch.uint8).numpy())
            data["image"] = (torch.from_numpy(temp1).to(torch.float) * 2. / 255. - 1.).permute((2, 0, 1)).contiguous()
            temp2 = self.photometric(((data["warped_image"] + 1) * 255 / 2).squeeze().permute(1, 2, 0).to(torch.uint8).numpy())
            data["warped_image"] = (torch.from_numpy(temp2).to(torch.float) * 2. / 255. - 1.).permute((2, 0, 1)).contiguous()
        return data

    def _get_data(self, image12, point, info, left, homo):
        image1, image2 = np.split(image12, 2, axis=1)
        if left:
            image = cv.resize(image1, dsize=(self.width, self.height), interpolation=cv.INTER_LINEAR)
            point = point["points_0"]
            desp_point_load = info["raw_desp_point1"]
        else:
            image = cv.resize(image2, dsize=(self.width, self.height), interpolation=cv.INTER_LINEAR)
            point = point["points_1"]
            desp_point_load = info["raw_desp_point2"]
        point_mask = np.ones_like(image).astype(np.float32)[:, :, 0].copy()

        # 1、由随机采样的单应变换得到第二副图像及其对应的关键点位置、原始掩膜和该单应变换
        if not homo:
            warped_image, warped_point_mask, warped_point, homography = \
                image.copy(), point_mask.copy(), point.copy(), np.eye(3)
        else:
            warped_image, warped_point_mask, warped_point, homography = self.homography(image, point, return_homo=True)
            warped_point_mask = warped_point_mask[:, :, 0].copy()


        # 2.1 得到第一副图点构成的热图
        heatmap = self._convert_points_to_heatmap(point)
        # corner1 = torch.from_numpy(sam_corner_info['corner'] // 255)
        # heatmap = torch.logical_or(heatmap, corner1).float()

        # 2.2 得到第二副图点构成的热图
        warped_heatmap = self._convert_points_to_heatmap(warped_point)
        # corner2 = torch.from_numpy(sam_corner_info['corner_warped'] // 255)
        # warped_heatmap = torch.logical_or(warped_heatmap, corner2).float()
        # display_image(warped_heatmap, image2)
        # 3、采样训练描述子要用的点
        if self.sydesp_type == 'random':
            desp_point = self._random_sample_point()
        else:
            desp_point = desp_point_load

        shape = image.shape

        warped_desp_point, valid_mask, not_search_mask = self._generate_warped_point(
            desp_point, homography, shape[0], shape[1])

        # debug use
        # image_point = draw_image_keypoints(image, desp_point, show=False)
        # warped_image_point = draw_image_keypoints(warped_image, warped_desp_point, show=False)
        # cat_all = np.concatenate((image, warped_image), axis=1)
        # cat_all = np.concatenate((image_point, warped_image_point), axis=1)
        # cv.imwrite("/home/yuyang/tmp/coco_tmp/%d.jpg" % idx, cat_all)
        # cv.imshow("cat_all", cat_all)
        # cv.waitKey()

        image = image.astype(np.float32) * 2. / 255. - 1.
        warped_image = warped_image.astype(np.float32) * 2. / 255. - 1.

        image = torch.from_numpy(image).permute((2, 0, 1))
        warped_image = torch.from_numpy(warped_image).permute((2, 0, 1))

        point_mask = torch.from_numpy(point_mask)
        warped_point_mask = torch.from_numpy(warped_point_mask)

        desp_point = torch.from_numpy(self._scale_point_for_sample(desp_point))
        warped_desp_point = torch.from_numpy(self._scale_point_for_sample(warped_desp_point))

        valid_mask = torch.from_numpy(valid_mask)
        not_search_mask = torch.from_numpy(not_search_mask)

        return {
            "image": image,  # [1,h,w]
            "point_mask": point_mask,  # [h,w]
            "heatmap": heatmap,  # [h,w]
            "warped_image": warped_image,  # [1,h,w]
            "warped_point_mask": warped_point_mask,  # [h,w]
            "warped_heatmap": warped_heatmap,  # [h,w]
            "desp_point": desp_point,  # [n,1,2]
            "warped_desp_point": warped_desp_point,  # [n,1,2]
            "valid_mask": valid_mask,  # [n]
            "not_search_mask": not_search_mask,  # [n,n]
        }


    @ staticmethod
    def _generate_warped_point(point, homography, height, width, threshold=16):
        """
        根据投影变换得到变换后的坐标点，有效关系及不参与负样本搜索的矩阵
        Args:
            point: [n,2] 与warped_point一一对应
            homography: 点对之间的变换关系

        Returns:
            not_search_mask: [n,n] type为float32的mask,不搜索的位置为1
        """
        # 得到投影点的坐标
        point = np.concatenate((point[:, ::-1], np.ones((point.shape[0], 1))), axis=1)[:, :, np.newaxis]  # [n,3,1]
        project_point = np.matmul(homography, point)[:, :, 0]
        project_point = project_point[:, :2] / project_point[:, 2:3]
        project_point = project_point[:, ::-1]  # 调换为y,x的顺序

        # 投影点在图像范围内的点为有效点，反之则为无效点
        boarder_0 = np.array((0, 0), dtype=np.float32)
        boarder_1 = np.array((height-1, width-1), dtype=np.float32)
        valid_mask = (project_point >= boarder_0) & (project_point <= boarder_1)
        valid_mask = np.all(valid_mask, axis=1)
        invalid_mask = ~valid_mask

        # 根据无效点及投影点之间的距离关系确定不搜索的负样本矩阵

        dist = np.linalg.norm(project_point[:, np.newaxis, :] - project_point[np.newaxis, :, :], axis=2)
        not_search_mask = ((dist <= threshold) | invalid_mask[np.newaxis, :]).astype(np.float32)
        return project_point.astype(np.float32), valid_mask.astype(np.float32), not_search_mask

    def _scale_point_for_sample(self, point):
        """
        将点归一化到[-1,1]的区间范围内，并调换顺序为x,y，方便采样
        Args:
            point: [n,2] y,x的顺序，原始范围为[0,height-1], [0,width-1]
        Returns:
            point: [n,1,2] x,y的顺序，范围为[-1,1]
        """
        org_size = np.array((self.height-1, self.width-1), dtype=np.float32)
        point = ((point * 2. / org_size - 1.)[:, ::-1])[:, np.newaxis, :].copy()
        return point

    def _random_sample_point(self):
        """
        根据预设的输入图像大小，随机均匀采样坐标点
        """
        grid = self.fix_grid.copy()
        # 随机选择指定数目个格子

        point_list = []
        for i in range(grid.shape[0]):
            y_start, x_start, y_end, x_end = grid[i]
            rand_y = np.random.randint(y_start, y_end)
            rand_x = np.random.randint(x_start, x_end)
            point_list.append(np.array((rand_y, rand_x), dtype=np.float32))
        point = np.stack(point_list, axis=0)

        return point

    def _generate_fixed_grid(self, option=None):
        """
        预先采样固定间隔的225个图像格子
        """
        if option == None:
            y_num = 20
            x_num = 20
        else:
            y_num = option[0]
            x_num = option[1]

        grid_y = np.linspace(0, self.height-1, y_num+1, dtype=np.int)
        grid_x = np.linspace(0, self.width-1, x_num+1, dtype=np.int)

        grid_y_start = grid_y[:y_num].copy()
        grid_y_end = grid_y[1:y_num+1].copy()
        grid_x_start = grid_x[:x_num].copy()
        grid_x_end = grid_x[1:x_num+1].copy()

        grid_start = np.stack((np.tile(grid_y_start[:, np.newaxis], (1, x_num)),
                               np.tile(grid_x_start[np.newaxis, :], (y_num, 1))), axis=2).reshape((-1, 2))
        grid_end = np.stack((np.tile(grid_y_end[:, np.newaxis], (1, x_num)),
                             np.tile(grid_x_end[np.newaxis, :], (y_num, 1))), axis=2).reshape((-1, 2))
        grid = np.concatenate((grid_start, grid_end), axis=1)

        return grid

    def _convert_points_to_heatmap(self, points):
        """
        将原始点位置经下采样后得到heatmap与incmap，heatmap上对应下采样整型点位置处的值为1，其余为0；incmap与heatmap一一对应，
        在关键点位置处存放整型点到亚像素角点的偏移量，以及训练时用来屏蔽非关键点inc量的incmap_valid
        Args:
            points: [n,2]

        Returns:
            heatmap: [h,w] 关键点位置为1，其余为0
            incmap: [2,h,w] 关键点位置存放实际偏移，其余非关键点处的偏移量为0
            incmap_valid: [h,w] 关键点位置为1，其余为0，用于训练时屏蔽对非关键点偏移量的训练，只关注关键点的偏移量

        """
        height = self.height
        width = self.width

        # localmap = self.localmap.clone()
        # padded_heatmap = torch.zeros(
        #     (height+self.g_paddings*2, width+self.g_paddings*2), dtype=torch.float)
        heatmap = torch.zeros((height, width), dtype=torch.float)

        num_pt = points.shape[0]
        if num_pt > 0:
            for i in range(num_pt):
                pt = points[i]
                pt_y_float, pt_x_float = pt

                pt_y_int = round(pt_y_float)
                pt_x_int = round(pt_x_float)

                pt_y = int(pt_y_int)  # 对真值点位置进行下采样,这里有量化误差
                pt_x = int(pt_x_int)

                # 排除掉经下采样后在边界外的点
                if pt_y < 0 or pt_y > height - 1:
                    continue
                if pt_x < 0 or pt_x > width - 1:
                    continue

                # 关键点位置在heatmap上置1，并在incmap上记录该点离亚像素点的偏移量
                heatmap[pt_y, pt_x] = 1.0

        return heatmap

    def convert_points_to_label(self, points):

        height = self.height
        width = self.width
        n_height = int(height / 8)
        n_width = int(width / 8)
        assert n_height * 8 == height and n_width * 8 == width

        num_pt = points.shape[0]
        label = torch.zeros((height * width))
        if num_pt > 0:
            points_h, points_w = torch.split(points, 1, dim=1)
            points_idx = points_w + points_h * width
            label = label.scatter_(dim=0, index=points_idx[:, 0], value=1.0).reshape((height, width))
        else:
            label = label.reshape((height, width))

        dense_label = space_to_depth(label)
        dense_label = torch.cat((dense_label, 0.5 * torch.ones((1, n_height, n_width))), dim=0)  # [65, 30, 40]
        sparse_label = torch.argmax(dense_label, dim=0)  # [30,40]

        return sparse_label
    @staticmethod
    def _format_file_list(mega_image_dir, mega_keypoint_dir, mega_despoint_dir):
        data_list = []

        # format megadepth related list
        mega_image_list = glob(os.path.join(mega_image_dir, '*.jpg'))
        mega_image_list = sorted(mega_image_list)
        data_type = 'real'
        for img in mega_image_list:
            img_name = img.split('/')[-1].split('.')[0]
            info = os.path.join(mega_despoint_dir, img_name + '.npz')
            label = os.path.join(mega_keypoint_dir, img_name + '.npz')
            data_list.append(
                {
                    'type': data_type,
                    'image': img,
                    'info': info,
                    'label': label,
                    'name': img_name,
                }
            )

        # format coco related list
        data_type = 'synthesis'
        my_mega_root = "/media/data/wjq/My_Mega/synthesis"
        for img in mega_image_list:
            img_name = img.split('/')[-1].split('.')[0]
            data_list.append(
                {
                    'type': data_type,
                    'name': img_name,
                    'data_root': my_mega_root
                }
            )

        return data_list

    def display_image(self, image):
        # Display the image
        plt.imshow(image)
        plt.axis('off')
        plt.show()

    def pinpoint_points_on_image(self, image, coordinates):
        fig, ax = plt.subplots()
        ax.imshow(image)
        ax.imshow(coordinates, cmap='jet', alpha=0.7, interpolation='bilinear', vmin=0, vmax=1)
        plt.show()

    def compare_ori_warped(self, data):
        image = data['image']
        image = ((image + 1) * 255 / 2).squeeze().permute(1, 2, 0).to(torch.uint8).numpy()
        self.display_image(image)
        warped_image = data['warped_image']
        warped_image = ((warped_image + 1) * 255 / 2).squeeze().permute(1, 2, 0).to(torch.uint8).numpy()
        self.display_image(warped_image)

    def plot_points_on_image(self, data):
        image = data['image']
        coordinates = data['heatmap']
        image = ((image + 1) * 255 / 2).squeeze().permute(1, 2, 0).to(torch.uint8).numpy()
        coordinates = coordinates.squeeze().numpy()


        self.pinpoint_points_on_image(image, coordinates)

        image = data['warped_image']
        coordinates = data['warped_heatmap']
        image = ((image + 1) * 255 / 2).squeeze().permute(1, 2, 0).to(torch.uint8).numpy()
        coordinates = coordinates.squeeze().numpy()
        self.pinpoint_points_on_image(image, coordinates)


    def pad_integer_string(self, filename):
        padded_integer = filename.zfill(7)
        return padded_integer

    def vis_mega_image(self, name):
        name = self.pad_integer_string(name)
        ori_img = '/media/data/wjq/Mega_train/image'
        img_path = os.path.join(ori_img, name) + '.jpg'
        img = mpimg.imread(img_path)
        self.display_image(img)

