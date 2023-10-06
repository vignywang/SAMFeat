#
# Created  on 2020/2/23
#
import os
import glob
import cv2 as cv
import numpy as np
from torch.utils.data import Dataset


class HPatchDataset(Dataset):

    def __init__(self, **configs):
        default_config = {
            'dataset_dir': '',
            'grayscale': False,
            'resize': False,
            'height': 240,
            'width': 320,
        }
        default_config.update(configs)

        self.hpatch_height = default_config['height']
        self.hpatch_width = default_config['width']
        self.resize = default_config['resize']
        if default_config['dataset_dir'] == '':
            assert False
        self.dataset_dir = default_config['dataset_dir']
        self.grayscale = default_config['grayscale']

        self.data_list = self._format_file_list()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        first_image_dir = self.data_list[idx]['first']
        second_image_dir = self.data_list[idx]['second']
        homo_dir = self.data_list[idx]['homo_dir']
        image_type = self.data_list[idx]['type']

        if self.grayscale:
            first_image = cv.imread(first_image_dir, cv.IMREAD_GRAYSCALE)
            second_image = cv.imread(second_image_dir, cv.IMREAD_GRAYSCALE)
        else:
            first_image = cv.imread(first_image_dir)[:, :, ::-1].copy()  # convert bgr to rgb
            second_image = cv.imread(second_image_dir)[:, :, ::-1].copy()
        homo = np.loadtxt(homo_dir, dtype=np.float)

        org_first_shape = [np.shape(first_image)[0], np.shape(first_image)[1]]
        org_second_shape = [np.shape(second_image)[0], np.shape(second_image)[1]]
        if self.resize:
            resize_shape = np.array((self.hpatch_height, self.hpatch_width), dtype=np.float)
            first_scale = resize_shape / org_first_shape
            second_scale = resize_shape / org_second_shape
            homo = self._generate_adjust_homography(first_scale, second_scale, homo)

            first_image = cv.resize(first_image, (self.hpatch_width, self.hpatch_height), interpolation=cv.INTER_LINEAR)
            second_image = cv.resize(second_image, (self.hpatch_width, self.hpatch_height),
                                     interpolation=cv.INTER_LINEAR)

            first_shape = resize_shape
            second_shape = resize_shape
        else:
            first_shape = np.array(org_first_shape)
            second_shape = np.array(org_second_shape)
        # scale is used to recover the location in original image scale

        sample = {
            'first_image': first_image, 'second_image': second_image,
            'image_type': image_type, 'gt_homography': homo,
            'first_shape': first_shape, 'second_shape': second_shape,
        }
        return sample

    @staticmethod
    def _generate_adjust_homography(first_scale, second_scale, homography):
        first_inv_scale_mat = np.diag((1. / first_scale[1], 1. / first_scale[0], 1))
        second_scale_mat = np.diag((second_scale[1], second_scale[0], 1))
        adjust_homography = np.matmul(second_scale_mat, np.matmul(homography, first_inv_scale_mat))
        return adjust_homography

    def _format_file_list(self):
        data_list = []
        with open(os.path.join(self.dataset_dir, 'illumination_list.txt'), 'r') as ilf:
            illumination_lines = ilf.readlines()
            for line in illumination_lines:
                line = line[:-1]
                first_dir, second_dir, homo_dir = line.split(',')
                dir_slice = {'first': first_dir, 'second': second_dir, 'homo_dir': homo_dir, 'type': 'illumination'}
                data_list.append(dir_slice)

        with open(os.path.join(self.dataset_dir, 'viewpoint_list.txt'), 'r') as vf:
            viewpoint_lines = vf.readlines()
            for line in viewpoint_lines:
                line = line[:-1]
                first_dir, second_dir, homo_dir = line.split(',')
                dir_slice = {'first': first_dir, 'second': second_dir, 'homo_dir': homo_dir, 'type': 'viewpoint'}
                data_list.append(dir_slice)

        return data_list


class OrgHPatchDataset(Dataset):

    def __init__(self, **configs):
        default_config = {
            'dataset_dir': '',
            'grayscale': False,
            'resize': False,
            'height': 240,
            'width': 320,
        }
        default_config.update(configs)

        self.hpatch_height = default_config['height']
        self.hpatch_width = default_config['width']
        self.resize = default_config['resize']
        if default_config['dataset_dir'] == '':
            assert False
        self.dataset_dir = default_config['dataset_dir']
        self.grayscale = default_config['grayscale']

        self.data_list = self._format_file_list()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        image_dir = self.data_list[idx]['image_dir']
        image_name = self.data_list[idx]['image_name']
        folder_name = self.data_list[idx]['folder_name']

        if self.grayscale:
            image = cv.imread(image_dir, cv.IMREAD_GRAYSCALE)
        else:
            image = cv.imread(image_dir)[:, :, ::-1].copy()  # onvert bgr to rgb

        if self.resize:
            image = cv.resize(image, (self.hpatch_width, self.hpatch_height), interpolation=cv.INTER_LINEAR)

        sample = {
            'image': image,
            'image_name': image_name,
            'folder_name': folder_name,
        }

        return sample

    def _format_file_list(self):
        data_list = []
        folder_list = os.listdir(self.dataset_dir)
        for folder in folder_list:
            images = glob.glob(os.path.join(self.dataset_dir, folder, "*.ppm"))
            images = sorted(images)
            for image in images:
                image_name = image.split('/')[-1].split('.')[0]
                data_list.append(
                    {
                        'image_dir': image,
                        'image_name': image_name,
                        'folder_name': folder,
                    }
                )

        return data_list


if __name__ == "__main__":
    # # uncomment to generate the data list
    # hpatch_dir = '/data/MegPoint/dataset/hpatch'
    # generate_hpatch_data_list(hpatch_dir)

    pass



