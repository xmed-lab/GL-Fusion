import os
import random
from monai.data import DataLoader
import monai.transforms
import numpy as np

import torch
import torch.nn as nn

import nibabel as nib


from collections import defaultdict
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

from monai.transforms import (
    AddChanneld,
    Compose,
    LoadImaged,
    RandSpatialCropd,
    CenterSpatialCropd,
    EnsureChannelFirstd,
    RandFlipd,
    Resized,
    ScaleIntensityRangePercentilesd,
    NormalizeIntensityd,
    Identity,
    EnsureTyped
)
np.set_printoptions(threshold=np.inf)
random.seed(6666)
np.random.seed(6666)

class PAHDataset(Dataset):
    def __init__(self, infos, root, is_train, data_list=None, set_select=['gy'], view_num=['1', '4'], spectrum_num=None, label_type='mPAP', transform=None):
        self.rort = root
        self.is_train = is_train
        self.set_select = set_select
        self.view_num = view_num
        self.spectrum_num = spectrum_num
        self.transform = self.get_transform(is_train, self.view_num)
        self.transform_spectrum = None if self.spectrum_num is None else self.get_transform(is_train, self.spectrum_num, single_frame=True)

        self.data_dict, self.queen = self.get_dict(infos)
        self.id_list = list(self.data_dict.keys())

        if is_train:
            self.train_list = random.sample(self.id_list, int(len(self.id_list) * 0.9))
            self.valid_list = random.sample(self.train_list, int(len(self.train_list) * 0.1))
            self.test_list = list(set(self.id_list).difference(set(self.train_list)))
            self.id_list = self.train_list
        elif (is_train is False and data_list is not None):
            self.id_list = data_list

    def __getitem__(self, index):

        def get_info_dict(id):
            current_image_dict = dict()
            current_spectrum_dict = dict()

            images = self.data_dict[id]['images']

            for k, v in images.items():
                if k in self.view_num and v is not None:
                    current_image_dict[k] = v

                if self.spectrum_num is not None:    
                    if k in self.spectrum_num and v is not None:
                        current_spectrum_dict[k] = v

            mPAP = self.data_dict[id]['mPAP']

            while (not current_image_dict and not np.isnan(mPAP)):
                current_image_dict, current_spectrum_dict, mPAP = get_info_dict(random.choice(self.id_list))

            return current_image_dict, current_spectrum_dict, mPAP

        current_image_dict, current_spectrum_dict, mPAP = get_info_dict(self.id_list[index])

        if np.isnan(mPAP):
            current_image_dict, current_spectrum_dict, mPAP = get_info_dict(random.choice(self.id_list))

        input_data = self.transform(current_image_dict)
        if self.spectrum_num is not None:
            input_spectrum = self.transform_spectrum(current_spectrum_dict)

        input_views, empty_frames = None, torch.zeros((1, 112, 112, 48))
        for key in self.view_num:
            if key not in input_data.keys():
                if input_views is None:
                    input_views = empty_frames
                else:
                    input_views = torch.cat((input_views, empty_frames))
            else:
                if input_views is None:
                    input_views = input_data[key]
                else:
                    input_views = torch.cat((input_views, input_data[key]))

        if self.spectrum_num is not None:
            for key in self.spectrum_num:
                if key not in input_spectrum.keys():
                    input_spectrum[key] = torch.zeros((1, 112, 112))
                elif input_spectrum[key] is None:
                    input_spectrum[key] = torch.zeros((1, 112, 112))

            input_spectrum = torch.cat([input_spectrum[key] for key in self.spectrum_num])

            return input_views / 255.0, input_spectrum / 255.0, mPAP

        else:
            return input_views / 255.0, mPAP

    def __len__(self):
        return len(self.id_list)

    def is_number(self, s):
        try:
            float(s)
            return True
        except ValueError:
            pass
     
        try:
            import unicodedata
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass
     
        return False

    def get_dict(self, infos):
        queen = list()
        selected_dict = dict()

        for k, v in infos.items():
            if v['dataset_name'] in self.set_select:
                if np.isnan(v['mPAP']):
                    continue
                else:
                    selected_dict[k] = {}
                    selected_dict[k]['images'] = v['views_images']
                    selected_dict[k]['masks'] = v['views_labels']
                    selected_dict[k]['fold'] = v['fold']
                    selected_dict[k]['mPAP'] = v['mPAP']
                    selected_dict[k]['Vmax'] = v['Vmax']
                    selected_dict[k]['dataset_name'] = v['dataset_name']
                    queen.append(v['mPAP'])

        return selected_dict, queen

    def get_transform(self, is_train, view_type, single_frame=False):
        all_keys = view_type
        spatial_size = (144, 144) if single_frame else (144, 144, 48)
        crop_size    = (112, 112) if single_frame else (112, 112, 48)
        if is_train:
            
            rf0 = RandFlipd(keys=all_keys, prob=0.5, spatial_axis=0)
            rf1 = RandFlipd(keys=all_keys, prob=0.5, spatial_axis=1)
            rf2 = RandFlipd(keys=all_keys, prob=0.5, spatial_axis=2)

            rf0.set_random_state(0)
            rf1.set_random_state(0)
            rf2.set_random_state(0)

            transform = Compose([
                    LoadImaged(keys=all_keys, allow_missing_keys=True),
                    AddChanneld(keys=all_keys, allow_missing_keys=True),
                    Resized(keys=all_keys, spatial_size=spatial_size, allow_missing_keys=True,mode='nearest'),
                    RandSpatialCropd(all_keys, crop_size, random_size=False, allow_missing_keys=True),
                    #ScaleIntensityRangePercentilesd(keys=all_keys, lower=5, upper=95, b_min=0., b_max=1., allow_missing_keys=True) if len(all_keys)>0 else Identity(),
                    #NormalizeIntensityd(keys=all_keys, subtrahend=0.5, divisor=0.5),
                    EnsureTyped(keys=all_keys, allow_missing_keys=True),
                ])
        else:
            transform = Compose([
                    LoadImaged(keys=all_keys, allow_missing_keys=True),
                    AddChanneld(keys=all_keys, allow_missing_keys=True),
                    Resized(keys=all_keys, spatial_size=spatial_size, allow_missing_keys=True,mode='nearest'),
                    RandSpatialCropd(all_keys, crop_size, random_size=False, allow_missing_keys=True),
                    #ScaleIntensityRangePercentilesd(keys=all_keys, lower=5, upper=95, b_min=0., b_max=1., allow_missing_keys=True) if len(all_keys)>0 else Identity(),
                    #NormalizeIntensityd(keys=all_keys, subtrahend=0.5, divisor=0.5),
                    EnsureTyped(keys=all_keys, allow_missing_keys=True),
                ])
        return transform


class Seg_PAHDataset(Dataset):
    def __init__(self, infos, root, is_train, data_list=None, set_select=['rmyy'], view_num=['2'], label_type='mPAP',
                 transform=None, single_frame=True, clip_length=32, seg_parts=True,require_id=False,is_unlab=False,all_mask_frames=False,gen_pseudo_label=False):
        self.rort = root
        self.is_train = is_train
        self.set_select = set_select
        self.view_num = view_num
        self.single_frame = single_frame
        self.require_id = require_id
        self.clip_length = clip_length
        self.seg_parts = seg_parts
        self.all_mask_frames = all_mask_frames
        self.transform = self.get_transform(is_train)
        self.is_unlab = is_unlab
        self.gen_pseudo_label = gen_pseudo_label
        self.data_dict = self.get_dict(infos)
        self.id_list = list(self.data_dict.keys())
        # self.time_list = self.data_dict['time']
        if data_list is not None:
            self.id_list = data_list
        elif is_train:
            # self.train_list = random.sample(self.id_list, int(len(self.id_list) * 0.9))
            # self.valid_list = random.sample(self.train_list, int(len(self.train_list) * 0.1))
            # self.test_list = list(set(self.id_list).difference(set(self.train_list)))
            # self.id_list = self.train_list
            self.train_list = random.sample(self.id_list, int(len(self.id_list) * 0.8))
            self.valid_list = random.sample(list(set(self.id_list).difference(set(self.train_list))),int(len(list(set(self.id_list).difference(set(self.train_list))))*0.5))
            self.test_list = list(set(self.id_list).difference(set(self.train_list)).difference(set(self.valid_list)))
            self.id_list = self.train_list




        # elif (is_train is False and data_list is not None):
        #     self.id_list = data_list


    def __getitem__(self, index):
        def get_info_dict(id):
            current_input_dir = dict()
            images = self.data_dict[id]['images']
            masks  = self.data_dict[id]['masks']
            for k in self.view_num:
                if (k in images.keys() and k in masks.keys()):
                    if (images[k] is not None and masks[k] is not None):
                        # print(id)
                        # print(images[k])
                        images = np.array(nib.load(images[k]).dataobj)#800*600*172
                        masks  = np.array(nib.load(masks[k]).dataobj)
                        # if self.single_frame == False:
                        #     select_images_, select_masks_, index = self.input_select(images, masks)
                        #     current_input_dir[k] = self.transform({'images': select_images_, 'masks': select_masks_})

                        if self.all_mask_frames==True:
                            select_images_, select_masks_, index = self.input_select(images, masks)
                            # print(select_images_.shape)
                            # print(select_masks_.shape)
                            frame_num = select_images_.shape[-1]


                            current_input_dir[k] = [self.transform({'images': select_images_[...,i], 'masks': select_masks_[...,i]}) for i in range(frame_num)]
                            current_input_dir[k] = {'images':torch.cat([tmp['images'].unsqueeze(3) for tmp in current_input_dir[k]],axis=-1),
                                                    'masks': torch.cat([tmp['masks'].unsqueeze(3) for tmp in current_input_dir[k]],axis=-1)}
                            # print(current_input_dir[k]['images'].shape)
                            current_input_dir[k] = {
                                'images': monai.data.MetaTensor(current_input_dir[k]['images']),
                                'masks': current_input_dir[k]['masks'].float()}
                        elif self.gen_pseudo_label:
                            self.clip_length = masks.shape[-1]
                            self.transform = self.get_transform(self.is_train)
                            current_input_dir[k] = self.transform({'images': images, 'masks': masks})
                        elif self.is_unlab==False:
                            select_images_, select_masks_, index = self.input_select(images, masks)
                            current_input_dir[k] = self.transform({'images':select_images_, 'masks':select_masks_})
                        else:
                            select_images_, select_masks_, index = images, masks,0
                            current_input_dir[k] = {'images':monai.data.MetaTensor(torch.from_numpy(select_images_).unsqueeze(0)), 'masks':torch.from_numpy(select_masks_).float()}
                    else:
                        if self.single_frame==True:
                            index = 0
                            current_input_dir[k] = {
                                'images': monai.data.MetaTensor(torch.zeros([1,112,112])),
                                'masks': torch.zeros([1,112,112]).float()}
                        elif self.all_mask_frames==True:
                            index = 0
                            current_input_dir[k] = {
                                'images': monai.data.MetaTensor(torch.zeros([1, 112, 112, 4])),
                                'masks': torch.zeros([1, 112, 112, 4]).float()}
                        else:
                            index = 0
                            current_input_dir[k] = {
                                'images': monai.data.MetaTensor(torch.zeros([1, 112, 112,self.clip_length])),
                                'masks': torch.zeros([1, 112, 112,self.clip_length]).float()}

            while not current_input_dir:
                current_input_dir, index = get_info_dict(random.choice(self.id_list))

            return current_input_dir, index

        if self.is_unlab and self.is_train and self.gen_pseudo_label==False:
            id = self.id_list[index // 24]
        elif self.is_train and self.gen_pseudo_label==False:
            id = self.id_list[index // 4]
        else:
            id = self.id_list[index]

        current_input_dir, index = get_info_dict(id)

        if self.seg_parts and self.is_unlab==False:
            if self.view_num   == ['1']:
                LV = torch.where(current_input_dir[self.view_num[0]]['masks'] == 1, 1, 0)
                RV = torch.where(current_input_dir[self.view_num[0]]['masks'] == 2, 1, 0)
                masks = torch.cat([LV, RV], dim=0)
            elif self.view_num == ['2']:
                PA = torch.where(current_input_dir[self.view_num[0]]['masks'] == 1, 1, 0)
                masks = PA
            elif self.view_num == ['3']:
                LV = torch.where(current_input_dir[self.view_num[0]]['masks'] == 1, 1, 0)
                RV = torch.where(current_input_dir[self.view_num[0]]['masks'] == 2, 1, 0)
                masks = torch.cat([LV, RV], dim=0)
            elif self.view_num == ['4']:
                LV = torch.where(current_input_dir[self.view_num[0]]['masks'] == 1, 1, 0)
                LA = torch.where(current_input_dir[self.view_num[0]]['masks'] == 2, 1, 0)
                RA = torch.where(current_input_dir[self.view_num[0]]['masks'] == 3, 1, 0)
                RV = torch.where(current_input_dir[self.view_num[0]]['masks'] == 4, 1, 0)
                masks = torch.cat([LV, LA, RA, RV], dim=0)
            masks = self.mask_to_allclass(masks, self.view_num[0])
        elif self.seg_parts and self.is_unlab==True:
            masks = current_input_dir[self.view_num[0]]['masks']

        else:
            masks = torch.where(current_input_dir[self.view_num[0]]['masks'] > 0, 1, 0)
        if self.is_unlab == True:
            images = current_input_dir[self.view_num[0]]['images']
        else:
            images = current_input_dir[self.view_num[0]]['images'] / 255.0

        if self.require_id == True:
            return images, masks, index,id
        else:
            return images, masks, index

    def __len__(self):
        if self.all_mask_frames or self.gen_pseudo_label:
            return len(self.id_list)
        if self.is_unlab and self.is_train:
            return len(self.id_list) * 24
        elif self.is_train:
            return len(self.id_list) * 4
        else:
            return len(self.id_list)

    def is_number(self, s):
        try:
            float(s)
            return True
        except ValueError:
            pass
     
        try:
            import unicodedata
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass
     
        return False

    def mask_to_allclass(self,masks,view):

        if len(masks.shape)==3:
            _, h, w = masks.shape
            if view == '1':
                # 胸骨旁左室长轴切面
                tmp_mask = torch.zeros(5,h,w)
                tmp_mask[1] = masks[1]
                tmp_mask[3] = masks[0]

            if view == '2':
                # 肺动脉长轴切面
                tmp_mask = torch.zeros(5, h, w)
                tmp_mask[4] = masks[0]

            if view == '3':
                # 左室短轴切面
                tmp_mask = torch.zeros(5, h, w)
                tmp_mask[1] = masks[1]
                tmp_mask[3] = masks[0]

            if view == '4':
                # 心尖四腔心切面
                tmp_mask = torch.zeros(5, h, w)
                tmp_mask[0] = masks[2]
                tmp_mask[1] = masks[3]
                tmp_mask[2] = masks[1]
                tmp_mask[3] = masks[0]
        else:
            _, h, w, l = masks.shape
            if view == '1':
                # 胸骨旁左室长轴切面

                tmp_mask = torch.zeros(5, h, w,l)
                tmp_mask[1] = masks[1]
                tmp_mask[3] = masks[0]

            if view == '2':
                # 肺动脉长轴切面
                tmp_mask = torch.zeros(5, h, w,l)
                tmp_mask[4] = masks[0]

            if view == '3':
                # 左室短轴切面
                tmp_mask = torch.zeros(5, h, w,l)
                tmp_mask[1] = masks[1]
                tmp_mask[3] = masks[0]

            if view == '4':
                # 心尖四腔心切面
                tmp_mask = torch.zeros(5, h, w,l)
                tmp_mask[0] = masks[2]
                tmp_mask[1] = masks[3]
                tmp_mask[2] = masks[1]
                tmp_mask[3] = masks[0]

        return tmp_mask

    def get_dict(self, infos):
        selected_dict = dict()

        for k, v in infos.items():
            if v['dataset_name'] in self.set_select:
                selected_dict[k] = {}
                selected_dict[k]['images'] = v['views_images']
                selected_dict[k]['masks']  = v['views_labels']
                selected_dict[k]['fold'] = v['fold']
                # selected_dict[k]['mPAP'] = v['mPAP']
                # selected_dict[k]['Vmax'] = v['Vmax']
                selected_dict[k]['dataset_name'] = v['dataset_name']

        return selected_dict 

    def input_select(self, images, masks):
        if len(masks.shape) >= 3:#800 * 600 * 127
            if self.is_unlab==False:
                mask_frames_ = np.sum(masks, axis=(0,1))
            else:
                mask_frames_ = np.sum(masks, axis=(0, 1, 2))
            mask_frames_ = np.where(mask_frames_ > 100, 1, 0)
            mask_frames_ = np.argwhere(mask_frames_ == 1)
            # print(mask_frames_.shape)
            # print(random.choice(mask_frames_))
            index = random.choice(mask_frames_)[0]
            np.squeeze(mask_frames_)
            mask_frames_ = list(mask_frames_)
            
            if self.single_frame and self.all_mask_frames==False:
                return images[:, :, index], masks[..., index], index
                # return images[:, :, index], masks[:, :, index], index
            elif self.all_mask_frames:
                return images[:, :, mask_frames_].squeeze(-1), masks[..., mask_frames_].squeeze(-1), mask_frames_
            else:
                if self.gen_pseudo_label:
                    self.clip_length = masks.shape[-1]
                elif masks.shape[-1] == 3:
                    images = np.tile(images[:, :, 1:2],(1,1,self.clip_length))
                    masks  = np.tile(masks[:, :, 1:2], (1,1,self.clip_length))
                else:     
                    r_index = random.randint(0, index if index < self.clip_length-1 else self.clip_length-1)
                    start = index - r_index
                    end = start + self.clip_length - 1
                    images = images[:, :, start:end]
                    # masks  = masks[:, :, start:end]
                    masks = masks[..., start:end]
                    index = r_index
                # print(self.clip_length)
                return images, masks, index
        else:
            if self.single_frame:
                return images, masks, 0
            else:
                return np.tile(images,(self.clip_length,1,1)).transpose(1,2,0), np.tile(masks,(self.clip_length,1,1)).transpose(1,2,0), 0

    def get_transform(self, is_train):
        all_keys = ['images', 'masks']
        crop_size = (112, 112) if self.single_frame or self.all_mask_frames else (112, 112, self.clip_length)
        spatial_size = (144, 144) if self.single_frame or self.all_mask_frames else (144, 144, self.clip_length)
        # crop_size = (84, 84) if self.single_frame or self.all_mask_frames else (84, 84, self.clip_length)
        # spatial_size = (112, 112) if self.single_frame or self.all_mask_frames else (112, 112, self.clip_length)
        # crop_size = (42, 42) if self.single_frame else (42, 42, self.clip_length)
        # spatial_size = (56, 56) if self.single_frame else (56, 56, self.clip_length)
        if is_train:
            
            rf0 = RandFlipd(keys=all_keys, prob=0.5, spatial_axis=0)
            rf1 = RandFlipd(keys=all_keys, prob=0.5, spatial_axis=1)
            rf2 = RandFlipd(keys=all_keys, prob=0.5, spatial_axis=2) if not self.single_frame else None

            rf0.set_random_state(0)
            rf1.set_random_state(0)
            if rf2 is not None:
                rf2.set_random_state(0)

            transform = Compose([
                    AddChanneld(keys=all_keys, allow_missing_keys=True),
                    Resized(keys=all_keys, spatial_size=spatial_size, allow_missing_keys=True,mode='nearest'),
                    # CenterSpatialCropd(all_keys, crop_size, allow_missing_keys=True),
                    RandSpatialCropd(all_keys, crop_size, random_size=False, allow_missing_keys=True),
                    #ScaleIntensityRangePercentilesd(keys=all_keys, lower=5, upper=95, b_min=0., b_max=1., allow_missing_keys=True) if len(all_keys)>0 else Identity(),
                    #NormalizeIntensityd(keys=all_keys, subtrahend=0.5, divisor=0.5),
                    EnsureTyped(keys=all_keys, allow_missing_keys=True),
                ])
        else:
            transform = Compose([
                    AddChanneld(keys=all_keys, allow_missing_keys=True),
                    Resized(keys=all_keys, spatial_size=spatial_size, allow_missing_keys=True,mode='nearest'),
                    CenterSpatialCropd(all_keys, crop_size,  allow_missing_keys=True),
                    # RandSpatialCropd(all_keys, crop_size, random_size=False, allow_missing_keys=True),
                    #ScaleIntensityRangePercentilesd(keys=all_keys, lower=5, upper=95, b_min=0., b_max=1., allow_missing_keys=True) if len(all_keys)>0 else Identity(),
                    #NormalizeIntensityd(keys=all_keys, subtrahend=0.5, divisor=0.5),
                    EnsureTyped(keys=all_keys, allow_missing_keys=True),
                ])
        return transform


class Couple_Seg_PAHDataset(Dataset):
    def __init__(self, infos, root, is_train, data_list=None, set_select=['gy'], view_num=['2'], label_type='mPAP',
                 transform=None, single_frame=True, clip_length=32, seg_parts=True, require_id=False, is_unlab=False):
        self.rort = root
        self.is_train = is_train
        self.set_select = set_select
        self.view_num = view_num
        self.single_frame = single_frame
        self.require_id = require_id
        self.clip_length = clip_length
        self.seg_parts = seg_parts
        self.transform = self.get_transform(is_train)
        self.is_unlab = is_unlab
        self.data_dict = self.get_dict(infos)
        self.id_list = list(self.data_dict.keys())
        # self.time_list = self.data_dict['time']
        if data_list is not None:
            self.id_list = data_list
        elif is_train:
            self.train_list = random.sample(self.id_list, int(len(self.id_list) * 0.9))
            self.valid_list = random.sample(self.train_list, int(len(self.train_list) * 0.1))
            self.test_list = list(set(self.id_list).difference(set(self.train_list)))
            self.id_list = self.train_list
        # elif (is_train is False and data_list is not None):
        #     self.id_list = data_list

    def __getitem__(self, index):
        def get_info_dict(id):
            current_input_dir = dict()
            images = self.data_dict[id]['images']
            masks = self.data_dict[id]['masks']
            for k in self.view_num:
                if (k in images.keys() and k in masks.keys()):
                    if (images[k] is not None and masks[k] is not None):
                        image = np.array(nib.load(images[k]).dataobj)  # 800*600*172
                        mask = np.array(nib.load(masks[k]).dataobj)
                        if self.is_unlab == False:
                            select_images_, select_masks_, index = self.input_select(image, mask)
                            current_input_dir[k] = self.transform({'images': select_images_, 'masks': select_masks_})
                        else:
                            select_images_, select_masks_, index = image, mask, 0
                            current_input_dir[k] = {
                                'images': monai.data.MetaTensor(torch.from_numpy(select_images_).unsqueeze(0)),
                                'masks': torch.from_numpy(select_masks_).float()}

            while not current_input_dir:
                current_input_dir, index = get_info_dict(random.choice(self.id_list))

            return current_input_dir, index

        id = self.id_list[index]

        current_input_dir, index = get_info_dict(id)
        images = {}
        masks = {}
        for view in self.view_num:
            images[view] = current_input_dir[view]['images']
            masks[view] = current_input_dir[view]['masks']

        if self.require_id == True:
            return images, masks, index, id
        else:
            return images, masks, index

    def __len__(self):

            return len(self.id_list)

    def is_number(self, s):
        try:
            float(s)
            return True
        except ValueError:
            pass

        try:
            import unicodedata
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass

        return False

    def mask_to_allclass(self, masks, view):

        if len(masks.shape) == 3:
            _, h, w = masks.shape
            if view == '1':
                # 胸骨旁左室长轴切面
                tmp_mask = torch.zeros(5, h, w)
                tmp_mask[1] = masks[1]
                tmp_mask[3] = masks[0]

            if view == '2':
                # 肺动脉长轴切面
                tmp_mask = torch.zeros(5, h, w)
                tmp_mask[4] = masks[0]

            if view == '3':
                # 左室短轴切面
                tmp_mask = torch.zeros(5, h, w)
                tmp_mask[1] = masks[1]
                tmp_mask[3] = masks[0]

            if view == '4':
                # 心尖四腔心切面
                tmp_mask = torch.zeros(5, h, w)
                tmp_mask[0] = masks[2]
                tmp_mask[1] = masks[3]
                tmp_mask[2] = masks[1]
                tmp_mask[3] = masks[0]
        else:
            _, h, w, l = masks.shape
            if view == '1':
                # 胸骨旁左室长轴切面

                tmp_mask = torch.zeros(5, h, w, l)
                tmp_mask[1] = masks[1]
                tmp_mask[3] = masks[0]

            if view == '2':
                # 肺动脉长轴切面
                tmp_mask = torch.zeros(5, h, w, l)
                tmp_mask[4] = masks[0]

            if view == '3':
                # 左室短轴切面
                tmp_mask = torch.zeros(5, h, w, l)
                tmp_mask[1] = masks[1]
                tmp_mask[3] = masks[0]

            if view == '4':
                # 心尖四腔心切面
                tmp_mask = torch.zeros(5, h, w, l)
                tmp_mask[0] = masks[2]
                tmp_mask[1] = masks[3]
                tmp_mask[2] = masks[1]
                tmp_mask[3] = masks[0]

        return tmp_mask

    def get_dict(self, infos):
        selected_dict = dict()

        for k, v in infos.items():
            if v['dataset_name'] in self.set_select:
                selected_dict[k] = {}
                selected_dict[k]['images'] = v['views_images']
                selected_dict[k]['masks'] = v['views_labels']
                selected_dict[k]['fold'] = v['fold']
                selected_dict[k]['mPAP'] = v['mPAP']
                selected_dict[k]['Vmax'] = v['Vmax']
                selected_dict[k]['dataset_name'] = v['dataset_name']

        return selected_dict

    def input_select(self, images, masks):
        if len(masks.shape) >= 3:  # 800 * 600 * 127
            if self.is_unlab == False:
                mask_frames_ = np.sum(masks, axis=(0, 1))
            else:
                mask_frames_ = np.sum(masks, axis=(0, 1, 2))
            mask_frames_ = np.where(mask_frames_ > 100, 1, 0)
            mask_frames_ = np.argwhere(mask_frames_ == 1)
            index = random.choice(mask_frames_)[0]

            if self.single_frame:
                return images[:, :, index], masks[..., index], index
                # return images[:, :, index], masks[:, :, index], index
            else:
                if masks.shape[-1] == 3:
                    images = np.tile(images[:, :, 1:2], (1, 1, self.clip_length))
                    masks = np.tile(masks[:, :, 1:2], (1, 1, self.clip_length))
                else:
                    r_index = random.randint(0, index if index < self.clip_length - 1 else self.clip_length - 1)
                    start = index - r_index
                    end = start + self.clip_length - 1
                    images = images[:, :, start:end]
                    # masks  = masks[:, :, start:end]
                    masks = masks[..., start:end]
                    index = r_index

                return images, masks, index
        else:
            if self.single_frame:
                return images, masks, 0
            else:
                return np.tile(images, (self.clip_length, 1, 1)).transpose(1, 2, 0), np.tile(masks, (
                self.clip_length, 1, 1)).transpose(1, 2, 0), 0

    def get_transform(self, is_train):
        all_keys = ['images', 'masks']
        crop_size = (112, 112) if self.single_frame else (112, 112, self.clip_length)
        spatial_size = (144, 144) if self.single_frame else (144, 144, self.clip_length)
        # crop_size = (84, 84) if self.single_frame else (84, 84, self.clip_length)
        # spatial_size = (112, 112) if self.single_frame else (112, 112, self.clip_length)
        # crop_size = (42, 42) if self.single_frame else (42, 42, self.clip_length)
        # spatial_size = (56, 56) if self.single_frame else (56, 56, self.clip_length)
        if is_train:

            rf0 = RandFlipd(keys=all_keys, prob=0.5, spatial_axis=0)
            rf1 = RandFlipd(keys=all_keys, prob=0.5, spatial_axis=1)
            rf2 = RandFlipd(keys=all_keys, prob=0.5, spatial_axis=2) if not self.single_frame else None

            rf0.set_random_state(0)
            rf1.set_random_state(0)
            if rf2 is not None:
                rf2.set_random_state(0)

            transform = Compose([
                AddChanneld(keys=all_keys, allow_missing_keys=True),
                Resized(keys=all_keys, spatial_size=spatial_size, allow_missing_keys=True,mode='nearest'),
                CenterSpatialCropd(all_keys, crop_size, allow_missing_keys=True),
                # RandSpatialCropd(all_keys, crop_size, random_size=False, allow_missing_keys=True),
                # ScaleIntensityRangePercentilesd(keys=all_keys, lower=5, upper=95, b_min=0., b_max=1., allow_missing_keys=True) if len(all_keys)>0 else Identity(),
                # NormalizeIntensityd(keys=all_keys, subtrahend=0.5, divisor=0.5),
                EnsureTyped(keys=all_keys, allow_missing_keys=True),
            ])
        else:
            transform = Compose([
                AddChanneld(keys=all_keys, allow_missing_keys=True),
                Resized(keys=all_keys, spatial_size=spatial_size, allow_missing_keys=True,mode='nearest'),
                CenterSpatialCropd(all_keys, crop_size, allow_missing_keys=True),
                # RandSpatialCropd(all_keys, crop_size, random_size=False, allow_missing_keys=True),
                # ScaleIntensityRangePercentilesd(keys=all_keys, lower=5, upper=95, b_min=0., b_max=1., allow_missing_keys=True) if len(all_keys)>0 else Identity(),
                # NormalizeIntensityd(keys=all_keys, subtrahend=0.5, divisor=0.5),
                EnsureTyped(keys=all_keys, allow_missing_keys=True),
            ])
        return transform


class Align_Seg_PAHDataset(Dataset):
    def __init__(self, infos, root, is_train, data_list=None, set_select=['rmyy'], view_num=['2'], label_type='mPAP',
                 transform=None, single_frame=True, clip_length=32, seg_parts=True, require_id=False, is_unlab=False,
                 all_mask_frames=False, gen_pseudo_label=False):
        self.rort = root
        self.is_train = is_train
        self.set_select = set_select
        self.view_num = view_num
        self.single_frame = single_frame
        self.require_id = require_id
        self.clip_length = clip_length
        self.seg_parts = seg_parts
        self.all_mask_frames = all_mask_frames
        self.transform = self.get_transform(is_train)
        self.is_unlab = is_unlab
        self.gen_pseudo_label = gen_pseudo_label
        self.data_dict = self.get_dict(infos)
        self.id_list = list(self.data_dict.keys())
        # self.time_list = self.data_dict['time']
        if data_list is not None:
            self.id_list = data_list
        elif is_train:

            self.train_list = random.sample(self.id_list, int(len(self.id_list) * 0.8))
            self.valid_list = random.sample(list(set(self.id_list).difference(set(self.train_list))),
                                            int(len(list(set(self.id_list).difference(set(self.train_list)))) * 0.5))
            self.test_list = list(set(self.id_list).difference(set(self.train_list)).difference(set(self.valid_list)))
            self.id_list = self.train_list

    def __getitem__(self, index):
        def get_info_dict(id):
            current_input_dir = dict()
            images = self.data_dict[id]['images']
            masks = self.data_dict[id]['masks']
            for k in self.view_num:
                if (k in images.keys() and k in masks.keys()):
                    if (images[k] is not None and masks[k] is not None):
                        # print(id)
                        # print(images[k])
                        images = np.array(nib.load(images[k]).dataobj)  # 800*600*172
                        masks = np.array(nib.load(masks[k]).dataobj)
                        index = 0
                        if self.is_unlab==False:
                            self.clip_length = masks.shape[-1]
                            self.transform = self.get_transform(is_train=False)
                            current_input_dir[k] = self.transform({'images': images, 'masks': masks})
                        else:
                            current_input_dir[k] = {
                                'images': monai.data.MetaTensor(torch.from_numpy(images).unsqueeze(0)),
                                'masks': torch.from_numpy(masks).float()}
                            # current_input_dir[k] = {'images': images, 'masks': masks}

            while not current_input_dir:
                current_input_dir, index = get_info_dict(random.choice(self.id_list))

            return current_input_dir, index

        id = self.id_list[index]
        current_input_dir, index = get_info_dict(id)

        if self.seg_parts and self.is_unlab == False:
            if self.view_num == ['1']:
                LV = torch.where(current_input_dir[self.view_num[0]]['masks'] == 1, 1, 0)
                RV = torch.where(current_input_dir[self.view_num[0]]['masks'] == 2, 1, 0)
                masks = torch.cat([LV, RV], dim=0)
            elif self.view_num == ['2']:
                PA = torch.where(current_input_dir[self.view_num[0]]['masks'] == 1, 1, 0)
                masks = PA
            elif self.view_num == ['3']:
                LV = torch.where(current_input_dir[self.view_num[0]]['masks'] == 1, 1, 0)
                RV = torch.where(current_input_dir[self.view_num[0]]['masks'] == 2, 1, 0)
                masks = torch.cat([LV, RV], dim=0)
            elif self.view_num == ['4']:
                LV = torch.where(current_input_dir[self.view_num[0]]['masks'] == 1, 1, 0)
                LA = torch.where(current_input_dir[self.view_num[0]]['masks'] == 2, 1, 0)
                RA = torch.where(current_input_dir[self.view_num[0]]['masks'] == 3, 1, 0)
                RV = torch.where(current_input_dir[self.view_num[0]]['masks'] == 4, 1, 0)
                masks = torch.cat([LV, LA, RA, RV], dim=0)
            masks = self.mask_to_allclass(masks, self.view_num[0])
        elif self.seg_parts and self.is_unlab == True:
            masks = current_input_dir[self.view_num[0]]['masks']

        else:
            masks = torch.where(current_input_dir[self.view_num[0]]['masks'] > 0, 1, 0)
        if self.is_unlab == True:
            images = current_input_dir[self.view_num[0]]['images']
        else:
            images = current_input_dir[self.view_num[0]]['images'] / 255.0

        if self.require_id == True:
            return images, masks, index, id
        else:
            return images, masks, index

    def __len__(self):
        return len(self.id_list)

    def is_number(self, s):
        try:
            float(s)
            return True
        except ValueError:
            pass

        try:
            import unicodedata
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass

        return False

    def mask_to_allclass(self, masks, view):

        if len(masks.shape) == 3:
            _, h, w = masks.shape
            if view == '1':
                # 胸骨旁左室长轴切面
                tmp_mask = torch.zeros(5, h, w)
                tmp_mask[1] = masks[1]
                tmp_mask[3] = masks[0]

            if view == '2':
                # 肺动脉长轴切面
                tmp_mask = torch.zeros(5, h, w)
                tmp_mask[4] = masks[0]

            if view == '3':
                # 左室短轴切面
                tmp_mask = torch.zeros(5, h, w)
                tmp_mask[1] = masks[1]
                tmp_mask[3] = masks[0]

            if view == '4':
                # 心尖四腔心切面
                tmp_mask = torch.zeros(5, h, w)
                tmp_mask[0] = masks[2]
                tmp_mask[1] = masks[3]
                tmp_mask[2] = masks[1]
                tmp_mask[3] = masks[0]
        else:
            _, h, w, l = masks.shape
            if view == '1':
                # 胸骨旁左室长轴切面

                tmp_mask = torch.zeros(5, h, w, l)
                tmp_mask[1] = masks[1]
                tmp_mask[3] = masks[0]

            if view == '2':
                # 肺动脉长轴切面
                tmp_mask = torch.zeros(5, h, w, l)
                tmp_mask[4] = masks[0]

            if view == '3':
                # 左室短轴切面
                tmp_mask = torch.zeros(5, h, w, l)
                tmp_mask[1] = masks[1]
                tmp_mask[3] = masks[0]

            if view == '4':
                # 心尖四腔心切面
                tmp_mask = torch.zeros(5, h, w, l)
                tmp_mask[0] = masks[2]
                tmp_mask[1] = masks[3]
                tmp_mask[2] = masks[1]
                tmp_mask[3] = masks[0]

        return tmp_mask

    def get_dict(self, infos):
        selected_dict = dict()

        for k, v in infos.items():
            if v['dataset_name'] in self.set_select:
                selected_dict[k] = {}
                selected_dict[k]['images'] = v['views_images']
                selected_dict[k]['masks'] = v['views_labels']
                selected_dict[k]['fold'] = v['fold']
                selected_dict[k]['dataset_name'] = v['dataset_name']
        return selected_dict

    def get_transform(self, is_train):
        all_keys = ['images', 'masks']
        crop_size = (112, 112) if self.single_frame else (112, 112, self.clip_length)
        spatial_size = (144, 144) if self.single_frame else (144, 144, self.clip_length)
        if is_train:

            rf0 = RandFlipd(keys=all_keys, prob=0.5, spatial_axis=0)
            rf1 = RandFlipd(keys=all_keys, prob=0.5, spatial_axis=1)
            rf2 = RandFlipd(keys=all_keys, prob=0.5, spatial_axis=2) if not self.single_frame else None

            rf0.set_random_state(0)
            rf1.set_random_state(0)
            if rf2 is not None:
                rf2.set_random_state(0)

            transform = Compose([
                AddChanneld(keys=all_keys, allow_missing_keys=True),
                Resized(keys=all_keys, spatial_size=spatial_size, allow_missing_keys=True, mode='nearest'),
                # CenterSpatialCropd(all_keys, crop_size, allow_missing_keys=True),
                RandSpatialCropd(all_keys, crop_size, random_size=False, allow_missing_keys=True),
                # ScaleIntensityRangePercentilesd(keys=all_keys, lower=5, upper=95, b_min=0., b_max=1., allow_missing_keys=True) if len(all_keys)>0 else Identity(),
                # NormalizeIntensityd(keys=all_keys, subtrahend=0.5, divisor=0.5),
                EnsureTyped(keys=all_keys, allow_missing_keys=True),
            ])
        else:
            transform = Compose([
                AddChanneld(keys=all_keys, allow_missing_keys=True),
                Resized(keys=all_keys, spatial_size=spatial_size, allow_missing_keys=True, mode='nearest'),
                CenterSpatialCropd(all_keys, crop_size, allow_missing_keys=True),
                # RandSpatialCropd(all_keys, crop_size, random_size=False, allow_missing_keys=True),
                # ScaleIntensityRangePercentilesd(keys=all_keys, lower=5, upper=95, b_min=0., b_max=1., allow_missing_keys=True) if len(all_keys)>0 else Identity(),
                # NormalizeIntensityd(keys=all_keys, subtrahend=0.5, divisor=0.5),
                EnsureTyped(keys=all_keys, allow_missing_keys=True),
            ])
        return transform

class Aligned_Video_Seg_PAHDataset(Dataset):
    def __init__(self, infos, root, is_train, data_list=None, set_select=['rmyy'], view_num=['2'], label_type='mPAP',
                 transform=None, single_frame=True, clip_length=32, seg_parts=True, require_id=False, is_unlab=False,
                 all_mask_frames=False, gen_pseudo_label=False,random_sample=False):
        self.rort = root
        self.is_train = is_train
        self.set_select = set_select
        self.view_num = view_num
        self.single_frame = single_frame
        self.require_id = require_id
        self.clip_length = clip_length
        self.random_sample = random_sample
        self.seg_parts = seg_parts
        self.all_mask_frames = all_mask_frames
        self.is_unlab = is_unlab
        self.gen_pseudo_label = gen_pseudo_label
        self.data_dict = self.get_dict(infos)
        self.id_list = list(self.data_dict.keys())
        # self.time_list = self.data_dict['time']
        if data_list is not None:
            self.id_list = data_list
        elif is_train:

            self.train_list = random.sample(self.id_list, int(len(self.id_list) * 0.8))
            self.valid_list = random.sample(list(set(self.id_list).difference(set(self.train_list))),
                                            int(len(list(set(self.id_list).difference(set(self.train_list)))) * 0.5))
            self.test_list = list(set(self.id_list).difference(set(self.train_list)).difference(set(self.valid_list)))
            self.id_list = self.train_list

    def __getitem__(self, index):
        def get_info_dict(id):
            current_input_dir = dict()
            images = self.data_dict[id]['images']
            masks = self.data_dict[id]['masks']
            for k in self.view_num:
                if (k in images.keys() and k in masks.keys()):
                    if (images[k] is not None and masks[k] is not None):
                        if self.random_sample:
                            images = np.array(nib.load(images[k]).dataobj)  # 800*600*172
                            masks = np.array(nib.load(masks[k]).dataobj)
                        else:
                            images = np.array(nib.load(images[k]).dataobj).squeeze(-1)  # 800*600*172
                            masks = np.array(nib.load(masks[k]).dataobj).squeeze(-1)
                        index = 0
                        if images.shape[-1]>self.clip_length:
                            if self.random_sample:
                                start = random.randint(0,images.shape[-1]-self.clip_length-1)
                                end = start + self.clip_length
                                images = images[:, :, start:end]
                                masks = masks[..., start:end]
                            else:
                                images = images[:,:,:self.clip_length]
                                masks = masks[:,:,:,:self.clip_length]
                        elif images.shape[-1]<self.clip_length:
                            images = np.concatenate([images,images],axis=2)
                            masks = np.concatenate([masks, masks], axis=3)
                        current_input_dir[k] = {
                            # 'images': monai.data.MetaTensor(torch.from_numpy(images).unsqueeze(0)),
                            'images': torch.from_numpy(images).unsqueeze(0),
                            'masks': torch.from_numpy(masks).float()}

            while not current_input_dir:
                current_input_dir, index = get_info_dict(random.choice(self.id_list))

            return current_input_dir, index

        id = self.id_list[index]
        current_input_dir, index = get_info_dict(id)

        images = current_input_dir[self.view_num[0]]['images']
        masks = current_input_dir[self.view_num[0]]['masks']
        if self.require_id == True:
            return images, masks, index, id
        else:
            return images, masks, index

    def __len__(self):
        return len(self.id_list)

    def is_number(self, s):
        try:
            float(s)
            return True
        except ValueError:
            pass

        try:
            import unicodedata
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass

        return False

    def get_dict(self, infos):
        selected_dict = dict()

        for k, v in infos.items():
            if v['dataset_name'] in self.set_select:
                selected_dict[k] = {}
                selected_dict[k]['images'] = v['views_images']
                selected_dict[k]['masks'] = v['views_labels']
                selected_dict[k]['fold'] = v['fold']
                selected_dict[k]['dataset_name'] = v['dataset_name']
        return selected_dict


class Test_Seg_PAHDataset(Dataset):
    def __init__(self, infos, root=None, is_train=False, data_list=None, set_select=['gy'], view_num=['2'], label_type='mPAP',
                 transform=None, single_frame=False, clip_length=32, seg_parts=True, require_id=False, is_unlab=False,
                 all_mask_frames=False):
        self.rort = root
        self.is_train = is_train
        self.set_select = set_select
        self.view_num = view_num
        self.single_frame = single_frame
        self.require_id = require_id
        self.clip_length = clip_length
        self.seg_parts = seg_parts
        self.all_mask_frames = all_mask_frames
        self.transform = self.get_transform(is_train)
        self.is_unlab = is_unlab

        self.data_dict = self.get_dict(infos)
        if data_list is not None:
            self.id_list = data_list
        else:
            self.id_list = list(self.data_dict.keys())


    def __getitem__(self, index):
        def get_info_dict(id):
            index = 0
            current_input_dir = dict()
            images = self.data_dict[id]['images']
            masks = self.data_dict[id]['masks']
            for k in self.view_num:
                if (k in images.keys() and k in masks.keys()):
                    if (images[k] is not None and masks[k] is not None):
                        images = np.array(nib.load(images[k]).dataobj)  # 800*600*172
                        masks = np.array(nib.load(masks[k]).dataobj)
                        # print(images.shape)
                        # print(masks.shape)
                        # select_images_, select_masks_, index = self.input_select(images, masks)

                        # current_input_dir[k] = {'images': torch.tensor(select_images_).unsqueeze(0), 'masks': torch.tensor(select_masks_).unsqueeze(0)}
                        # current_input_dir[k] = self.transform({'images': select_images_, 'masks': select_masks_})
                        current_input_dir[k] ={'images': images, 'masks': masks}
                    else:
                        if self.single_frame == True:
                            index = 0
                            current_input_dir[k] = {
                                'images': monai.data.MetaTensor(torch.zeros([1, 112, 112])),
                                'masks': torch.zeros([5, 112, 112]).float()}
                        else:
                            index = 0
                            current_input_dir[k] = {
                                'images': monai.data.MetaTensor(torch.zeros([1, 112, 112, self.clip_length])),
                                'masks': torch.zeros([5, 112, 112, self.clip_length]).float()}

            while not current_input_dir:
                current_input_dir, index = get_info_dict(random.choice(self.id_list))

            return current_input_dir, index


        id = self.id_list[index]

        current_input_dir, index = get_info_dict(id)

        # if self.seg_parts and self.is_unlab == False:
        #     if self.view_num == ['1']:
        #         LV = torch.where(current_input_dir[self.view_num[0]]['masks'] == 1, 1, 0)
        #         RV = torch.where(current_input_dir[self.view_num[0]]['masks'] == 2, 1, 0)
        #         masks = torch.cat([LV, RV], dim=0)
        #     elif self.view_num == ['2']:
        #         PA = torch.where(current_input_dir[self.view_num[0]]['masks'] == 1, 1, 0)
        #         masks = PA
        #     elif self.view_num == ['3']:
        #         LV = torch.where(current_input_dir[self.view_num[0]]['masks'] == 1, 1, 0)
        #         RV = torch.where(current_input_dir[self.view_num[0]]['masks'] == 2, 1, 0)
        #         masks = torch.cat([LV, RV], dim=0)
        #     elif self.view_num == ['4']:
        #         LV = torch.where(current_input_dir[self.view_num[0]]['masks'] == 1, 1, 0)
        #         LA = torch.where(current_input_dir[self.view_num[0]]['masks'] == 2, 1, 0)
        #         RA = torch.where(current_input_dir[self.view_num[0]]['masks'] == 3, 1, 0)
        #         RV = torch.where(current_input_dir[self.view_num[0]]['masks'] == 4, 1, 0)
        #
        #         masks = torch.cat([LV, LA, RA, RV], dim=0)
        #     masks = self.mask_to_allclass(masks, self.view_num[0])
        # elif self.seg_parts and self.is_unlab == True:
        #     masks = current_input_dir[self.view_num[0]]['masks']
        #
        # else:
        #     masks = torch.where(current_input_dir[self.view_num[0]]['masks'] > 0, 1, 0)
        # if self.is_unlab == True:
        #     images = current_input_dir[self.view_num[0]]['images']
        # else:
        #     images = current_input_dir[self.view_num[0]]['images'] / 255.0
        masks = current_input_dir[self.view_num[0]]['masks']
        images = current_input_dir[self.view_num[0]]['images'] / 255.0
        if self.require_id == True:
            return images, masks, index, id
        else:
            return images, masks, index

    def __len__(self):
        if self.all_mask_frames:
            return len(self.id_list)
        if self.is_unlab and self.is_train:
            return len(self.id_list) * 24
        elif self.is_train:
            return len(self.id_list) * 4
        else:
            return len(self.id_list)

    def is_number(self, s):
        try:
            float(s)
            return True
        except ValueError:
            pass

        try:
            import unicodedata
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass

        return False

    def mask_to_allclass(self, masks, view):

        if len(masks.shape) == 3:
            _, h, w = masks.shape
            if view == '1':
                # 胸骨旁左室长轴切面
                tmp_mask = torch.zeros(5, h, w)
                tmp_mask[1] = masks[1]
                tmp_mask[3] = masks[0]

            if view == '2':
                # 肺动脉长轴切面
                tmp_mask = torch.zeros(5, h, w)
                tmp_mask[4] = masks[0]

            if view == '3':
                # 左室短轴切面
                tmp_mask = torch.zeros(5, h, w)
                tmp_mask[1] = masks[1]
                tmp_mask[3] = masks[0]

            if view == '4':
                # 心尖四腔心切面
                tmp_mask = torch.zeros(5, h, w)
                tmp_mask[0] = masks[2]
                tmp_mask[1] = masks[3]
                tmp_mask[2] = masks[1]
                tmp_mask[3] = masks[0]
        else:
            _, h, w, l = masks.shape
            if view == '1':
                # 胸骨旁左室长轴切面
                tmp_mask = torch.zeros(5, h, w, l)
                tmp_mask[1] = masks[1]
                tmp_mask[3] = masks[0]

            if view == '2':
                # 肺动脉长轴切面
                tmp_mask = torch.zeros(5, h, w, l)
                tmp_mask[4] = masks[0]

            if view == '3':
                # 左室短轴切面
                tmp_mask = torch.zeros(5, h, w, l)
                tmp_mask[1] = masks[1]
                tmp_mask[3] = masks[0]

            if view == '4':
                # 心尖四腔心切面
                tmp_mask = torch.zeros(5, h, w, l)
                tmp_mask[0] = masks[2]
                tmp_mask[1] = masks[3]
                tmp_mask[2] = masks[1]
                tmp_mask[3] = masks[0]

        return tmp_mask

    def get_dict(self, infos):
        selected_dict = dict()

        for k, v in infos.items():
            if v['dataset_name'] in self.set_select:
                selected_dict[k] = {}
                selected_dict[k]['images'] = v['views_images']
                selected_dict[k]['masks'] = v['views_labels']
                selected_dict[k]['fold'] = v['fold']
                selected_dict[k]['mPAP'] = v['mPAP']
                selected_dict[k]['Vmax'] = v['Vmax']
                selected_dict[k]['dataset_name'] = v['dataset_name']

        return selected_dict

    def input_select(self, images, masks):
        if len(masks.shape) >= 3:  # 800 * 600 * 127
            mask_frames_ = np.sum(masks, axis=(0, 1, 2))
            mask_frames_ = np.where(mask_frames_ > 100, 1, 0)
            mask_frames_ = np.argwhere(mask_frames_ == 1)
            index = random.choice(mask_frames_)[0]
            np.squeeze(mask_frames_)
            # mask_frames_ = list(mask_frames_)

            if self.single_frame:
                return images[:, :, index], masks[..., index], index
            else:
                if masks.shape[-1] == 3:
                    images = np.tile(images[:, :, 1:2], (1, 1, self.clip_length))
                    masks = np.tile(masks[:, :, 1:2], (1, 1, self.clip_length))
                else:
                    r_index = random.randint(0, index if index < self.clip_length - 1 else self.clip_length - 1)
                    start = index - r_index
                    end = start + self.clip_length - 1
                    images = images[:, :, start:end]
                    # masks  = masks[:, :, start:end]
                    masks = masks[..., start:end]
                    index = r_index

                return images, masks, index
        else:
            if self.single_frame:
                return images, masks, 0
            else:
                return np.tile(images, (self.clip_length, 1, 1)).transpose(1, 2, 0), np.tile(masks, (
                self.clip_length, 1, 1)).transpose(1, 2, 0), 0

    def get_transform(self, is_train):
        all_keys = ['images', 'masks']
        # crop_size = (112, 112) if self.single_frame else (112, 112, self.clip_length)
        # spatial_size = (144, 144) if self.single_frame else (144, 144, self.clip_length)
        crop_size = (467, 467) if self.single_frame else (467, 467, self.clip_length)
        spatial_size = (600, 600) if self.single_frame else (600, 600, self.clip_length)
        if is_train:

            rf0 = RandFlipd(keys=all_keys, prob=0.5, spatial_axis=0)
            rf1 = RandFlipd(keys=all_keys, prob=0.5, spatial_axis=1)
            rf2 = RandFlipd(keys=all_keys, prob=0.5, spatial_axis=2) if not self.single_frame else None

            rf0.set_random_state(0)
            rf1.set_random_state(0)
            if rf2 is not None:
                rf2.set_random_state(0)

            transform = Compose([
                AddChanneld(keys=all_keys, allow_missing_keys=True),
                Resized(keys=all_keys, spatial_size=spatial_size, allow_missing_keys=True,mode='nearest'),
                # CenterSpatialCropd(all_keys, crop_size, allow_missing_keys=True),
                RandSpatialCropd(all_keys, crop_size, random_size=False, allow_missing_keys=True),
                # ScaleIntensityRangePercentilesd(keys=all_keys, lower=5, upper=95, b_min=0., b_max=1., allow_missing_keys=True) if len(all_keys)>0 else Identity(),
                # NormalizeIntensityd(keys=all_keys, subtrahend=0.5, divisor=0.5),
                EnsureTyped(keys=all_keys, allow_missing_keys=True),
            ])
        else:
            transform = Compose([
                AddChanneld(keys=all_keys, allow_missing_keys=True),
                Resized(keys=all_keys, spatial_size=spatial_size, allow_missing_keys=True,mode='nearest'),
                CenterSpatialCropd(all_keys, crop_size, allow_missing_keys=True),
                # RandSpatialCropd(all_keys, crop_size, random_size=False, allow_missing_keys=True),
                # ScaleIntensityRangePercentilesd(keys=all_keys, lower=5, upper=95, b_min=0., b_max=1., allow_missing_keys=True) if len(all_keys)>0 else Identity(),
                # NormalizeIntensityd(keys=all_keys, subtrahend=0.5, divisor=0.5),
                EnsureTyped(keys=all_keys, allow_missing_keys=True),
            ])
        return transform


class Seg_PAHDataset_all_mask(Dataset):
    def __init__(self, infos, root, is_train, data_list=None, set_select=['rmyy'], view_num=['2'], label_type='mPAP',
                 transform=None, single_frame=True, clip_length=5, seg_parts=True, require_id=False, is_unlab=False,
                 all_mask_frames=False, gen_pseudo_label=False):
        self.rort = root
        self.is_train = is_train
        self.set_select = set_select
        self.view_num = view_num
        self.single_frame = single_frame
        self.require_id = require_id
        self.clip_length = clip_length
        self.seg_parts = seg_parts
        self.all_mask_frames = all_mask_frames
        self.transform = self.get_transform(is_train)
        self.is_unlab = is_unlab
        self.gen_pseudo_label = gen_pseudo_label
        self.data_dict = self.get_dict(infos)
        self.id_list = data_list

    def __getitem__(self, index):
        def get_info_dict(id):
            current_input_dir = dict()
            images = self.data_dict[id]['images']
            masks = self.data_dict[id]['masks']
            for k in self.view_num:
                if (k in images.keys() and k in masks.keys()):
                    if (images[k] is not None and masks[k] is not None):
                        images = np.array(nib.load(images[k]).dataobj)  # 800*600*172
                        masks = np.array(nib.load(masks[k]).dataobj)
                        select_images_, select_masks_, index = self.input_select(images, masks)
                        frame_num = select_images_.shape[-1]
                        current_input_dir[k] = self.transform({'images': select_images_, 'masks': select_masks_})
                        current_input_dir[k] = {
                            'images': monai.data.MetaTensor(current_input_dir[k]['images']),
                            'masks': current_input_dir[k]['masks'].float()}

            while not current_input_dir:
                current_input_dir, index = get_info_dict(random.choice(self.id_list))

            return current_input_dir, index

        id = self.id_list[index]

        current_input_dir, index = get_info_dict(id)

        if self.seg_parts and self.is_unlab == False:
            if self.view_num == ['1']:
                LV = torch.where(current_input_dir[self.view_num[0]]['masks'] == 1, 1, 0)
                RV = torch.where(current_input_dir[self.view_num[0]]['masks'] == 2, 1, 0)
                masks = torch.cat([LV, RV], dim=0)
            elif self.view_num == ['2']:
                PA = torch.where(current_input_dir[self.view_num[0]]['masks'] == 1, 1, 0)
                masks = PA
            elif self.view_num == ['3']:
                LV = torch.where(current_input_dir[self.view_num[0]]['masks'] == 1, 1, 0)
                RV = torch.where(current_input_dir[self.view_num[0]]['masks'] == 2, 1, 0)
                masks = torch.cat([LV, RV], dim=0)
            elif self.view_num == ['4']:
                LV = torch.where(current_input_dir[self.view_num[0]]['masks'] == 1, 1, 0)
                LA = torch.where(current_input_dir[self.view_num[0]]['masks'] == 2, 1, 0)
                RA = torch.where(current_input_dir[self.view_num[0]]['masks'] == 3, 1, 0)
                RV = torch.where(current_input_dir[self.view_num[0]]['masks'] == 4, 1, 0)
                masks = torch.cat([LV, LA, RA, RV], dim=0)
            masks = self.mask_to_allclass(masks, self.view_num[0])
        elif self.seg_parts and self.is_unlab == True:
            masks = current_input_dir[self.view_num[0]]['masks']

        else:
            masks = torch.where(current_input_dir[self.view_num[0]]['masks'] > 0, 1, 0)
        if self.is_unlab == True:
            images = current_input_dir[self.view_num[0]]['images']
        else:
            images = current_input_dir[self.view_num[0]]['images'] / 255.0

        if self.require_id == True:
            return images, masks, index, id
        else:
            return images, masks, index

    def __len__(self):

        return len(self.id_list)

    def is_number(self, s):
        try:
            float(s)
            return True
        except ValueError:
            pass

        try:
            import unicodedata
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass

        return False

    def mask_to_allclass(self, masks, view):

        if len(masks.shape) == 3:
            _, h, w = masks.shape
            if view == '1':
                # 胸骨旁左室长轴切面
                tmp_mask = torch.zeros(5, h, w)
                tmp_mask[1] = masks[1]
                tmp_mask[3] = masks[0]

            if view == '2':
                # 肺动脉长轴切面
                tmp_mask = torch.zeros(5, h, w)
                tmp_mask[4] = masks[0]

            if view == '3':
                # 左室短轴切面
                tmp_mask = torch.zeros(5, h, w)
                tmp_mask[1] = masks[1]
                tmp_mask[3] = masks[0]

            if view == '4':
                # 心尖四腔心切面
                tmp_mask = torch.zeros(5, h, w)
                tmp_mask[0] = masks[2]
                tmp_mask[1] = masks[3]
                tmp_mask[2] = masks[1]
                tmp_mask[3] = masks[0]
        else:
            _, h, w, l = masks.shape
            if view == '1':
                # 胸骨旁左室长轴切面

                tmp_mask = torch.zeros(5, h, w, l)
                tmp_mask[1] = masks[1]
                tmp_mask[3] = masks[0]

            if view == '2':
                # 肺动脉长轴切面
                tmp_mask = torch.zeros(5, h, w, l)
                tmp_mask[4] = masks[0]

            if view == '3':
                # 左室短轴切面
                tmp_mask = torch.zeros(5, h, w, l)
                tmp_mask[1] = masks[1]
                tmp_mask[3] = masks[0]

            if view == '4':
                # 心尖四腔心切面
                tmp_mask = torch.zeros(5, h, w, l)
                tmp_mask[0] = masks[2]
                tmp_mask[1] = masks[3]
                tmp_mask[2] = masks[1]
                tmp_mask[3] = masks[0]

        return tmp_mask

    def get_dict(self, infos):
        selected_dict = dict()

        for k, v in infos.items():
            if v['dataset_name'] in self.set_select:
                selected_dict[k] = {}
                selected_dict[k]['images'] = v['views_images']
                selected_dict[k]['masks'] = v['views_labels']
                selected_dict[k]['fold'] = v['fold']
                # selected_dict[k]['mPAP'] = v['mPAP']
                # selected_dict[k]['Vmax'] = v['Vmax']
                selected_dict[k]['dataset_name'] = v['dataset_name']

        return selected_dict

    def input_select(self, images, masks):
        if len(masks.shape) >= 3:  # 800 * 600 * 127
            if self.is_unlab == False:
                mask_frames_ = np.sum(masks, axis=(0, 1))
            else:
                mask_frames_ = np.sum(masks, axis=(0, 1, 2))
            mask_frames_ = np.where(mask_frames_ > 100, 1, 0)
            mask_frames_ = np.argwhere(mask_frames_ == 1)
            index = random.choice(mask_frames_)[0]
            np.squeeze(mask_frames_)
            mask_frames_ = list(mask_frames_)
            return images[:, :, mask_frames_].squeeze(-1), masks[..., mask_frames_].squeeze(-1), mask_frames_


    def get_transform(self, is_train):
        all_keys = ['images', 'masks']
        crop_size = (112, 112, self.clip_length)
        spatial_size = (144, 144, self.clip_length)

        if is_train:

            rf0 = RandFlipd(keys=all_keys, prob=0.5, spatial_axis=0)
            rf1 = RandFlipd(keys=all_keys, prob=0.5, spatial_axis=1)
            rf2 = RandFlipd(keys=all_keys, prob=0.5, spatial_axis=2) if not self.single_frame else None

            rf0.set_random_state(0)
            rf1.set_random_state(0)
            if rf2 is not None:
                rf2.set_random_state(0)

            transform = Compose([
                AddChanneld(keys=all_keys, allow_missing_keys=True),
                Resized(keys=all_keys, spatial_size=spatial_size, allow_missing_keys=True, mode='nearest'),
                # CenterSpatialCropd(all_keys, crop_size, allow_missing_keys=True),
                RandSpatialCropd(all_keys, crop_size, random_size=False, allow_missing_keys=True),
                # ScaleIntensityRangePercentilesd(keys=all_keys, lower=5, upper=95, b_min=0., b_max=1., allow_missing_keys=True) if len(all_keys)>0 else Identity(),
                # NormalizeIntensityd(keys=all_keys, subtrahend=0.5, divisor=0.5),
                EnsureTyped(keys=all_keys, allow_missing_keys=True),
            ])
        else:
            transform = Compose([
                AddChanneld(keys=all_keys, allow_missing_keys=True),
                Resized(keys=all_keys, spatial_size=spatial_size, allow_missing_keys=True, mode='nearest'),
                CenterSpatialCropd(all_keys, crop_size, allow_missing_keys=True),
                # RandSpatialCropd(all_keys, crop_size, random_size=False, allow_missing_keys=True),
                # ScaleIntensityRangePercentilesd(keys=all_keys, lower=5, upper=95, b_min=0., b_max=1., allow_missing_keys=True) if len(all_keys)>0 else Identity(),
                # NormalizeIntensityd(keys=all_keys, subtrahend=0.5, divisor=0.5),
                EnsureTyped(keys=all_keys, allow_missing_keys=True),
            ])
        return transform


if __name__ == '__main__':
    config = {
        "train": {
            "cudnn": True,
            "enable_GPUs_id": [3],
            "device_ids": [3, 4, 6, 7],
            "batch_size": 8,
            "num_workers": 8,
            "num_epochs": 200,
            "view_num": ['1', '2', '3', '4'],
            "test_view": ['1', '2', '3', '4'],
            "seg_parts": True,
            "record_params": False,
            "save_dir": './result/model/view_4',
            "log_dir": './result/log_info/log_01',
            "use_data": ['gy'],  # 'gy','rmyy','szfw'
            "alpha": 0.8
            # "device":torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        },

        "net": {
            "opt": {
                "opt_name": 'Adam',
                "lr": 3e-4,
                "step_size": 50,
                "params": (0.9, 0.999),
                "weight_decay": 1e-5,
            },
        },

    }
    discrete_infos = np.load(f'/home/xmli/zyzheng/Dataset/dataset_pa_iltrasound_nii_files_3rdcenters/save_infos_reg_v2.npy',
                             allow_pickle=True).item()
    # infos = np.load(f'/home/listu/zyzheng/PAH/rmyy_40iter_pseudo_dataset/infos_unlab.npy', allow_pickle=True).item()
    infos = np.load(f'/home/xmli/zyzheng/PAH/rmyy_fix_pseudo_dataset/infos_unlab.npy', allow_pickle=True).item()


    view_num = ['1','2','3','4']
    data_dict = dict()
    root = '/home/xmli/zyzheng/dataset_pa_iltrasound_nii_files_3rdcenters'

    root = '/home/xmli/zyzheng/dataset_pa_iltrasound_nii_files_3rdcenters'
    infos = np.load(f'/home/xmli/zyzheng/dataset_pa_iltrasound_nii_files_3rdcenters/save_infos_seg.npy',
                    allow_pickle=True).item()
    infos_cyc = np.load(f'/home/listu/zyzheng/dataset_pa_iltrasound_nii_files_3rdcenters/save_infos_reg.npy',
                        allow_pickle=True).item()
    test_info = np.load(f'/home/listu/zyzheng/PAH/test_data/save_infos_seg.npy',
                        allow_pickle=True).item()
    data_list = np.load(f'/home/listu/zyzheng/PAH/data_list/align_list/align_list_64.npy')
    seg_parts = config['train']['seg_parts']

    dataset = Test_Seg_PAHDataset(test_info, root, is_train=False,
                                  set_select=['rmyy'],clip_length = 40,
                                  view_num=['4'], seg_parts=seg_parts)

    imgs,masks,_ = dataset[0]
    plt.imshow(imgs[0, :, :,0].detach().cpu().numpy())
    plt.savefig('/home/listu/zyzheng/PAH/result/visual_img_mask/img.png')
    plt.close()
    tmp_mask = masks[ :, :, :,0]
    # visual_mask = torch.argmax(torch.cat([0.5 + torch.zeros_like(tmp_mask[0]).unsqueeze(0), tmp_mask],dim=0), dim=0)
    for organ in range(5):
        plt.imshow(tmp_mask[organ].detach().cpu().numpy())
        plt.savefig(f'/home/listu/zyzheng/PAH/result/visual_img_mask/gnd{organ}.png')
        plt.close()
    print('end')

    # train_dataset['1'][1]

    # for test_view in test_views:
    #     if os.path.exists(os.path.join(root,test_view))==False:
    #         os.mkdir(os.path.join(root,test_view))
    #     for i in range(train_dataset[test_view].__len__()):
    #         img,mask,_ = train_dataset[test_view][i]
    #         img = img.array.squeeze(0)
    #         mask = np.array(mask.argmax(dim=0))
    #
    #         plt.imshow(1-img,cmap='Greys')
    #         if os.path.exists(os.path.join(root, test_view,'train')) == False:
    #             os.mkdir(os.path.join(root, test_view,'train'))
    #         plt.savefig(os.path.join(root,test_view,'train',f'img{i}'))
    #         # plt.show()
    #         plt.close()
    #         plt.imshow(1-mask,cmap='Greys')
    #         plt.savefig(os.path.join(root,test_view, 'train', f'mask{i}'))
    #         # plt.show()
    #         plt.close()
    #     for i in range(test_dataset[test_view].__len__()):
    #         img,mask,_ = test_dataset[test_view][i]
    #         img = img.array.squeeze(0)
    #         mask = np.array(mask.argmax(dim=0))
    #         plt.imshow(1-img,cmap='Greys')
    #         if os.path.exists(os.path.join(root, test_view,'test')) == False:
    #             os.mkdir(os.path.join(root, test_view,'test'))
    #         plt.savefig(os.path.join(root,test_view,'test',f'img{i}'))
    #         # plt.show()
    #         plt.close()
    #         plt.imshow(1-mask,cmap='Greys')
    #         plt.savefig(os.path.join(root,test_view, 'test', f'mask{i}'))
    #         # plt.show()
    #         plt.close()
    #     for i in range(valid_dataset[test_view].__len__()):
    #         img, mask, _ = valid_dataset[test_view][i]
    #         img = img.array.squeeze(0)
    #         mask = np.array(mask.argmax(dim=0))
    #         plt.imshow(1 - img, cmap='Greys')
    #         if os.path.exists(os.path.join(root, test_view,'val')) == False:
    #             os.mkdir(os.path.join(root, test_view,'val'))
    #         plt.savefig(os.path.join(root,test_view, 'val', f'img{i}'))
    #         # plt.show()
    #         plt.close()
    #         plt.imshow(1 - mask, cmap='Greys')
    #         plt.savefig(os.path.join(root,test_view, 'val', f'mask{i}'))
    #         # plt.show()
    #         plt.close()
