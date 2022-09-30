import pickle
import torch.utils.data as data
from os import listdir
from os.path import join
import numpy as np
import nibabel as nib
import glob
import cv2
from scipy import ndimage
from helper import *
import os
import matplotlib.pyplot as plt

class TrainDatasetACDC(data.Dataset):
    def __init__(self, data_path):
        super(TrainDatasetACDC, self).__init__()
        self.data_path = data_path
        self.filename = [f for f in sorted(listdir(self.data_path))]

    def __getitem__(self, index):
        input, fix_seg, fix_seg_myomask, mov_seg = load_data_ACDC(self.data_path, self.filename[index], size=96, rand_frame=None, load_seg_mask=True)

        mov = input[:1]
        fix = input[1:]

        # print(image.shape)
        # print(image_pred.shape)

        return mov, fix, mov_seg, fix_seg, fix_seg_myomask

    def __len__(self):
        return len(self.filename)


class TestDatasetACDC(data.Dataset):
    def __init__(self, data_path):
        super(TestDatasetACDC, self).__init__()
        self.data_path = data_path
        self.filename = [f for f in sorted(listdir(self.data_path))]

    def __getitem__(self, index):
        input, fix_seg, fix_seg_myomask, mov_seg = load_data_ACDC(self.data_path, self.filename[index], size=96, rand_frame=0, load_seg_mask=True)

        mov = input[:1]
        fix = input[1:]

        file_name = self.filename[index]

        return mov, fix, mov_seg, fix_seg, fix_seg_myomask, file_name

    def __len__(self):
        return len(self.filename)


def load_data_ACDC(data_path, filename, size, rand_frame=None, load_seg_mask=False):

    # Load configuration file to get ED and ES frame number
    info_file_path = os.path.join(data_path, filename, 'Info.cfg')
    with open(info_file_path) as f:
        line1 = f.readline()
        line2 = f.readline()
    
    ED_idx = '{:02d}'.format(int(line1.split(':')[1]))
    ES_idx = '{:02d}'.format(int(line2.split(':')[1])) # type str

    # Load images and labels
    nim = nib.load(join(data_path, filename, filename+'_frame'+ED_idx+'.nii.gz')) # load ED image (source)
    image = nim.get_data()[:, :, :] # h x w x z
    image = np.array(image, dtype='float32')

    # generate random index for t and z dimension
    if rand_frame is not None:
        # rand_t = rand_frame
        # rand_t = 1
        rand_z = rand_frame
        # print('this is rand_t', rand_t)
        # print('this is rand_z', rand_z)
    else:
        # rand_t = np.random.randint(0, image.shape[3])
        rand_z = np.random.randint(1, image.shape[2]-1)

    # preprocessing
    image_max = np.max(np.abs(image))
    image /= image_max
    image_ED = image[..., rand_z]
    image_ED = image_ED[np.newaxis]


    nim = nib.load(join(data_path, filename, filename+'_frame'+ES_idx+'.nii.gz')) # load ES image (target)
    image = nim.get_data()[:, :, :]
    image = np.array(image, dtype='float32')

    nim_seg = nib.load(join(data_path, filename, filename+'_frame'+ES_idx+'_gt.nii.gz')) # load ES mask (target)
    seg = nim_seg.get_data()[:, :, :]

    image_ES = image[..., rand_z]
    image_ES /= image_max
    seg_ES = seg[..., rand_z]

    slice = (seg[..., image.shape[2]//2] == 2).astype(np.uint8) # get the middle slice in the z stack
    centre = ndimage.measurements.center_of_mass(slice)
    centre = np.round(centre).astype(np.uint8)
    # print(centre)

    image_ES = image_ES[np.newaxis]
    seg_ES = seg_ES[np.newaxis]

    image_bank = np.concatenate((image_ED, image_ES), axis=0)

    image_bank = centre_crop(image_bank, size, centre)
    seg_ES = centre_crop(seg_ES, size, centre)
    image_bank = np.transpose(image_bank, (0, 2, 1))
    seg_ES = np.transpose(seg_ES, (0, 2, 1))
    image_bank = np.array(image_bank, dtype='float32')
    seg_ES = np.array(seg_ES, dtype='int16')

    mask = (seg_ES == 2).astype(np.uint8)
    # mask_new = mask
    # print(mask.shape)
    # mask = centre_crop(mask, size, centre)
    # print(np.max(mask_new - mask))
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask[0], kernel, iterations=3)
    mask = np.array(mask[np.newaxis], dtype='int16')

    if load_seg_mask:
        nim_seg_ED = nib.load(join(data_path, filename, filename+'_frame'+ED_idx+'_gt.nii.gz')) # load ED mask (source)
        seg_ED = nim_seg_ED.get_data()[:, :, :]
        seg_ED = seg_ED.copy()[..., rand_z]
        seg_ED = seg_ED[np.newaxis]
        seg_ED = centre_crop(seg_ED, size, centre)
        seg_ED = np.transpose(seg_ED, (0, 2, 1))
        seg_ED = np.array(seg_ED, dtype='int16')

    if not load_seg_mask:
        return image_bank, seg_ES, mask # (mov, fix), fix_seg, fix_seg_myo
    if load_seg_mask:
        return image_bank, seg_ES, mask, seg_ED # (mov, fix), fix_seg, fix_seg_myo, mov_seg


class TrainDatasetLVQuant(data.Dataset):
    def __init__(self, data_path):
        super(TrainDatasetLVQuant, self).__init__()
        self.data_path = data_path
        self.filename = [f for f in sorted(listdir(self.data_path))]

    def __getitem__(self, index):
        input, fix_seg, fix_seg_myomask, mov_seg = load_data_LVQuant(self.data_path, self.filename[index], size=96, load_seg_mask=True)

        mov = input[:1]
        fix = input[1:]

        # print(image.shape)
        # print(image_pred.shape)

        return mov, fix, mov_seg, fix_seg, fix_seg_myomask

    def __len__(self):
        return len(self.filename)


class TestDatasetLVQuant(data.Dataset):
    def __init__(self, data_path):
        super(TestDatasetLVQuant, self).__init__()
        self.data_path = data_path
        self.filename = [f for f in sorted(listdir(self.data_path))]

    def __getitem__(self, index):
        input, fix_seg, fix_seg_myomask, mov_seg = load_data_LVQuant(self.data_path, self.filename[index], size=96, load_seg_mask=True)

        mov = input[:1]
        fix = input[1:]

        file_name = self.filename[index]

        return mov, fix, mov_seg, fix_seg, fix_seg_myomask, file_name

    def __len__(self):
        return len(self.filename)
    
def load_data_LVQuant(data_path, filename, size, load_seg_mask=False):
    
    # load file 
    with open(os.path.join(data_path, filename), 'rb') as f:
        file = pickle.load(f)

    mov_image = np.array(file['mov'], dtype='float32')
    fix_image = np.array(file['fix'], dtype='float32')
    mov_seg = file['mov_seg']
    fix_seg = file['fix_seg']
    
    # preprocessing
    mov_image /= np.max(np.abs(mov_image))
    mov_image = mov_image[np.newaxis]
    mov_seg = mov_seg[np.newaxis]

    fix_image /= np.max(np.abs(fix_image))
    fix_image = fix_image[np.newaxis]
    fix_seg = fix_seg[np.newaxis]

    myo_slice = (fix_seg[0] == 2).astype(np.uint8)

    # plt.imshow(myo_slice, cmap='gray')
    # plt.show()

    centre = ndimage.measurements.center_of_mass(myo_slice) # myo_slice has to be two-dimensional
    centre = np.round(centre).astype(np.uint8)

    # print(centre)

    image_bank = np.concatenate((mov_image, fix_image), axis=0)
    fix_seg = centre_crop(fix_seg, size, centre)
    image_bank = centre_crop(image_bank, size, centre)
    
    fix_seg = np.transpose(fix_seg, (0,2,1))
    image_bank = np.transpose(image_bank, (0,2,1))

    mask = (fix_seg == 2).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask[0], kernel, iterations=3)
    mask = np.array(mask[np.newaxis], dtype='int16')


    if load_seg_mask:
        mov_seg = centre_crop(mov_seg, size, centre)
        mov_seg = np.transpose(mov_seg, (0, 2, 1))
        mov_seg = np.array(mov_seg, dtype='int16')

    if not load_seg_mask:
        return image_bank, fix_seg, mask
    if load_seg_mask:
        return image_bank, fix_seg, mask, mov_seg
