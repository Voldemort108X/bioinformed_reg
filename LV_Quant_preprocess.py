# %%
import os
import numpy as np
from scipy.io import loadmat, savemat
from skimage.transform import resize
import pickle
import matplotlib.pyplot as plt

def func_imresize(image, size):
    return resize(image, (size, size))

size = 256
data_path = '../../Dataset/LV_Quant_Challenge/Original_data/TrainingData_LVQuan19'
train_path = '../../Dataset/LV_Quant_Challenge/training'
val_path = '../../Dataset/LV_Quant_Challenge/validation'
test_path = '../../Dataset/LV_Quant_Challenge/testing'

if not os.path.exists(train_path):
    os.makedirs(train_path)

if not os.path.exists(val_path):
    os.makedirs(val_path)

if not os.path.exists(test_path):
    os.makedirs(test_path)

file_name_list = os.listdir(data_path)

# mov_list, fix_list, mov_endo_seg_list, fix_endo_seg_list, mov_epi_seg_list, fix_epi_seg_list = [],[],[],[],[],[]
mov_list, fix_list, mov_seg_list, fix_seg_list = [],[],[],[]
save_file_name_list = []

for file_name in file_name_list:
    patient_data = loadmat(os.path.join(data_path, file_name))
    image = patient_data['image']
    endo = patient_data['endo']
    epi = patient_data['epi']

    height, width, z_size = image.shape
    for z_idx in [0, 10]: # ED/ES to ES/ED
    # for z_idx in range(z_size-1): # frame to frame
        mov = func_imresize(image[:,:,z_idx], size=size) 
        fix = func_imresize(image[:,:,z_idx+1], size=size) 
        mov_endo_seg = np.where(func_imresize(endo[:,:,z_idx], size=size)/np.max(func_imresize(endo[:,:,z_idx], size=size))>0.5, 1, 0) 
        fix_endo_seg = np.where(func_imresize(endo[:,:,z_idx+9], size=size)/np.max(func_imresize(endo[:,:,z_idx+9], size=size))>0.5, 1, 0) # ED/ES to ES/ED
        # fix_endo_seg = np.where(func_imresize(endo[:,:,z_idx+1], size=size)/np.max(func_imresize(endo[:,:,z_idx+1], size=size))>0.5, 1, 0) # frame to frame

        mov_epi_seg = np.where(func_imresize(epi[:,:,z_idx], size=size)/np.max(func_imresize(epi[:,:,z_idx], size=size))>0.5, 1, 0) # ED/ES to ES/ED
        fix_epi_seg = np.where(func_imresize(epi[:,:,z_idx+9], size=size)/np.max(func_imresize(epi[:,:,z_idx+9], size=size))>0.5, 1, 0) # ED/ES to ES/ED
        # fix_epi_seg = np.where(func_imresize(epi[:,:,z_idx+1], size=size)/np.max(func_imresize(epi[:,:,z_idx+1], size=size))>0.5, 1, 0) # frame to frame

        mov_myo_mask = mov_epi_seg - mov_endo_seg
        fix_myo_mask = fix_epi_seg - fix_endo_seg

        mov_myo_mask = np.where(mov_myo_mask == 1, 2, 0)
        fix_myo_mask = np.where(fix_myo_mask == 1, 2, 0)

        mov_seg = mov_endo_seg + mov_myo_mask
        fix_seg = fix_endo_seg + fix_myo_mask

        mov_list.append(mov)
        fix_list.append(fix)
        mov_seg_list.append(mov_seg)
        fix_seg_list.append(fix_seg)

        # plt.imshow(mov_seg, cmap='gray')
        # plt.show()
        # plt.imshow(fix_seg, cmap='gray')
        # plt.show()

        save_file_name_list.append(file_name.split('.')[0]+'_f_{}_f_{}.pkl'.format(str(z_idx+1), str(z_idx+10))) 
        # save_file_name_list.append(file_name.split('.')[0]+'_f_{}_f_{}.pkl'.format(str(z_idx+1), str(z_idx+2))) # frame to frame
    # break


numOfSlices = len(mov_list)
train_indices = np.arange(0, int(0.6*numOfSlices))
val_indices = np.arange(int(0.6*numOfSlices), int(0.8*numOfSlices))
test_indices = np.arange(int(0.8*numOfSlices), int(numOfSlices))

for train_idx in train_indices:
    # file = {'mov': mov_list[train_idx], 'fix':fix_list[train_idx], 'mov_endo_seg':mov_endo_seg_list[train_idx], 'fix_endo_seg':fix_endo_seg_list[train_idx], 'mov_epi_seg':mov_epi_seg_list[train_idx],'fix_epi_seg':fix_epi_seg_list[train_idx]}
    file = {'mov': mov_list[train_idx], 'fix':fix_list[train_idx], 'mov_seg':mov_seg_list[train_idx], 'fix_seg':fix_seg_list[train_idx]}
    f = open(os.path.join(train_path, save_file_name_list[train_idx]), 'wb')
    pickle.dump(file, f)
    f.close()

for val_idx in val_indices:
    # file = {'mov': mov_list[val_idx], 'fix':fix_list[val_idx], 'mov_endo_seg':mov_endo_seg_list[val_idx], 'fix_endo_seg':fix_endo_seg_list[val_idx], 'mov_epi_seg':mov_epi_seg_list[val_idx],'fix_epi_seg':fix_epi_seg_list[val_idx]}
    file = {'mov': mov_list[val_idx], 'fix':fix_list[val_idx], 'mov_seg':mov_seg_list[val_idx], 'fix_seg':fix_seg_list[val_idx]}
    f = open(os.path.join(val_path, save_file_name_list[val_idx]), 'wb')
    pickle.dump(file, f)
    f.close()

for test_idx in test_indices:
    # file = {'mov': mov_list[test_idx], 'fix':fix_list[test_idx], 'mov_endo_seg':mov_endo_seg_list[test_idx], 'fix_endo_seg':fix_endo_seg_list[test_idx], 'mov_epi_seg':mov_epi_seg_list[test_idx],'fix_epi_seg':fix_epi_seg_list[test_idx]}
    file = {'mov': mov_list[test_idx], 'fix':fix_list[test_idx], 'mov_seg':mov_seg_list[test_idx], 'fix_seg':fix_seg_list[test_idx]}
    f = open(os.path.join(test_path, save_file_name_list[test_idx]), 'wb')
    pickle.dump(file, f)
    f.close()

# %%
