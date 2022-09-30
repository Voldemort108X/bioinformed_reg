from cv2 import imshow
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from medpy import metric
import os
import pickle
import matplotlib.pyplot as plt
import torch.nn.functional as F
from network import generate_grid
import torch
from registration import func_simpleElastix
from skimage.registration import optical_flow_tvl1
from skimage.transform import warp
import matplotlib.pyplot as plt
import SimpleITK as sitk

# def func_jac_fx(dvf):
#     # assume the dvf is HxWx2
#     n_row, n_col = dvf.shape[0], dvf.shape[1]
#     grid_x, grid_y = np.meshgrid(np.arange(n_row), np.arange(n_col))
#     t_x = dvf[0] + grid_x
#     t_y = dvf[1] + grid_y
#     return np.stack([t_x, t_y], axis=2)

# def func_computeJac(dvf):
#     jacobian_fx = jacobian(func_jac_fx)
#     return jacobian_fx(dvf)

def func_rearangeDVF(dvf, type):
    assert type in ['False', 'optflow', 'bspline', 'noreg']
    dvf_out = np.zeros((dvf.shape[1], dvf.shape[1], 2))
    # print(dvf_out.shape)
    if type == 'False':
        dvf_out[:,:,0] = dvf[1,:,:]
        dvf_out[:,:,1] = dvf[0,:,:]
    if type == 'optflow':
        dvf_out[:,:,0] = dvf[:,:,1]
        dvf_out[:,:,1] = dvf[:,:,0]
    if type == 'bspline':
        dvf_out[:,:,0] = dvf[:,:,0]
        dvf_out[:,:,1] = dvf[:,:,1]
    return dvf_out

def func_computeDetJac(dvf, type):
    dvf_out = func_rearangeDVF(dvf, type)
    dvf_sitk = sitk.GetImageFromArray(dvf_out, isVector=True)
    # print(dvf_sitk.type)
    return sitk.GetArrayFromImage(sitk.DisplacementFieldJacobianDeterminant(dvf_sitk))


def func_computeImgMetrics(fix, moved):
    # normalize fix and moved
    fix = (fix-np.min(fix)) / (np.max(fix) - np.min(fix))
    moved = (moved-np.min(moved)) / (np.max(moved) - np.min(moved))

    # plt.imshow(fix,cmap='gray')
    # plt.show()
    # plt.imshow(moved,cmap='gray')
    # plt.show()

    ssim_index = ssim(fix, moved)
    mse_index = mse(fix, moved)

    return ssim_index, mse_index

def func_computeSegMetrics2D(pred, gt):

    dice = metric.binary.dc(pred, gt)
    jc = metric.binary.jc(pred, gt)
    hd = metric.binary.hd(pred, gt, voxelspacing=[1, 1, 1.25, 1.25])
    asd = metric.binary.asd(pred, gt, voxelspacing=[1, 1, 1.25, 1.25])

    return dice, jc, hd, asd


# def func_computeSegMetricsEachOrgan(mov_mask, fix_mask, dvf, dice_list, jc_list, hd_list, asd_list, req_reg='False'):
def func_computeSegMetricsEachOrgan(mov_mask, fix_mask, dvf, dice_list, jc_list, hd_list, asd_list, mov, fix, moved, req_reg='False'):
    assert req_reg in ['False', 'optflow', 'bspline', 'noreg']

    if req_reg == 'False':
        grid = generate_grid(torch.from_numpy(mov_mask.astype(np.float32)).to('cuda'), torch.from_numpy(dvf).to('cuda'))
        moved_mask = F.grid_sample(torch.from_numpy(mov_mask.astype(np.float32)).to('cuda'), grid, mode='bilinear').cpu().numpy()
        manual_moved = F.grid_sample(torch.from_numpy(mov.astype(np.float32)).to('cuda'), grid, mode='bilinear').cpu().numpy() # moved image for visual check

    if req_reg == 'optflow':
        nr, nc = mov_mask.shape[2], mov_mask.shape[3]
        row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')
        moved_mask = warp(mov_mask[0,0,:,:], np.array([row_coords + dvf[0, :,:,0], col_coords + dvf[0, :,:, 1]]), mode='nearest') 
        moved_mask = np.reshape(moved_mask, (1,1, nr, nc))
        manual_moved = warp(mov[0,0,:,:], np.array([row_coords + dvf[0, :,:,0], col_coords + dvf[0, :,:, 1]]), mode='nearest') 

    if req_reg == 'bspline':
        nr, nc = mov_mask.shape[2], mov_mask.shape[3]
        row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')
        moved_mask = warp(mov_mask[0,0,:,:], np.array([row_coords + dvf[0, :,:,1], col_coords + dvf[0, :,:, 0]]), mode='nearest') 
        moved_mask = np.reshape(moved_mask, (1,1,nr, nc))
        manual_moved = warp(mov[0,0,:,:], np.array([row_coords + dvf[0, :,:,1], col_coords + dvf[0, :,:, 0]]), mode='nearest')
    # if req_reg == 'noreg':
    #     # if there is no registration 
    #     moved_mask = mov_mask
    #     manual_moved = mov
    
    # if req_reg == 'False':
    #     plt.imshow(manual_moved[0,0,:,:], cmap='gray')
    #     plt.show()
    # else:
    #     plt.imshow(manual_moved, cmap='gray')
    #     plt.show()
    # plt.imshow(moved[0,0,:,:], cmap='gray')
    # plt.show()
    if req_reg == 'noreg':
        try:
            dice, jc, hd, asd = func_computeSegMetrics2D(mov_mask, fix_mask)
            dice_list.append(dice), jc_list.append(jc), hd_list.append(hd), asd_list.append(asd)
        except:
            pass
    else: 
        mov_mask = (mov_mask - np.min(mov_mask)) / (np.max(mov_mask) - np.min(mov_mask))
        fix_mask = (fix_mask - np.min(fix_mask)) / (np.max(fix_mask) - np.min(fix_mask))
        moved_mask = (moved_mask - np.min(moved_mask)) / (np.max(moved_mask) - np.min(moved_mask))

        mov_mask = np.where(mov_mask>0.5, 1, 0)
        fix_mask = np.where(fix_mask>0.5, 1, 0)
        moved_epi_mask = np.where(moved_mask>0.5, 1, 0)

        try:
            dice, jc, hd, asd = func_computeSegMetrics2D(moved_epi_mask, fix_mask)
            dice_list.append(dice), jc_list.append(jc), hd_list.append(hd), asd_list.append(asd)
        except:
            pass
        
    return dice_list, jc_list, hd_list, asd_list

def func_computeAllMetrics(model_name, dataset, organ, req_reg='False'):
    assert dataset in ['ACDC17', 'LVQuant19']
    assert req_reg in ['False', 'optflow', 'bspline', 'noreg']
    if dataset == 'ACDC17':
        assert organ in ['LV', 'RV', 'Epi']
    if dataset == 'LVQuant19':
        assert organ in ['Endo', 'Epi']

    ssim_index_list, mse_index_list = [], []
    dice_list, jc_list, hd_list, asd_list = [], [], [], []
    detjac_list = []
    # if not req_reg == 'False':

    result_path = './results'
    result_names_list = os.listdir(os.path.join(result_path, model_name))
    for result_name in result_names_list:
        result_load_path = os.path.join(result_path, model_name, result_name)
        with open(result_load_path, 'rb') as f:
            prediction = pickle.load(f)
        
        mov = prediction['mov']
        fix = prediction['fix']
        moved = prediction['pred_moved']
        dvf = prediction['pred_dvf']
        mov_seg = prediction['mov_seg']
        fix_seg = prediction['fix_seg']

        # print(mov.shape)
        # print(fix.shape)
        # print(moved.shape)
        # print(dvf.shape)
        # print(mov_seg.shape)
        # print(fix_seg.shape)

        ssim_index, mse_index = func_computeImgMetrics(fix[0,0,:,:], moved[0,0,:,:])
        ssim_index_list.append(ssim_index), mse_index_list.append(mse_index)
        
        detjac_list.append(np.abs(func_computeDetJac(dvf[0, :, :, :], req_reg) - 1))
        # warp the mov segmentation
        # mov_seg = mov_seg[0,0,:,:] 
        # fix_seg = fix_seg[0,0,:,:]

        if dataset == 'ACDC17':
            mov_LV_mask = (mov_seg==3).astype(np.uint8)
            fix_LV_mask = (fix_seg==3).astype(np.uint8)

            mov_myo_mask = (mov_seg==2).astype(np.uint8)
            fix_myo_mask = (fix_seg==2).astype(np.uint8)

            mov_RV_mask = (mov_seg==1).astype(np.uint8)
            fix_RV_mask = (fix_seg==1).astype(np.uint8)

            mov_epi_mask = mov_LV_mask + mov_myo_mask
            fix_epi_mask = fix_LV_mask + fix_myo_mask

            if organ == 'LV':
                # dice_list, jc_list, hd_list, asd_list = func_computeSegMetricsEachOrgan(mov_LV_mask, fix_LV_mask, dvf, dice_list, jc_list, hd_list, asd_list, req_reg)
                dice_list, jc_list, hd_list, asd_list = func_computeSegMetricsEachOrgan(mov_LV_mask, fix_LV_mask, dvf, dice_list, jc_list, hd_list, asd_list, mov, fix, moved, req_reg)
            if organ == 'RV':
                # dice_list, jc_list, hd_list, asd_list = func_computeSegMetricsEachOrgan(mov_RV_mask, fix_RV_mask, dvf, dice_list, jc_list, hd_list, asd_list, req_reg)
                dice_list, jc_list, hd_list, asd_list = func_computeSegMetricsEachOrgan(mov_RV_mask, fix_RV_mask, dvf, dice_list, jc_list, hd_list, asd_list, mov, fix, moved, req_reg)
            if organ == 'Epi':
                # dice_list, jc_list, hd_list, asd_list = func_computeSegMetricsEachOrgan(mov_epi_mask, fix_epi_mask, dvf, dice_list, jc_list, hd_list, asd_list, req_reg)
                dice_list, jc_list, hd_list, asd_list = func_computeSegMetricsEachOrgan(mov_epi_mask, fix_epi_mask, dvf, dice_list, jc_list, hd_list, asd_list, mov, fix, moved, req_reg)
            
        # TODO
        if dataset =='LVQuant19':
            mov_endo_mask = (mov_seg==1).astype(np.uint8)
            fix_endo_mask = (fix_seg==1).astype(np.uint8)

            mov_myo_mask = (mov_seg==2).astype(np.uint8)
            fix_myo_mask = (fix_seg==2).astype(np.uint8)

            mov_epi_mask = mov_endo_mask + mov_myo_mask
            fix_epi_mask = fix_endo_mask + fix_myo_mask

            if organ == 'Endo':
                # dice_list, jc_list, hd_list, asd_list = func_computeSegMetricsEachOrgan(mov_endo_mask, fix_endo_mask, dvf, dice_list, jc_list, hd_list, asd_list, req_reg)
                dice_list, jc_list, hd_list, asd_list = func_computeSegMetricsEachOrgan(mov_endo_mask, fix_endo_mask, dvf, dice_list, jc_list, hd_list, asd_list, mov, fix, moved,  req_reg)
            if organ == 'Epi':
                # dice_list, jc_list, hd_list, asd_list = func_computeSegMetricsEachOrgan(mov_epi_mask, fix_epi_mask, dvf, dice_list, jc_list, hd_list, asd_list, req_reg)
                dice_list, jc_list, hd_list, asd_list = func_computeSegMetricsEachOrgan(mov_epi_mask, fix_epi_mask, dvf, dice_list, jc_list, hd_list, asd_list, mov, fix, moved,  req_reg)



        # print('after dice:', dice)
        # break
        # break
    
    return ssim_index_list, mse_index_list, dice_list, jc_list, hd_list, asd_list, detjac_list

