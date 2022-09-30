# %%
import SimpleITK as sitk
from SimpleITK.SimpleITK import Elastix
from scipy.io import loadmat, savemat
from visualize import func_findParentPath
import os
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from skimage.registration import optical_flow_tvl1
from skimage.transform import warp
import pickle

def func_simpleElastix(mov, fix):
    # mov_nii = nib.Nifti1Image(mov, np.eye(4))
    # fix_nii = nib.Nifti1Image(fix, np.eye(4))

    # nib.save(mov_nii, 'mov.nii')
    # nib.save(fix_nii, 'fix.nii')

    plt.imsave('mov.png', mov)
    plt.imsave('fix.png', fix)

    # initialize elastix filter
    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetParameterMap(elastixImageFilter.GetDefaultParameterMap('nonrigid'))
    # elastixImageFilter.SetMovingImage(sitk.ReadImage('mov.nii'))
    # elastixImageFilter.SetFixedImage(sitk.ReadImage('fix.nii'))

    elastixImageFilter.SetMovingImage(sitk.ReadImage('mov.png', sitk.sitkUInt8))
    elastixImageFilter.SetFixedImage(sitk.ReadImage('fix.png', sitk.sitkUInt8))

    # run registration
    elastixImageFilter.Execute()
    transformParameterMap = elastixImageFilter.GetTransformParameterMap()

    result_image = elastixImageFilter.GetResultImage()
    result_image = sitk.Cast(result_image, sitk.sitkUInt8)

    # get deformation map
    transformix = sitk.TransformixImageFilter()
    transformix.ComputeDeformationFieldOn()
    transformix.SetTransformParameterMap(transformParameterMap)
    transformix.Execute()
    deformationField = transformix.GetDeformationField()
    deformationField_arr = sitk.GetArrayFromImage(deformationField)

    return sitk.GetArrayFromImage(result_image), deformationField_arr

def func_plotDVF(path_save, mov, fix, moved, DVF, method, mov_idx, fix_idx, n_step=4):
    grid_x, grid_y = np.meshgrid(np.arange(0, DVF.shape[0], n_step), np.arange(0, DVF.shape[1], n_step))

    if moved.ndim == 3:
        moved = moved[:,:,0]
    
    # normalize for plot
    thres_mov, thres_fix, thres_moved = np.float(np.max(mov)), np.float(np.max(fix)), np.float(np.max(moved))
    fig, axes = plt.subplots(1, 6, figsize=(30,4))

    # fig.set_size_inches(24, 4)
    axes[0].imshow(mov, cmap='gray')
    # axes[0].set_yticklabels([])
    # axes[0].set_xticklabels([])
    axes[0].set_xlabel('moving')
    axes[1].imshow(fix, cmap='gray')
    axes[1].set_xlabel('fixed')
    axes[2].imshow(moved, cmap='gray')
    axes[2].set_xlabel('moved')
    axes[3].imshow(mov/thres_mov - fix/thres_fix, cmap='gray')
    axes[3].set_xlabel('diff before')
    axes[4].imshow(moved/thres_moved - fix/thres_fix, cmap='gray')
    axes[4].set_xlabel('diff after')
    axes[5].quiver(grid_x, grid_y, DVF[::n_step,::n_step,0], DVF[::n_step,::n_step,1]) # make the vector field more sparse
    axes[5].set_xlabel('DVF')
    plt.suptitle('Deformation between frame {} (mov) and frame {} (fix) using {}'.format(int(mov_idx), int(fix_idx), str(method)))
    plt.show()
    fig.savefig(os.path.join(path_save,'{}_f{}_to_f{}.pdf'.format(str(method), int(mov_idx), int(fix_idx))), bbox_inches='tight')


def func_runRegistration(model_name, dataset, req_reg):
    assert dataset in ['ACDC17', 'LVQuant19']
    assert req_reg in ['optflow', 'bspline']


    result_path = './results'
    result_names_list = os.listdir(os.path.join(result_path, model_name))
    for result_name in result_names_list:
        result_load_path = os.path.join(result_path, model_name, result_name)
        with open(result_load_path, 'rb') as f:
            prediction = pickle.load(f)
        
        mov = prediction['mov']
        fix = prediction['fix']
        mov_seg = prediction['mov_seg']
        fix_seg = prediction['fix_seg']
        
        nr, nc = mov.shape[2], mov.shape[3]
        row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc), indexing='ij')

        # dvf prediction
        if req_reg == 'bspline':
            _, dvf = func_simpleElastix(mov[0,0,:,:], fix[0,0,:,:])
            moved = warp(mov[0,0,:,:], np.array([row_coords + dvf[:,:,1], col_coords + dvf[:,:,0]]), mode='nearest') # the coordinate is reversed for simpleElastix
            # dvf = torch.from_numpy(dvf).view((1,2,height, width))
        if req_reg == 'optflow':
            u, v = optical_flow_tvl1(fix[0,0,:,:], mov[0,0,:,:])
            dvf = np.stack((u, v), axis=2)
            moved = warp(mov[0,0,:,:], np.array([row_coords + dvf[:,:,0], col_coords + dvf[:,:, 1]]), mode='nearest') 
            # dvf = torch.from_numpy(np.stack((u, v), axis=2)).view((1,2,height, width))

        moved = np.reshape(moved, (1,1, nr, nc))

        model_name_reg = str(dataset) + '_' + str(req_reg)
        result_save_path = os.path.join(result_path, model_name_reg)
        if not os.path.exists(result_save_path):
            os.makedirs(result_save_path)

        prediction = {'mov': mov, 'fix': fix, 'pred_moved': moved, 'pred_dvf':np.reshape(dvf,(1, nr, nc, 2)), 'mov_seg':mov_seg, 'fix_seg':fix_seg}
        prediction_save_path = os.path.join(result_save_path, result_name)

        f = open(prediction_save_path, 'wb')
        pickle.dump(prediction, f)
        f.close()
