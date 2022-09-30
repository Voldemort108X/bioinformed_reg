import imp
import os
import PIL
from numpy.core.fromnumeric import repeat
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from PIL import Image
from scipy.ndimage import zoom
from torch import prelu
from torchvision.transforms import ToTensor
import io

def func_findParentPath(org_path,num_itr=1):
    parentPath = org_path
    for i in range(num_itr):
        parentPath = os.path.abspath(os.path.join(parentPath,os.pardir))
    return parentPath

def func_plotEndoEpiArea(endo_area, epi_area):
    plt.figure()
    plt.xlabel('Frame number')
    plt.ylabel('Area')
    plt.xticks(np.arange(1, endo_area.shape[0]+1, 1.0))
    plt.plot(np.arange(1,endo_area.shape[0]+1), endo_area, label='Endocardium')
    plt.plot(np.arange(1,epi_area.shape[0]+1), epi_area, label='Epicardium')
    plt.legend()
    plt.show()

def func_plotEndoEpiMaskOverlay(image, epi_mask, endo_mask, numOfFrame):
    plt.figure()
    for frame_index in range(numOfFrame):
        plt.imshow(image[:,:,frame_index], cmap='gray')
        plt.imshow(epi_mask[:,:,frame_index], alpha=0.3)
        plt.imshow(endo_mask[:,:,frame_index], alpha=0.3)
        plt.title('Frame {}'.format(str(frame_index+1)))
        plt.show()



def func_saveEachFrame(dest_path, image, endo_mask, epi_mask, numOfFrame):
    for frame_index in range(numOfFrame):
        plt.axis('off')
        plt.tight_layout()
        plt.imsave(os.path.join(dest_path, 'image_frame{:d}.png'.format(frame_index+1)), image[:,:,frame_index], cmap='gray')
        plt.imsave(os.path.join(dest_path, 'epi_mask_frame{:d}.png'.format(frame_index+1)), epi_mask[:,:,frame_index], cmap='gray')
        plt.imsave(os.path.join(dest_path, 'endo_mask_frame{:d}.png'.format(frame_index+1)), endo_mask[:,:,frame_index], cmap='gray')


def func_transformDVF(DVF):
    # The DVF produced by the network for F.grid_sample() seems to need some transform to match the image location for plot
    # horizontal flip
    DVF = np.flip(DVF, axis=1)
    # DVF = np.flip(DVF)
    return DVF

def func_plotDVF(mov, fix, moved, DVF, n_step=4, req_reg = 'False'):
    assert req_reg in ['False', 'optflow', 'bspline']
    if req_reg == 'False':
        grid_x, grid_y = np.meshgrid(np.arange(0, DVF.shape[1], n_step), np.arange(0, DVF.shape[2], n_step))
    if req_reg == 'optflow' or req_reg == 'bspline':
        grid_x, grid_y = np.meshgrid(np.arange(0, DVF.shape[0], n_step), np.arange(0, DVF.shape[1], n_step))
    # print(grid_x.shape)
    if moved.ndim == 3:
        moved = moved[:,:,0]
    
    # normalize for plot
    thres_mov, thres_fix, thres_moved = np.float(np.max(mov)), np.float(np.max(fix)), np.float(np.max(moved))
    fig, axes = plt.subplots(1, 6, figsize=(30,4))

    # transform the DVF for plot purpose
    if req_reg == 'False':
        DVF = func_transformDVF(DVF)

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
    if req_reg == 'False':
        axes[5].quiver(grid_x, grid_y, DVF[0,::n_step,::n_step], DVF[1, ::n_step,::n_step]) # make the vector field more sparse
    if req_reg == 'optflow':
        # print(grid_x.shape)
        # print(grid_y.shape)
        # print(DVF.shape)
        axes[5].quiver(grid_x, grid_y, DVF[::n_step,::n_step,0], DVF[::n_step,::n_step,1])
    if req_reg == 'bspline':
        axes[5].quiver(grid_x, grid_y, DVF[::n_step,::n_step,1], DVF[::n_step,::n_step,0])

    axes[5].set_xlabel('DVF')

def func_ACDC2D_training_visual_check(visual_check_path, batch_idx, mov, fix, manual_moved, mov_LV_mask, mov_RV_mask, mov_epi_mask, fix_LV_mask, fix_RV_mask, fix_epi_mask, moved_LV_mask, moved_RV_mask, moved_epi_mask):
    mov_slice = mov.cpu().numpy()[0,0,:,:]
    fix_slice = fix.cpu().numpy()[0,0,:,:]
    moved_slice = manual_moved.detach().cpu().numpy()[0,0,:,:]

    mov_LV_slice = mov_LV_mask.cpu().numpy()[0,0,:,:]
    mov_RV_slice = mov_RV_mask.cpu().numpy()[0,0,:,:]
    mov_epi_slice = mov_epi_mask.cpu().numpy()[0,0,:,:]

    fix_LV_slice = fix_LV_mask.cpu().numpy()[0,0,:,:]
    fix_RV_slice = fix_RV_mask.cpu().numpy()[0,0,:,:]
    fix_epi_slice = fix_epi_mask.cpu().numpy()[0,0,:,:]

    moved_LV_slice = moved_LV_mask.detach().cpu().numpy()[0,0,:,:]
    moved_RV_slice = moved_RV_mask.detach().cpu().numpy()[0,0,:,:]
    moved_epi_slice = moved_epi_mask.detach().cpu().numpy()[0,0,:,:]


    plt.imsave(os.path.join(visual_check_path, 'idx_{}_mov_slice.png'.format(str(batch_idx))), mov_slice)
    plt.imsave(os.path.join(visual_check_path, 'idx_{}_fix_slice.png'.format(str(batch_idx))), fix_slice)
    plt.imsave(os.path.join(visual_check_path, 'idx_{}_moved_slice.png'.format(str(batch_idx))), moved_slice)

    plt.imsave(os.path.join(visual_check_path, 'idx_{}_mov_LV_slice.png'.format(str(batch_idx))), mov_LV_slice)
    plt.imsave(os.path.join(visual_check_path, 'idx_{}_fix_LV_slice.png'.format(str(batch_idx))), fix_LV_slice)
    plt.imsave(os.path.join(visual_check_path, 'idx_{}_moved_LV_slice.png'.format(str(batch_idx))), moved_LV_slice)

    plt.imsave(os.path.join(visual_check_path, 'idx_{}_mov_RV_slice.png'.format(str(batch_idx))), mov_RV_slice)
    plt.imsave(os.path.join(visual_check_path, 'idx_{}_fix_RV_slice.png'.format(str(batch_idx))), fix_RV_slice)
    plt.imsave(os.path.join(visual_check_path, 'idx_{}_moved_RV_slice.png'.format(str(batch_idx))), moved_RV_slice)

    plt.imsave(os.path.join(visual_check_path, 'idx_{}_mov_epi_slice.png'.format(str(batch_idx))), mov_epi_slice)
    plt.imsave(os.path.join(visual_check_path, 'idx_{}_fix_epi_slice.png'.format(str(batch_idx))), fix_epi_slice)
    plt.imsave(os.path.join(visual_check_path, 'idx_{}_moved_epi_slice.png'.format(str(batch_idx))), moved_epi_slice)


def func_ACDC2D_image_check_dict(mov, fix, manual_moved, mov_LV_mask, mov_RV_mask, mov_epi_mask, fix_LV_mask, fix_RV_mask, fix_epi_mask, moved_LV_mask, moved_RV_mask, moved_epi_mask):

    mov_slice = mov.cpu().numpy()[0,:,:,:]
    fix_slice = fix.cpu().numpy()[0,:,:,:]
    moved_slice = manual_moved.detach().cpu().numpy()[0,:,:,:]

    # mov_LV_slice = mov_LV_mask.cpu().numpy()[0,:,:,:]
    # mov_RV_slice = mov_RV_mask.cpu().numpy()[0,:,:,:]
    # mov_epi_slice = mov_epi_mask.cpu().numpy()[0,:,:,:]

    fix_LV_slice = fix_LV_mask.cpu().numpy()[0,:,:,:]
    fix_RV_slice = fix_RV_mask.cpu().numpy()[0,:,:,:]
    fix_epi_slice = fix_epi_mask.cpu().numpy()[0,:,:,:]

    moved_LV_slice = moved_LV_mask.detach().cpu().numpy()[0,:,:,:]
    moved_RV_slice = moved_RV_mask.detach().cpu().numpy()[0,:,:,:]
    moved_epi_slice = moved_epi_mask.detach().cpu().numpy()[0,:,:,:]

    return {'mov_slice': mov_slice, 'fix_slice': fix_slice, 'moved_slice': moved_slice, 
    'diff_LV': moved_LV_slice/np.float(np.max(moved_LV_slice)) - fix_LV_slice/np.float(np.max(fix_LV_slice)), 
    'diff_RV': moved_RV_slice/np.float(np.max(moved_RV_slice)) - fix_RV_slice/np.float(np.max(fix_RV_slice)),
    'diff_epi': moved_epi_slice/np.float(np.max(moved_epi_slice)) - fix_epi_slice/np.float(np.max(fix_epi_slice))}



def func_LVQuant_training_visual_check(visual_check_path, batch_idx, mov, fix, manual_moved, mov_endo_mask, mov_epi_mask, fix_endo_mask, fix_epi_mask, moved_endo_mask, moved_epi_mask):
    mov_slice = mov.cpu().numpy()[0,0,:,:]
    fix_slice = fix.cpu().numpy()[0,0,:,:]
    moved_slice = manual_moved.detach().cpu().numpy()[0,0,:,:]

    mov_endo_slice = mov_endo_mask.cpu().numpy()[0,0,:,:]
    mov_epi_slice = mov_epi_mask.cpu().numpy()[0,0,:,:]

    fix_endo_slice = fix_endo_mask.cpu().numpy()[0,0,:,:]
    fix_epi_slice = fix_epi_mask.cpu().numpy()[0,0,:,:]

    moved_endo_slice = moved_endo_mask.detach().cpu().numpy()[0,0,:,:]
    moved_epi_slice = moved_epi_mask.detach().cpu().numpy()[0,0,:,:]


    plt.imsave(os.path.join(visual_check_path, 'idx_{}_mov_slice.png'.format(str(batch_idx))), mov_slice)
    plt.imsave(os.path.join(visual_check_path, 'idx_{}_fix_slice.png'.format(str(batch_idx))), fix_slice)
    plt.imsave(os.path.join(visual_check_path, 'idx_{}_moved_slice.png'.format(str(batch_idx))), moved_slice)

    plt.imsave(os.path.join(visual_check_path, 'idx_{}_mov_endo_slice.png'.format(str(batch_idx))), mov_endo_slice)
    plt.imsave(os.path.join(visual_check_path, 'idx_{}_fix_endo_slice.png'.format(str(batch_idx))), fix_endo_slice)
    plt.imsave(os.path.join(visual_check_path, 'idx_{}_moved_endo_slice.png'.format(str(batch_idx))), moved_endo_slice)

    plt.imsave(os.path.join(visual_check_path, 'idx_{}_mov_epi_slice.png'.format(str(batch_idx))), mov_epi_slice)
    plt.imsave(os.path.join(visual_check_path, 'idx_{}_fix_epi_slice.png'.format(str(batch_idx))), fix_epi_slice)
    plt.imsave(os.path.join(visual_check_path, 'idx_{}_moved_epi_slice.png'.format(str(batch_idx))), moved_epi_slice)


def func_LVQuant_image_check_dict(mov, fix, manual_moved, mov_endo_mask, mov_epi_mask, fix_endo_mask, fix_epi_mask, moved_endo_mask, moved_epi_mask):

    mov_slice = mov.cpu().numpy()[0,:,:,:]
    fix_slice = fix.cpu().numpy()[0,:,:,:]
    moved_slice = manual_moved.detach().cpu().numpy()[0,:,:,:]

    # mov_endo_slice = mov_endo_mask.cpu().numpy()[0,:,:,:]
    # mov_epi_slice = mov_epi_mask.cpu().numpy()[0,:,:,:]

    fix_endo_slice = fix_endo_mask.cpu().numpy()[0,:,:,:]
    fix_epi_slice = fix_epi_mask.cpu().numpy()[0,:,:,:]

    moved_endo_slice = moved_endo_mask.detach().cpu().numpy()[0,:,:,:]
    moved_epi_slice = moved_epi_mask.detach().cpu().numpy()[0,:,:,:]

    return {'mov_slice': mov_slice, 'fix_slice': fix_slice, 'moved_slice': moved_slice, 
    'diff_endo': moved_endo_slice/np.float(np.max(moved_endo_slice)) - fix_endo_slice/np.float(np.max(fix_endo_slice)), 
    'diff_epi': moved_epi_slice/np.float(np.max(moved_epi_slice)) - fix_epi_slice/np.float(np.max(fix_epi_slice))}
