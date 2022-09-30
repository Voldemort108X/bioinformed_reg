import torch
from helper import compute_strain
import torch.nn as nn

def BMIloss(dvf, nup=0.4):
    # construct 2D material property matrix (3x3)
    C_mat_inv = torch.Tensor([[1, -nup, 0],[-nup, 1, 0],[0, 0, 2*(1+nup)]]).to('cuda')
    C_mat = torch.inverse(C_mat_inv)

    # assume dvf has shape bsize x csize x height x width
    bsize, csize, height, width = dvf.size()
    strain_vector = compute_strain(dvf) # bsize x 3 x height x width
    strain_left_vector = strain_vector.view(bsize, height, width, 1, 3)
    strain_right_vector = strain_vector.view(bsize, 3, 1, height, width)

    # compute the probability matrix
    p_mat_l = torch.matmul(strain_left_vector, C_mat) # bsize x height x width x 1 x 3
    p_mat = torch.einsum('bijlm,bjklm->biklm', p_mat_l.view(bsize, 1, 3, height, width), strain_right_vector) # bsize x 1 x 1 x height x width

    # print(p_mat.shape)

    bmi = torch.norm(p_mat)

    return bmi


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.contiguous().view(-1)
        targets = targets.contiguous().view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice