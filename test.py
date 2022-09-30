
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import pickle

import argparse
from ast import parse
from network import *
from dataset import *
from helper import *


def test(model_name, model_path, result_path):

    Tensor = torch.cuda.FloatTensor

    # configure all the paths
    model_save_path = os.path.join(model_path, model_name)
    result_save_path = os.path.join(result_path, model_name)
    if not os.path.exists(result_save_path):
        os.makedirs(result_save_path)

    # load the model
    model = Registration_Net()
    model.load_state_dict(torch.load(model_save_path+'.pth'))
    model = model.cuda()
    model.eval()

    # prepare the testdataset
    if 'ACDC17' in model_name:
        data_path = '../../Dataset/ACDC2017/'
        test_set = TestDatasetACDC(os.path.join(data_path, 'testing'))
        testing_data_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=1, shuffle=False)

    if 'LVQuant19' in model_name:
        data_path = '../../Dataset/LV_Quant_Challenge/'
        test_set = TestDatasetLVQuant(os.path.join(data_path, 'testing'))
        testing_data_loader = DataLoader(dataset=test_set, num_workers=4, batch_size=1, shuffle=False)

    for batch_idx, batch in tqdm(enumerate(testing_data_loader, 1),
                                 total=len(testing_data_loader)):
        # print('len of testset',len(testing_data_loader))
        x, x_pred, x_gnd, x_pred_gnd, mask, file_name = batch
        x_c = Variable(x.type(Tensor))
        x_predc = Variable(x_pred.type(Tensor))
        mask = Variable(mask.type(Tensor))

        net = model(x_c, x_predc, x_c)

        pred_moved = net['fr_st'].detach().cpu().numpy()
        pred_dvf = net['out'].detach().cpu().numpy()

        # be very careful of how the segmentation is stored. x_gnd is fix_seg and x_pred_gnd is mov_seg !!!
        prediction = {'mov': x.cpu().numpy(), 'fix': x_pred.cpu().numpy(), 'pred_moved':pred_moved, 'pred_dvf':pred_dvf, 'mov_seg':x_gnd.cpu().numpy(), 'fix_seg':x_pred_gnd.cpu().numpy()}

        # print(x_pred.shape)
        # print(x.shape)
        prediction_save_path = os.path.join(result_save_path, str(file_name[0])+'.pkl')

        f = open(prediction_save_path, 'wb')
        pickle.dump(prediction, f)
        f.close()

        # np.save(os.path.join(result_save_path, 'moving.npy'), x)
        # np.save(os.path.join(result_save_path, 'fixed.npy'), x_pred)
        # np.save(os.path.join(result_save_path, 'pred_moved.npy'), pred_moved)
        # np.save(os.path.join(result_save_path, 'pred_dvf.npy'), pred_dvf)


if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', required=True, type=str, help='model to use for prediction')
    parser.add_argument('--modelpath', default='./models/', help='model save path')
    parser.add_argument('--resultpath', default='./results', help='where to save the results')
    args = parser.parse_args()

    print('Testing model used', args)
    test(args.model_name, args.modelpath, args.resultpath)