#!/usr/bin/env python
# coding: utf-8

import fnmatch
import serial
import time
from datetime import datetime, timedelta
import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import random
import warnings
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from skimage.io import imread, imsave
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms
#import pydicom as pyd
import cv2
import pandas as pd
import archs
from utils import str2bool, count_params
import losses
import joblib
import scipy.io
# import gradio as gr
import matplotlib.pyplot as plt

arch_names = list(archs.__dict__.keys())
loss_names = list(losses.__dict__.keys())

def parse_args():
    # 參數集
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='NestedUNet',
                        choices=arch_names,
                        help='model architecture: ' +
                            ' | '.join(arch_names) +
                            ' (default: NestedUNet)')
    parser.add_argument('--deepsupervision', default='True', type=str2bool)
    parser.add_argument('--dataset', default='DDHLineBase',
                        help='dataset name')
    parser.add_argument('--input-channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--image-ext', default='bmp',
                        help='image file extension')
    parser.add_argument('--mask-ext', default='dcm',
                        help='mask file extension')
    parser.add_argument('--aug', default='True', type=str2bool)
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=loss_names,
                        help='loss: ' +
                            ' | '.join(loss_names) +
                            ' (default: BCEDiceLoss)')
    parser.add_argument('--epochs', default= 50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--early-stop', default=50, type=int,
                        metavar='N', help='early stopping (default: 20)')
    parser.add_argument('-b', '--batch-size', default=1, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                            ' | '.join(['Adam', 'SGD']) +
                            ' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')
    parser.add_argument('--m_class', default=5, type=float,
                        help='4')

    args = parser.parse_args(args=[])

    return args

def get_max_preds(batch_heatmaps):
    # 在heatmap上面求最大點的座標
    '''
    get predictions from score maps
    heatmaps: numpy.ndarray([batch_size, num_joints, height, width])
    '''
    assert isinstance(batch_heatmaps, np.ndarray),         'batch_heatmaps should be numpy.ndarray'
    assert batch_heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals

args = parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

# create model
print("=> loading model " )

model = archs.NestedUNet(args)
model_dict = model.state_dict()
pretrained_dict = torch.load('model_best.pth', map_location='cpu')
key = list(pretrained_dict.keys())[0]
if (str(key).startswith('module.')):
    pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if
                        k[7:] in model_dict and v.size() == model_dict[k[7:]].size()}
else:
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                           k in model_dict and v.size() == model_dict[k].size()}
# 2. overwrite entries in the existing state dict
model_dict.update(pretrained_dict)
# 3. load the new state dict
model.load_state_dict(model_dict)

def keypoint_detection(image):
    # 原圖處理
    osi = image.shape
    image = cv2.resize(image,(512,512),interpolation=cv2.INTER_LINEAR)
    image = image.astype('float32') / 255
    image = image[:,:,:,np.newaxis]       
    
    image = image.transpose((3, 2, 0, 1))        
    input = np.asarray(image)    
    logName = 'Model_gradiotest_' + datetime.now().strftime('%Y%m%dT%H%M') + '.csv'

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        with torch.no_grad():       
                
            input = torch.from_numpy(input)   
            # compute output
            if args.deepsupervision:
                output = model(input)[-1]
            else:
                output = model(input)
            # inference的結果
            output = output.cpu().detach().numpy() 
            
            for ij in range(output.shape[0]):                            
                    log = []
                    log = pd.DataFrame(index=[], columns=[
                        'Point Name', 'Xaxis', 'Yaxis', 'Probability value'
                    ])
                    
                    preds_tmp = output[ij,:,:,:]                    
                    preds_tmp  = preds_tmp[:,:,np.newaxis]
                    preds_tmp = preds_tmp.transpose((2, 0, 1, 3))
                    
                    out_preds, out_maxvals = get_max_preds(preds_tmp)                    
                    out_preds = out_preds.squeeze()
                    out_maxvals = out_maxvals.squeeze()
                    
                    # 每個點儲存成CSV
                    for Poi in range(5):
                        if Poi == 4:
                            tmp = pd.Series([
                                'Point'+str(3),                            
                                out_preds[2][0]/512*osi[1],
                                out_preds[2][1]/512*osi[0],
                                out_maxvals[2]                            
                            ], index=['Point Name', 'Xaxis', 'Yaxis', 'Probability value'])

                            log = log.append(tmp, ignore_index=True)

                            tmp = pd.Series([
                                'Point'+str(Poi+1),                            
                                out_preds[Poi][0]/512*osi[1],
                                out_preds[Poi][1]/512*osi[0],
                                out_maxvals[Poi]                            
                            ], index=['Point Name', 'Xaxis', 'Yaxis', 'Probability value'])

                            log = log.append(tmp, ignore_index=True)
                        else:
                            tmp = pd.Series([
                                'Point'+str(Poi+1),                            
                                out_preds[Poi][0]/512*osi[1],
                                out_preds[Poi][1]/512*osi[0],
                                out_maxvals[Poi]                            
                            ], index=['Point Name', 'Xaxis', 'Yaxis', 'Probability value'])

                            log = log.append(tmp, ignore_index=True)

                    log.to_csv(logName, index=False)

                    # A角度計算
                    x1 = out_preds[0][0]/512*osi[1]
                    y1 = out_preds[0][1]/512*osi[0]
                    x2 = out_preds[1][0]/512*osi[1]
                    y2 = out_preds[1][1]/512*osi[0]
                    point12_start, point12_end = (int(x1) , int(y1) ), (int(x2) , int(y2) )
                    lm1 =(y2-y1)/(x2-x1)
                    if x2==x1:
                        lm1 = 90
                    
                    
                    x1 = out_preds[2][0]/512*osi[1]
                    y1 = out_preds[2][1]/512*osi[0]
                    x2 = out_preds[3][0]/512*osi[1]
                    y2 = out_preds[3][1]/512*osi[0]
                    point34_start, point34_end = (int(x1) , int(y1) ), (int(x2) , int(y2) )
                    lm2 =(y2-y1)/(x2-x1)
                    if x2==x1:
                        lm2 = 90
                    
                    AAngle =  abs(180*math.atan((lm1 - lm2)/(1+lm1*lm2))/math.pi)
                    
                    # B角度計算
                    x1 = out_preds[2][0]/512*osi[1]
                    y1 = out_preds[2][1]/512*osi[0]
                    x2 = out_preds[4][0]/512*osi[1]
                    y2 = out_preds[4][1]/512*osi[0]
                    point35_start, point35_end = (int(x1) , int(y1) ), (int(x2) , int(y2) )
                    lm2 =(y2-y1)/(x2-x1)
                    if x2==x1:
                        lm2 = 90
                   
                    BAngle =  abs(180*math.atan((lm1 - lm2)/(1+lm1*lm2))/math.pi)
                
        torch.cuda.empty_cache()
        
    #test
    heatmap = output[0,0,:,:].squeeze()
    heatmap = cv2.resize(heatmap,(osi[1],osi[0]),interpolation=cv2.INTER_LINEAR)
    # 恢復成原圖尺度
    pred_img = image.squeeze().astype('float32')*255   
    pred_img = pred_img.astype('uint8')    
    pred_img= pred_img.transpose((1, 2, 0))   
    pred_img = cv2.resize(pred_img,(osi[1],osi[0]),interpolation=cv2.INTER_LINEAR)
    '''
    #繪製PLT圖
    fig = plt.figure()    
    plt.imshow(pred_img)
    plt.imshow(heatmap,cmap = ('viridis'),alpha=0.45)
    #線段
    #plt.plot(log.Xaxis.values.tolist(), log.Yaxis.values.tolist(), color = "r", marker = "o", linestyle= "--", linewidth = 3,
    #    markersize = 7)
    '''
        
    new_img = cv2.line(pred_img, point12_start, point12_end, (255,0,0),5)
    new_img = cv2.line(new_img, point34_start, point34_end, (0,255,0),5)   
    new_img = cv2.line(new_img, point35_start, point35_end, (0,0,255),5)
    #角度的文字
    #plt.text((out_preds[3][0]/512*osi[1])+100,out_preds[3][1]/512*osi[0],'%.4f'%AAngle,ha = 'center', color = "r",fontsize=12)
    #plt.text((out_preds[4][0]/512*osi[1])+100,out_preds[4][1]/512*osi[0],'%.4f'%BAngle,ha = 'center', color = "r",fontsize=12)
    new_img = cv2.putText(new_img, '%.1f'%AAngle, ((int(out_preds[3][0]/512*osi[1])+50),int(out_preds[3][1]/512*osi[0])), cv2.FONT_HERSHEY_SIMPLEX ,  
                   1.5, (0, 255, 255), 2)
    new_img = cv2.putText(new_img, '%.1f'%BAngle, ((int(out_preds[4][0]/512*osi[1])+50),int(out_preds[4][1]/512*osi[0])), cv2.FONT_HERSHEY_SIMPLEX ,  
                   1.5, (0, 255, 255), 2) 
    '''
    plt.close()
    cv2.imshow("123",new_img)
    '''
    return new_img # pred_img, fig, str(AAngle), str(BAngle)
####VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
def auto_detect_serial_unix(preferred_list=['*']):
    '''try to auto-detect serial ports on win32'''
    import glob
    glist = glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*')
    ret = []

    # try preferred ones first
    for d in glist:
        for preferred in preferred_list:
            if fnmatch.fnmatch(d, preferred):
                ret.append(d)
    if len(ret) > 0:
        return ret
    # now the rest
    for d in glist:
        ret.append(d)
    return ret
    ####^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

def main():
    ####VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
    #### connected with arduino and action senser automatically
    available_ports = auto_detect_serial_unix()##*****************
    port = serial.Serial(available_ports[0], 115200,timeout=0.1)##*****************
    ####^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    dummy_input = torch.randn(1, 3, 512, 512, device='cpu')
    model_onnx = model

    input_names = ['in']
    output_names = ['out']

    # torch.onnx.export(model, dummy_input, "alexnet.onnx", verbose=True, input_names=input_names, output_names=output_names)
    torch.onnx.export(model_onnx, dummy_input, 'test_onnx.onnx', verbose=True, input_names=input_names, output_names=output_names, opset_version=11)
    import numpy as np
    import cv2
    cap = cv2.VideoCapture(0)
    print("open screen ？: {}".format(cap.isOpened()))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    cv2.namedWindow('image_win',flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)

    img_count = 1 # 拍照順序

    helpInfo = '''
    按键Q: 退出
    按键C: 拍照
    '''
    print(helpInfo)
    #### connected with arduino and action senser

    while(True):
        ret, frame = cap.read()

        ####VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV
        ## read senser parameters
        rawdata = port.readline()##*****************
        try:##*****************
            data = rawdata.decode()##*****************
        except:##*****************
            pass##*****************
        print(data)##*****************

        ####^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
        if not ret:
            print("fail to capture a frame")
            break
        # image = keypoint_detection(frame[97:1013, 353:1558])
        # image = keypoint_detection(frame)
        # image = keypoint_detection(frame[0:918, 345:1556]) # captured from for 1920 * 1080 <=> X600, X700 images
        # image = keypoint_detection(frame[0:1076, 236:1701]) # iphone8 <=> X600, X700 imagse
        image = keypoint_detection(frame[0:853, 457:1336]) # iphone8 <=> LELTEK
        cv2.imshow('image_win', image)
        print()
        # key = cv2.waitKey(1) # 等待按键事件发生 等待1ms
        if key == ord('q'):
            print("程序正常退出...Bye 不要想我哦")
            break
        elif key == ord('c'):
            cv2.imwrite("{}.png".format(img_count), frame)
            print("截图，并保存为  {}.png".format(img_count))
            img_count += 1

    # 释放VideoCapture
    cap.release()
    # 销毁所有的窗口
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
