# -*- coding: utf-8 -*-
from data import test_dataloader
import os
import torch
import argparse
from torch.backends import cudnn
from torchvision.transforms import functional as F
from models.LAIBNet import build_LAIBNet
import time
import numpy as np
import cv2
parser = argparse.ArgumentParser()
parser.add_argument('--test_model', type=str, default='./weights/Final_epoch_320.pkl')
parser.add_argument('--test_data_dir', type=str, default='./dataset/test')
args = parser.parse_args()
if __name__ == '__main__':
    # CUDNN
    cudnn.benchmark = True
    model = build_LAIBNet(16)
    state_dict = torch.load(args.test_model)
    model.load_state_dict(state_dict['model'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')#
    model.eval().to(device) 
    dummy = torch.randn(1,3,600,960).to(device)
    with torch.no_grad():
        model(dummy)                            
        torch.cuda.synchronize()
    #data:WSIs
    subfolders = os.listdir(args.test_data_dir)
    folders = ["LAIBNet_flat","LAIBNet_correction"]
    for subfolder in subfolders:
        for folder in folders:
            outpath = os.path.join(args.test_data_dir,subfolder,folder)
            if not os.path.exists(outpath):
                os.makedirs(outpath)
    #start inference
    with torch.no_grad():
        for subfolder in subfolders:
            dataloader = test_dataloader(os.path.join(args.test_data_dir,subfolder), batch_size=1, num_workers=0)
            for iter_idx, data in enumerate(dataloader):
                src,correction,distortion,name = data
                src_img = src.to(device)
                #infernece
                start_time = time.time()
                pred_flat_img = model(src_img)
                pred_clip_flat_img = torch.clamp(pred_flat_img, 0, 1)
                #flat field image
                pred_flat_mean = torch.mean(pred_clip_flat_img,axis=(2,3),keepdims=True)
                pred_corr = src_img/pred_clip_flat_img*pred_flat_mean
                pred_corr_clip = torch.clamp(pred_corr, 0, 1)
                end_time = time.time()
                inference_time = (end_time - start_time) * 1000
                print(f"{inference_time:.2f} ms")
                
                #rgb->bgr
                pred_flat = (np.round(pred_clip_flat_img.squeeze(0).permute(1,2,0).cpu().numpy()*255.0)).astype(np.uint8)
                pred_flat_bgr = cv2.cvtColor(pred_flat, cv2.COLOR_RGB2BGR)
                correction = (np.round(pred_corr_clip.squeeze(0).permute(1,2,0).cpu().numpy()*255.0)).astype(np.uint8)
                pred_correction_bgr = cv2.cvtColor(correction, cv2.COLOR_RGB2BGR)
                #save images
                flat_save_name = os.path.join(args.test_data_dir,subfolder,folders[0],name[0])
                cv2.imwrite(flat_save_name, pred_flat_bgr)
                correction_save_name = os.path.join(args.test_data_dir,subfolder,folders[1],name[0])
                cv2.imwrite(correction_save_name, pred_correction_bgr)
    
