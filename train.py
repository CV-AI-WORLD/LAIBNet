# -*- coding: utf-8 -*-
import torch
from data import train_dataloader
import os
from pytorch_msssim  import  MS_SSIM
import numpy as np
def _train(model, args):
    #device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #optomizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, args.lr_steps, args.gamma)
    l1_loss = torch.nn.L1Loss()
    ms_ssim_loss = MS_SSIM(data_range=1.0, win_size = 5 ,size_average=True, channel=3)
    #dataLoader
    dataloader = train_dataloader(args.data_dir, args.batch_size, args.num_worker)    
    epoch = 1
    loss_l1_cor_list = []
    loss_ssim_cor_list = []
    loss_l1_flat_list = []
    loss_ssim_flat_list = []
    if args.resume:
        state = torch.load(args.resume)
        model.load_state_dict(state['model'])
    for epoch_idx in range(epoch, args.num_epoch + 1):
        total_every_epoch_l1_cor = 0
        total_every_epoch_ssim_cor = 0
        total_every_epoch_l1_flat =0 
        total_every_epoch_ssim_flat  = 0
        num = 0
        for iter_idx, batch_data in enumerate(dataloader):
            src,GTC,GTF,name = batch_data
            src_img = src.to(device)
            correction_img = GTC.to(device)
            flat_img = GTF.to(device)
            #predict
            pred_flat_img = model(src_img)
            pred_correction_img = src_img/(pred_flat_img)*torch.mean(pred_flat_img,dim=(2,3),keepdim=True)
            #correction loss
            mae_loss_correction = l1_loss(pred_correction_img,correction_img)
            ssim_loss_correction = 1.0- ms_ssim_loss(pred_correction_img,correction_img)
            #flat loss
            mae_loss_flat = l1_loss(pred_flat_img,flat_img)
            ssim_loss_flat = 1.0- ms_ssim_loss(pred_flat_img,flat_img)
            loss = 0.5*mae_loss_flat+10*ssim_loss_flat + mae_loss_correction+10*ssim_loss_correction
            #
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()        
            print(epoch_idx,scheduler.get_last_lr(),mae_loss_correction.item(),mae_loss_flat.item())
         
            total_every_epoch_l1_cor = total_every_epoch_l1_cor + mae_loss_correction.item()
            total_every_epoch_ssim_cor = total_every_epoch_ssim_cor + ssim_loss_correction.item()
            total_every_epoch_l1_flat = total_every_epoch_l1_flat + mae_loss_flat.item()
            total_every_epoch_ssim_flat = total_every_epoch_ssim_flat + ssim_loss_flat.item()
            num = num+1
        scheduler.step()
        loss_l1_cor_list.append(total_every_epoch_l1_cor/num)
        loss_ssim_cor_list.append(total_every_epoch_ssim_cor/num)
        loss_l1_flat_list.append(total_every_epoch_l1_flat/num)
        loss_ssim_flat_list.append(total_every_epoch_ssim_flat/num)

        print("every epoch correction l1 loss:",loss_l1_cor_list)
        print("every epoch correction ssim loss:",loss_ssim_cor_list)
        print("every epoch flat l1 loss:",loss_l1_flat_list)
        print("every epoch flat ssim loss:",loss_ssim_flat_list)
        if epoch_idx%10 == 0:
            save_name = os.path.join(args.model_save_dir, 'Final_epoch_'+str(epoch_idx)+'.pkl')
            torch.save({'model': model.state_dict()}, save_name)
    np.savetxt(os.path.join(args.model_save_dir, 'loss_correction_l1.txt'), np.array(loss_l1_cor_list))
    np.savetxt(os.path.join(args.model_save_dir, 'loss_correction_ssim.txt'), np.array(loss_ssim_cor_list))
    np.savetxt(os.path.join(args.model_save_dir, 'loss_flat_l1.txt'), np.array(loss_l1_flat_list))
    np.savetxt(os.path.join(args.model_save_dir, 'loss_flat_ssim.txt'), np.array(loss_ssim_flat_list))

