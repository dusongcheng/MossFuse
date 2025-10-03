# -*- coding: utf-8 -*-

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  
from dataset import HyperDatasetTrain, HyperDatasetValid
from MossFuseNet import MossFuse
import datetime
import itertools
import sys
import time
import cv2
import hdf5storage as hdf5
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import cc, record_loss, show, PSNR_SSIM_cal, cal_decomp_loss

'''
------------------------------------------------------------------------------
Configure our network
------------------------------------------------------------------------------
'''
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
GPU_number = os.environ['CUDA_VISIBLE_DEVICES']
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)



init_lr = 1e-3*0.1*0.9
num_epochs = 500 
batch_size = 2

scale = 32
if_show = True     # whether save the psf and srf images

model_name = 'CAVE_'+str(scale)

# Model
model = nn.DataParallel(MossFuse(dim=48, num_blocks=3, scale=scale)).to(device)
Hyper_train = HyperDatasetTrain(scale=scale)
Hyper_test = HyperDatasetValid(scale=scale)

datalen = Hyper_train.__len__()
T_max = (datalen//(batch_size))*200
optimizer = torch.optim.Adam(itertools.chain(model.parameters()),lr = init_lr,betas=(0.9,0.999),weight_decay=0)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=1e-5*0.1, last_epoch=-1) 

timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
writer = SummaryWriter(log_dir="models/"+'Tensorboard_'+model_name+'_'+timestamp+"CAVE_32:lr=1e-4*0.9_alpha_2=0.1")
writer.add_text('Training mode: ', timestamp)

MSELoss = nn.MSELoss()  
L1Loss = nn.L1Loss()


trainloader = DataLoader(Hyper_train,batch_size=batch_size,shuffle=False, num_workers=8, pin_memory=False, drop_last=True)
testloader = DataLoader(Hyper_test,batch_size=1,shuffle=False, num_workers=4, pin_memory=False, drop_last=True)


loss_csv_train = open(os.path.join("models/"+timestamp+'train.csv'), 'a+')
loss_csv_test = open(os.path.join("models/"+timestamp+'test.csv'), 'a+')
record_loss(loss_csv_train, 'epoch', 'cc_sim', 'cc_unsim', 'mse_MSI', 'mse_HSI', 'mse_HSI_R', 'mse_MSI_R', 'mse_srf', 'mse_psf')
record_loss(loss_csv_test, 'epoch', 'cc_sim', 'cc_unsim', 'mse_MSI', 'mse_HSI', 'mse_HSI_R', 'mse_MSI_R', 'mse_srf', 'mse_psf')



'''
------------------------------------------------------------------------------
Train
------------------------------------------------------------------------------
'''

step = 0
torch.backends.cudnn.benchmark = True
prev_time = time.time()
srf_g = torch.Tensor(hdf5.loadmat('./dataset/resp.mat')['resp']).to(device)
gaussian_kernel = cv2.getGaussianKernel(scale-1, 2)
psf_g = torch.Tensor(gaussian_kernel * gaussian_kernel.T).to(device)



for epoch in range(num_epochs):
    ''' train '''
    cc_sim_ = []
    cc_unsim_ = []
    mse_MSI_ = []
    mse_HSI_ = []
    mse_HSI_R_ = []
    mse_MSI_R_ = []
    mse_srf_ = []
    mse_psf_ = []
    mse_lr_msi_ = []

    mse_HSI_R_fHSI_ = []
    mse_MSI_R_fHSI_ = []
    mse_HR_HSI_ = []
    loss_decomp_ = []

    for msi, hsi, hsi_g in tqdm(trainloader):
        hsi_ = torch.nn.functional.interpolate(hsi, scale_factor=(scale,scale), mode='bilinear')
        msi, hsi, hsi_, hsi_g = msi.to(device), hsi.to(device), hsi_.to(device), hsi_g.to(device)
        model.train()
        model.zero_grad()
        optimizer.zero_grad()

        msi_spatial_spectral, msi_spatial, hsi_spatial_spectral, hsi_spectral, msi_out, hsi_out, lr_msi_fhsi, lr_msi_fmsi, lr_msi_out, srf, psf, HR_HSI, hsi_fhsi, msi_fhsi = model(msi, hsi)

        cc_loss_sim = L1Loss(msi_spatial_spectral, hsi_spatial_spectral)
        cc_loss_unsim = cc(msi_spatial, hsi_spectral)
        
        mse_loss_msi = L1Loss(msi, msi_out)
        mse_loss_hsi = L1Loss(hsi, hsi_out[:,:,(scale//2-1)::scale,(scale//2-1)::scale])

        mse_LR_MSI = L1Loss(lr_msi_fhsi, lr_msi_fmsi)
        lr_msi_fhsi = torch.nn.functional.interpolate(lr_msi_fhsi, scale_factor=(scale,scale), mode='bilinear')
        lr_msi_fmsi = torch.nn.functional.interpolate(lr_msi_fmsi, scale_factor=(scale,scale), mode='bilinear')
        mse_HSI_R = L1Loss(lr_msi_fhsi, lr_msi_out)
        mse_MSI_R = L1Loss(lr_msi_fmsi, lr_msi_out)

        # loss_decomp = (cc_loss_D) ** 2/ (1.01 + cc_loss_B)  
        loss_decomp = cal_decomp_loss(msi_spatial_spectral, msi_spatial, hsi_spatial_spectral, hsi_spectral)

        mse_srf = L1Loss(srf_g, srf)
        mse_psf = L1Loss(psf_g, psf)

        mse_hsi_fhrHSI = L1Loss(hsi, hsi_fhsi)
        mse_msi_fhrHSI = L1Loss(msi, msi_fhsi)
        mse_HSI = L1Loss(hsi_g, HR_HSI)

        cc_sim_.append(cc_loss_sim.data.cpu().numpy())
        cc_unsim_.append(cc_loss_unsim.data.cpu().numpy())
        mse_MSI_.append(mse_loss_msi.data.cpu().numpy())
        mse_HSI_.append(mse_loss_hsi.data.cpu().numpy())
        mse_HSI_R_.append(mse_HSI_R.data.cpu().numpy())
        mse_MSI_R_.append(mse_MSI_R.data.cpu().numpy())
        mse_srf_.append(mse_srf.data.cpu().numpy())
        mse_psf_.append(mse_psf.data.cpu().numpy())
        mse_lr_msi_.append(mse_LR_MSI.data.cpu().numpy())
        mse_HSI_R_fHSI_.append(mse_hsi_fhrHSI.data.cpu().numpy())
        mse_MSI_R_fHSI_.append(mse_msi_fhrHSI.data.cpu().numpy())
        mse_HR_HSI_.append(mse_HSI.data.cpu().numpy())
        loss_decomp_.append(loss_decomp.data.cpu().numpy())

        # loss1 = mse_loss_msi + mse_loss_hsi + loss_decomp + mse_hsi_fhrHSI + mse_msi_fhrHSI + mse_HSI_R + mse_MSI_R
        loss1 = 0.1*loss_decomp + mse_hsi_fhrHSI + mse_msi_fhrHSI
        for name, param in model.named_parameters():
            if ("blind" in name):
                param.requires_grad = False
            else:
                param.requires_grad = True
        loss1.backward(retain_graph=True)

        loss2 = mse_LR_MSI
        for name, param in model.named_parameters():
            if ("blind" in name):
                param.requires_grad = True
            else:
                param.requires_grad = False
        loss2.backward(retain_graph=True)

        for name, param in model.named_parameters():
            param.requires_grad = True

        optimizer.step()
    record_loss(loss_csv_train, epoch,np.mean(np.array(cc_sim_)),np.mean(np.array(loss_decomp_)),np.mean(np.array(mse_MSI_)),np.mean(np.array(mse_HSI_)),
           np.mean(np.array(mse_HSI_R_)),np.mean(np.array(mse_MSI_R_)),np.mean(np.array(mse_srf_)),np.mean(np.array(mse_psf_)))
    print("Train epoch:%d, sim:%.5f, decom:%.5f, MSI:%.5f, HSI:%.5f, HSI_R:%.5f, MSI_R:%.5f, srf:%.5f, psf:%.5f, lr_msi:%.5f, HR-MSI_R:%.5f, LR-HSI_R:%.5f, HSI:%.5f"%
          (epoch,np.mean(np.array(cc_sim_)),np.mean(np.array(loss_decomp_)),np.mean(np.array(mse_MSI_)),np.mean(np.array(mse_HSI_)),
           np.mean(np.array(mse_HSI_R_)),np.mean(np.array(mse_MSI_R_)),np.mean(np.array(mse_srf_)),np.mean(np.array(mse_psf_)), 
           np.mean(np.array(mse_lr_msi_)), np.mean(np.array(mse_MSI_R_fHSI_)), np.mean(np.array(mse_HSI_R_fHSI_)), np.mean(np.array(mse_HR_HSI_))))

    writer.add_scalar('train/sim', np.mean(np.array(cc_sim_)), epoch)
    writer.add_scalar('train/unsim', np.mean(np.array(cc_unsim_)), epoch)
    writer.add_scalar('train/loss_decomp_', np.mean(np.array(loss_decomp_)), epoch)
    writer.add_scalar('train/MSI_Decoder', np.mean(np.array(mse_MSI_)), epoch)
    writer.add_scalar('train/HSI_Decoder', np.mean(np.array(mse_HSI_)), epoch)
    writer.add_scalar('train/MSI_Reverse', np.mean(np.array(mse_MSI_R_)), epoch)
    writer.add_scalar('train/HSI_Reverse', np.mean(np.array(mse_HSI_R_)), epoch)
    writer.add_scalar('train/SRF', np.mean(np.array(mse_srf_)), epoch)
    writer.add_scalar('train/PSF', np.mean(np.array(mse_psf_)), epoch)

# Test
    loss = 0
    cc_sim_ = []
    cc_unsim_ = []
    mse_MSI_ = []
    mse_HSI_ = []
    mse_HSI_R_ = []
    mse_MSI_R_ = []
    mse_srf_ = []
    mse_psf_ = []
    mse_lr_msi_ = []
    mse_HSI_R_fHSI_ = []
    mse_MSI_R_fHSI_ = []
    mse_HR_HSI_ = []
    PSNR_HR_HSI_ = []
    SSIM_HR_HSI_ = []
    loss_decomp_ = []

    for msi, hsi, hsi_g in tqdm(testloader):
        with torch.no_grad():
            msi, hsi, hsi_, hsi_g = msi.cuda(), hsi.cuda(), hsi_.cuda(), hsi_g.cuda()
            hsi_ = torch.nn.functional.interpolate(hsi, scale_factor=(scale,scale), mode='bilinear')
            model.eval()
            msi_spatial_spectral, msi_spatial, hsi_spatial_spectral, hsi_spectral, msi_out, hsi_out, lr_msi_fhsi, lr_msi_fmsi, lr_msi_out, srf, psf, HR_HSI, hsi_fhsi, msi_fhsi = model(msi, hsi)

            cc_loss_sim = cc(msi_spatial_spectral, hsi_spatial_spectral)
            cc_loss_unsim = cc(msi_spatial, hsi_spectral)
            
            mse_loss_msi = L1Loss(msi, msi_out)
            mse_loss_hsi = L1Loss(hsi, hsi_out[:,:,(scale//2-1)::scale,(scale//2-1)::scale])

            mse_LR_MSI = L1Loss(lr_msi_fhsi, lr_msi_fmsi)
            lr_msi_fhsi = torch.nn.functional.interpolate(lr_msi_fhsi, scale_factor=(scale,scale), mode='bilinear')
            lr_msi_fmsi = torch.nn.functional.interpolate(lr_msi_fmsi, scale_factor=(scale,scale), mode='bilinear')
            mse_HSI_R = L1Loss(lr_msi_fhsi, lr_msi_out)
            mse_MSI_R = L1Loss(lr_msi_fmsi, lr_msi_out)

            loss_decomp = cal_decomp_loss(msi_spatial_spectral, msi_spatial, hsi_spatial_spectral, hsi_spectral)
            mse_srf = L1Loss(srf_g, srf)
            mse_psf = L1Loss(psf_g, psf)            

            mse_hsi_fhrHSI = L1Loss(hsi, hsi_fhsi)
            mse_msi_fhrHSI = L1Loss(msi, msi_fhsi)
            mse_HSI = L1Loss(hsi_g, HR_HSI)
            psnr_HSI, ssim_HSI = PSNR_SSIM_cal(hsi_g, HR_HSI)
 
            cc_sim_.append(np.array(cc_loss_sim.data.cpu()))
            cc_unsim_.append(np.array(cc_loss_unsim.data.cpu()))
            mse_MSI_.append(np.array(mse_loss_msi.data.cpu()))
            mse_HSI_.append(np.array(mse_loss_hsi.data.cpu()))
            mse_HSI_R_.append(np.array(mse_HSI_R.data.cpu()))
            mse_MSI_R_.append(np.array(mse_MSI_R.data.cpu()))
            mse_srf_.append(np.array(mse_srf.data.cpu()))
            mse_psf_.append(np.array(mse_psf.data.cpu()))
            mse_lr_msi_.append(mse_LR_MSI.data.cpu().numpy())
            mse_HSI_R_fHSI_.append(mse_hsi_fhrHSI.data.cpu().numpy())
            mse_MSI_R_fHSI_.append(mse_msi_fhrHSI.data.cpu().numpy())
            mse_HR_HSI_.append(mse_HSI.data.cpu().numpy())
            PSNR_HR_HSI_.append(psnr_HSI)
            SSIM_HR_HSI_.append(ssim_HSI)
            loss_decomp_.append(loss_decomp.data.cpu().numpy())

    record_loss(loss_csv_train, epoch,np.mean(np.array(cc_sim_)),np.mean(np.array(cc_unsim_)),np.mean(np.array(mse_MSI_)),np.mean(np.array(mse_HSI_)),
           np.mean(np.array(mse_HSI_R_)),np.mean(np.array(mse_MSI_R_)),np.mean(np.array(mse_srf_)),np.mean(np.array(mse_psf_)))
    print("Test  epoch:%d, sim:%.5f, decom:%.5f, MSI:%.5f, HSI:%.5f, HSI_R:%.5f, MSI_R:%.5f, srf:%.5f, psf:%.5f, lr_msi:%.5f, HR-MSI_R:%.5f, LR-HSI_R:%.5f, HSI:%.5f, psnr:%.5f, ssim:%.5f"%
          (epoch,np.mean(np.array(cc_sim_)),np.mean(np.array(loss_decomp_)),np.mean(np.array(mse_MSI_)),np.mean(np.array(mse_HSI_)),
           np.mean(np.array(mse_HSI_R_)),np.mean(np.array(mse_MSI_R_)),np.mean(np.array(mse_srf_)),np.mean(np.array(mse_psf_)), 
           np.mean(np.array(mse_lr_msi_)), np.mean(np.array(mse_MSI_R_fHSI_)), np.mean(np.array(mse_HSI_R_fHSI_)), 
           np.mean(np.array(mse_HR_HSI_)), np.mean(np.array(PSNR_HR_HSI_)), np.mean(np.array(SSIM_HR_HSI_))))
    if if_show:
        show(epoch, srf, srf_g, psf, psf_g)

    writer.add_scalar('test/sim', np.mean(np.array(cc_sim_)), epoch)
    writer.add_scalar('test/unsim', np.mean(np.array(cc_unsim_)), epoch)
    writer.add_scalar('test/loss_decomp_', np.mean(np.array(loss_decomp_)), epoch)
    writer.add_scalar('test/MSI_Decoder', np.mean(np.array(mse_MSI_)), epoch)
    writer.add_scalar('test/HSI_Decoder', np.mean(np.array(mse_HSI_)), epoch)
    writer.add_scalar('test/MSI_Reverse', np.mean(np.array(mse_MSI_R_)), epoch)
    writer.add_scalar('test/HSI_Reverse', np.mean(np.array(mse_HSI_R_)), epoch)
    writer.add_scalar('test/SRF', np.mean(np.array(mse_srf_)), epoch)
    writer.add_scalar('test/PSF', np.mean(np.array(mse_psf_)), epoch)
    writer.add_scalar('test/LR_MSI', np.mean(np.array(mse_lr_msi_)), epoch)
    writer.add_scalar('test/PSNR', np.mean(np.array(PSNR_HR_HSI_)), epoch)
    writer.add_scalar('test/SSIM', np.mean(np.array(SSIM_HR_HSI_)), epoch)
    if if_show:
        srf_img = torch.from_numpy(np.transpose(cv2.cvtColor(cv2.imread('models/fig/src_epoch'+str(epoch)+'.png'), cv2.COLOR_BGR2RGB), [2,0,1]))
        psf_img = torch.from_numpy(np.transpose(cv2.cvtColor(cv2.imread('models/fig/psf_epoch'+str(epoch)+'.png'), cv2.COLOR_BGR2RGB), [2,0,1]))
        writer.add_image('fig/SRF_image', srf_img, global_step=None)
        writer.add_image('fig/PSF_image', psf_img, global_step=None)

    # adjust the learning rate
    scheduler.step()  

    if optimizer.param_groups[0]['lr'] <= 1e-6:
        optimizer.param_groups[0]['lr'] = 1e-6
    
# if True:
    checkpoint = {
        'Model_stage1': model.state_dict(),
    }
    if not os.path.exists(os.path.join("models", "pth")):
                          os.mkdir(os.path.join("models", "pth"))
    if not os.path.exists(os.path.join("models", "pth", timestamp)):
        os.mkdir(os.path.join("models", "pth", timestamp))
    torch.save(checkpoint, os.path.join("models", "pth", timestamp, model_name+"_epoch:"+str(epoch)+".pth"))



