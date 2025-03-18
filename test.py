import argparse
import os
import torch
import numpy as np
import cv2
import hdf5storage as hdf5
import time
from utils import Loss_valid, AverageMeter_valid, load_model, show
from dataset import HyperDatasetTest
from net import MossFuse
from tqdm import tqdm
import time
import datetime
scale = 32
def generate_psf_srf(scale):
    srf_g = torch.Tensor(hdf5.loadmat('./dataset/resp.mat')['resp']).cuda()
    gaussian_kernel = cv2.getGaussianKernel(scale-1, 2)
    psf_g = torch.Tensor(gaussian_kernel * gaussian_kernel.T).cuda()
    return psf_g, srf_g

def validate(val_loader, model, criterion, save, save_path):
    model.eval()
    losses = AverageMeter_valid()
    timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
    inference_time = 0
    psf_g, srf_g = generate_psf_srf(scale)
    for msi, hsi, hsi_g, img_name in tqdm(val_loader):
        with torch.no_grad():
            msi, hsi, hsi_g = msi.cuda(), hsi.cuda(), hsi_g.cuda()
            model.eval()
            start_time = time.time()
            _, _, _, _, _, _, _, _, _, srf, psf, HR_HSI, _, _ = model(msi, hsi)
            inference_time = inference_time + time.time() - start_time
            show(srf, srf_g, psf, psf_g)
            print('Inferen time of %s: %6f'%(img_name[0], time.time() - start_time))
            print('Mean L1 loss of %s: %6f'%(img_name[0], torch.mean(torch.abs(hsi_g-HR_HSI)).data.cpu().numpy()))
        if save:
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            if not os.path.exists(os.path.join(save_path, timestamp)):
                os.mkdir(os.path.join(save_path, timestamp))
            out = HR_HSI.clone().data.permute(0, 2, 3, 1).cpu().numpy()[0, :, :, :].astype(np.float32)
            save_img_path = os.path.join(save_path, timestamp, img_name[0])
            print(save_img_path)
            hdf5.write(data=out, path='cube', filename=save_img_path, matlab_compatible=True)
        loss = criterion(hsi_g, HR_HSI)
        losses.update(loss.data)
    return losses.avg

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.chdir('./MossFuse-master')
    dataset_root = './dataset/CAVE'
    model_name = './model/CAVE_32_model.pth'
    save_path = './results'
    model = MossFuse(dim=48, num_blocks=3, scale=scale)
    model = load_model(model=model, model_name=model_name, model_var='Model_stage1')
    test_dataset = HyperDatasetTest(base_root=dataset_root, scale=scale)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)
    print('Network name: %s;' %model_name)
    criterion_valid = Loss_valid(scale).cuda()
    loss = validate(test_loader, model, criterion_valid, save=False, save_path=save_path)
    print('psnr:        rmse:       ssim:       sam:        ergas:      UIQI:')
    print("%5f,   %5f,   %5f,   %5f,   %5f,   %5f"%(loss[0][2], loss[0][1], loss[0][0], loss[0][4], loss[0][3], loss[0][5]))

