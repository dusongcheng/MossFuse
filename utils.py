import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from torch.nn.functional import cosine_similarity
from sewar.full_ref import uqi


def record_loss(loss_csv,epoch, cc_B, cc_D, mse_MSI, mse_HSI, mse_HSI_R, mse_MSI_R, mse_srf, mse_psf):
    """ Record many results."""
    loss_csv.write('{},{},{},{},{},{},{},{},{}\n'.format(epoch, cc_B, cc_D, mse_MSI, mse_HSI, mse_HSI_R, mse_MSI_R, mse_srf, mse_psf))
    loss_csv.flush()    
    loss_csv.close
    
def show(srf, srf_g, psf, psf_g):
    srf = np.array(srf.data.cpu())
    srf_g = np.array(srf_g.data.cpu())
    psf = np.array(psf.data.cpu())
    psf_g = np.array(psf_g.data.cpu())
    # show SRF
    channel = range(31)
    plt.figure(figsize=(10, 6), facecolor='lightgray', edgecolor='black')
    plt.plot(channel, srf[0,:], marker='o', linestyle='--', color='b')
    plt.plot(channel, srf_g[0,:], marker='o', linestyle='-', color='b')
    plt.plot(channel, srf[1,:], marker='o', linestyle='--', color='g')
    plt.plot(channel, srf_g[1,:], marker='o', linestyle='-', color='g')
    plt.plot(channel, srf[2,:], marker='o', linestyle='--', color='r')
    plt.plot(channel, srf_g[2,:], marker='o', linestyle='-', color='r')
    plt.title('Spectral Response Curve')
    plt.xlabel('Spectral')
    plt.ylabel('Response')
    plt.grid(True)
    plt.savefig('./results/src.png')
    #show(PSF)
    plt.figure(figsize=(10, 4), facecolor='lightgray', edgecolor='black')
    plt.subplot(131), plt.imshow(psf, cmap='hot', interpolation='nearest'), plt.title('PSF')
    plt.subplot(132), plt.imshow(psf_g, cmap='hot', interpolation='nearest'), plt.title('PSF_GT')
    plt.subplot(133), plt.imshow(np.abs(psf-psf_g)*10**4, cmap='hot', interpolation='nearest', vmin=0, vmax=1), plt.title('PSF_error x 10**4')
    print(np.abs(psf-psf_g).max())
    plt.savefig('results/psf.png')
    plt.close('all')


def load_model(model, model_name, model_var='Model_stage1'):
    model_param = torch.load(model_name, weights_only=True)[model_var]
    model_dict = {}
    for k1, k2 in zip(model.state_dict(), model_param):
        model_dict[k1] = model_param[k2]
    model.load_state_dict(model_dict)
    return model.cuda()

def PSNR_SSIM_cal(gt, rec):
    gt = gt.detach().cpu().numpy()
    rec = rec.detach().cpu().numpy()
    psnr = cal_psnr(gt[0,:,:,:], rec[0,:,:,:])
    gt = np.transpose(gt,(0,2,3,1))[0,:,:,:]
    rec = np.transpose(rec,(0,2,3,1))[0,:,:,:]
    ssim = compare_ssim(gt, rec, K1 = 0.01, K2 = 0.03, channel_axis=-1, data_range=1)
    return psnr, np.mean(np.array(ssim))

def cal_cos_loss(l1, l2):
    l1 = l1.view(-1)
    l2 = l2.view(-1)
    similarity = cosine_similarity(l1, l2, dim=-1)
    return similarity


def cal_decomp_loss(RGB_spatial_spectral, RGB_spatial, HSI_spatial_spectral, HSI_spectral):
    positive = torch.exp(cal_cos_loss(RGB_spatial_spectral, HSI_spatial_spectral))
    negative = torch.exp(cal_cos_loss(RGB_spatial_spectral, RGB_spatial)) + torch.exp(cal_cos_loss(HSI_spatial_spectral, HSI_spectral)) + torch.exp(cal_cos_loss(RGB_spatial, HSI_spectral))
    decomp_loss = -torch.log(positive/(positive+negative))
    return decomp_loss


def cal_psnr(label, output):

    img_c, img_w, img_h = label.shape
    ref = label.reshape(img_c, -1)
    tar = output.reshape(img_c, -1)
    msr = np.mean((ref - tar) ** 2, 1)
    max1 = np.max(ref, 1)

    psnrall = 10 * np.log10(1 / msr)
    out_mean = np.mean(psnrall)
    # return out_mean, max1
    return out_mean


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class AverageMeter_valid(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = np.zeros([1,6])
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val*np.array(n)
        self.count += n
        self.avg = self.sum / self.count

class Loss_valid(nn.Module):
    def __init__(self, scale=8):
        super(Loss_valid, self).__init__()
        self.scale=scale

    def forward(self, label_image, rec_image):
        self.batch_size = label_image.shape[0]
        assert self.batch_size == 1
        self.label = label_image.data.cpu().squeeze(0).numpy()
        self.output = rec_image.data.cpu().squeeze(0).numpy()
        self.output = np.clip(self.output, 0, 1)
        valid_error = np.zeros([1, 6])

        valid_error[0, 0] = self.ssim()
        valid_error[0, 1] = self.cal_rmse()
        valid_error[0, 2] = self.cal_psnr()
        valid_error[0, 3] = self.cal_ergas()
        valid_error[0, 4] = self.sam()
        valid_error[0, 5] = self.cal_uqi()
        return valid_error

    def cal_mrae(self):
        error = np.abs(self.output - self.label) / self.label
        # error = torch.abs(outputs - label)
        mrae = np.mean(error.reshape(-1))
        return mrae

    def cal_rmse(self):
        rmse = np.sqrt(np.mean((self.label-self.output)**2))
        return rmse

    def cal_psnr(self):
        
        assert self.label.ndim == 3 and self.output.ndim == 3

        img_c, img_w, img_h = self.label.shape
        ref = self.label.reshape(img_c, -1)
        tar = self.output.reshape(img_c, -1)
        msr = np.mean((ref - tar) ** 2, 1)
        max1 = np.max(ref, 1)

        psnrall = 10 * np.log10(1 / msr)
        out_mean = np.mean(psnrall)
        # return out_mean, max1
        return out_mean

    def cal_ergas(self, scale=32):
        d = self.label - self.output
        ergasroot = 0
        for i in range(d.shape[0]):
            ergasroot = ergasroot + np.mean(d[i, :, :] ** 2) / np.mean(self.label[i, :, :]) ** 2
        ergas = (100 / scale) * np.sqrt(ergasroot/(d.shape[0]+1))
        return ergas

    def cal_sam(self):
        assert self.label.ndim == 3 and self.label.shape == self.label.shape

        c, w, h = self.label.shape
        x_true = self.label.reshape(c, -1)
        x_pred = self.output.reshape(c, -1)

        x_pred[np.where((np.linalg.norm(x_pred, 2, 1)) == 0),] += 0.0001

        sam = (x_true * x_pred).sum(axis=1) / (np.linalg.norm(x_true, 2, 1) * np.linalg.norm(x_pred, 2, 1))

        sam = np.arccos(sam) * 180 / np.pi
        # sam = np.arccos(sam)
        mSAM = sam.mean()
        var_sam = np.var(sam)
        # return mSAM, var_sam
        return mSAM

    def cal_ssim(self, data_range=1, multidimension=False):
        """
        :param x_true:
        :param x_pred:
        :param data_range:
        :param multidimension:
        :return:
        """
        mssim = [
            compare_ssim(X=self.label[i, :, :], Y=self.output[i, :, :], data_range=data_range, multidimension=multidimension)
            for i in range(self.label.shape[0])]
        return np.mean(mssim)

    def cal_uqi(self):
        fout = np.transpose(self.output, [1,2,0])
        hsi_g = np.transpose(self.label, [1,2,0])
        uqi_ = uqi(hsi_g, fout)
        return uqi_

    def ssim(self):
        fout_0 = np.transpose(self.output, [1,2,0])
        hsi_g_0 = np.transpose(self.label, [1,2,0])
        # ssim_result = compare_ssim(fout_0, hsi_g_0, data_range=1)
        ssim = compare_ssim(hsi_g_0, fout_0, K1 = 0.01, K2 = 0.03, channel_axis=-1, data_range=1)
        return ssim
    
    def psnr(self):
        fout = self.output
        hsi_g = self.label
        psnr_g = []
        for i in range(31):
            psnr_g.append(compare_psnr(hsi_g[i,:,:],fout[i,:,:]))
        return np.mean(np.array(psnr_g))

    def sam(self):
        """
        cal SAM between two images
        :param groundTruth: ground truth reference image. (Height x Width x Spectral_Dimension)
        :param recovered: image under evaluation. (Height x Width x Spectral_Dimension)
        :return: Spectral Angle Mapper between `recovered` and `groundTruth`.
        """
        groundTruth = np.transpose(self.label, [1,2,0])
        recovered = np.transpose(self.output, [1,2,0])
        assert groundTruth.shape == recovered.shape, "Size not match for groundtruth and recovered spectral images"

        nom = np.sum(groundTruth * recovered, 2)
        denom1 = np.sqrt(np.sum(groundTruth**2, 2))
        denom2 = np.sqrt(np.sum(recovered ** 2, 2))
        sam = np.arccos(np.divide(nom, denom1*denom2))
        sam = np.divide(sam, np.pi) * 180.0
        sam = np.mean(sam)

        return sam
    








