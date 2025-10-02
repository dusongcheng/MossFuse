import torch
import torch.nn as nn
import torch.nn.functional as F



def kernel_generator(Q,
                     kernel_size: int,
                     scale_factor: int,
                     shift='center'):
    """
    modified version of https://github.com/zsyOAOA/BSRDM
    """
    mask = torch.tensor([[1.0, 0.0],
                         [1.0, 1.0]], dtype=torch.float32).to(Q.device)
    M = Q * mask
    INV_SIGMA = torch.mm(M.t(), M)

    # Set expectation position (shifting kernel for aligned image)
    if shift.lower() == 'left':
        MU = kernel_size // 2 - 0.5 * (scale_factor - 1)
    elif shift.lower() == 'center':
        MU = kernel_size // 2
    elif shift.lower() == 'right':
        MU = kernel_size // 2 + 0.5 * (scale_factor - 1)

    # Create meshgrid for Gaussian
    X, Y = torch.meshgrid(torch.arange(kernel_size), torch.arange(kernel_size))
    Z = torch.stack((X, Y), dim=2).unsqueeze(3).to(Q.device)  # k x k x 2 x 1

    # Calcualte Gaussian for every pixel of the kernel
    ZZ = Z - MU
    ZZ = ZZ.type(torch.float32)
    ZZ_t = ZZ.permute(0, 1, 3, 2)  # k x k x 1 x 2
    raw_kernel = torch.exp(-0.5 * torch.squeeze(ZZ_t.matmul(INV_SIGMA).matmul(ZZ)))

    # Normalize the kernel and return
    kernel = raw_kernel / torch.sum(raw_kernel)  # k x k

    return kernel.unsqueeze(0).unsqueeze(0)

class GaussianKernel(nn.Module):
    def __init__(self, kernel_size, scale_factor):
        super(GaussianKernel, self).__init__()
        self.kernel_size = kernel_size
        self.scale_factor = scale_factor

        self.KernelParam = nn.Parameter(5 * torch.eye(2, 2))

    def re_init(self):
        self.KernelParam = nn.Parameter(5 * torch.eye(2, 2))

    def forward(self, Z):
        """
        :param Z: [batchsize, bands, block_height, block_width]
        :return:
        """
        _,c,_,_ = Z.shape
        self.KernelAdaption = kernel_generator(self.KernelParam, self.kernel_size, self.scale_factor, shift='center')
        X_r = F.conv2d(Z, self.KernelAdaption.repeat(c, 1, 1, 1), groups=c)
        X_r = X_r[:, :, 0::self.scale_factor, 0::self.scale_factor]

        return X_r, self.KernelAdaption[0,0,:,:]

class BlurDown(object):
    def __init__(self, ratio):
        self.ratio = ratio
        #self.shift_h = shift_h
        #self.shift_w = shift_w
        #self.stride = stride

    def __call__(self, input_tensor, psf):
        if psf.dim() == 2:
            psf = psf.unsqueeze(0).unsqueeze(0)
        psf = psf.repeat(input_tensor.shape[1], 1, 1, 1) #8X1X8X8
        output_tensor = F.conv2d(input_tensor, psf, None, (self.ratio, self.ratio),  groups=input_tensor.shape[1]) #ratio为步长 None代表bias为0，padding默认为无
        return output_tensor

class SRF_Down(object):
    def __init__(self):
        pass
    def __call__(self, input_tensor, srf):
        if srf.dim() == 2:
            srf = srf.unsqueeze(2).unsqueeze(3)
        output_tensor = F.conv2d(input_tensor, srf, None)
        return output_tensor

class BlindNet(nn.Module):
    def __init__(self, hs_bands=31, ms_bands=3, ker_size=7, ratio=8):
        super().__init__()
        self.hs_bands = hs_bands
        self.ms_bands = ms_bands
        self.ker_size = ker_size #8
        self.ratio = ratio #8
        
        #psf = torch.rand([1, 1, self.ker_size, self.ker_size]) #0-1均匀分布
        # psf = torch.ones([1, 1, self.ker_size, self.ker_size]) * (1.0 / (self.ker_size ** 2))
        # self.psf = nn.Parameter(psf)
        
        #srf = torch.rand([self.ms_bands, self.hs_bands, 1, 1]) #0-1均匀分布
        # srf = torch.ones([self.ms_bands, self.hs_bands, 1, 1]) * (1.0 / self.hs_bands) 
        srf = torch.ones([self.ms_bands, self.hs_bands, 1, 1])
        self.srf = nn.Parameter(srf)
        self.blur_down = GaussianKernel(ker_size, ratio)

    def forward(self, lr_hsi, hr_msi):
        
        srf_div = torch.sum(self.srf, dim=1, keepdim=True) # 8 x 1x 1 x 1
        srf_div = torch.div(1.0, srf_div)     #8 x 1x 1 x 1
        srf_div = torch.transpose(srf_div, 0, 1)

        lr_msi_fhsi = F.conv2d(lr_hsi, self.srf, None) #(1,8,30, 30)
        lr_msi_fhsi = torch.mul(lr_msi_fhsi, srf_div)
        lr_msi_fhsi = torch.clamp(lr_msi_fhsi, 0.0, 1.0)
        lr_msi_fmsi, psf = self.blur_down(hr_msi)
        lr_msi_fmsi = torch.clamp(lr_msi_fmsi, 0.0, 1.0)
        return_srf = torch.div(self.srf, torch.sum(self.srf, dim=1, keepdim=True))[:,:,0,0]
        return_psf = psf

        return lr_msi_fhsi, lr_msi_fmsi, return_srf, return_psf

