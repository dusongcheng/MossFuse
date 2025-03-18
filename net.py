import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from LKNet import UniRepLKNetBlock
import numbers
os.environ['CUDA_VISIBLE_DEVICES'] = '0'



##########################################################################
## Layer Norm
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3,
                              stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x





class Spatial_Attention(nn.Module):
    def __init__(self, dim=32, expansion_factor=2):
        super(Spatial_Attention, self).__init__()
        self.spatial_atten = nn.Sequential(
            nn.Conv2d(dim, dim//8, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace = True),
            nn.Conv2d(dim//8, dim//8, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace = True),
            nn.Conv2d(dim//8, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_att = self.spatial_atten(x)
        return x_att

class Spectral_Attention(nn.Module):
    def __init__(self, dim=32, expansion_factor=2):
        super(Spectral_Attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.spectral_atten = nn.Sequential(
            nn.Linear(dim, dim//8, bias=False),
            nn.GELU(),
            nn.Linear(dim//8, dim//8, bias=False),
            nn.GELU(),
            nn.Linear(dim//8, dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.spectral_atten(y).view(b, c, 1, 1)
        return y.expand_as(x)


class Cross_Spatial_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Cross_Spatial_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.Spatial_Att = Spatial_Attention(dim)
        self.Spectral_Att = Spectral_Attention(dim)
        self.DWConv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        spatial_att = self.Spatial_Att(y)
        q = q*spatial_att

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)
        out_spectral_att = self.Spectral_Att(out)

        y = self.DWConv(y)*out_spectral_att

        out = self.project_out(out+y)
        return out


class Cross_Spectral_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Cross_Spectral_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.Spatial_Att = Spatial_Attention(dim)
        self.Spectral_Att = Spectral_Attention(dim)
        self.DWConv = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        spectral_att = self.Spectral_Att(y)
        q = q*spectral_att

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)
        out_spatial_att = self.Spatial_Att(out)

        y = self.DWConv(y)*out_spatial_att

        out = self.project_out(out+y)
        return out


class Cross_Spatial_TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(Cross_Spatial_TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Cross_Spatial_Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, y):
        x = x + self.attn(self.norm1(x), y)
        x = x + self.ffn(self.norm2(x))
        return x


class Cross_Spectral_TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(Cross_Spectral_TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Cross_Spectral_Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x, y):
        x = x + self.attn(self.norm1(x), y)
        x = x + self.ffn(self.norm2(x))
        return x

class Restormer_Encoder_MSI(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=32,
                 dim=64,
                 num_blocks=4,
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):

        super(Restormer_Encoder_MSI, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks)])
        self.baseFeature = nn.Sequential(UniRepLKNetBlock(dim=dim, kernel_size=7, attempt_use_lk_impl=False),
                                         UniRepLKNetBlock(dim=dim, kernel_size=0, attempt_use_lk_impl=False),
                                         UniRepLKNetBlock(dim=dim, kernel_size=0, attempt_use_lk_impl=False),
                                         TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
                                         )
        self.detailFeature = nn.Sequential(UniRepLKNetBlock(dim=dim, kernel_size=7, attempt_use_lk_impl=False),
                                         UniRepLKNetBlock(dim=dim, kernel_size=0, attempt_use_lk_impl=False),
                                         UniRepLKNetBlock(dim=dim, kernel_size=0, attempt_use_lk_impl=False)
                                         )
             
    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        base_feature = self.baseFeature(out_enc_level1)
        detail_feature = self.detailFeature(out_enc_level1)
        return base_feature, detail_feature, out_enc_level1


class Restormer_Encoder_HSI(nn.Module):
    def __init__(self,
                 inp_channels=31,
                 out_channels=32,
                 dim=64,
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):

        super(Restormer_Encoder_HSI, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks)])
        self.baseFeature = nn.Sequential(TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type),
                                         UniRepLKNetBlock(dim=dim, kernel_size=7, attempt_use_lk_impl=False),
                                         UniRepLKNetBlock(dim=dim, kernel_size=0, attempt_use_lk_impl=False),
                                         UniRepLKNetBlock(dim=dim, kernel_size=0, attempt_use_lk_impl=False)
                                         )
        self.detailFeature = TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
             
    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        base_feature = self.baseFeature(out_enc_level1)
        detail_feature = self.detailFeature(out_enc_level1)
        return base_feature, detail_feature, out_enc_level1

class Restormer_Decoder_MSI(nn.Module):
    def __init__(self,
                 inp_channels=32,
                 out_channels=3,
                 dim=64,
                 num_blocks=4,
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):

        super(Restormer_Decoder_MSI, self).__init__()
        self.reduce_channel = nn.Conv2d(int(dim*2), int(dim), kernel_size=1, bias=bias)
        self.fuse = Cross_Spatial_TransformerBlock(dim=dim, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks-1)])
        self.output = nn.Sequential(
            nn.Conv2d(int(dim), int(dim)//2, kernel_size=3,
                      stride=1, padding=1, bias=bias),
            nn.LeakyReLU(),
            nn.Conv2d(int(dim)//2, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=bias),)
        self.sigmoid = nn.Sigmoid()              
    def forward(self, base_feature, detail_feature):
        out_enc_level0 = self.fuse(base_feature, detail_feature)
        out_enc_level1 = self.encoder_level2(out_enc_level0)
        out_enc_level1 = self.output(out_enc_level1)
        return out_enc_level0, out_enc_level1


class Restormer_Decoder_HSI(nn.Module):
    def __init__(self,
                 inp_channels=32,
                 out_channels=31,
                 dim=64,
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):

        super(Restormer_Decoder_HSI, self).__init__()
        self.reduce_channel = nn.Conv2d(int(dim*2), int(dim), kernel_size=1, bias=bias)
        self.fuse = Cross_Spectral_TransformerBlock(dim=dim, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type)
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks-1)])
        self.output = nn.Sequential(
            nn.Conv2d(int(dim), int(dim)//2, kernel_size=3,
                      stride=1, padding=1, bias=bias),
            nn.LeakyReLU(),
            nn.Conv2d(int(dim)//2, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=bias),)
        self.sigmoid = nn.Sigmoid()              
    def forward(self, base_feature, detail_feature):
        out_enc_level0 = self.fuse(base_feature, detail_feature)
        out_enc_level1 = self.encoder_level2(out_enc_level0)
        out_enc_level1 = self.output(out_enc_level1)
        return out_enc_level0, out_enc_level1


class Restormer_Decoder_LR(nn.Module):
    def __init__(self,
                 inp_channels=32,
                 out_channels=3,
                 dim=64,
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):

        super(Restormer_Decoder_LR, self).__init__()
        self.reduce_channel = nn.Conv2d(int(dim*2), int(dim), kernel_size=1, bias=bias)
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks)])
        self.output = nn.Sequential(
            nn.Conv2d(int(dim), int(dim)//2, kernel_size=3,
                      stride=1, padding=1, bias=bias),
            nn.LeakyReLU(),
            nn.Conv2d(int(dim)//2, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=bias),)
        self.sigmoid = nn.Sigmoid()              
    def forward(self, base_feature, detail_feature):
        out_enc_level0 = torch.cat((base_feature, detail_feature), dim=1)
        out_enc_level0 = self.reduce_channel(out_enc_level0)
        out_enc_level1 = self.encoder_level2(out_enc_level0)
        out_enc_level1 = self.output(out_enc_level1)
        return out_enc_level1


class Restormer_Decoder_HR(nn.Module):
    def __init__(self,
                 inp_channels=32,
                 out_channels=31,
                 dim=64,
                 num_blocks=[4, 4],
                 heads=[8, 8, 8],
                 ffn_expansion_factor=2,
                 bias=False,
                 LayerNorm_type='WithBias',
                 ):

        super(Restormer_Decoder_HR, self).__init__()
        self.reduce_channel = nn.Conv2d(int(dim*2), int(dim), kernel_size=1, bias=bias)
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor,
                                            bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks)])
        self.output = nn.Sequential(
            nn.Conv2d(int(dim), int(dim)//2, kernel_size=3,
                      stride=1, padding=1, bias=bias),
            nn.LeakyReLU(),
            nn.Conv2d(int(dim)//2, out_channels, kernel_size=3,
                      stride=1, padding=1, bias=bias),)
        self.sigmoid = nn.Sigmoid()              
    def forward(self, base_feature, detail_feature):
        out_enc_level0 = torch.cat((base_feature, detail_feature), dim=1)
        out_enc_level0 = self.reduce_channel(out_enc_level0)
        out_enc_level1 = self.encoder_level2(out_enc_level0)
        out_enc_level1 = self.output(out_enc_level1)
        return out_enc_level1



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



class MossFuse(nn.Module):
    def __init__(self, dim=48, num_blocks=3, scale=8):
        super(MossFuse, self).__init__()
        self.scale = scale
        self.modelE_MSI = Restormer_Encoder_MSI(dim=dim, num_blocks=num_blocks)
        self.modelE_HSI = Restormer_Encoder_HSI(dim=dim, num_blocks=num_blocks)
        self.modelD_MSI = Restormer_Decoder_MSI(dim=dim, num_blocks=num_blocks)
        self.modelD_HSI = Restormer_Decoder_HSI(dim=dim, num_blocks=num_blocks)
        self.blind = BlindNet(ker_size=scale-1, ratio=scale)
        self.modelD_LR = Restormer_Decoder_LR(dim=dim, num_blocks=num_blocks)
        self.modelD_HR = Restormer_Decoder_HR(dim=dim, num_blocks=num_blocks)
        self.spatial_down = BlurDown(ratio=scale)
        self.spectral_down = SRF_Down()

    def forward(self, MSI, HSI):
        HSI_ = torch.nn.functional.interpolate(HSI, scale_factor=(self.scale, self.scale), mode='bilinear')
        RGB_E = self.modelE_MSI(MSI)
        HSI_E = self.modelE_HSI(HSI_)
        RGB_f, RGB_D = self.modelD_MSI(RGB_E[0], RGB_E[1])
        HSI_f, HSI_D = self.modelD_HSI(HSI_E[0], HSI_E[1])
        lr_msi_fhsi, lr_msi_fmsi, srf, psf = self.blind(HSI, MSI)
        LR_D = self.modelD_LR(RGB_E[0], HSI_E[0])
        # lr_msi_fhsi = torch.nn.functional.interpolate(lr_msi_fhsi, scale_factor=(8,8), mode='bilinear')
        # lr_msi_fmsi = torch.nn.functional.interpolate(lr_msi_fmsi, scale_factor=(8,8), mode='bilinear')

        HR_HSI = self.modelD_HR(RGB_f, HSI_f)
        msi_fhsi = self.spectral_down(HR_HSI, srf)
        hsi_fhsi = self.spatial_down(HR_HSI, psf)

        return RGB_E[0], RGB_E[1], HSI_E[0], HSI_E[1], RGB_D, HSI_D, lr_msi_fhsi, lr_msi_fmsi, LR_D, srf, psf, HR_HSI, hsi_fhsi, msi_fhsi


def count_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)



if __name__ == '__main__':

    height = 128
    width = 128
    window_size = 8
    input_tensor1 = torch.rand(2, 3, height, width).cuda()
    input_tensor2 = torch.rand(2, 31, 16, 16).cuda()
    model = MossFuse().cuda()
    with torch.no_grad():
        RGB_E0, EGB_E1, HSI_E0, HSI_E1, RGB_D, HSI_D, lr_msi_fhsi, lr_msi_fmsi, LR_D, srf, psf, HR_HSI, hsi_fhsi, msi_fhsi = model(input_tensor1, input_tensor2)
    print(RGB_D.size())
    print(HSI_D.size())
    print('Parameters number of modelE_MSI is ', sum(param.numel() for param in model.parameters()))
    print(torch.__version__)

    # 遍历模型的每个模块并打印参数量
    for name, module in model.named_modules():
        if name != '':  # 排除根模块
            params = count_parameters(module)
            if name in ['modelE_MSI', 'modelE_HSI', 'modelD_MSI', 'modelD_HSI', 'blind', 'modelD_LR', 'modelD_HR', 'spatial_down', 'spectral_down']:
                print(f"Module: {name}, Parameters: {params}")

    # input_tensor1 = torch.rand(1, 64, 128, 128).cuda()
    # input_tensor2 = torch.rand(1, 64, 128, 128).cuda()
    # model2 = Model_stage2().cuda()
    # with torch.no_grad():
    #     HSI = model2((input_tensor1, input_tensor2), (input_tensor1, input_tensor2))
    # print(HSI.shape)

    # input_tensor1 = torch.rand(1, 31, 128, 128).cuda()
    # input_tensor2 = torch.rand(1, 31, 128, 128).cuda()
    # model = Model_stage1().cuda()
    # lr_msi_fhsi, lr_msi_fmsi = model(input_tensor1, input_tensor2, 0)
    # print(lr_msi_fhsi.shape)