import random
import h5py
import numpy as np
import torch
import torch.utils.data as udata
import glob
import os
import hdf5storage as hdf5
import scipy.io as scio
import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm

valid_name_list = ['hairs_ms', 'balloons_ms', 'real_and_fake_peppers_ms', 'stuffed_toys_ms', 'thread_spools_ms', 'fake_and_real_tomatoes_ms', 'fake_and_real_lemons_ms', 'egyptian_statue_ms', 'clay_ms', 'real_and_fake_apples_ms', 'fake_and_real_beers_ms', 'fake_and_real_peppers_ms']
train_name_list = ['watercolors_ms', 'beads_ms', 'fake_and_real_sushi_ms', 'pompoms_ms', 'sponges_ms', 'cloth_ms', 'oil_painting_ms', 'flowers_ms', 'cd_ms', 'superballs_ms', 'fake_and_real_lemon_slices_ms', 'fake_and_real_food_ms', 'paints_ms', 'face_ms', 'feathers_ms', 'chart_and_stuffed_toy_ms', 'jelly_beans_ms', 'photo_and_face_ms', 'fake_and_real_strawberries_ms', 'glass_tiles_ms']
# train_name_list = ['watercolors_ms', 'beads_ms']



class HyperDatasetValid(udata.Dataset):
    def __init__(self, base_root, scale=32):
        data_path = os.path.join(base_root)
        data_names = valid_name_list
        self.res_path = './dataset/resp.mat'
        self.keys = data_names
        self.keys.sort()
        res = hdf5.loadmat(self.res_path)['resp']
        res = np.transpose(res, (1, 0))
        self.hyper_list = []
        self.hyper1_list = []
        self.rgb_list = []
        for i in range(len(self.keys)):
            mat = hdf5.loadmat(os.path.join(data_path, self.keys[i]))
            hyper = np.float32(np.array(mat['rad'])/(2**16-1))
            hyper1 = cv2.GaussianBlur(hyper,((scale-1),(scale-1)),2)[(scale//2-1)::scale,(scale//2-1)::scale,:]
            rgb = np.tensordot(hyper, res, (-1, 0))
            hyper1 = np.transpose(hyper1, [2, 0, 1])
            hyper = np.transpose(hyper, [2, 0, 1])
            rgb = np.transpose(rgb, [2, 0, 1])
            hyper = torch.Tensor(hyper)
            hyper1 = torch.Tensor(hyper1)
            rgb = torch.Tensor(rgb)
            self.hyper_list.append(hyper)
            self.hyper1_list.append(hyper1)
            self.rgb_list.append(rgb)
        print(str(len(self.keys))+' test image pairs loaded!')

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        rgb = self.rgb_list[index]
        hyper1 = self.hyper1_list[index]
        hyper = self.hyper_list[index]
        return rgb, hyper1, hyper


class HyperDatasetTrain(udata.Dataset):
    def __init__(self, base_root, scale=32):
        self.baseroot = os.path.join(base_root)
        self.res_path = './dataset/resp.mat'
        self.keys = train_name_list
        self.num_pre_img = 7
        self.train_len = len(self.keys)*(self.num_pre_img)**2
        self.lr_size = 128//scale
        self.msi_list = []
        self.hsi_g_list = []
        self.hsi_list = []
        res = hdf5.loadmat(self.res_path)['resp']
        res = np.transpose(res, (1, 0))
        for i in range(len(self.keys)):
            mat_path = os.path.join(self.baseroot, self.keys[i])
            mat = hdf5.loadmat(mat_path)
            hyper = np.float32(np.array(mat['rad'])/(2**16-1))
            hyper1 = cv2.GaussianBlur(hyper,((scale-1),(scale-1)),2)[(scale//2-1)::scale,(scale//2-1)::scale,:]
            rgb = np.tensordot(hyper, res, (-1, 0))
            self.hsi_g_list.append(hyper)
            self.hsi_list.append(hyper1)
            self.msi_list.append(rgb)
        print(str(len(self.keys))+' train image pairs loaded!')

    def __len__(self):
        return len(self.keys)*(self.num_pre_img)**2

    def __getitem__(self, index):
        index_img = index // self.num_pre_img**2 
        index_inside_image = index % self.num_pre_img**2 
        index_row = index_inside_image // self.num_pre_img 
        index_col = index_inside_image % self.num_pre_img

        hsi_g = self.hsi_g_list[index_img][index_row*64:(index_row*64+128),index_col*64:(index_col*64+128),:]
        hsi = self.hsi_list[index_img][index_row*(self.lr_size//2):(index_row*(self.lr_size//2)+self.lr_size),index_col*(self.lr_size//2):(index_col*(self.lr_size//2)+self.lr_size),:]
        msi = self.msi_list[index_img][index_row*64:(index_row*64+128),index_col*64:(index_col*64+128),:]
        
        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)

        for j in range(rotTimes):
            hsi_g = np.rot90(hsi_g)
            hsi = np.rot90(hsi)
            msi = np.rot90(msi)

        # Random vertical Flip   
        for j in range(vFlip):
            hsi_g = np.flip(hsi_g,axis=1)
            hsi = np.flip(hsi,axis=1)
            msi = np.flip(msi,axis=1)
    
        # Random Horizontal Flip
        for j in range(hFlip):
            hsi_g = np.flip(hsi_g,axis=0)
            hsi = np.flip(hsi,axis=0)
            msi = np.flip(msi,axis=0)

        hsi = np.transpose(hsi,(2,0,1)).copy()
        msi = np.transpose(msi,(2,0,1)).copy()
        hsi_g = np.transpose(hsi_g,(2,0,1)).copy()

        hsi = torch.Tensor(hsi)
        msi = torch.Tensor(msi)
        hsi_g = torch.Tensor(hsi_g)

        return msi, hsi, hsi_g


class HyperDatasetTest(udata.Dataset):
    def __init__(self, base_root, scale=32):
        self.baseroot = os.path.join(base_root)
        self.res_path = './dataset/resp.mat'
        self.keys = valid_name_list
        self.keys.sort()
        res = hdf5.loadmat(self.res_path)['resp']
        res = np.transpose(res, (1, 0))
        self.hyper_list = []
        self.hyper1_list = []
        self.rgb_list = []
        for i in range(len(self.keys)):
            mat = hdf5.loadmat(os.path.join(self.baseroot, self.keys[i]))
            hyper = np.float32(np.array(mat['rad'])/(2**16-1))
            hyper1 = cv2.GaussianBlur(hyper,((scale-1),(scale-1)),2)[(scale//2-1)::scale,(scale//2-1)::scale,:]
            rgb = np.tensordot(hyper, res, (-1, 0))
            hyper1 = np.transpose(hyper1, [2, 0, 1])
            hyper = np.transpose(hyper, [2, 0, 1])
            rgb = np.transpose(rgb, [2, 0, 1])
            hyper = torch.Tensor(hyper)
            hyper1 = torch.Tensor(hyper1)
            rgb = torch.Tensor(rgb)
            self.hyper_list.append(hyper)
            self.hyper1_list.append(hyper1)
            self.rgb_list.append(rgb)
        print(str(len(self.keys))+' test image pairs loaded!')

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        rgb = self.rgb_list[index]
        hyper1 = self.hyper1_list[index]
        hyper = self.hyper_list[index]
        img_name = self.keys[index].split('/')[-1]
        return rgb, hyper1, hyper, img_name


if __name__ == '__main__':
    train_data1 = HyperDatasetTrain(mode='train', scale=32)
    train_loader = DataLoader(dataset=train_data1, batch_size=16, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    for j in range(10):
        for i, (images, labels, hyper_downsapmle) in tqdm(enumerate(train_loader)):
            # print(i)
            print(images.dtype, end=', ')
            print(images.shape, end=', ')
            print(labels.shape, end=', ')
            print(hyper_downsapmle.shape)
