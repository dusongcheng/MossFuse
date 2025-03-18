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


model_name = './model/CAVE_32_model.pth'
model_var = 'Model_stage1'
model = MossFuse(dim=48, num_blocks=3, scale=32)

checkpoint = {
    'model': model.state_dict(),
}
torch.save(checkpoint, os.path.join("models.pth"))