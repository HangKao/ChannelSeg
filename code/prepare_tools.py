import torch
import keras
from torch.utils.data import Dataset
import os
import numpy as np

class ChannDataset(Dataset):
    def __init__(self, dpth, fpth ,dimension, chann, num_sample):
        self.dpth = dpth
        self.fpth = fpth
        dpth_list = os.listdir(self.dpth)
        self.dfile = [self.dpth + file for file in dpth_list]
        fpth_list = os.listdir(self.fpth)
        self.ffile = [self.fpth + file for file in fpth_list]
        self.num_sample = num_sample
        self.dfile = self.dfile[:self.num_sample]
        self.ffile = self.ffile[:self.num_sample]
        self.dim = dimension
        self.chann = chann
        self.num_file = len(dpth_list)

    # def __getitem__(self, item):
    #     dres = np.fromfile(self.dfile[item])
    #     dres = dres.reshape(self.dimension)
    #     fres = np.fromfile(self.ffile[item])
    #     fres = fres.reshape(self.dimension)
    #     rots = np.random.randint(0, 4)
    #     img = np.rot90(dres, rots, (0, 1))
    #     label = np.rot90(fres, rots, (0, 1))
    #     img = img[np.newaxis,:]
    #     img = img - np.min(img)
    #     img = img / np.max(img)
    #     img = img * 255
    #     img.astype(np.float32)
    #     return (img,label)

    def __getitem__(self, item):
        gx = np.fromfile(self.dfile[item],dtype=np.single)
        fx = np.fromfile(self.ffile[item],dtype=np.single)
        gx = np.reshape(gx, self.dim)
        fx = np.reshape(fx, self.dim)
        gx = gx - np.min(gx)
        gx = gx / np.max(gx)
        #gx = gx * 255

        # augment
        X = np.zeros((4, 1, *self.dim), dtype=np.single)
        for i in range(4):
            X[i, :] = np.reshape(np.rot90(gx, i, (0, 1)), (1,*self.dim))
        Y = np.zeros((4, *self.dim), dtype=np.single)
        for i in range(4):
            Y[i, :] = np.reshape(np.rot90(fx, i, (0, 1)), (self.dim))

        # X = gx
        # Y = fx
        # X = X[np.newaxis, :]
        return X, Y

    def __len__(self):
        return self.num_file



if __name__ == '__main__':
    dim = (128,128,128)
    chann = 1
    datas = ChannDataset('G:/datas/Train/seis/','G:/datas/Train/channel/',dim,chann)
    (a,b) = datas[4]
    print(a.shape,b.shape)
    print(b.shape)
    # print(len(datas))

