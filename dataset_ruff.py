# from dataset import VOCAnnotationTransform

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import re


class RuffDataset(Dataset):

    max_size = 1600

    def __init__(self, path, target_transform=None):
        dir = os.path.expanduser(path)
        self.file = []
        self.f = open('./logs.txt', 'w')
        for file_path in sorted(os.listdir(dir)):
            if os.path.isfile(file_path):
                continue
            self.file.append(path + '/' + file_path)


    def __getitem__(self, index):
        path = self.file[index]
        data, x = self.to_data(path)
        # 返回数据
        return data, x
    
    def to_data(self, file_path):
        if not os.path.isfile(file_path):
            return
        file = open(file_path, 'r', encoding='utf-8')
        wave_len = -1
        data = np.zeros(self.max_size)
        i = 0
        wave_len = 0
        for line in file:
            try:
                if i >= self.max_size:
                    self.f.write(file_path + ' out of max\n')
                    break
                is_data = line.find('#')
                if is_data != -1:
                    continue

                mlist = line.split(",")
                if (len(mlist) < 2):
                    continue
                y_val = ''.join(re.findall("\S", mlist[1]))
                if i == 0:
                    wave_len = int(np.float64(mlist[0]))
                try:
                    y_val = np.float64(y_val)
                except:
                    continue
                data[i] = y_val
                i += 1
            
            except:
                self.f.write(file_path + ' got exception\n')
                continue

            
        #记录label
        amplitude =  torch.from_numpy(data)
        if wave_len < 0:
            self.f.write(file_path+' x=' + str(wave_len))
            wave_len = 0
        return amplitude, wave_len
    
    def __len__(self):
        return len(self.file)


if __name__ == '__main__':
    data_dir = 'E:\work\python\gan-raman\data\LR-Raman'
    dataset = RuffDataset(data_dir)
    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=32,
                                               shuffle=True)
    print('load finish')
