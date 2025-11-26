
import torch
import numpy as np
import os
import re


class BaseRamanDataset():

    def to_data(self, path, x, timeunit=2):
        return


class RuffDataset(BaseRamanDataset):

    def to_data(self, path, x=0, timeunit=1):
        if not os.path.isfile(path):
            return

        count = 0
        for line in open(path):
            count += 1
        #print('数据长度:' + str(count))
        file = open(path, 'r', encoding='utf-8')
        data = []
        wave = []
        i = 0
        index = 0
        count = int(count / 1000)
        for line in file:
            try:
                # i += 1
                # if i < count:
                #     continue
                # if i == count:
                #     i = 0

                if i >= 8100:
                    self.f.write(path + ' out of 8100\n')
                    break
                is_data = line.find('#')
                if is_data != -1:
                    continue

                mlist = line.split(",")
                if (len(mlist) < 2):
                    continue

                y_val = ''.join(re.findall("\S", mlist[1]))

                y_val = np.float32(y_val)
                x_val = np.float32(mlist[0])

                data.append(y_val)
                wave.append(x_val)
                index += 1

            except:
                self.f.write(path + ' got exception\n')
                continue

        #记录label

        return wave, data, index


class MyDataset(BaseRamanDataset):

    def to_data(self, path, x, timeunit=1):
        if not os.path.isfile(path):
            return

        count = 0
        for line in open(path):
            count += 1
        # print('数据长度:' + str(count))
        file = open(path, 'r', encoding='utf-8')
        data = []
        wave = []
        i = 0
        index = 0
        
        x_val = x
        for line in file:
            try:
                is_data = line.find('x')
                if is_data != -1:
                    continue

                x_val += timeunit
                num = line
                y_val = np.float32(num)

                data.append(y_val)
                wave.append(x_val)
                index += 1

            except:
                print(path + ' got exception\n')
                continue

        #记录label

        return wave, data, index

class InsertDataset(BaseRamanDataset):

    def to_data(self, path, x, timeunit=1):
        if not os.path.isfile(path):
            return

        count = 0
        for line in open(path):
            count += 1
        # print('数据长度:' + str(count))
        file = open(path, 'r', encoding='utf-8')
        file1 = open(path+'_out', 'w', encoding='utf-8')
        data = []
        wave = []
        i = 0
        index = 0
        
        x_val = x
        for line in file:
            try:
                is_data = line.find('x')
                if is_data != -1:
                    continue

                x_val += timeunit
                num = line
                y_val = np.float32(num)

                data.append(y_val)
                wave.append(x_val)
                index += 1
                text = str(x_val) + ',' + line
                file1.write(text)

            except:
                print(path + ' got exception\n')
                continue

        #记录label
        
        file1.close()
        return wave, data, index
    

class ResetDataset(BaseRamanDataset):

    def to_data(self, path, x, timeunit=1):
        if not os.path.isfile(path):
            return

        count = 0
        for line in open(path):
            count += 1
        # print('数据长度:' + str(count))
        file = open(path, 'r', encoding='utf-8')
        file1 = open(path+'_out', 'w', encoding='utf-8')
        data = []
        wave = []
        i = 0
        index = 0
        
        x_val = x
        for line in file:
            try:
                is_data = line.find('x')
                if is_data != -1:
                    continue

                x_val += timeunit
                mlist = line.split(",")
                if (len(mlist) < 2):
                    continue

                y_val = ''.join(re.findall("\S", mlist[1]))


                data.append(y_val)
                wave.append(x_val)
                index += 1
                text = str(x_val) + ',' + y_val + '\n'
                file1.write(text)

            except:
                print(path + ' got exception\n')
                continue

        #记录label
        
        file1.close()
        return wave, data, index

if __name__ == '__main__':
    dataset = ResetDataset()
    
    root = 'dataset/train-lunwen-1600/3'

    #批量转化
    filelist = os.listdir(root)
    for file_path in sorted(filelist):
        if os.path.isfile(file_path):
            continue
        path = root + '/' + file_path
        dataset.to_data(path, 0, 1)
    
    print('load finish')
