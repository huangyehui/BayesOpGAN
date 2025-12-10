from sys import prefix
from tkinter import N
import numpy as np
import math
from regex import F
from scipy.fftpack import fft, ifft, fftfreq
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
from raman_loader import RuffDataset, MyDataset
from PIL import Image
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import shutil


ruff_dataset = RuffDataset()
synthesis_dataset = MyDataset()


def rruff_source_dataset_fourier(path):
    files = os.listdir(path)
    ffts = []
    arr_x = []
    for file_name in files:
        rruff_file = path + '/' + file_name
        x, y, size = ruff_dataset.to_data(rruff_file, 0)
        fft_ruff = fft(y)  #快速傅里叶变换
        ffts.append(fft_ruff)
        arr_x.append(x)
    return arr_x, ffts


def min_max_normalize(arr):
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    # 避免除零，如果最大值和最小值相等，则返回全0.5（或处理为常数）
    if arr_max == arr_min:
        return np.zeros_like(arr) + 0.5  # 或者返回0，或者根据需求处理
    return (arr - arr_min) / (arr_max - arr_min)


def cale_fourier_distance(x, rruff_ffts, path, is_normal):

    x1, y1, size = synthesis_dataset.to_data(path, 0)
    if is_normal:
        y1 = min_max_normalize(y1)
    fft_synthesis = fft(y1)  #快速傅里叶变换
    fft_ce = []
    idx = 0
    for ffts in rruff_ffts:
        amplitude_synthesis = np.abs(fft_synthesis)
        amplitude_ruff = np.abs(ffts)

        x_rruff = x[idx]
        len_min = min(len(x_rruff), len(x1))
        if len(x_rruff) == len_min:
            minus = len(x1) - len(x_rruff)
            padding_l = int(minus / 2)
            padding_r = minus - padding_l
            amplitude_synthesis = amplitude_synthesis[padding_l:len(x1) -
                                                      padding_r]
        else:
            minus = len(x_rruff) - len(x1)
            padding_l = int(minus / 2)
            padding_r = minus - padding_l
            amplitude_ruff = amplitude_ruff[padding_l:len(x_rruff) - padding_r]

        amp_synthesis_normal = amplitude_synthesis / np.sum(
            amplitude_synthesis)
        amp_rruff_normal = amplitude_ruff / np.sum(amplitude_ruff)

        # 计算相对熵（kl）
        kl = np.sum(amp_synthesis_normal *
                    np.log(amp_synthesis_normal / amp_rruff_normal))
        fft_ce.append(kl)
        idx += 1
    res = np.mean(fft_ce)
    return res


def cale_raw_fourier_distance(x, rruff_ffts, x1, fft_synthesis):

    fft_ce = []
    idx = 0
    for ffts in rruff_ffts:
        amplitude_synthesis = np.abs(fft_synthesis)
        amplitude_ruff = np.abs(ffts)

        x_rruff = x[idx]
        len_min = min(len(x_rruff), len(x1))
        if len(x_rruff) == len_min:
            minus = len(x1) - len(x_rruff)
            padding_l = int(minus / 2)
            padding_r = minus - padding_l
            amplitude_synthesis = amplitude_synthesis[padding_l:len(x1) -
                                                      padding_r]
        else:
            minus = len(x_rruff) - len(x1)
            padding_l = int(minus / 2)
            padding_r = minus - padding_l
            amplitude_ruff = amplitude_ruff[padding_l:len(x_rruff) - padding_r]

        # 将幅值归一化
        amp_synthesis_normal = amplitude_synthesis / np.sum(
            amplitude_synthesis)
        amp_rruff_normal = amplitude_ruff / np.sum(amplitude_ruff)

        # 计算相对熵（kl）
        kl = np.sum(amp_synthesis_normal *
                    np.log(amp_synthesis_normal / amp_rruff_normal))
        # mean = np.mean(abs_ampl_minus)
        fft_ce.append(kl)
        idx += 1
    res = np.mean(fft_ce)
    print('光谱{}的傅里叶距离为{}'.format(idx, res))
    return res


#论文里正式计算傅立叶距离方法
def cale_batch_fft_dis_with_rruff(src_path, target_path, is_normal):
    x, y = rruff_source_dataset_fourier(target_path)
    files = os.listdir(src_path)
    total_ffts = []
    ffts_file = []
    for file_name in files:
        synthesis_file = src_path + '/' + file_name
        fft_dis = cale_fourier_distance(x, y, synthesis_file, is_normal)
        total_ffts.append(fft_dis)
        ffts_file.append(file_name)
    batch_fft_distance = np.mean(total_ffts)
    print(f"calc {src_path} 的傅里叶距离:{batch_fft_distance}")
    return batch_fft_distance


# 求生成数据和rruff之间的傅立叶距离
def fourier_distance(path_rruff, path_comp):
    ruff = RuffDataset()
    x, y, size = ruff.to_data(path_rruff)

    fft_ruff = fft(y)  #快速傅里叶变换

    gan = MyDataset()
    x1, y1, size = gan.to_data(path_comp, 0)
    fft_gan = fft(y1)  #快速傅里叶变换

    amplitude_gan = np.abs(fft_gan)
    amplitude_ruff = np.abs(fft_ruff)


    x_comp = min(len(x), len(x1))
    amplitude_gan = amplitude_gan[0:x_comp]

    ampl_minus = amplitude_gan - amplitude_ruff
    abs_ampl_minus = np.abs(ampl_minus)
    #module = np.abs(fft_minus)
    mean = np.mean(abs_ampl_minus)
    return mean


# 求rruff数据之间的傅立叶距离
def fourier_distance_rruff(path_rruff, path_comp):
    ruff = RuffDataset()
    x, y, size = ruff.to_data(path_rruff)

    fft_ruff = fft(y)  #快速傅里叶变换

    gan = MyDataset()
    x1, y1, size = ruff.to_data(path_comp)
    fft_gan = fft(y1)  #快速傅里叶变换

    amplitude_gan = np.abs(fft_gan)
    amplitude_ruff = np.abs(fft_ruff)

    plt.figure()
    plt.plot(x, amplitude_ruff)
    plt.title('rruff')

    plt.figure()
    plt.plot(x1, amplitude_gan)
    plt.title(path_comp)
    plt.show()

    x_comp = min(len(x), len(x1))
    if len(x) == x_comp:
        amplitude_gan = amplitude_gan[0:x_comp]
    else:
        amplitude_ruff = amplitude_ruff[0:x_comp]

    ampl_minus = amplitude_gan - amplitude_ruff
    abs_ampl_minus = np.abs(ampl_minus)
    #module = np.abs(fft_minus)
    mean = np.mean(abs_ampl_minus)
    return mean







if __name__ == '__main__':
    cale_batch_fft_dis_with_rruff('./out/synthesis', './out/beryl', False)