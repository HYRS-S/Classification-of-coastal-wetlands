import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.decomposition import PCA
#########ss annotation
#  高光谱图像（HSI）数据集的构建类，
# 主要作用是把一个原始的高光谱数据立方体和对应标签转成能直接喂给 PyTorch 模型的 Dataset 对象，
# 同时支持像素光谱和空间邻域块的联合输入。
class HSIDataset(Dataset):
    def __init__(self, data, label, patchsz=1):
        '''
        :param data: [h, w, bands]
        :param label: [h, w]
        :param n_components: scale
        :param patchsz: scale
        '''
        super(HSIDataset, self).__init__()
        self.data = data # [h, w, bands]
        self.label = label # [h, w]
        self.patchsz = patchsz
        # 原始数据的维度
        self.h, self.w, self.bands = self.data.shape
        self.Normalize()
        # self.get_mean()
        # # 数据中心化
        # self.data -= self.mean
        self.addMirror()

    # 数据归一化
    def Normalize(self):
        data = self.data.reshape((self.h * self.w, self.bands))
        data -= np.min(data, axis=0)
        data /= np.max(data, axis=0)
        self.data = data.reshape((self.h, self.w, self.bands))


    # 添加镜像
    def addMirror(self):
        dx = self.patchsz // 2
        if dx != 0:
            mirror = np.zeros((self.h + 2 * dx, self.w + 2 * dx, self.bands))
            mirror[dx:-dx, dx:-dx, :] = self.data
            for i in range(dx):
                # 填充左上部分镜像
                mirror[:, i, :] = mirror[:, 2 * dx - i, :]
                mirror[i, :, :] = mirror[2 * dx - i, :, :]
                # 填充右下部分镜像
                mirror[:, -i - 1, :] = mirror[:, -(2 * dx - i) - 1, :]
                mirror[-i - 1, :, :] = mirror[-(2 * dx - i) - 1, :, :]
            self.data = mirror

    def __len__(self):
        return self.h * self. w

    def __getitem__(self, index):
        '''
        :param index:
        :return: 元素光谱信息， 元素的空间信息， 标签
        '''
        l = index // self.w
        c = index % self.w
        # 领域: [patchsz, patchsz, bands]
        neighbor_region = self.data[l:l + self.patchsz, c:c + self.patchsz, :]
        # 取均值
        # neighbor_region_mean = neighbor_region
        neighbor_region_mean = np.mean(neighbor_region, axis=-1, keepdims=True)
        # 中心像素的光谱
        spectra = self.data[l + self.patchsz // 2, c + self.patchsz // 2]
        # 类别
        target = self.label[l, c] - 1
        return (torch.tensor(spectra, dtype=torch.float32), torch.tensor(neighbor_region_mean, dtype=torch.float32)), \
        torch.tensor(target, dtype=torch.long)

class HSIDatasetV1(HSIDataset):
    def __init__(self, data, label, patchsz=1):
        super().__init__(data, label, patchsz)
        self.sampleIndex = list(zip(*np.nonzero(self.label)))

    def __len__(self):
        return len(self.sampleIndex)

    def __getitem__(self, index):
        l, c = self.sampleIndex[index]
        spectra = self.data[l + self.patchsz // 2, c + self.patchsz // 2]
        neighbor_region = self.data[l:l + self.patchsz, c:c + self.patchsz, :]
        # 标签从0开始编码
        target = self.label[l, c] - 1
        # 取均值
        neighbor_region_mean = np.mean(neighbor_region, axis=-1, keepdims=True)
        return (torch.tensor(spectra, dtype=torch.float32), torch.tensor(neighbor_region_mean, dtype=torch.float32)), \
                torch.tensor(target, dtype=torch.long)

class HSIDatasetV2(HSIDataset):
    def __init__(self, data, label, patchsz=1):
        super().__init__(data, label, patchsz)
        self.sampleIndex = list(zip(*np.nonzero(self.label)))

    def __len__(self):
        return len(self.sampleIndex)

    def __getitem__(self, index):
        l, c = self.sampleIndex[index]
        spectra = self.data[l + self.patchsz // 2, c + self.patchsz // 2]
        neighbor_region = self.data[l:l + self.patchsz, c:c + self.patchsz, :]
        # 标签从0开始编码
        target = self.label[l, c] - 1
        # 取均值
        neighbor_region_mean = np.mean(neighbor_region, axis=-1, keepdims=True)
        return (torch.tensor(spectra, dtype=torch.float32), torch.tensor(neighbor_region_mean, dtype=torch.float32)), \
                torch.tensor(target, dtype=torch.long), l, c


# from scipy.io import loadmat
# import numpy as np
# m = loadmat('data/PaviaU/PaviaU.mat')
# data = m['paviaU']
# m = loadmat('data/PaviaU/PaviaU_gt.mat')
# label = m['paviaU_gt']
# data, label = data.astype(np.float32), label.astype(np.long)
# dataset = HSIDataset(data, label, patchsz=21)
# w = data.shape[1]
# index = 150
# l, c = index // w, index % w
# (spectra, neighbor_region), target = dataset[index]
# print(torch.equal(spectra, neighbor_region[21 // 2, 21 // 2]))
# print(target - 1)
# print(label[l, c])

#ss annotation
#将高光谱影像（HSI）与标签转为 PyTorch 可迭代数据集。
#在取样时同时提供光谱信息和空间邻域块，方便光谱–空间联合模型训练。
#支持全图采样（基类）、只采有标签像素（V1/V2）、以及返回坐标信息（V2）。
#自动完成归一化与镜像填充，让模型直接接收干净的输入数据。

