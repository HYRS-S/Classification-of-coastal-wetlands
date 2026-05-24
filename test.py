import copy
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from HSIDataset import HSIDatasetV1
from model import MSSTANet
from utils import loadData

isExists = lambda path: os.path.exists(path)

SEED = 971104
torch.manual_seed(SEED)
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
NAME = 'LHK'
SAMPLE_PER_CLASS = 0.75

def run_test(model, dataLoader):
    model.to(DEVICE)
    model.eval()
    prediction_pairs = []
    for (spectra, neighbor_region), target in dataLoader:
        spectra, neighbor_region, target = spectra.to(DEVICE), neighbor_region.to(DEVICE), target.to(DEVICE)
        spectra = spectra.unsqueeze(1)
        neighbor_region = neighbor_region.permute((0, 3, 1, 2))
        logits = model(neighbor_region, spectra)
        pred = torch.argmax(logits, dim=-1).to('cpu').numpy().squeeze()
        gt = target.to('cpu').numpy()
        prediction_pairs.append((pred, gt))
    return prediction_pairs

def main(datasetName, n_sample_per_class, run):
    label_path = r'trainTestSplit/{}/sample{}_run{}.npy'.format(datasetName,  n_sample_per_class, run)
    data = loadData(datasetName)
    bands = data.shape[2]
    isExists(label_path)
    Labels = np.load(label_path)
    nc = np.max(Labels)
    trainLabel = Labels[0]
    testLabel = Labels[1]
    data, trainLabel, testLabel = data.astype(np.float32), trainLabel.astype(int), testLabel.astype(int)
    testDataset = HSIDatasetV1(data, testLabel, patchsz=21)
    testLoader = DataLoader(testDataset, batch_size=1024, shuffle=True)

    model = MSSTANet(bands, nc)

    root = 'MS3Net/{}/{}/{}'.format(datasetName, n_sample_per_class, run)
    modelSavepath = os.path.join(root, 'MSSTANet_sample{}_run{}_best.pkl'.format(n_sample_per_class, run))
    ModelDict = torch.load(modelSavepath, map_location=torch.device('cpu'))

    modelDict = model.state_dict()
    modelDict.update(ModelDict)
    model.load_state_dict(modelDict)
    cMat = np.zeros((nc, nc))
    prediction_pairs = run_test(model, testLoader)
    for i, (ps,ts) in enumerate(prediction_pairs):
        for j in range(len(ps)):
            cMat[ts[j],ps[j]] += 1
    return cMat, nc

if __name__ == '__main__':

    print('*'*5+'TEST'+'*'*5)
    print('*' * 5 + 'DATASET NAME:{}'.format(NAME) + '*' * 5)
    print('*' * 5 + 'SAMPLE_PER_CLASS:{}'.format(SAMPLE_PER_CLASS) + '*' * 5)
    cMat,nc = main(NAME,SAMPLE_PER_CLASS,0)
    oa = np.trace(cMat) / np.sum(cMat)
    po = copy.deepcopy(oa)
    x = np.matmul(np.sum(cMat, axis=0) , np.sum(cMat, axis=1))
    pe = x / np.sum(cMat)**2
    kappa = (po-pe)/(1-pe)
    ca = []
    for c in range(nc):
        ca.append(cMat[c,c]/np.sum(cMat[c,:]))
    aa = np.mean(ca)

    for c in range(nc):
        print('class {} : {:.6f}%'.format(c+1,ca[c]))

    print('OA : {:6f}%'.format(oa * 100))
    print('AA : {:6f}%'.format(aa * 100))
    print('Kappa : {:6f}%'.format(kappa * 100))

