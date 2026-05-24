import torch
from torch.nn import init
from torch import nn
import os
import random
import numpy as np
from osgeo import gdal

DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

def weight_init(m):
    if isinstance(m, nn.Linear):
        init.normal_(m.weight, 0, 5e-2)
        init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        init.normal_(m.weight, 0, 5e-2)
        init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm1d):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)

def loadData(datasetName):
    vn_path = r"data\{}_VN.tif".format(datasetName)
    sw_path = r"data\{}_SW.tif".format(datasetName)

    vn = gdal.Open(vn_path).ReadAsArray()
    sw = gdal.Open(sw_path).ReadAsArray()

    data = np.concatenate((vn, sw), axis=0)
    bands, h, w = data.shape
    data = np.reshape(data,(h,w,bands))
    return data

def train(model, criterion, optimizer, dataLoader, model_name=None, data=None, patchsz=21, print_num=30, debug=False):
    model.train()
    model.to(DEVICE)
    trainLoss = []

    for step, batch in enumerate(dataLoader):
        (spectra, neighbor_region), target = batch
        spectra, target = spectra.to(DEVICE), target.to(DEVICE)

        neighbor_region = neighbor_region.to(DEVICE)

        if neighbor_region.dim() == 4:
            neighbor_region = neighbor_region.permute(0, 3, 1, 2)

        out = model(neighbor_region, spectra.unsqueeze(1))

        loss = criterion(out, target)
        trainLoss.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % print_num == 0:
            print(f"step:{step} loss:{loss.item():.6f}")

    return model, float(np.mean(trainLoss))

def run_test(model, criterion, dataLoader, model_name=None):
    model.eval()
    evalLoss = []
    correct = 0
    with torch.no_grad():
        for batch in dataLoader:
            if len(batch) == 4:
                (spectra, _), target, l, c = batch
                spectra, target = spectra.to(DEVICE), target.to(DEVICE)
                continue
            else:
                (spectra, neighbor_region), target = batch
                spectra, target = spectra.to(DEVICE), target.to(DEVICE)

                if neighbor_region.dim() == 4:
                    neighbor_region = neighbor_region.permute(0, 3, 1, 2)
                neighbor_region = neighbor_region.to(DEVICE)

            logits = model(neighbor_region, spectra.unsqueeze(1))
            loss = criterion(logits, target)
            evalLoss.append(loss.item())
            pred = torch.argmax(logits, dim=-1)
            correct += torch.sum(pred == target).item()

    acc = float(correct) / len(dataLoader.dataset) if len(dataLoader.dataset) > 0 else 0.0
    return acc, float(np.mean(evalLoss))