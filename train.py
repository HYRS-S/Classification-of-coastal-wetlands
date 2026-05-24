import torch
from torch import nn, optim
import numpy as np
from utils import weight_init, loadData
from HSIDataset import HSIDatasetV1
from model import MSSTANet
from torch.utils.data import DataLoader
import os
import argparse
from utils import train, run_test

isExists = lambda path: os.path.exists(path)
SAMPLE_PER_CLASS = [0.75]
RUN = 3
EPOCHS = 10
LR = 1e-1
SEED = 971104
torch.manual_seed(SEED)
DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
ROOT = None
MODEL_NAME = 'MSSTANet'

EARLY_STOP = 10

def main(datasetName, train_rate, run):
    # 加载数据和标签
    label_path = r'trainTestSplit/{}/sample{}_run{}.npy'.format(datasetName, train_rate, run)
    data = loadData(datasetName)
    bands = data.shape[-1]
    isExists(label_path)
    Labels = np.load(label_path)
    trainLabel = Labels[0]
    testLabel = Labels[1]
    # 数据转换
    data, trainLabel, testLabel = data.astype(np.float32), trainLabel.astype(int), testLabel.astype(int)
    nc = int(np.max(trainLabel))
    trainDataset = HSIDatasetV1(data, trainLabel, patchsz=21)
    testDataset = HSIDatasetV1(data, testLabel, patchsz=21)
    trainLoader = DataLoader(trainDataset, batch_size=256, shuffle=True)
    testLoader = DataLoader(testDataset, batch_size=256, shuffle=True)
    model = MSSTANet(bands, nc)
    modelSavepath = os.path.join(ROOT,'MSSTANet2_sample{}_run{}_best.pkl'.format(train_rate, run))
    modelDict = model.state_dict()
    if os.path.exists(modelSavepath):
        print('*'*5, 'loading saved model', '*'*5)
        ModelDict = torch.load(modelSavepath, map_location=torch.device('cpu'))
        modelDict.update(ModelDict)
        model.load_state_dict(modelDict)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    best_acc = 0.0
    num = 0

    for epoch in range(EPOCHS):
        print('*'*5 + 'Epoch:{}'.format(epoch) + '*'*5)
        model, trainLoss = train(model, criterion=criterion, optimizer=optimizer, dataLoader=trainLoader, print_num=30, model_name=MODEL_NAME)
        acc, evalLoss = run_test(model, criterion=criterion, dataLoader=testLoader, model_name=MODEL_NAME)
        print('epoch:{} trainLoss:{:.8f} evalLoss:{:.8f} acc:{:.4f} best acc:{:.4f}'.format(epoch, trainLoss, evalLoss, acc,best_acc))
        print('*'*18)
        if (acc - best_acc) >1e-4:
            torch.save(model.state_dict(),
                       os.path.join(ROOT, 'MSSTANet2_sample{}_run{}_best.pkl'.format(train_rate, run)))
            best_acc = acc
            num = 0
        else: num+=1
        if num == EARLY_STOP:
            break

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train MS3Net')
    parser.add_argument('--name', type=str, default='LHK',
                        help='The name of dataset')
    parser.add_argument('--epoch', type=int, default=200,
                        help='模型的训练次数')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate')

    args = parser.parse_args()
    EPOCHS = args.epoch
    datasetName = args.name
    LR = args.lr


    print('*'*5 + datasetName + '*'*5)
    for i, num in enumerate(SAMPLE_PER_CLASS):
        print('*' * 5 + 'Train rate:{}'.format(num) + '*' * 5)
        for r in range(RUN):
            print('*' * 5 + 'run:{}'.format(r) + '*' * 5)
            ROOT = 'MS3Net/{}/{}/{}'.format(datasetName, num, r)
            if not os.path.isdir(ROOT):
                os.makedirs(ROOT)
            main(datasetName, num, r)
    print('*'*5 + 'FINISH' + '*'*5)