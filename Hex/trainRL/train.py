import torch
import network
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.utils.data as data
import torch.nn.functional as F
import glob
import numpy as np


class hexDataSet(data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        feature = self.data['features'][item]
        action = self.data['pis'][item]
        result = self.data['results'][item]
        return {'feature': feature,
                'pi': action,
                'result': result
                }

    def __len__(self):
        return len(self.data['results'])


class hexDataSetPC(data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        feature = self.data['features'][item]
        action = self.data['pis'][item]
        result = self.data['results'][item]
        aveValue = self.data['aveValues'][item]
        return {'feature': feature,
                'pi': action,
                'result': result,
                'aveValue': aveValue
                }

    def __len__(self):
        return len(self.data['results'])


def train(net, data, lr, PC=False):
    net.train()
    optimizer = optim.SGD(net.parameters(), weight_decay=0.00001, lr=lr, momentum=0.9)
    if PC:
        trainData = hexDataSetPC(data)
    else:
        trainData = hexDataSet(data)
    trainLoader = DataLoader(trainData, batch_size=256, shuffle=True, num_workers=8)
    for batch in trainLoader:
        position_batch = Variable(torch.FloatTensor(batch['feature'].float()).cuda())
        pi_batch = Variable(torch.FloatTensor(batch['pi'].float()).cuda())
        result_batch = Variable(torch.FloatTensor(batch['result'].float()).cuda())
        if PC:
            aveValue_batch = Variable(torch.FloatTensor(batch['aveValue'].float()).cuda())
        optimizer.zero_grad()
        policy, value = net(position_batch)
        value_loss = F.mse_loss(value.view(-1), result_batch.view(-1))
        policy_loss = -torch.mean(torch.sum(pi_batch * policy, 1))
        loss = value_loss + policy_loss
        if PC:
            pc_loss = F.mse_loss(value.view(-1), aveValue_batch.view(-1))
            loss = loss + pc_loss
        loss.backward()
        optimizer.step()
        valueLoss = value_loss.data.item()
        policyLoss = policy_loss.data.item()
        if PC:
            pcLoss = pc_loss.data.item()
        fr = open('result.txt', 'a')
        line = str(valueLoss)+'\t'+str(policyLoss)
        if PC:
            line = line + '\t' + str(pcLoss)
        fr.write(line+'\n')
        fr.close()
    path_file_number = glob.glob(pathname='./model/*.model')
    modelName = './model/model' + str(len(path_file_number)) + '.model'
    torch.save(net.state_dict(), modelName)



path_file_number = glob.glob(pathname='./dataTrain/*.npy')
dataName = './dataTrain/data'+str(len(path_file_number)-1)+'.npy'
data = np.load(dataName, allow_pickle=True).item()
net = network.Network(channel=128, numBlock=10).cuda()
path_file_number = glob.glob(pathname='./model/*.model')
modelName = './model/model'+str(len(path_file_number)-1)+'.model'
net.load_state_dict(torch.load(modelName))
train(net, data, lr=0.0001, PC=True)
