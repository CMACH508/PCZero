import torch
import network
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.utils.data as data
import torch.nn.functional as F
import glob
import numpy as np


class OthelloDataSet(data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        feature = self.data['features'][item]
        pi = self.data['pis'][item]
        result = self.data['results'][item]
        return {
            'feature': feature,
            'pi': pi,
            'result': result
        }

    def __len__(self):
        return len(self.data['results'])

class OthelloDataSetPC(data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        feature = self.data['features'][item]
        pi = self.data['pis'][item]
        result = self.data['results'][item]
        aveValue = self.data['aveValues'][item]
        return {
            'feature': feature,
            'pi': pi,
            'result': result,
            'aveValue': aveValue
        }

    def __len__(self):
        return len(self.data['results'])


def train(net, data, lr, lamd):
    net.train()
    optimizer = optim.SGD(net.parameters(), weight_decay=0.00001, lr=lr, momentum=0.9)
    trainData = OthelloDataSetPC(data)
    trainLoader = DataLoader(trainData, batch_size=256, shuffle=True, num_workers=8)
    for batch in trainLoader:
        position_batch = Variable(torch.FloatTensor(batch['feature'].float()).cuda())
        pi_batch = Variable(torch.FloatTensor(batch['pi'].float()).cuda())
        result_batch = Variable(torch.FloatTensor(batch['result'].float()).cuda())
        aveValue_batch = Variable(torch.FloatTensor(batch['aveValue'].float()).cuda())
        optimizer.zero_grad()
        policy, value = net(position_batch)
        value_loss = F.mse_loss(value.view(-1), result_batch.view(-1))
        policy_loss = -torch.mean(torch.sum(pi_batch * policy, 1))
        pc_loss = F.mse_loss(value.view(-1), aveValue_batch.view(-1))
        loss = value_loss + policy_loss + lamd * pc_loss
        loss.backward()
        optimizer.step()
        valueLoss = value_loss.data.item()
        policyLoss = policy_loss.data.item()
        pcLoss = pc_loss.data.item()
        fr = open('result.txt', 'a')
        fr.write(str(valueLoss)+'\t'+str(policyLoss)+'\t'+str(pcLoss)+'\n')
        fr.close()
    path_file_number = glob.glob(pathname='./model/*.model')
    modelName = './model/model' + str(len(path_file_number)) + '.model'
    torch.save(net.state_dict(), modelName)

lamd=1.0
net = network.Network().cuda()
path_file_number = glob.glob(pathname='./model/*.model')
if len(path_file_number) > 0:
    modelName = './model/model'+str(len(path_file_number)-1)+'.model'
    net.load_state_dict(torch.load(modelName))
if len(path_file_number) < 200:
    lr = 0.01
elif len(path_file_number) < 350:
    lr = 0.001
else:
    lr = 0.0001
path_file_number = glob.glob(pathname='./dataTrain/*.npy')
dataName = './dataTrain/data'+str(len(path_file_number)-1)+'.npy'
data = np.load(dataName, allow_pickle=True).item()
train(net, data, lr, lamd)
