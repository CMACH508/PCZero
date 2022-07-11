import torch
import network
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.utils.data as data
import torch.nn.functional as F
import glob
import numpy as np
import random


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


class hexPCDataSet(data.Dataset):
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


def train(net, trainFiles, lr, lam, num, PC):
    net.eval()
    features = []
    pis = []
    results = []
    if PC:
        aveValues = []
    for file in trainFiles:
        batch = np.load(file, allow_pickle=True).item()
        if len(batch['features']) == 0:
            continue
        for i in range(len(batch['features'])):
            features.append(batch['features'][i])
            pis.append(batch['pis'][i].reshape(13*13))
            results.append(batch['results'][i])
        if PC:
            position_batch = Variable(torch.FloatTensor(batch['features']).cuda(num))
            _, neighborValue, _ = net(position_batch)
            neighborValue = neighborValue.data.cpu().numpy()
            for k in range(len(neighborValue)):
                fIndex = np.max([0, k - 1])
                lIndex = np.min([len(neighborValue), k + 2])
                aveValues.append(np.mean(neighborValue[fIndex : lIndex]))
    if PC:
        data = {
            'features': np.array(features),
            'pis': np.array(pis),
            'results': np.array(results),
            'aveValues': np.array(aveValues)
        }
    else:
        data = {
            'features': np.array(features),
            'pis': np.array(pis),
            'results': np.array(results)
        }
    
    net.train()
    optimizer = optim.SGD(net.parameters(), weight_decay=0.00001, lr=lr, momentum=0.9)
    if PC:
        trainData = hexPCDataSet(data)
    else:
        trainData = hexDataSet(data)
    trainLoader = DataLoader(trainData, batch_size=256, shuffle=True, num_workers=8)
    for batch in trainLoader:
        position_batch = Variable(torch.FloatTensor(batch['feature'].float()).cuda(num))
        pi_batch = Variable(torch.FloatTensor(batch['pi'].float()).cuda(num))
        result_batch = Variable(torch.FloatTensor(batch['result'].float()).cuda(num))
        if PC:
            aveValue_batch = Variable(torch.FloatTensor(batch['aveValue'].float()).cuda(num))
        optimizer.zero_grad()
        policy, value, _ = net(position_batch)
        value_loss = F.mse_loss(value.view(-1), result_batch.view(-1))
        policy_loss = -torch.mean(torch.sum(pi_batch * policy, 1))
        if PC:
            pc_loss = F.mse_loss(value.view(-1), aveValue_batch.view(-1))
            loss = value_loss + policy_loss + lam * pc_loss
        else:
            loss = value_loss + policy_loss
        loss.backward()
        optimizer.step()
        valueLoss = value_loss.data.item()
        policyLoss = policy_loss.data.item()
        if PC:
            pcLoss = pc_loss.data.item()
            fr = open('resultPC_13_' + str(lam) + '_1.txt', 'a')
            fr.write(str(valueLoss)+'\t'+str(policyLoss)+'\t'+str(pcLoss)+'\n')
            fr.close()
        else:
            fr = open('resultNonPC_13.txt', 'a')
            fr.write(str(valueLoss)+'\t'+str(policyLoss)+'\n')
            fr.close()
    return net


if __name__ == '__main__':
    net = network.Network().cuda(0)
    files = glob.glob(pathname='./Hex_13x13/*.npy')
    for i in range(500):
        trainFiles = random.sample(files, 1000)
        net = train(net, trainFiles, 0.01, lam=1.0, num=0, PC=True)
    torch.save(net.state_dict(), 'model_13_1.model')

