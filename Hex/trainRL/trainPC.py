import torch
import network
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.utils.data as data
import torch.nn.functional as F
import numpy as np
import hex


class hexDataSet(data.Dataset):
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


def train(net, data, lr, i):
    net.train()
    optimizer = optim.SGD(net.parameters(), weight_decay=0.00001, lr=lr, momentum=0.9)
    trainData = hexDataSet(data)
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
        loss = value_loss + policy_loss + pc_loss
        loss.backward()
        optimizer.step()
        valueLoss = value_loss.data.item()
        policyLoss = policy_loss.data.item()
        pcLoss = pc_loss.data.item()
        fr = open('resultPC.txt', 'a')
        fr.write(str(valueLoss)+'\t'+str(policyLoss)+'\t'+str(pcLoss)+'\n')
        fr.close()
    modelName = './modelPC/model' + str(i) + '.model'
    torch.save(net.state_dict(), modelName)


for i in range(900):
    print(i)
    dataName = './dataTrain/data'+str(i)+'.npy'
    data = np.load(dataName, allow_pickle=True).item()
    seperate = []
    j = 0
    while j < data['features'].shape[0]:
        if np.sum(data['features'][j][0][1:hex.SIZE+1, 1:hex.SIZE+1]) == 1:
            seperate.append(j)
            j += 1
        j += 1
    net = network.Network(channel=128, numBlock=10).cuda()
    if i > 0:
    	modelName = './modelPC/model'+str(i - 1)+'.model'
    	net.load_state_dict(torch.load(modelName))
    net.eval()
    batchSize = 256
    batchNum = data['features'].shape[0] // 256
    values = []
    for batchIndex in range(batchNum):
        features = Variable(torch.FloatTensor(data['features'][batchIndex*batchSize:(batchIndex+1)*batchSize]).cuda())
        _, value = net(features)
        value = value.data.cpu().numpy().reshape(-1)
        for v in value:
            values.append(v)
    features = Variable(torch.FloatTensor(data['features'][batchIndex*batchSize:]).cuda())
    _, value = net(features)
    value = value.data.cpu().numpy().reshape(-1)
    for v in value:
        values.append(v)
    values = np.array(values)
    aveValues = []
    for j in range(len(seperate)):
        start = seperate[j]
        if j == len(seperate) - 1:
            end = data['features'].shape[0]
        else:
            end = seperate[j+1]
        for k in range(start, end):
            fIndex = np.max([start, k - 2])
            lIndex = np.min([end, k + 3])
            aveValues.append(np.mean(values[fIndex : lIndex]))
    data['aveValues'] = np.array(aveValues)
    if i < 75:
        lr = 0.01
    elif i < 250:
        lr = 0.001
    else:
        lr = 0.0001
    train(net, data, lr, i)