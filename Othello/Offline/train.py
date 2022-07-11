import random
import glob
import torch
import numpy as np
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
from networkFeature import Network


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

class OthelloDataSetPCFeature(data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        feature = self.data['features'][item]
        pi = self.data['pis'][item]
        result = self.data['results'][item]
        aveFeature = self.data['aveFeatures'][item]
        return {
            'feature': feature,
            'pi': pi,
            'result': result,
            'aveFeature': aveFeature
        }

    def __len__(self):
        return len(self.data['results'])

class OthelloDataSetPCMix(data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        feature = self.data['features'][item]
        pi = self.data['pis'][item]
        result = self.data['results'][item]
        aveValue = self.data['aveValues'][item]
        aveFeature = self.data['aveFeatures'][item]
        return {
            'feature': feature,
            'pi': pi,
            'result': result,
            'aveValue': aveValue,
            'aveFeature': aveFeature
        }

    def __len__(self):
        return len(self.data['results'])

def integrateFile(net, trainFiles, num, PC, featurePC):
    net.eval()
    features = []
    pis = []
    results = []
    if PC:
        aveValues = []
    if featurePC:
        aveFeatures = []
    for file in trainFiles:
        batch = np.load(file, allow_pickle=True).item()
        if len(batch['features']) == 0:
            continue
        for idx in range(len(batch['results'])):
            features.append(batch['features'][idx])
            pis.append(batch['pis'][idx])
            results.append(batch['results'][idx] * 2 - 1)
        if PC or featurePC:
            feature_batch = Variable(torch.FloatTensor(batch['features']).cuda(num))
            _, value, valueConv = net(feature_batch)
            value = value.data.cpu().numpy()
            valueConv = valueConv.data.cpu().numpy()
            for idx in range(len(value)):
                fIndex = np.max([0, idx-2])
                lIndex = np.min([len(value), idx+3])
                if PC:
                    aveValues.append(np.mean(value[fIndex:lIndex]))
                if featurePC:
                    aveFeatures.append(np.mean(valueConv[fIndex:lIndex], axis=0))
    data = {
        'features': np.array(features),
        'pis': np.array(pis),
        'results': np.array(results)
    }
    if PC and featurePC:
        data['aveValues'] = np.array(aveValues)
        data['aveFeatures'] = np.array(aveFeatures)
        data = OthelloDataSetPCMix(data)
    elif PC and not featurePC:
        data['aveValues'] = np.array(aveValues)
        data = OthelloDataSetPC(data)
    elif not PC and featurePC:
        data['aveFeatures'] = np.array(aveFeatures)
        data = OthelloDataSetPCFeature(data)
    else:
        data = OthelloDataSet(data)
    return data

def train(net, trainFiles, lr, num, lamd, beta, PC, featurePC):
    trainData = integrateFile(net, trainFiles, num, PC, featurePC)
    net.train()
    optimizer = optim.SGD(net.parameters(), weight_decay=0.00001, lr=lr, momentum=0.9)
    trainLoader = DataLoader(trainData, batch_size=256, shuffle=True, num_workers=8)
    idx = 0.0
    for batch in trainLoader:
        feature_batch = Variable(torch.FloatTensor(batch['feature'].float()).cuda(num))
        pi_batch = Variable(torch.FloatTensor(batch['pi'].float()).cuda(num))
        result_batch = Variable(torch.FloatTensor(batch['result'].float()).cuda(num))
        if PC:
            aveValue_batch = Variable(torch.FloatTensor(batch['aveValue'].float()).cuda(num))
        if featurePC:
            aveFeature_batch = Variable(torch.FloatTensor(batch['aveFeature'].float()).cuda(num))
        optimizer.zero_grad()
        policy, value, valueConv = net(feature_batch)
        value_loss = F.mse_loss(value.view(-1), result_batch.view(-1))
        policy_loss = -torch.mean(torch.sum(pi_batch * policy, 1))
        loss = value_loss + policy_loss
        if PC:
            pc_loss = F.mse_loss(value.view(-1), aveValue_batch.view(-1))
            loss += lamd * pc_loss
        if featurePC:
            featurepc_loss = F.mse_loss(valueConv, aveFeature_batch)
            loss += beta * featurepc_loss
        loss.backward()
        optimizer.step()
        valueLoss = value_loss.data.item()
        policyLoss = policy_loss.data.item()
        if PC and featurePC:
            pcLoss = pc_loss.data.item()
            featurePCLoss = featurepc_loss.data.item()
            fr = open('./loss/PC_' + str(lamd) + '_Feature_' + str(beta) + '.txt', 'a')
            fr.write(str(valueLoss)+'\t'+str(policyLoss)+'\t'+str(pcLoss)+'\t'+str(featurePCLoss)+'\n')
            fr.close()
        elif PC and not featurePC:
            pcLoss = pc_loss.data.item()
            fr = open('./loss/PC_' + str(lamd) + '.txt', 'a')
            fr.write(str(valueLoss)+'\t'+str(policyLoss)+'\t'+str(pcLoss)+'\n')
            fr.close()
        elif not PC and featurePC:
            featurePCLoss = featurepc_loss.data.item()
            fr = open('./loss/Feature_' + str(beta) + '.txt', 'a')
            fr.write(str(valueLoss)+'\t'+str(policyLoss)+'\t'+str(featurePCLoss)+'\n')
            fr.close()
        else:
            fr = open('./loss/NonPC.txt', 'a')
            fr.write(str(valueLoss)+'\t'+str(policyLoss)+'\n')
            fr.close()
    return net

def test(net, num, lamd, beta, PC, featurePC):
    files = glob.glob('/cmach-data/zhaodengwei/Othello/test/*.npy')
    epoch = len(files) // 1000
    net.eval()
    correctNum = 0.0
    totalNum = 0.0
    valueLoss = 0.0
    policyLoss = 0.0
    pcLoss = 0.0
    featureLoss = 0.0
    for i in range(epoch + 1):
        if i == epoch:
            testFiles = files[epoch * 1000:]
        else:
            testFiles = files[epoch * 1000: (epoch + 1) * 1000]
        testData = integrateFile(net, testFiles, num, True, True)
        testLoader = DataLoader(testData, batch_size=256, shuffle=True, num_workers=8)
        for batch in testLoader:
            position_batch = Variable(torch.FloatTensor(batch['feature'].float()).cuda(num))
            pi_batch = Variable(torch.FloatTensor(batch['pi'].float()).cuda(num))
            result_batch = Variable(torch.FloatTensor(batch['result'].float()).cuda(num))
            aveValue_batch = Variable(torch.FloatTensor(batch['aveValue'].float()).cuda(num))
            aveFeature_batch = Variable(torch.FloatTensor(batch['aveFeature'].float()).cuda(num))
            policy, value, valueConv = net(position_batch)
            predict = torch.argmax(policy, dim=1)
            action = torch.argmax(pi_batch, dim=1)
            correctNum += (predict == action).sum().data.item()
            totalNum += len(predict)
            value_loss = F.mse_loss(value.view(-1), result_batch.view(-1))
            valueLoss += value_loss.data.item() * len(predict)
            policy_loss = -torch.mean(torch.sum(pi_batch * policy, 1))
            policyLoss += policy_loss.data.item() * len(predict)
            pc_loss = F.mse_loss(value.view(-1), aveValue_batch.view(-1))
            pcLoss += pc_loss.data.item() * len(predict)
            featurepc_loss = F.mse_loss(valueConv, aveFeature_batch)
            featureLoss += featurepc_loss.data.item() * len(predict)
    print('Predict Accuracy: {:.2%}'.format(correctNum/totalNum))
    if PC and featurePC:
        fileName = './testResult/PC_' + str(lamd) + '_Feature_' + str(beta) + '.txt'
    elif PC and not featurePC:
        fileName = './testResult/PC_' + str(lamd) + '.txt'
    elif not PC and featurePC:
        fileName = './testResult/Feature_' + str(beta) + '.txt'
    else:
        fileName = './testResult/NonPC.txt'
    fr = open(fileName, 'a')
    fr.write(str(policyLoss/totalNum) + '\t' + str(valueLoss/totalNum) + '\t' + str(pcLoss/totalNum) + '\t' + str(featureLoss/totalNum) + '\t' + str(correctNum/totalNum) + '\n')
    fr.close()



def trainModel():
    sample_num = 1000
    epochs = 10
    gpu_num = 1
    lr = 0.01
    lamd = 2.0
    beta = 0.6
    num_block = 3
    num_channel = 256
    path = './Othello/train/*.npy'
    files= glob.glob(pathname=path)
    np.random.shuffle(files)
    PC = True
    featurePC = True
    net = Network(channel=num_channel, numBlock=num_block)
    net = net.cuda(gpu_num)
    for i in range(epochs):
        for j in range(100):
            trainFiles = files[j * sample_num : (j + 1) * sample_num]
            net = train(net, trainFiles, lr, num=gpu_num, lamd=lamd, beta=beta, PC=PC, featurePC=featurePC)
        test(net, gpu_num, lamd, beta, PC, featurePC)
        if PC and featurePC:
            modelName = './model/model_PC_' + str(lamd) + '_Feature_' + str(beta) + '_' + str(i+1)  + '.model'
        elif PC and not featurePC:
            modelName = './model/model_PC_' + str(lamd) + '_' + str(i+1)  + '.model'
        elif not PC and featurePC:
            modelName = './model/model_Feature_' + str(beta) + '_' + str(i+1)  + '.model'
        else:
            modelName = './model/model_NonPC_' + str(i+1) + ' .model'
        torch.save(net.state_dict(), modelName)

trainModel()
