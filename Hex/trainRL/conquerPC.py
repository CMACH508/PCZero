import os
import random
import numpy as np
import glob
import torch
import network
from torch.autograd import Variable

sample_num = 1000
filelist = []
for name in os.listdir('./data'):
    filename = './data/' + name
    filelist.append(filename)
while len(filelist) >= 5000:
    f = min(filelist)
    filelist.remove(f)
    os.remove(f)
sample_list = random.sample(filelist, sample_num)
features = []
pis = []
results = []
aveValues = []
path_file_number = glob.glob(pathname='./model/*.model')
modelName = './model/model'+str(len(path_file_number)-1)+'.model'
net = network.Network(channel=128, numBlock=10).cuda()
net.load_state_dict(torch.load(modelName))
net.eval()
for game in sample_list:
    data = np.load(game, allow_pickle=True).item()
    feature = Variable(torch.FloatTensor(data['features']).cuda())
    _, values = net(feature)
    values = values.data.cpu().numpy().reshape(-1)
    for i in range(data['features'].shape[0]):
        features.append(data['features'][i])
        pis.append(data['pis'][i])
        results.append(data['results'][i])
        fIndex = np.max([0, i-2])
        lIndex = np.min([len(data['results']), i+3])
        aveValues.append(np.mean(values[fIndex: lIndex]))
    os.remove(game)
index = [i for i in range(len(results))]
train_data = {
    'features': np.array(features)[index],
    'pis': np.array(pis)[index],
    'results': np.array(results)[index],
    'aveValues': np.array(aveValues)[index]
}
path_file_number = glob.glob(pathname='./dataTrain/*.npy')
dataName = './dataTrain/data'+str(len(path_file_number))+'.npy'
np.save(dataName, train_data)
