import os
import random
import numpy as np
import glob


sample_num = 500
filelist = []
for name in os.listdir('./data'):
    filename = './data/' + name
    filelist.append(filename)
while len(filelist) >= 1000:
    f = min(filelist)
    filelist.remove(f)
    os.remove(f)
sample_list = random.sample(filelist, sample_num)
features = []
pis = []
results = []
aveValues = []
for game in sample_list:
    data = np.load(game, allow_pickle=True).item()
    for i in range(data['features'].shape[0]):
        features.append(data['features'][i])
        pis.append(data['pis'][i])
        results.append(data['results'][i])
        aveValues.append(data['aveValues'][i])
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
