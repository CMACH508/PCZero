import glob
import numpy as np
import shutil

files = glob.glob('/cmach-data/zhaodengwei/Hex/train_9/*.npy')
index = np.array([i for i in range(len(files))])
np.random.shuffle(index)
testIndex = index[: 10420]
for fileIndex in testIndex:
    despath = '/cmach-data/zhaodengwei/Hex/test_9/' + files[fileIndex].split('/')[-1]
    shutil.move(files[fileIndex], despath)
    


