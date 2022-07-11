import hex
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np


class ResBlock(nn.Module):
    def __init__(self, channel=128):
        super(ResBlock, self).__init__()
        self.channel = channel
        self.block = nn.Sequential(
            nn.Conv2d(self.channel, self.channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.channel),
            nn.ReLU(),
            nn.Conv2d(self.channel, self.channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.channel)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.block(x) + x)


class Resnet(nn.Module):
    def __init__(self, channel, numBlock):
        super(Resnet, self).__init__()
        self.blocks = []
        self.numBlock = numBlock
        self.channel = channel
        for _ in range(self.numBlock):
            self.blocks.append(ResBlock(channel=self.channel))
        self.blocks = nn.ModuleList(self.blocks)
        self.resnet = nn.Sequential(*self.blocks)

    def forward(self, x):
        return self.resnet(x)


class Network(nn.Module):
    def __init__(self, channel=128, numBlock=10):
        super(Network, self).__init__()
        self.channel = channel
        self.numBlock = numBlock
        self.head = nn.Sequential(
            nn.Conv2d(in_channels=5, out_channels=self.channel, kernel_size=3, padding=0),
            nn.BatchNorm2d(self.channel),
            nn.ReLU()
        )
        self.resnet = Resnet(channel=self.channel, numBlock=self.numBlock)
        self.policy_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.channel, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
        self.policy_fc = nn.Sequential(
            #nn.Linear(2*hex.SIZE*hex.SIZE, hex.SIZE*hex.SIZE),
            nn.LogSoftmax(dim=1)
        )
        self.value_conv = nn.Sequential(
            nn.Conv2d(in_channels=self.channel, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
        self.value_fc = nn.Sequential(
            nn.Linear(hex.SIZE * hex.SIZE, 1),
            nn.Tanh()
        )

    def valueFeature(self, feature):
        feature = self.head(feature)
        feature = self.resnet(feature)
        policy = self.policy_conv(feature)
        policy = policy.view(-1, hex.SIZE*hex.SIZE)
        policy = self.policy_fc(policy)
        value = self.value_conv(feature)
        value = value.view(-1, hex.SIZE*hex.SIZE)
        return value

    def forward(self, feature):
        feature = self.head(feature)
        feature = self.resnet(feature)
        policy = self.policy_conv(feature)
        policy = policy.view(-1, hex.SIZE*hex.SIZE)
        policy = self.policy_fc(policy)
        value = self.value_conv(feature)
        valueConv = value.view(-1, hex.SIZE*hex.SIZE)
        value = self.value_fc(valueConv)
        return policy, value, valueConv


class PV():
    def __init__(self, model_path=None, channel=128, numBlock=10, num=0):
        self.net = Network(channel=channel, numBlock=numBlock)
        self.net.cuda(num)
        self.num = num
        cudnn.benchmark = True
        if model_path is not None:
            loc1 = 'cuda:0'
            loc2 = 'cuda:' + str(num)
            self.net.load_state_dict(torch.load(model_path, map_location={loc1:loc2}))
        self.net.eval()

    def run(self, board):
        probs, values = self.run_many([board])
        return probs[0], values[0]

    def run_many(self, boards):
        fs = [board.to_feature() for board in boards]
        fs = np.array(fs, dtype=np.float32)
        fs = torch.FloatTensor(fs)
        log_act_probs, value, _ = self.net(Variable(fs.cuda(self.num)))
        act_probs = np.exp(log_act_probs.data.cpu().numpy())
        value = value.data.cpu().numpy()
        return act_probs,  value
