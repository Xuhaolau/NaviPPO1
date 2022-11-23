import torch
import torch.nn as nn
from torch.nn import functional as F


class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()
        self.logstd = nn.Parameter(torch.zeros(2))

        self.act_fea_cv1 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.act_fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.act_fc1 = nn.Linear(128*32, 256)
        self.act_fc2 = nn.Linear(256+2+2, 128)
        self.actor1 = nn.Linear(128, 1)
        self.actor2 = nn.Linear(128, 1)

    def forward(self, scan, goal, speed):
        a = F.relu(self.act_fea_cv1(scan))
        a = F.relu(self.act_fea_cv2(a))
        a = a.view(a.shape[0], -1)
        a = F.relu(self.act_fc1(a))
        a = torch.cat((a, goal, speed), dim=-1)
        a = F.relu(self.act_fc2(a))
        mean1 = F.sigmoid(self.actor1(a))
        mean2 = F.tanh(self.actor2(a))

        mean = torch.cat((mean1, mean2), dim=-1).squeeze()
        logstd = self.logstd.expand_as(mean)

        std = torch.exp(logstd).squeeze()
        return mean, std


class ValueNet(nn.Module):
    def __init__(self):
        super(ValueNet, self).__init__()
        self.crt_fea_cv1 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=5, stride=2, padding=1)
        self.crt_fea_cv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.crt_fc1 = nn.Linear(128*32, 256)
        self.crt_fc2 = nn.Linear(256+2+2, 128)
        self.critic = nn.Linear(128, 1)

    def forward(self, scan, goal, speed):
        v = F.relu(self.crt_fea_cv1(scan))
        v = F.relu(self.crt_fea_cv2(v))             # ( n, 32, 128)
        v = v.view(v.shape[0], -1)                  # (n, 4096)
        v = F.relu(self.crt_fc1(v))                 # (n, 256)

        v = torch.cat((v, goal, speed), dim=-1)
        v = F.relu(self.crt_fc2(v))
        v = self.critic(v).squeeze()

        print (v)
        print ('-----------------------')
        return v


