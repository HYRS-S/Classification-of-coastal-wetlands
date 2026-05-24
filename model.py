import torch
from torch import nn
import torch.nn.functional as F
from math import sqrt

class SelfAttention(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v):
        super(SelfAttention, self).__init__()
        self.dim_q = dim_q
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.linear_q = nn.Linear(dim_q, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_q, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_q, dim_v, bias=False)
        self._norm_fact = 1 / sqrt(dim_k)

    def forward(self, x):
        batch, n, dim_q = x.shape
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)
        dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact
        dist = torch.softmax(dist, dim=-1)
        att = torch.bmm(dist, v)
        return att

class Inception(nn.Module):
    def __init__(self, in_channels):
        super(Inception, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=1)
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=1),
            nn.Conv2d(16, 24, kernel_size=5, padding=2)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=1),
            nn.Conv2d(16, 24, kernel_size=3, padding=1),
            nn.Conv2d(24, 24, kernel_size=3, padding=1)
        )
        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)

    def forward(self, x):
        branch1 = self.branch1(x)
        branch5 = self.branch5(x)
        branch3 = self.branch3(x)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        return torch.cat((branch1, branch5, branch3, branch_pool), dim=1)

class SaBranch(nn.Module):
    def __init__(self):
        super(SaBranch, self).__init__()
        self.conv1 = nn.Conv2d(1, 30, 1)
        self.conv2 = nn.Conv2d(88, 88, 3, 1, 1)
        self.inception = Inception(in_channels=30)
        self.mp = nn.MaxPool2d(2)

    def forward(self, x):
        x = F.relu(self.mp(self.conv1(x)))
        x = self.inception(x)
        out = F.relu(self.mp(self.conv2(x)))
        return out

class SEblock(nn.Module):
    def __init__(self, channel, r=0.5):
        super(SEblock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, int(channel * r)),
            nn.ReLU(),
            nn.Linear(int(channel * r), channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        branch = self.global_avg_pool(x)
        branch = branch.view(branch.size(0), -1)
        weight = self.fc(branch)
        h, w = weight.shape
        weight = torch.reshape(weight, (h, w, 1, 1))
        scale = weight * x
        return scale

class AFP(nn.Module):
    def __init__(self, n):
        super(AFP, self).__init__()
        self.branch1 = nn.Sequential(
            nn.MaxPool2d(3, 1, padding=1),
        )
        self.branch2 = nn.Sequential(
            nn.AvgPool2d(3, 1, padding=1),
        )
        self.branch3_1 = nn.Sequential(
            nn.Conv2d(n, n//3, 1),
            nn.Conv2d(n//3, n//3, 3, padding=1),
            nn.Conv2d(n//3, n//3, 3, padding=1),
        )
        self.branch3_2 = nn.Sequential(
            nn.Conv2d(n, n//3*2, 1),
            nn.Conv2d(n//3*2, n//3*2, 3, padding=1)
        )
        self.branch_SE = SEblock(channel=n)
        self.w = nn.Parameter(torch.ones(4))

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3_1 = self.branch3_1(x)
        b3_2 = self.branch3_2(x)
        b3_Combine = torch.cat((b3_1, b3_2), dim=1)
        b3 = self.branch_SE(b3_Combine)
        b4 = x
        w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        w3 = torch.exp(self.w[2]) / torch.sum(torch.exp(self.w))
        w4 = torch.exp(self.w[3]) / torch.sum(torch.exp(self.w))
        x_out = b1 * w1 + b2 * w2 + b3 * w3 + b4 * w4
        return x_out

class MSTA(nn.Module):
    def __init__(self, in_channels, hidden_dim=32):
        super(MSTA, self).__init__()
        self.key_conv   = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        self.query_fc   = nn.Linear(in_channels, hidden_dim)
        self.out_conv   = nn.Conv2d(hidden_dim, in_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.size()
        center = x[:, :, H//2, W//2]
        q = self.query_fc(center)
        q = q.unsqueeze(1)
        k = self.key_conv(x).view(B, -1, H*W)
        v = self.value_conv(x).view(B, -1, H*W)
        att = torch.bmm(q, k) / (k.size(1)**0.5)
        att = torch.softmax(att, dim=-1)
        out = torch.bmm(att, v.transpose(1,2))
        out = out.transpose(1,2).view(B, -1, 1, 1)
        out = out.expand(-1, -1, H, W)
        out = self.out_conv(out)
        return out + x

class MSSTANet(nn.Module):
    def __init__(self, bands, classes):
        super(MSSTANet, self).__init__()
        self.sa_net = SaBranch()
        self.se_net = SelfAttention(bands, 200, 200)
        self.msta = MSTA(in_channels=96, hidden_dim=32)
        self.ft_fusion = AFP(96)
        self.cls = nn.Sequential(
            nn.Linear(96*5*5, 1200),
            nn.ReLU(inplace=True),
            nn.Linear(1200, 400),
            nn.ReLU(inplace=True),
            nn.Linear(400, classes)
        )

    def forward(self, x_spa, x_spe):
        B = x_spa.size(0)
        spa_ft = self.sa_net(x_spa)
        spe_ft = self.se_net(x_spe)
        spe_ft = spe_ft.reshape(B, 8, 5, 5)
        ss = torch.cat([spa_ft, spe_ft], dim=1)
        ss = self.msta(ss)
        fusion = self.ft_fusion(ss)
        fusion = fusion.view(B, -1)
        logits = self.cls(fusion)
        return logits