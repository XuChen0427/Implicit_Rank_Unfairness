import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
# 创建 RankNet 模型
class RankNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(RankNet, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            #nn.Sigmoid()
        )

    def forward(self, x):
        #fusion_feature = torch.cat((x1,x2),dim=-1)
        out = self.model(x)
        out = F.softmax(out, dim=-1)
        return out
        #return out

class MLPclassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLPclassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, x):
        out = self.model(x)
        out = F.softmax(out,dim=-1)
        return out

def load_point_model(input_dim, output_dim):

    loaded_model = MLPclassifier(input_dim, output_dim)  # 创建一个模型实例

    return loaded_model

def load_pair_model(input_dim, output_dim):

    loaded_model = RankNet(input_dim=input_dim, output_dim=output_dim)  # 创建一个模型实例

    return loaded_model



