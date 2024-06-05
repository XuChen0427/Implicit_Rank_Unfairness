import copy

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import pandas as pd
import os
###Here we adapt two ways to train a probing: pair-wise and point-wise
import argparse
import json
import re
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, TensorDataset
#import pandas as pd
import ast

from rank_model import RankNet, MLPclassifier, load_point_model, load_pair_model
from prepare_data import read_embs_from_file, PrepareDataForPairProbing

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score




def train(path, type='point', topk=5, label_length=2):


    print("loading datas from file...")

    embs, label, name, rank = read_embs_from_file(path)

    #exit(0)
    # According to different LLMs' format, parse the format to csv
    # | emb  | label | name | rank |
    # |---------|---------|---------|
    # | text response embedding | $s\in\mathcal{S}$ | user names| position in the ranking list |

    # In the paper, we will give an example



    input_dim = len(embs[0])


    output_dim = label_length
    # Gender:2 Race:3 Continent:5



    x1, x2, label, names = PrepareDataForPairProbing(embs, label, name, topk)

    x1 = torch.tensor(x1, dtype=torch.float32)
    x2 = torch.tensor(x2, dtype=torch.float32)
    label = torch.tensor(label, dtype=torch.long)

    x = torch.cat((x1, x2), dim=-1)

    model = load_pair_model(input_dim=input_dim, output_dim=output_dim)


    embs_train = torch.tensor(x, dtype=torch.float32)
    embs_test = torch.tensor(copy.copy(x), dtype=torch.float32)

    label_train = torch.tensor(label, dtype=torch.long)
    label_test = torch.tensor(copy.copy(label), dtype=torch.long)

    train = TensorDataset(embs_train, label_train)
    test = TensorDataset(embs_test, label_test)


    #print("start {} eval on names: {} model for {} domain for testing {} attribute".format(type, model_type, domain, sensitive_type))

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    batch_size = 16
    shuffle = True

    data_loader = DataLoader(train, batch_size=batch_size, shuffle=shuffle)

    batch_size = 128
    #shuffle = False

    test_data_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    num_epochs= 5
    for epoch in trange(num_epochs):

        for batch_x, batch_label in data_loader:
            outputs = model(batch_x)

            loss = criterion(outputs, batch_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    label = []
    predict = []
    model.eval()

    #due to no shuffle, we can order the name
    name_id = 0
    name_set = set([i for i in names])
    name_dict = {name: 0 for name in name_set}

    criterion = nn.CrossEntropyLoss(reduce=False)
    for batch_emb, batch_label in test_data_loader:

        outputs = model(batch_emb)
        loss = criterion(outputs, batch_label).detach().numpy()

        # outputs = outputs.detach().numpy()
        # predicts = np.argmax(outputs, axis=-1, keepdims=False)
        # labels = batch_label.detach().numpy()
        for i in range(len(loss)):
            name = names[name_id]
            name_dict[name] += loss[i]
            name_id += 1
        # predict.extend(predicts)
        # label.extend(labels)

    print(name_dict)





if __name__ == "__main__":
    K = 5
    train("emb_example.csv", type='pair', topk=K, label_length=2)
