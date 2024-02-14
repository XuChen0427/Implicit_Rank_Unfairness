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

from const import Const as C
from rank_model import RankNet, MLPclassifier, load_point_model, load_pair_model
from prepare_prob_data import read_embs_from_file

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score




def train(path, type='point', topk=5, label_length=2):


    print("loading datas from file...")

    embs, label, rank = read_embs_from_file(path)
    # According to different LLMs' format, parse the format to csv
    # | emb  | label | rank |
    # |---------|---------|---------|
    # | text response embedding | $s\in\mathcal{S}$ | position in the ranking list |

    # In the paper, we utilize following code
    # model = SentenceTransformer("t5-large")
    # stop_word = []
    # with open("stopwords.txt", 'r') as f:
    #     for word in f:
    #         stop_word.append(word.strip())
    # input sentence s
    # filter_s = s.lower()
    # word_list = tokenizer.tokenize(filter_s)
    # word_list = [token for token in word_list if token not in stop_word]
    # keep_words = []
    # for w in word_list:
    #     if w not in stop_word:
    #         keep_words.append(w)
    # keep_s = ' '.join(keep_words)
    # embedding = model.encode([s])

    input_dim = len(embs[0])


    output_dim = label_length
    # Gender:2 Race:3 Continent:5


    if type == 'point':
        x, label = PrepareDataForPointProbing(embs, label)

        model = load_point_model(model_type=model_type, domain=domain,
                                 sensitive_type=sensitive_type, input_dim=input_dim, output_dim=output_dim)
    else:
        x1, x2, label = PrepareDataForPairProbing(embs, label, topk)

        x1 = torch.tensor(x1, dtype=torch.float32)
        x2 = torch.tensor(x2, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        x = torch.cat((x1, x2), dim=-1)

        model = load_pair_model(model_type=model_type, domain=domain,
                                sensitive_type=sensitive_type, input_dim=input_dim, output_dim=output_dim)

    embs_train = torch.tensor(x, dtype=torch.float32)
    embs_test = torch.tensor(copy.copy(x), dtype=torch.float32)

    label_train = torch.tensor(label, dtype=torch.long)
    label_test = torch.tensor(copy.copy(label), dtype=torch.long)

    train = TensorDataset(embs_train, label_train)
    test = TensorDataset(embs_test, label_test)


    #print("start {} eval on names: {} model for {} domain for testing {} attribute".format(type, model_type, domain, sensitive_type))

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    batch_size = 16  # 指定批大小
    shuffle = True  # 打乱数据

    data_loader = DataLoader(train, batch_size=batch_size, shuffle=shuffle)

    batch_size = 128  # 指定批大小
    shuffle = False  # 打乱数据

    test_data_loader = DataLoader(test, batch_size=batch_size, shuffle=shuffle)
    criterion = nn.CrossEntropyLoss()

    num_epochs= 50
    for epoch in trange(num_epochs):
        # 前向传播
        for batch_x, batch_label in data_loader:
            outputs = model(batch_x)
            # 计算损失
            loss = criterion(outputs, batch_label)
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    label = []
    predict = []
    model.eval()

    weight = []
    for batch_emb, batch_label in test_data_loader:
        outputs = model(batch_emb)
        outputs = outputs.detach().numpy()
        predicts = np.argmax(outputs, axis=-1, keepdims=False)
        for i, p in enumerate(predicts):
            weight.append(outputs[i,p])
        labels = batch_label.detach().numpy()
        predict.extend(predicts)
        label.extend(labels)

    label = np.array(label)
    predict = np.array(predict)

    accuracy = accuracy_score(label, predict)
    weight = np.array(weight)

    weight = weight.reshape((K,-1))
    weight =np.mean(weight, axis=0, keepdims=False)
    average = np.mean(weight)
    weight = weight/average

    np.save("weight.npy", weight)
    #weight for your evaluation for each sample

    print("acc:",accuracy)




if __name__ == "__main__":
    K = 5
    train("your parsed path", type='point', topk=K, label_length=2)
