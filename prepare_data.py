import torch
import numpy as np
import pandas as pd
import os

import json
import re
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, TensorDataset
# import pandas as pd
import ast



K = 5




def PrepareDataForPointProbing(embs, label):
    # ids = []
    return embs, label


def PrepareDataForPairProbing(embs, label, name, topk=K):
    labels = []
    pos_emb = []
    neg_emb = []
    names = []
    length = len(embs)
    assert length % topk == 0
    for i in range(0, length, topk):
        # batch = i // topk
        for j in range(topk):
            if j != topk - 1:
                for k in range(j + 1, topk):
                    pos_emb.append(embs[i + j])
                    neg_emb.append(embs[i + k])
                    labels.append(label[i + j])
                    names.append(name[i + j])

    return np.array(pos_emb), np.array(neg_emb), np.array(labels), names


def read_embs_from_file(path):

    #According to different LLMs' format, parse the format to csv
    # | emb  | label | name | rank |
    # |---------|---------|---------|
    # | text response embedding | $s\in\mathcal{S}$ | user names | position in the ranking list |

    datas = pd.read_csv(path)


    embs_str = list(datas['emb'].values)

    embs = []
    for index, e in enumerate(embs_str):
        embs.append(ast.literal_eval(e))
    embs = np.array(embs)

    label = np.array(list(datas['label'].values))

    rank = np.array(list(datas['rank'].values))

    names = list(datas['name'].values)

    return embs, label, names, rank


