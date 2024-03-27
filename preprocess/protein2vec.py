# import numpy as np
# import pandas as pd
# import os

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# data = pd.read_csv('/home/xiazhiliang/Project/data/s_sequence_2200_10label.csv', skiprows=0, dtype=object)  #s_sequence_new.csv
# data = np.array(data).flatten()   # 转换为一维数组
# data = data.tolist()            # 转换为list


# Protvec = {}
# label = []
# with open('/home/xiazhiliang/Project/data/protVec_100d_3grams.csv', 'r') as f:
#     for line in f:
#         k_mer = line[1:4]
#         line1 = line[4:-2].strip().split('\t')
#         vector = [float(x) for x in line1]
#         label.append(str(k_mer))
#         Protvec[k_mer] = vector


# def segmentation(seq):
#     vec = []
#     if len(seq) > 3:
#         segments = [seq[j:j + 3] for j in range(len(seq) - 2)]
#         seg = [''.join([str(j) for j in i]) for i in segments]
#         for i in seg:
#             if i not in label:
#                 i = 'unk'
#             vec.append(Protvec[i])
        
#     else:
#         vec = np.array([Protvec['unk']])
#     vec = np.array(vec)
#     return vec

# def full_map(x):
#     temp = segmentation(x)
#     return temp


# def embeddings(x):
#     data = np.array([full_map(l) for l in x])

#     return data


# p2v = embeddings(data)
# print(p2v, p2v.shape)
# np.savez('Alt_amino_1_protein2vec.npz', p2v)  #try_protein2vec_13label

import numpy as np
import pandas as pd
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

data = pd.read_csv('/home/xiazhiliang/Project/data/s_sequence_2200_10label.csv', skiprows=0, dtype=object)
data = np.array(data).flatten()   # 转换为一维数组
data = data.tolist()              # 转换为list

Protvec = {}
label = []
with open('/home/xiazhiliang/Project/data/protVec_100d_3grams.csv', 'r') as f:
    for line in f:
        k_mer = line[1:4]
        line1 = line[4:-2].strip().split('\t')
        vector = [float(x) for x in line1]
        label.append(str(k_mer))
        Protvec[k_mer] = vector

def segmentation(seq):
    vec = []
    if len(seq) > 3:
        segments = [seq[j:j + 3] for j in range(len(seq) - 2)]
        seg = [''.join([str(j) for j in i]) for i in segments]
        for i in seg:
            if i not in label:
                i = 'unk'
            vec.append(Protvec[i])
    else:
        vec = [Protvec['unk']]
    return np.array(vec)

def full_map(x):
    return segmentation(x)

def embeddings(x):
    return np.array([full_map(l) for l in x], dtype=object)

p2v = embeddings(data)
print(p2v, p2v.shape)
np.savez('same_protein2vec_2200_10label.npz', p2v)  #try_protein2vec_13label