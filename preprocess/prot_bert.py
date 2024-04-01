# -*- coding: utf-8 -*-
from transformers import BertModel, BertTokenizer
import re
import pandas as pd
import numpy as np
import torch
import os

tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert",do_lower_case=False)
model = BertModel.from_pretrained("Rostlab/prot_bert")


#读取数据
data = pd.read_csv('/home/xiazhiliang/Project/data/s_sequence_2200_10label.csv')   #s_sequence_new

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = model.to(device)
model = model.eval()

max_length=512

features = [] 

# 循环处理每个序列
for sequence in data['Sequence']:
    sequence = re.sub(r"[UZOB]", "X", sequence)
    sequence = " ".join(sequence)

    # 将序列分割成多个段落
    chunks = [sequence[i:i+max_length] for i in range(0, len(sequence), max_length)]

    # 创建一个空列表来存储每个分段的嵌入
    chunk_embeddings = []

    # 处理每个分段
    for chunk in chunks:
        # 编码输入序列
        inputs = tokenizer.encode_plus(chunk, add_special_tokens=True, padding='longest', truncation=True, max_length=max_length, return_tensors="pt")
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        # 将输入传入模型获取嵌入
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # 提取嵌入特征
        embedding = outputs.last_hidden_state.squeeze(dim=0)  # 取最后一层隐藏状态
        seq_len = torch.sum(attention_mask).item() - 2  # 减去特殊标记 [CLS] 和 [SEP]
        seq_emd = embedding[1:seq_len+1].cpu().numpy()  # 从第2个位置开始到seq_len的嵌入

        # 将分段的嵌入添加到列表中
        chunk_embeddings.append(seq_emd)

    # 对分段的嵌入进行处理，例如拼接或其他操作，得到整个序列的表示
    combined_embedding = np.concatenate(chunk_embeddings, axis=0)
    # print(combined_embedding,combined_embedding.shape)
    # print(seq_emd,seq_emd.shape)
    features.append(combined_embedding)

features=np.array(features,dtype = object)
# 打印特征列表
print(features,features.shape)
np.savez('same_ProtBert_2200_10label.npz', features)  

