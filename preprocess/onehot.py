import numpy as np
import pandas as pd

# 读取CSV文件
data = pd.read_csv('/home/xiazhiliang/Project/data/s_sequence_2200_10label.csv')

# 获取蛋白质序列列
sequences = data['Sequence']

# 创建字典，映射氨基酸到整数编码
aa_dict = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
    'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19
}

# 将每个氨基酸编码为One-hot向量
num_aa = len(aa_dict)
one_hot_sequences = []
for seq in sequences:
    one_hot_seq = np.zeros((len(seq), num_aa))
    for i, aa in enumerate(seq):
        if aa in aa_dict:
            aa_index = aa_dict[aa]
            one_hot_seq[i, aa_index] = 1
    one_hot_sequences.append(one_hot_seq)

# 转换为 numpy 数组
one_hot_sequences = np.array(one_hot_sequences, dtype=object)

# 输出One-hot编码后的序列列表
print(one_hot_sequences, one_hot_sequences.shape)
np.savez('same_onehot_2200_10label.npz', one_hot_sequences)

# #fasta文件处理
# import pandas as pd
# import numpy as np
# import torch
# from transformers import AutoTokenizer, AutoModel

# tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D")
# model = AutoModel.from_pretrained("facebook/esm2_t36_3B_UR50D")

# # Fasta file path
# fasta_file = '/home/xiazhiliang/Project/data/Alt_amino.fasta'

# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# # if torch.cuda.device_count() > 1:
# #     model = torch.nn.DataParallel(model)  # Use DataParallel to replicate the model on multiple GPUs

# model = model.to(device)
# model = model.eval()
# # List to store sequence representations
# sequence_representations = []
# count = 0
# # Process sequences and compute representations
# with open(fasta_file, "r") as file:
#     for line in file:
#         count+=1
#         sequence = line.strip()
#         inputs = tokenizer(sequence, truncation=True, padding=True, return_tensors="pt",max_length=2560)
#         input_ids = inputs["input_ids"].to(device)
#         attention_mask = inputs["attention_mask"].to(device)

#         with torch.no_grad():
#             outputs = model(input_ids, attention_mask=attention_mask)
#         token_representations = outputs.last_hidden_state

#         # Compute sequence representation via averaging
#         sequence_representation = token_representations[:, 1:-1, :].squeeze(0).cpu().numpy()
#         print(count,"------",sequence_representation.shape)

#         sequence_representations.append(sequence_representation)

# # Convert the list of sequence representations to a numpy array
# sequence_representations = np.array(sequence_representations)

# # Print the sequence representations and shape
# # print(sequence_representations)
# print(sequence_representations.shape)
# np.savez('Alt_amino_esm2_3B.npz', sequence_representations)

# import numpy as np

# # 从NPZ文件中加载数据
# data = np.load('/home/xiazhiliang/Project/preprocessing/a_new_file.npz',allow_pickle=True)

# # 获取原始数据中的所有键（通常有一个或多个键）
# keys = data.files

# # 访问每个键的数据
# for key in keys:
#     # 获取数组的形状
#     shape = data[key].shape
#     print(f"Array '{key}' 的形状为：{shape}")
    
#     # 查看最后几条数据
#     if len(shape) == 2:
#         # 如果数组是二维的
#         last_few_rows = data[key][-5:]  # 获取最后5行数据（可以根据需要修改数字）
#         print(f"Array '{key}' 的最后几条数据为：\n{last_few_rows}")
#     else:
#         # 如果数组是其他维度的，可以根据需要自行处理
#         print(f"Array '{key}' 不是二维数组，无法查看最后几条数据")

# # 关闭文件
# data.close()

# import numpy as np

# # 加载 .npz 文件
# data = np.load('/home/xiazhiliang/Project/preprocessing/a_new_file.npz',allow_pickle=True)

# # 获取文件中的数组名称
# array_names = data.files

# # 循环遍历数组名称
# for array_name in array_names:
#     array_data = data[array_name]
    
#     # 获取数组的形状
#     shape = array_data.shape
    
#     # 打印数组的最后五条数据
#     if len(shape) == 1:
#         print(f"数组 '{array_name}' 的最后五条数据:")
#         print(array_data[-5:])
#     elif len(shape) == 2:
#         print(f"数组 '{array_name}' 的最后五条数据:")
#         print(array_data[-5:, :])
#     elif len(shape) == 3:
#         print(f"数组 '{array_name}' 的最后五条数据:")
#         print(array_data[-5:, :, :])
#     # 添加更多维度的情况（根据需要）
