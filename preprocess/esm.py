# import torch
# import pandas as pd
# import numpy as np
# # from transformers import AutoTokenizer, AutoModel

# # tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
# # model = AutoModel.from_pretrained("facebook/esm2_t33_650M_UR50D")

# # Load model directly
# from transformers import AutoTokenizer, AutoModel

# tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D")
# model = AutoModel.from_pretrained("facebook/esm2_t36_3B_UR50D")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

# data = pd.read_csv('/home/xiazhiliang/Project/data/s_sequence_new.csv')

# def encode_sequence(sequence, tokenizer, model):
#     encoded_input = tokenizer(sequence, truncation=True, return_tensors='pt', padding=True)
#     encoded_input = {k: v.to(device) for k, v in encoded_input.items()}  # Move tensors to device
#     with torch.no_grad():
#         # output = model(**encoded_input).last_hidden_state
#         output = model(**encoded_input).last_hidden_state
    
#     return output.squeeze(0).cpu().numpy()

# outputs = []
# for sequence in data['Sequence']:
#     features = encode_sequence(sequence, tokenizer, model)
#     outputs.append(features)


# # Convert outputs to a numpy array
# outputs = np.array(outputs)
# print(outputs,outputs.shape)
# # Save the array as an npz file
# np.savez('try_esm2_3B_13label.npz', outputs)


##csv
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D")
model = AutoModel.from_pretrained("facebook/esm2_t36_3B_UR50D")

# Load the protein sequences from CSV file
data = pd.read_csv('/home/xiazhiliang/Project/data/s_sequence_2200_10label.csv')  #s_sequence_new
sequences = data["Sequence"].tolist()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = model.to(device)
model = model.eval()

# List to store sequence representations
sequence_representations = []

# Process sequences and compute representations
# for i,sequence in enumerate(sequences):
for sequence in sequences:
    inputs = tokenizer(sequence, truncation=True, padding=True, return_tensors="pt",max_length=2000)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    token_representations = outputs.last_hidden_state

    # Compute sequence representation via averaging
    sequence_representation = token_representations[:, 1:-1, :].squeeze(0).cpu().numpy()
 
    sequence_representations.append(sequence_representation)
   
# Convert the list of sequence representations to a torch tensor
# sequence_representations = torch.stack(sequence_representations, dim=0)

# Convert the torch tensor to a numpy array
sequence_representations=np.array(sequence_representations,dtype=object)

# Print the sequence representations and shape
print(sequence_representations)
print(sequence_representations.shape)
np.savez('same_esm2_3B_2200_10label.npz', sequence_representations) #try_esm2_3B_13label_new



# import pandas as pd
# import numpy as np
# import torch
# from transformers import AutoTokenizer, AutoModel

# tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t36_3B_UR50D")
# model = AutoModel.from_pretrained("facebook/esm2_t36_3B_UR50D")

# # Fasta file path
# fasta_file = '/home/xiazhiliang/Project/data/Ref_amino.fasta'

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# if torch.cuda.device_count() > 1:
#     model = torch.nn.DataParallel(model)  # Use DataParallel to replicate the model on multiple GPUs

# model = model.to(device)
# model = model.eval()
# # List to store sequence representations
# sequence_representations = []
# count= 0
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
# np.savez('Ref_amino_esm2_3B.npz', sequence_representations)



