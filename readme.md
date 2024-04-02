# RecGOBD
The RecGOBD model is a protein function prediction model based on TensorFlow 2.6. We integrate embeddings of sequences obtained from various pretrained models, input them into a bidirectional LSTM network, and combine them with Gene Ontology (GO) through self-attention mechanisms to establish relationships between sequences and GO terms. Using a category dense layer, we predict the correlation between protein sequences and GO terms. Finally, we evaluate the model's performance using the AUPR and AUROC metrics. After merging the four types of embeddings, we achieve better results, with an average AUROC of 0.917 and an average AUPR of 0.694.
## USAGE
We train our model on Nvidia GeForce 3090. Our dataset includes independent training, validation, and testing sets.
### 1. Data Acquisition
#### Pretrained Model Requirements
python 3.7 | pytorch 11.8 | cuda 12.0 | cudnn 8.1
#### Convert Sequences to Embeddings
```python
python preprocess/esm.py
python preprocess/prot_bert.py
python preprocess/protein2vec.py
python preprocess/onehot.py
```
#### Data Splitting

```python
python preprocess/preprocess.py
```
### 2. Model Training Requirements
python 3.7 | tensorflow 2.6 | cuda 12.0| cudnn 8.1
### 3.Train
```python
python main.py -e train -c ./config/config_test.json
```
### 4.Test
```python
python main.py -e test -c ./config/config_test.json
```
### 5.Result
If you trained your model in ./4_embeddings, you can find the results in ./4_embeddings/result. Other n_embeddings are similar.
