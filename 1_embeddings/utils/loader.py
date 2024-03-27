# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import random

class SequenceLoader(object):
    def __init__(self, data_dir='./mydata/', batch_size=32, shuffle=True, num_workers=None):
        self.dataset_name = 'same_protein2vec_2200_10label'
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.train_steps = int(np.ceil(len(np.load(f'./mydata/{self.dataset_name}_process_data/{self.dataset_name}_train.npz')['features'])/batch_size))
        self.valid_steps = int(np.ceil(len(np.load(f'./mydata/{self.dataset_name}_process_data/{self.dataset_name}_val.npz')['features'])/batch_size))
        self.test_steps = int(np.ceil(len(np.load(f'./mydata/{self.dataset_name}_process_data/{self.dataset_name}_test.npz')['features'])/batch_size))

        data = np.load(f'/home/xiazhiliang/Project/Bioinfor-DeepATT-main/mydata/{self.dataset_name}.npz', allow_pickle=True)['arr_0']
        data = data.take(1)
        self.embedding_size = data.shape[-1]

    def train_data_generator(self):
        data = np.load(f'./mydata/{self.dataset_name}_process_data/{self.dataset_name}_train.npz')
        x_train, y_train = data['features'], data['labels']
        index = 0
        
        while True:
            # Generate or load the next batch of training data
            batch_x = x_train[index: index + self.batch_size]
            batch_y = y_train[index: index + self.batch_size]

            # Update the index for the next batch
            index += self.batch_size
            if index >= len(x_train):
                index = 0
                
            if len(batch_x) < self.batch_size:
                break
              
            yield batch_x, batch_y

    def get_train_data(self):
        train_dataset = tf.data.Dataset.from_generator(
            self.train_data_generator,
            output_signature=(
                tf.TensorSpec(shape=(self.batch_size, 2000,self.embedding_size), dtype=tf.float32),
                tf.TensorSpec(shape=(self.batch_size,10), dtype=tf.float32)
            )
        )
        if self.shuffle:
            buffer_size = len(np.load(f'./mydata/{self.dataset_name}_process_data/{self.dataset_name}_train.npz')['features'])
            # buffer_size = max(buffer_size // 2,1)  # 设置为训练集样本数量的一半或四分之一
            buffer_size = 192
            train_dataset = train_dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)

        train_dataset = train_dataset.repeat()
        train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return train_dataset
    
    def get_valid_data(self):
        data = np.load(f'./mydata/{self.dataset_name}_process_data/{self.dataset_name}_val.npz')
        x_val, y_val = data['features'], data['labels']
        dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        dataset = dataset.batch(self.batch_size, drop_remainder=False)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset
    

    def get_test_data(self):
        data = np.load(f'./mydata/{self.dataset_name}_process_data/{self.dataset_name}_test.npz')
        x_test, y_test = data['features'], data['labels']
        dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        dataset = dataset.batch(self.batch_size, drop_remainder=False)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset
    
    def get_train_data_iterator(self):
        train_dataset = self.get_train_data()
        return iter(train_dataset)
