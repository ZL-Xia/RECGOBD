# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import random

class SequenceLoader(object):
    def __init__(self, data_dir='./mydata/', batch_size=32, shuffle=True, num_workers=None):
        self.dataset_name =  ['same_esm2_3B_2200_10label','same_prot_bert_2200_10label','same_onehot_2200_10label']
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.train_steps = int(np.ceil(len(np.load(f'./mydata/{self.dataset_name[0]}_process_data/{self.dataset_name[0]}_train.npz')['features'])/batch_size))
        self.valid_steps = int(np.ceil(len(np.load(f'./mydata/{self.dataset_name[0]}_process_data/{self.dataset_name[0]}_val.npz')['features'])/batch_size))
        self.test_steps = int(np.ceil(len(np.load(f'./mydata/{self.dataset_name[0]}_process_data/{self.dataset_name[0]}_test.npz')['features'])/batch_size))
        
        self.embedding_sizes = []
        for dataset_name in self.dataset_name:
            data = np.load(f'/home/xiazhiliang/Project/preprocessing/{dataset_name}.npz', allow_pickle=True)['arr_0']
            data = data.take(1)
            embedding_size = data.shape[-1]
            self.embedding_sizes.append(embedding_size)
    
    def get_train_data_iterator(self):
        train_dataset = self.get_train_data()
        return iter(train_dataset)
    
    
    def train_data_generator_multi_embedding(self):
        data_1 = np.load(f'./mydata/{self.dataset_name[0]}_process_data/{self.dataset_name[0]}_train.npz')
        data_2 = np.load(f'./mydata/{self.dataset_name[1]}_process_data/{self.dataset_name[1]}_train.npz')
        data_3 = np.load(f'./mydata/{self.dataset_name[2]}_process_data/{self.dataset_name[2]}_train.npz')

        x_train_1, y_train_1 = data_1['features'], data_1['labels']
        x_train_2, y_train_2 = data_2['features'], data_2['labels']
        x_train_3, y_train_3 = data_3['features'], data_3['labels']

        index = 0
        while True:
            batch_x_1 = x_train_1[index: index + self.batch_size]
            batch_y_1 = y_train_1[index: index + self.batch_size]

            batch_x_2 = x_train_2[index: index + self.batch_size]
            batch_y_2 = y_train_2[index: index + self.batch_size]

            batch_x_3 = x_train_3[index: index + self.batch_size]
            batch_y_3 = y_train_3[index: index + self.batch_size]


            index += self.batch_size
            if index >= len(x_train_1):
                index = 0

            if len(batch_x_1) < self.batch_size:
                break

            yield ((batch_x_1, batch_x_2, batch_x_3), batch_y_1)

    def get_train_data(self):
        train_dataset = tf.data.Dataset.from_generator(
            self.train_data_generator_multi_embedding,
            output_signature=(
                (
                    tf.TensorSpec(shape=(self.batch_size, 2000, self.embedding_sizes[0]), dtype=tf.float32),
                    tf.TensorSpec(shape=(self.batch_size, 2000, self.embedding_sizes[1]), dtype=tf.float32),
                    tf.TensorSpec(shape=(self.batch_size, 2000, self.embedding_sizes[2]), dtype=tf.float32),
                ),
                tf.TensorSpec(shape=(self.batch_size, 10), dtype=tf.float32)
            )
        )
        if self.shuffle:
            buffer_size = len(np.load(f'./mydata/{self.dataset_name[0]}_process_data/{self.dataset_name[0]}_train.npz')['features'])
            # buffer_size = max(buffer_size // 2,1)  # 设置为训练集样本数量的一半或四分之一
            buffer_size = 200
            train_dataset = train_dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)
            
        train_dataset = train_dataset.repeat()
        train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return train_dataset

    def get_valid_data(self):
        data_1 = np.load(f'./mydata/{self.dataset_name[0]}_process_data/{self.dataset_name[0]}_val.npz')
        data_2 = np.load(f'./mydata/{self.dataset_name[1]}_process_data/{self.dataset_name[1]}_val.npz')
        data_3 = np.load(f'./mydata/{self.dataset_name[2]}_process_data/{self.dataset_name[2]}_val.npz')

        x_val_1, y_val_1 = data_1['features'], data_1['labels']
        x_val_2, y_val_2 = data_2['features'], data_2['labels']
        x_val_3, y_val_3 = data_3['features'], data_3['labels']


        dataset = tf.data.Dataset.from_tensor_slices(
        (
            (x_val_1, x_val_2, x_val_3),
            y_val_1
        )
        )

        dataset = dataset.batch(self.batch_size, drop_remainder=False)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset

    def get_test_data(self):
        data_1 = np.load(f'./mydata/{self.dataset_name[0]}_process_data/{self.dataset_name[0]}_test.npz')
        data_2 = np.load(f'./mydata/{self.dataset_name[1]}_process_data/{self.dataset_name[1]}_test.npz')
        data_3 = np.load(f'./mydata/{self.dataset_name[2]}_process_data/{self.dataset_name[2]}_test.npz')

        x_test_1, y_test_1 = data_1['features'], data_1['labels']
        x_test_2, y_test_2 = data_2['features'], data_2['labels']
        x_test_3, y_test_3 = data_3['features'], data_3['labels']

        dataset = tf.data.Dataset.from_tensor_slices(
            (
                (x_test_1, x_test_2, x_test_3),
                y_test_1
            )
        )

        dataset = dataset.batch(self.batch_size, drop_remainder=False)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset




