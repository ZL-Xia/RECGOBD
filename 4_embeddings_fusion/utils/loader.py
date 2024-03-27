# -*- coding: utf-8 -*-
# import numpy as np
# import tensorflow as tf
# import scipy.io as sio


# class SequenceLoader(object):
#     # def __init__(self, data_dir='./data/', batch_size=64, shuffle=True, num_workers=None):
#     def __init__(self, data_dir='./mydata/', batch_size=64, shuffle=True, num_workers=None):
#         self.data_dir = data_dir
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         self.num_workers = num_workers
#         # self.train_steps = int(np.ceil(4400000/batch_size))
#         # self.valid_steps = int(np.ceil(8000/100))
#         # self.test_steps = int(np.ceil(455024/100))
#         self.train_steps = int(np.ceil(2429*0.8/batch_size))
#         self.valid_steps = int(np.ceil(2429*0.1/batch_size))
#         self.test_steps = int(np.ceil(2429*0.1/batch_size))

#     def get_train_data(self):
#         filenames = ['./data/traindata-00.tfrecord', './data/traindata-01.tfrecord',
#                      './data/traindata-02.tfrecord', './data/traindata-03.tfrecord']
#         dataset = tf.data.TFRecordDataset(filenames, buffer_size=100000, num_parallel_reads=None)
#         if self.shuffle == True:
#             dataset = dataset.shuffle(buffer_size=10000, reshuffle_each_iteration=True)
#         dataset = dataset.map(map_func=self.parse_function, num_parallel_calls=self.num_workers)
#         dataset = dataset.batch(self.batch_size, drop_remainder=False)
#         dataset = dataset.prefetch(buffer_size=10000)
#         return dataset # 4400000/64 = 68750

#     def get_valid_data(self):
#         data = sio.loadmat('./data/valid.mat')
#         x = data['validxdata']  # shape = (8000, 4, 1000)
#         y = data['validdata']  # shape = (8000, 919)
#         x = np.transpose(x, (0, 2, 1)).astype(dtype=np.float32)  # shape = (8000, 1000, 4)
#         y = np.transpose(y, (0, 1)).astype(dtype=np.int32)  # shape = (8000, 919)
#         dataset = tf.data.Dataset.from_tensor_slices((x, y))
#         dataset = dataset.batch(100)
#         dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
#         return dataset # 8000/100 = 80

#     def get_test_data(self):
#         filenames = ['./data/testdata.tfrecord']
#         dataset = tf.data.TFRecordDataset(filenames, buffer_size=10000, num_parallel_reads=None)
#         dataset = dataset.map(map_func=self.parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#         dataset = dataset.batch(100, drop_remainder=False)
#         dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
#         return dataset # 455024/64 = 7109.75 = 7110

#     @staticmethod
#     def parse_function(example_proto):
#         dics = {
#             'x': tf.io.FixedLenFeature([1000, 4], tf.int64),
#             'y': tf.io.FixedLenFeature([919], tf.int64),
#         }
#         parsed_example = tf.io.parse_single_example(example_proto, dics)
#         x = tf.reshape(parsed_example['x'], [1000, 4])
#         y = tf.reshape(parsed_example['y'], [919])
#         x = tf.cast(x, tf.float32)
#         y = tf.cast(y, tf.int32)
#         return (x, y)

# import os
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from tensorflow.keras.preprocessing.sequence import pad_sequences


# class SequenceLoader(object):
#     def __init__(self, data_dir='./mydata/', batch_size=64, shuffle=True, num_workers=None):
#         self.data_dir = data_dir
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         self.num_workers = num_workers
#         self.train_steps = int(np.ceil(2429*0.8/batch_size))
#         self.valid_steps = int(np.ceil(2429*0.1/batch_size))
#         self.test_steps = int(np.ceil(2429*0.1/batch_size))
#         # Load data
#         npz_file = '/home/xiazhiliang/Project/Bioinfor-DeepATT-main/mydata/prot_bert.npz'
#         csv_file = '/home/xiazhiliang/Project/Bioinfor-DeepATT-main/mydata/prot_func.csv'
#         label_columns = ['GO0001666', 'GO0002250', 'GO0006954','GO0006955','GO0006974','GO0007399','GO0007420','GO0042493','GO0045087']
#         self.train_dataset, self.valid_dataset, self.test_dataset = self.create_dataset(npz_file, csv_file, label_columns, batch_size)
        

#     def create_dataset(self,npz_file, csv_file, label_columns, batch_size, buffer_size=1000, prefetch_size=200):
#         # Load csv file
#         labels_df = pd.read_csv(csv_file)

#         # Load npz file
#         data = np.load(npz_file, allow_pickle=True)['arr_0']
#         data=data.take(1)
#         embedding_size = data.shape[-1]
        
#         # Compute maximum sequence length
#         maxlen=2000

#         # Convert labels to one-hot encoding
#         labels = labels_df[label_columns].values
#         labels = tf.convert_to_tensor(labels, dtype=tf.int32).numpy()


#         # Create generator function to load npz file
        # def load_npz_file(npz_file, maxlen):
        #     with np.load(npz_file, allow_pickle=True) as data:
        #         for sample_data in data['arr_0']:
        #             padded_data = pad_sequences([sample_data], maxlen=maxlen, dtype=np.float32, padding='post', truncating='post', value=0)
        #             yield padded_data[0]

#         dataset = tf.data.Dataset.from_generator(
#             lambda: load_npz_file(npz_file, maxlen),
#             output_signature=tf.TensorSpec(shape=(maxlen, embedding_size), dtype=tf.float32)
#         )

#         # Convert labels to dataset and zip with features
#         labels_dataset = tf.data.Dataset.from_tensor_slices(labels)
#         dataset = tf.data.Dataset.zip((dataset, labels_dataset))
#         dataset = dataset.map(map_func=self.parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
#         dataset = dataset.shuffle(buffer_size)

#         # Split dataset into train, validation, and test sets
#         dataset_size = len(labels)
#         train_size = int(0.8 * dataset_size)
#         val_size = int(0.1 * dataset_size)
#         test_size = dataset_size - train_size - val_size
#         # test_size = int(0.1 * dataset_size)
#         # val_size = int(0.1 * dataset_size)
#         # train_size = dataset_size - test_size - val_size

#         train_dataset = dataset.take(train_size)
#         val_dataset = dataset.skip(train_size).take(val_size)
#         test_dataset = dataset.skip(train_size + val_size).take(test_size)
#         # test_dataset = dataset.take(test_size)
#         # val_dataset = dataset.skip(test_size).take(val_size)
#         # train_dataset = dataset.skip(test_size + val_size).take(train_size)


#         train_dataset = train_dataset.batch(batch_size)
#         train_dataset = train_dataset.prefetch(prefetch_size)

#         val_dataset = val_dataset.batch(batch_size)
#         val_dataset = val_dataset.prefetch(prefetch_size)

#         test_dataset = test_dataset.batch(batch_size)
#         test_dataset = test_dataset.prefetch(prefetch_size)

#         return train_dataset, val_dataset, test_dataset
#     # Example usage

#     def get_train_dataset(self):
#         return self.train_dataset

#     def get_valid_dataset(self):
#         return self.valid_dataset

#     def get_test_dataset(self):
#         return self.test_dataset

#     @staticmethod
#     def parse_function(features, labels):
#         features = tf.cast(features, tf.float32)
#         labels = tf.cast(labels, tf.int32)
#         return features, labelsa


import os
import tensorflow as tf
import numpy as np
import pandas as pd
import random

class SequenceLoader(object):
    def __init__(self, data_dir='./mydata/', batch_size=32, shuffle=True, num_workers=None):
        self.dataset_name =  ['same_esm2_3B_2200_10label','same_prot_bert_2200_10label','same_protein2vec_2200_10label','same_onehot_2200_10label']
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
        data_4 = np.load(f'./mydata/{self.dataset_name[3]}_process_data/{self.dataset_name[3]}_train.npz')

        x_train_1, y_train_1 = data_1['features'], data_1['labels']
        x_train_2, y_train_2 = data_2['features'], data_2['labels']
        x_train_3, y_train_3 = data_3['features'], data_3['labels']
        x_train_4, y_train_4 = data_4['features'], data_4['labels']

        index = 0
        while True:
            batch_x_1 = x_train_1[index: index + self.batch_size]
            batch_y_1 = y_train_1[index: index + self.batch_size]

            batch_x_2 = x_train_2[index: index + self.batch_size]
            batch_y_2 = y_train_2[index: index + self.batch_size]

            batch_x_3 = x_train_3[index: index + self.batch_size]
            batch_y_3 = y_train_3[index: index + self.batch_size]

            
            batch_x_4 = x_train_4[index: index + self.batch_size]
            batch_y_4 = y_train_4[index: index + self.batch_size]


            index += self.batch_size
            if index >= len(x_train_1):
                index = 0

            if len(batch_x_1) < self.batch_size:
                break

            yield ((batch_x_1, batch_x_2, batch_x_3,batch_x_4), batch_y_1)

    def get_train_data(self):
        train_dataset = tf.data.Dataset.from_generator(
            self.train_data_generator_multi_embedding,
            output_signature=(
                (
                    tf.TensorSpec(shape=(self.batch_size, 2000, self.embedding_sizes[0]), dtype=tf.float32),
                    tf.TensorSpec(shape=(self.batch_size, 2000, self.embedding_sizes[1]), dtype=tf.float32),
                    tf.TensorSpec(shape=(self.batch_size, 2000, self.embedding_sizes[2]), dtype=tf.float32),
                    tf.TensorSpec(shape=(self.batch_size, 2000, self.embedding_sizes[3]), dtype=tf.float32),
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
        data_4 = np.load(f'./mydata/{self.dataset_name[3]}_process_data/{self.dataset_name[3]}_val.npz')

        x_val_1, y_val_1 = data_1['features'], data_1['labels']
        x_val_2, y_val_2 = data_2['features'], data_2['labels']
        x_val_3, y_val_3 = data_3['features'], data_3['labels']
        x_val_4, y_val_4 = data_4['features'], data_4['labels']


        dataset = tf.data.Dataset.from_tensor_slices(
        (
            (x_val_1, x_val_2, x_val_3,x_val_4),
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
        data_4 = np.load(f'./mydata/{self.dataset_name[3]}_process_data/{self.dataset_name[3]}_test.npz')

        x_test_1, y_test_1 = data_1['features'], data_1['labels']
        x_test_2, y_test_2 = data_2['features'], data_2['labels']
        x_test_3, y_test_3 = data_3['features'], data_3['labels']
        x_test_4, y_test_4 = data_4['features'], data_4['labels']

        dataset = tf.data.Dataset.from_tensor_slices(
            (
                (x_test_1, x_test_2, x_test_3,x_test_4),
                y_test_1
            )
        )

        dataset = dataset.batch(self.batch_size, drop_remainder=False)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return dataset




