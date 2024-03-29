# -*- coding: utf-8 -*-
import os
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences

class MyDataset:
    @classmethod
    def create_dataset(cls, npz_file, csv_file, label_columns, buffer_size=1000, prefetch_size=200, save_dir=None):
        # Load csv file
        labels_df = pd.read_csv(csv_file)

        # Load npz file
        data = np.load(npz_file, allow_pickle=True)['arr_0']
        data = data.take(1)
        embedding_size = data.shape[-1]

        # Compute maximum sequence length
        maxlen = 2000

        # Convert labels to one-hot encoding
        labels = labels_df[label_columns].values
        labels = tf.convert_to_tensor(labels, dtype=tf.int32).numpy()

        # Create generator function to load npz file
        def load_npz_file(npz_file, maxlen):
            with np.load(npz_file, allow_pickle=True) as data:
                for sample_data in data['arr_0']:
                    padded_data = pad_sequences([sample_data], maxlen=maxlen, dtype=np.float32, padding='post', truncating='post', value=0)
                    yield padded_data[0]

        dataset = tf.data.Dataset.from_generator(
            lambda: load_npz_file(npz_file, maxlen),
            output_signature=tf.TensorSpec(shape=(maxlen, embedding_size), dtype=tf.float32)
        )



        # Convert labels to dataset and zip with features
        labels_dataset = tf.data.Dataset.from_tensor_slices(labels)
        # labels_dataset = labels_dataset.shuffle(buffer_size)
        dataset = tf.data.Dataset.zip((dataset, labels_dataset))
        dataset = dataset.map(map_func=cls.parse_function, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # dataset = dataset.map(lambda x, y: (tf.cast(x, tf.float32), tf.cast(y, tf.int32)))
       

        # # Split dataset into train, validation, and test sets
        dataset_size = len(labels)
        train_size = int(0.8 * dataset_size)
        val_size = int(0.1 * dataset_size)
        test_size = dataset_size - train_size - val_size

        train_dataset = dataset.take(train_size)
        val_dataset = dataset.skip(train_size).take(val_size)
        test_dataset = dataset.skip(train_size + val_size).take(test_size)

        # train_dataset = train_dataset.batch(batch_size)
        # train_dataset = train_dataset.prefetch(prefetch_size)

        # val_dataset = val_dataset.batch(batch_size)
        # val_dataset = val_dataset.prefetch(prefetch_size)

        # test_dataset = test_dataset.batch(batch_size)
        # test_dataset = test_dataset.prefetch(prefetch_size)

        if save_dir is not None:
            npz_file_name = os.path.splitext(os.path.basename(npz_file))[0]
            process_data_dir = os.path.join(os.path.dirname(npz_file), f'{npz_file_name}_process_data')
            os.makedirs(process_data_dir, exist_ok=True)

            # Save train, val, and test datasets as npz files
            train_features, train_labels = zip(*[(x.numpy(), y.numpy()) for x, y in train_dataset])
            val_features, val_labels = zip(*[(x.numpy(), y.numpy()) for x, y in val_dataset])
            test_features, test_labels = zip(*[(x.numpy(), y.numpy()) for x, y in test_dataset])

            np.savez(os.path.join(process_data_dir, f'{npz_file_name}_train.npz'), features=train_features, labels=train_labels)
            np.savez(os.path.join(process_data_dir, f'{npz_file_name}_val.npz'), features=val_features, labels=val_labels)
            np.savez(os.path.join(process_data_dir, f'{npz_file_name}_test.npz'), features=test_features, labels=test_labels)


    @staticmethod
    def parse_function(features, labels):
        features = tf.cast(features, tf.float32)
        labels = tf.cast(labels, tf.int32)
        return features, labels

if __name__ == '__main__':
    # Load data
    npz_file = '/home/Project/mydata/try_esm2_3B_2200_10label.npz'
    csv_file = '/home/Project/mydata/s_prot_func_2200_10label.csv'   #prot_func.csv   
    label_columns = ['GO0002250','GO0004984','GO0006955','GO0007420','GO0042742','GO0045087','GO0045471','GO0071277','GO0071456','GO0150104'] #10label
    # Call create_dataset function
    MyDataset.create_dataset(npz_file=npz_file, csv_file=csv_file,label_columns=label_columns,save_dir='/home/Project/mydata/')
