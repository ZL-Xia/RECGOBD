# -*- coding: utf-8 -*-
# import h5py
# import numpy as np
# import tensorflow as tf
# import scipy.io as sio
# # from mat4py import mat4py
# from tqdm import tqdm


# def serialize_example(x, y):
#     # Create a dictionary mapping the feature name to the tf.Example-compatible data type.
#     example = {
#         'x': tf.train.Feature(int64_list=tf.train.Int64List(value=x.flatten())),
#         'y': tf.train.Feature(int64_list=tf.train.Int64List(value=y.flatten()))}

#     # Create a Features message using tf.train.Example.
#     example = tf.train.Features(feature=example)
#     example = tf.train.Example(features=example)
#     serialized_example = example.SerializeToString()
#     return serialized_example

# def traindata_to_tfrecord():
#     filename = './data/train.mat'
#     with h5py.File(filename, 'r') as file:
#         x = file['trainxdata'] # shape = (1000, 4, 4400000)
#         y = file['traindata'] # shape = (919, 4400000)
#         x = np.transpose(x, (2, 0, 1)) # shape = (4400000, 1000, 4)
#         y = np.transpose(y, (1, 0)) # shape = (4400000, 919)

#     for file_num in range(4):
#         with tf.io.TFRecordWriter('./data/traindata-%.2d.tfrecord' % file_num) as writer:
#             for i in tqdm(range(file_num*1100000, (file_num+1)*1100000), desc="Processing Train Data {}".format(file_num), ascii=True):
#                 example_proto = serialize_example(x[i], y[i])
#                 writer.write(example_proto)

  
# def testdata_to_tfrecord():
#     filename = './data/test.mat' 
#     data = sio.loadmat(filename)
#     # print(data)
#     x = data['testxdata'] # shape = (455024, 4, 1000)
#     y = data['testdata'] # shape = (455024, 919)
#     x = np.transpose(x, (0, 2, 1)) # shape = (455024, 1000, 4)
#     y = np.transpose(y, (0, 1)) # shape = (455024, 919)

#     with tf.io.TFRecordWriter('./data/testdata.tfrecord') as writer:
#         for i in tqdm(range(len(y)), desc="Processing Test Data", ascii=True):
#             example_proto = serialize_example(x[i], y[i])
#             writer.write(example_proto)

# if __name__ == '__main__':
#     # Write the train data and test data to .tfrecord file.
#     traindata_to_tfrecord()
#     testdata_to_tfrecord()


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
        train_dataset = train_dataset.shuffle(buffer_size)

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
    npz_file = '/home/xiazhiliang/Project/Bioinfor-DeepATT-main/mydata/try_esm2_3B_2200_10label.npz'
    csv_file = '/home/xiazhiliang/Project/Bioinfor-DeepATT-main/mydata/s_prot_func_2200_10label.csv'   #prot_func.csv   
    # label_columns = ['GO0001666', 'GO0002250', 'GO0006954','GO0006955','GO0006974','GO0007399','GO0007420','GO0042493','GO0045087']  #9label
    # label_columns = ['GO0001666','GO0002250','GO0004984','GO0006954','GO0006955','GO0006974','GO0006979','GO0007399','GO0007417','GO0007420','GO0009615','GO0032355','GO0032496','GO0032869','GO0042493','GO0042742','GO0045087','GO0045471','GO0051607','GO0071222','GO0071260','GO0071277','GO0071356','GO0071456','GO0150104']  #25label
   # label_columns = ['GO0002250','GO0004984','GO0006955','GO0007417','GO0007420','GO0042742','GO0045087','GO0045471','GO0051607','GO0071277','GO0071356','GO0071456','GO0150104']  #13label
    label_columns = ['GO0002250','GO0004984','GO0006955','GO0007420','GO0042742','GO0045087','GO0045471','GO0071277','GO0071456','GO0150104'] #10label
    # Call create_dataset function
    MyDataset.create_dataset(npz_file=npz_file, csv_file=csv_file,label_columns=label_columns,save_dir='/home/xiazhiliang/Project/Bioinfor-DeepATT-main/mydata/')
