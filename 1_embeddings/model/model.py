# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow import keras

from .layers.bidirection_rnn import BidLSTM, BidGRU
from .layers.multihead_attention import MultiHeadAttention
from .layers.category_dense import CategoryDense

class RecGOBD_1_embedding(keras.Model):
    def __init__(self):
        super(RecGOBD_1_embedding, self).__init__()

        self.bidirectional_rnn = BidLSTM(512)

        self.category_encoding = tf.eye(10)[tf.newaxis, :, :]

        self.multi_head_attention = MultiHeadAttention(400, 4)

        self.dropout = keras.layers.Dropout(0.2)

        self.point_wise_dense_1 = keras.layers.Dense(
            units=100,
            activation='relu')

        self.point_wise_dense_2 = keras.layers.Dense(
            units=1,
            activation='sigmoid')

    def call(self, inputs, training=None, mask=None):
        """
        Forward propagation of RecGOBD model.
        :param inputs: shape = (batch_size, length, c)
        :param training: training or not.
        :param kwargs: None
        :return: shape = (batch_size, 919)
        """
        batch_size = tf.shape(inputs)[0]
        
        temp, _ = self.bidirectional_rnn(inputs, training=training, mask=mask)

        query = tf.tile(self.category_encoding, multiples=[batch_size, 1, 1])

        temp, _ = self.multi_head_attention(query, k=temp, v=temp)
        
        temp = self.dropout(temp, training=training)
        temp = self.point_wise_dense_1(temp)

        output = self.point_wise_dense_2(temp)

        output = tf.reshape(output, [-1, 10])
        return output


class RecGOBD_2_embeddings(keras.Model):
    def __init__(self):
        super(RecGOBD_2_embeddings, self).__init__()

        self.bidirectional_rnn = BidLSTM(512)
        
        self.category_encoding = tf.eye(10)[tf.newaxis, :, :]

        self.multi_head_attention = MultiHeadAttention(400, 4)


        self.dropout = keras.layers.Dropout(0.2)

        self.point_wise_dense_1 = keras.layers.Dense(
            units=100,
            activation='relu')

        self.point_wise_dense_2 = keras.layers.Dense(
            units=1,
            activation='sigmoid')       

    def call(self, inputs, training=None, mask=None):
        """
        Forward propagation of RecGOBD model.
        :param inputs: list of 2 tensors, each with shape (batch_size, length, c)
        :param training: training or not.
        :param mask: None
        :return: shape = (batch_size, 10)
        """
        batch_size = tf.shape(inputs[0])[0]

        temp = tf.concat([inputs[0], inputs[1]], axis=-1)
        
        # print("拼接后的维度",temp.shape)    
        
        temp, _ = self.bidirectional_rnn(temp, training=training, mask=mask) 
        
        query = tf.tile(self.category_encoding, multiples=[batch_size, 1, 1])

        temp_1, _ = self.multi_head_attention(query, k=temp, v=temp)   
        
        temp_1 = self.dropout(temp_1, training=training)

        
        temp = self.point_wise_dense_1(temp_1)

        output = self.point_wise_dense_2(temp)

        output = tf.reshape(output, [-1, 10])
        return output
    
class RecGOBD_3_embeddings(keras.Model):
    def __init__(self):
        super(RecGOBD_3_embeddings, self).__init__()

        self.bidirectional_rnn = BidLSTM(512)

        self.category_encoding = tf.eye(10)[tf.newaxis, :, :]

        self.multi_head_attention = MultiHeadAttention(400, 4)

        self.dropout = keras.layers.Dropout(0.2)
        
        # Fully connected layer for the second input feature (inputs[1])
        self.point_wise_dense_0 = keras.layers.Dense(
            units=800,
            activation='relu')

        self.point_wise_dense_1 = keras.layers.Dense(
            units=100,
            activation='relu')

        self.point_wise_dense_2 = keras.layers.Dense(
            units=1,
            activation='sigmoid')

    def call(self, inputs, training=None, mask=None):
        """
        Forward propagation of RecGOBD model.
        :param inputs: list of 3 tensors, each with shape (batch_size, length, c)
        :param training: training or not.
        :param mask: None
        :return: shape = (batch_size, 10)
        """
        batch_size = tf.shape(inputs[0])[0]
        
        temp = tf.concat([inputs[0],inputs[1],inputs[2]], axis=-1)
        
        # print("拼接后的维度",temp.shape)
        temp, _ = self.bidirectional_rnn(temp, training=training, mask=mask)

        query = tf.tile(self.category_encoding, multiples=[batch_size, 1, 1])

        temp, _ = self.multi_head_attention(query, k=temp, v=temp)
        temp = self.dropout(temp, training=training)
        
        temp = self.point_wise_dense_1(temp)


        output = self.point_wise_dense_2(temp)

        output = tf.reshape(output, [-1, 10])
        return output
    
class RecGOBD_4_embeddings(keras.Model):
    def __init__(self):
        super(RecGOBD_4_embeddings, self).__init__()
        
        self.bidirectional_rnn = BidLSTM(512)

        self.category_encoding = tf.eye(10)[tf.newaxis, :, :]

        self.multi_head_attention = MultiHeadAttention(400, 4)

        self.dropout = keras.layers.Dropout(0.2)

        self.point_wise_dense_1 = keras.layers.Dense(
            units=100,
            activation='relu')

        self.point_wise_dense_2 = keras.layers.Dense(
            units=1,
            activation='sigmoid')

    def call(self, inputs, training=None, mask=None):
        """
        Forward propagation of RecGOBD model.
        :param inputs: list of 4 tensors, each with shape (batch_size, length, c)
        :param training: training or not.
        :param mask: None
        :return: shape = (batch_size, 10)
        """
        batch_size = tf.shape(inputs[0])[0]

        temp = tf.concat([inputs[0], inputs[1], inputs[2], inputs[3]], axis=-1)

        temp, _ = self.bidirectional_rnn(temp, training=training, mask=mask)

        query = tf.tile(self.category_encoding, multiples=[batch_size, 1, 1])

        temp, _ = self.multi_head_attention(query, k=temp, v=temp)
        
        temp = self.dropout(temp, training=training)

        temp = self.point_wise_dense_1(temp)

        output = self.point_wise_dense_2(temp)

        output = tf.reshape(output, [-1, 10])
        return output


