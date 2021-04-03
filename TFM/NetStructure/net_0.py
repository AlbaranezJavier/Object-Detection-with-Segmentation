from tensorflow.keras.layers import Conv2D, Concatenate, BatchNormalization, Conv2DTranspose
from tensorflow.keras.regularizers import L2
import numpy as np
from tensorflow.keras import Model
import cv2
from tensorflow import keras, random_normal_initializer, Variable, zeros_initializer
import tensorflow as tf

'''
This script contains the proposed network structures.
'''


# class Conv2d_fixed(keras.layers.Layer):
#     def __init__(self):
#         super(Conv2d_fixed, self).__init__()
#
#         sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], tf.float32)
#         sobel_y = tf.constant([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], tf.float32)
#         self.sobel_x_filter = tf.reshape(sobel_x, [1, 1, 3, 3])
#         self.sobel_y_filter = tf.reshape(sobel_y, [1, 1, 3, 3])
#
#
# def call(self, inputs):
#     # tf.nn.conv2d(   input, filters, strides, padding, data_format='NHWC', dilations=None,   name=None)
#     print("fixed")
#     filtered_x = tf.nn.conv2d(inputs, self.sobel_x_filter, strides=[1, 1, 1, 1], padding='SAME')
#     filtered_y = tf.nn.conv2d(inputs, self.sobel_y_filter, strides=[1, 1, 1, 1], padding='SAME')
#     # print("conv",convolucion.shape)
#     return filtered_x

class Net_0(Model):
    def __init__(self, learn_reg):
        super(Net_0, self).__init__()

        self.bn = BatchNormalization()
        self.cnt = Concatenate(axis=3)

        self.l_720_3 = Conv2D(filters=2, kernel_size=(3, 3), kernel_regularizer=L2(learn_reg), padding="same", activation="relu")
        self.l_720_5 = Conv2D(filters=5, kernel_size=(5, 5), kernel_regularizer=L2(learn_reg), padding="same", activation="relu")

        self.l_360_1 = Conv2D(filters=5, kernel_size=(1, 1), kernel_regularizer=L2(learn_reg), strides=(2,2), padding="valid", activation="relu")
        self.l_360_3 = Conv2D(filters=2, kernel_size=(3, 3), kernel_regularizer=L2(learn_reg), padding="same", activation="relu")
        self.l_360_5 = Conv2D(filters=5, kernel_size=(5, 5), kernel_regularizer=L2(learn_reg), padding="same", activation="relu")

        self.l_3_1 = Conv2D(filters=5, kernel_size=(1, 1), kernel_regularizer=L2(learn_reg), strides=(2, 2), padding="valid", activation="relu")
        self.l_3_3 = Conv2D(filters=2, kernel_size=(3, 3), kernel_regularizer=L2(learn_reg), padding="same", activation="relu")
        self.l_3_5 = Conv2D(filters=5, kernel_size=(5, 5), kernel_regularizer=L2(learn_reg), padding="same", activation="relu")

        self.l_2_5 = Conv2D(filters=32, kernel_size=(5, 5), kernel_regularizer=L2(learn_reg), strides=(2, 2), padding="valid", activation="relu")
        self.l_2_3 = Conv2D(filters=64, kernel_size=(3, 3), kernel_regularizer=L2(learn_reg), padding="same", activation="relu")

        self.m_1_5 = Conv2D(filters=64, kernel_size=(5, 5), kernel_regularizer=L2(learn_reg), strides=(2, 2), padding="valid", activation="relu")
        self.m_1_3 = Conv2D(filters=128, kernel_size=(3, 3), kernel_regularizer=L2(learn_reg), padding="same", activation="relu")

        self.r_2_5 = Conv2DTranspose(filters=64, kernel_size=(5, 5), kernel_regularizer=L2(learn_reg), strides=(2, 2), padding="valid", activation="relu")
        self.r_2_3 = Conv2D(filters=32, kernel_size=(3, 3), kernel_regularizer=L2(learn_reg), padding="same", activation="relu")
        self.r_2_55 = Conv2DTranspose(filters=32, kernel_size=(5, 5), kernel_regularizer=L2(learn_reg), strides=(2, 2), padding="valid", activation="relu")

        self.r_3_1 = Conv2D(filters=16, kernel_size=(1, 1), kernel_regularizer=L2(learn_reg), padding="same", activation="relu")
        self.r_3_11 = Conv2DTranspose(filters=5, kernel_size=(1, 1), kernel_regularizer=L2(learn_reg), strides=(2, 2), padding="valid", activation="relu")

        self.r_360_1 = Conv2D(filters=5, kernel_size=(1, 1), kernel_regularizer=L2(learn_reg), padding="same", activation="relu")
        self.r_360_11 = Conv2D(filters=5, kernel_size=(1, 1), kernel_regularizer=L2(learn_reg), padding="same", activation="softmax")
        self.r_360_111 = Conv2DTranspose(filters=5, kernel_size=(1, 1), kernel_regularizer=L2(learn_reg), strides=(2, 2),
                                      padding="valid", activation="relu")

        self.r_720_1 = Conv2D(filters=5, kernel_size=(1, 1), kernel_regularizer=L2(learn_reg), padding="same", activation="relu")
        self.r_720_11 = Conv2D(filters=5, kernel_size=(1, 1), kernel_regularizer=L2(learn_reg), padding="same", activation="softmax")

    def call(self, input_720):
        # Inputs
        _input_360 = tf.image.resize(input_720, [640, 360], method=tf.image.ResizeMethod.BICUBIC)
        _input_180 = tf.image.resize(input_720, [320, 180], method=tf.image.ResizeMethod.BICUBIC)
        # L 720
        _temp1 = self.l_720_3(input_720)
        _temp2 = self.l_720_5(input_720)
        _temp3 = self.cnt([_temp1, _temp2])
        _l_720_bn = self.bn(_temp3)

        # L 360
        _temp1 = self.l_360_1(_l_720_bn)
        _temp2 = self.l_360_3(_input_360)
        _temp3 = self.l_360_5(_input_360)
        _temp4 = self.cnt([_temp1, _temp2, _temp3])
        _l_360_bn = self.bn(_temp4)

        # L 3
        _temp1 = self.l_180_1(_l_360_bn)
        _temp2 = self.l_180_3(_input_180)
        _temp3 = self.l_180_5(_input_180)
        _temp4 = self.cnt([_temp1, _temp2, _temp3])
        _l_3_bn = self.bn(_temp4)

        # L 2
        _temp1 = self.l_2_5(_l_3_bn)
        _temp2 = self.l_2_3(_temp1)
        _l_2_bn = self.bn(_temp2)

        # 1
        _temp1 = self.m_1_5(_l_2_bn)
        _temp2 = self.m_1_3(_temp1)
        _m_1_bn = self.bn(_temp2)

        # R 2
        _temp1 = self.cnt([_l_2_bn, _m_1_bn])
        _temp2 = self.r_2_5(_temp1)
        _temp3 = self.r_2_3(_temp2)
        _temp4 = self.r_2_55(_temp3)
        _r_2_bn = self.bn(_temp4)

        # R 3
        _temp1 = self.cnt([_l_3_bn, _r_2_bn])
        _temp2 = self.r_3_1(_temp1)
        _temp3 = self.r_3_11(_temp2)
        _r_3_bn = self.bn(_temp3)

        # R 360
        _temp1 = self.cnt([_l_360_bn, _r_3_bn])
        _temp2 = self.r_360_1(_temp1)
        _temp3 = self.bn(_temp2)
        output_360 = self.r_360_11(_temp3)
        _temp4 = self.r_360_111(_temp3)
        _r_360_bn = self.bn(_temp4)

        # R 720
        _temp1 = self.cnt([_l_720_bn, _r_360_bn])
        _temp2 = self.r_720_1(_temp1)
        _temp3 = self.bn(_temp2)
        output_720 = self.r_720_11(_temp3)

        return output_360, output_720
