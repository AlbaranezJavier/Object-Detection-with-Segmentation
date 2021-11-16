from tensorflow.keras.layers import Conv2D, concatenate, BatchNormalization, Conv2DTranspose, Flatten, Dense
from tensorflow.keras.regularizers import L2, L1
import numpy as np
from tensorflow.keras import Model, Input
import cv2, glob
from tensorflow import keras, random_normal_initializer, Variable, zeros_initializer
import tensorflow as tf


'''
This script contains the proposed network structures.
'''
class Filters:
    def __init__(self, type, out_channels):
        if type == "border":
            filter = np.array([[[[0]], [[-1]], [[0]]],
                                    [[[-1]], [[4]], [[-1]]],
                                    [[[0]], [[-1]], [[0]]]])
            self.strides = [1, 1, 1, 1]
            self.pad = "SAME"
        elif type == "bilinear":
            filter = np.array([[[[.25]], [[.5]], [[.25]]],
                                    [[[.5]], [[1]], [[.5]]],
                                    [[[.25]], [[.5]], [[.25]]]])
            self.strides = [1, 2, 2, 1]
            self.pad = [[0, 0], [1, 1], [1, 1], [0, 0]]
        elif type == "linear":
            filter = np.array([[[[0]], [[.5]], [[.5]]],
                                    [[[.5]], [[1]], [[.5]]],
                                    [[[.5]], [[.5]], [[0]]]])
            self.strides = [1, 2, 2, 1]
            self.pad = [[0, 0], [1, 1], [1, 1], [0, 0]]
        elif type == "average":
            filter = np.array([[[[1 / 9]], [[1 / 9]], [[1 / 9]]],
                                    [[[1 / 9]], [[1 / 9]], [[1 / 9]]],
                                    [[[1 / 9]], [[1 / 9]], [[1 / 9]]]])
            self.strides = [1, 2, 2, 1]
            self.pad = [[0, 0], [1, 1], [1, 1], [0, 0]]
        else:
            raise Exception("In Conv2d_fixed select: border, bilinear, linear or average.")

        self.filter = np.repeat(filter.astype(np.float32), [out_channels], axis=3)

class Conv2DFixed(keras.layers.Layer, Filters):
    def __init__(self, kernel_type, out_channels):
        super(Conv2DFixed, self).__init__()
        Filters.__init__(self, kernel_type, out_channels)
        self.w = tf.Variable(initial_value=tf.constant(self.filter, dtype=tf.float32), trainable=False)
        self.bn = BatchNormalization()

    def call(self, inputs):
        channels = tf.nn.conv2d(inputs, self.w, strides=self.strides, padding=self.pad)
        norm = self.bn(channels)
        return norm

class Conv2DFixed_Transpose(keras.layers.Layer, Filters):
    def __init__(self, kernel_type, out_shape):
        super(Conv2DFixed_Transpose, self).__init__()
        Filters.__init__(self, kernel_type, out_shape[3])
        self.out_shape = out_shape
        self.w = tf.Variable(initial_value=tf.constant(self.filter, dtype=tf.float32), trainable=False)
        self.bn = BatchNormalization()

    def call(self, inputs):
        channels = tf.nn.conv2d_transpose(inputs, self.w, output_shape=self.out_shape, strides=self.strides,
                                         padding=self.pad)
        norm = self.bn(channels)
        return norm

class Conv2D_NA(keras.layers.Layer):
    def __init__(self, k_dim, output_channel, stride, padding='VALID', k_reg=None):
        super(Conv2D_NA, self).__init__()
        self.stride = stride
        self.padding = padding

        self.conv = Conv2D(filters=output_channel, kernel_size=(k_dim, k_dim), strides=(stride, stride),
                           padding=padding, kernel_regularizer=k_reg, use_bias=False)
        self.bn = BatchNormalization()

    def call(self, inputs):
        channels = self.conv(inputs)
        norm = self.bn(channels)
        return tf.nn.relu(norm + self.conv.bias)

class Net_test(Model):
    def __init__(self):
        super(Net_test, self).__init__()
        self.l_720 = Conv2DFixed("border", 3)
        # self.l_720 = Conv2DFixed("bilinear", 3)
        # self.d_720 = Conv2DFixed_Transpose("bilinear", [1, 720, 1280, 3])
        # self.l_720 = Conv2DFixed("linear", 3)
        # self.l_720 = Conv2DFixed("average", 3)

    def call(self, inputs):
        x = self.l_720(inputs)
        # x = self.d_720(x)
        return x

def CNet_5L0(inputs, batch, learn_reg=1e-2):
    # Variables
    l2 = L2(learn_reg)

    # Left side = Li, Mid = M y right side = Ld
    # - Level 1, Li
    n1Li = Conv2D_NA(k_dim=3, output_channel=2, stride=1, padding="SAME", k_reg=l2)(inputs)

    # - Level 2, Li
    n2Li = Conv2DFixed("bilinear", out_channels=2)(n1Li)
    n2Li = Conv2D_NA(k_dim=3, output_channel=4, stride=1, padding="SAME", k_reg=l2)(n2Li)

    # - Level 3, Li
    n3Li = Conv2DFixed("bilinear", out_channels=4)(n2Li)
    n3Li = Conv2D_NA(k_dim=3, output_channel=8, stride=1, padding="SAME", k_reg=l2)(n3Li)

    # - Level 4, Li
    n4Li = Conv2DFixed("bilinear", out_channels=8)(n3Li)
    n4Li = Conv2D_NA(k_dim=5, output_channel=64, stride=1, padding="SAME", k_reg=l2)(n4Li)

    # - Level 5, M
    n5M = Conv2D_NA(k_dim=5, output_channel=128, stride=1, padding="SAME", k_reg=l2)(n4Li)

    # - Level 4, Ld
    n4Ld = concatenate([n4Li, n5M], axis=3)
    n4Ld = Conv2D_NA(k_dim=5, output_channel=64, stride=1, padding="SAME", k_reg=l2)(n4Ld)
    n4Ld = Conv2DFixed_Transpose("bilinear", [batch, 180, 320, 64])(n4Ld)

    # - Level 3, Ld
    n3Ld = concatenate([n3Li, n4Ld])
    n3Ld = Conv2D_NA(k_dim=3, output_channel=5, stride=1, padding="SAME", k_reg=l2)(n3Ld)
    n3Ld = Conv2DFixed_Transpose("bilinear", [batch, 360, 640, 5])(n3Ld)

    # - Level 2, Ld
    n2Ld = concatenate([n2Li, n3Ld])
    n2Ld = Conv2D_NA(k_dim=3, output_channel=5, stride=1, padding="SAME", k_reg=l2)(n2Ld)
    n2Ld = Conv2DFixed_Transpose("bilinear", [batch, 720, 1280, 5])(n2Ld)

    # - Level 1, Ld
    n1Ld = concatenate([n1Li, n2Ld])
    n1Ld = Conv2D(filters=5, kernel_size=(1, 1), kernel_regularizer=l2, padding="SAME", activation="softmax")(n1Ld)
    return n1Ld

def CNet_5L1(inputs, batch, learn_reg=1e-2):
    # Variables
    l2 = L2(learn_reg)

    # Left side = Li, Mid = M y right side = Ld
    # - Level 1, Li
    n1Li = Conv2D_NA(k_dim=3, output_channel=2, stride=1, padding="SAME", k_reg=l2)(inputs)

    # - Level 2, Li
    n2Li = Conv2DFixed("bilinear", out_channels=2)(n1Li)
    n2Li = Conv2D_NA(k_dim=3, output_channel=4, stride=1, padding="SAME", k_reg=l2)(n2Li)

    # - Level 3, Li
    n3Li = Conv2DFixed("bilinear", out_channels=4)(n2Li)
    n3Li = Conv2D_NA(k_dim=3, output_channel=8, stride=1, padding="SAME", k_reg=l2)(n3Li)

    # - Level 4, Li
    n4Li = Conv2DFixed("bilinear", out_channels=8)(n3Li)
    n4Li = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n4Li)

    # - Level 5, Li
    n5Li = Conv2DFixed("bilinear", out_channels=32)(n4Li)
    n5Li = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n5Li)

    # - Level 6, M
    n6M = Conv2D_NA(k_dim=5, output_channel=64, stride=1, padding="SAME", k_reg=l2)(n5Li)

    # - Level 5, Ld
    n5Ld = concatenate([n5Li, n6M], axis=3)
    n5Ld = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n5Ld)
    n5Ld = Conv2DFixed_Transpose("bilinear", [batch, 90, 160, 32])(n5Ld)

    # - Level 4, Ld
    n4Ld = concatenate([n4Li, n5Ld], axis=3)
    n4Ld = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n4Ld)
    n4Ld = Conv2DFixed_Transpose("bilinear", [batch, 180, 320, 32])(n4Ld)

    # - Level 3, Ld
    n3Ld = concatenate([n3Li, n4Ld])
    n3Ld = Conv2D_NA(k_dim=3, output_channel=5, stride=1, padding="SAME", k_reg=l2)(n3Ld)
    n3Ld = Conv2DFixed_Transpose("bilinear", [batch, 360, 640, 5])(n3Ld)

    # - Level 2, Ld
    n2Ld = concatenate([n2Li, n3Ld])
    n2Ld = Conv2D_NA(k_dim=3, output_channel=5, stride=1, padding="SAME", k_reg=l2)(n2Ld)
    n2Ld = Conv2DFixed_Transpose("bilinear", [batch, 720, 1280, 5])(n2Ld)

    # - Level 1, Ld
    n1Ld = concatenate([n1Li, n2Ld])
    n1Ld = Conv2D(filters=5, kernel_size=(1, 1), kernel_regularizer=l2, padding="SAME", activation="softmax")(n1Ld)
    return n1Ld

def CNet_5L2(inputs, batch, learn_reg=1e-2):
    # Variables
    l2 = L2(learn_reg)

    # Left side = Li, Mid = M y right side = Ld
    # - Level 1, Li
    fixed_n1Li = Conv2DFixed("border", out_channels=3)(inputs)
    input_fixed = concatenate([fixed_n1Li, inputs], axis=3)

    n1Li = Conv2D_NA(k_dim=3, output_channel=8, stride=1, padding="SAME", k_reg=l2)(input_fixed)

    # - Level 2, Li
    n2Li = Conv2DFixed("bilinear", out_channels=8)(n1Li)
    inputs_n2Li = tf.image.resize(input_fixed, [360, 640], antialias=False)
    n2Li = concatenate([n2Li, inputs_n2Li], axis=3)

    n2Li = Conv2D_NA(k_dim=3, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n2Li)

    # - Level 3, Li
    n3Li = Conv2DFixed("bilinear", out_channels=32)(n2Li)
    inputs_n3Li = tf.image.resize(input_fixed, [180, 320], antialias=False)
    n3Li = concatenate([n3Li, inputs_n3Li], axis=3)

    n3Li = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n3Li)

    # - Level 4, Li
    n4Li = Conv2DFixed("bilinear", out_channels=32)(n3Li)
    inputs_n4Li = tf.image.resize(input_fixed, [90, 160], antialias=False)
    n4Li = concatenate([n4Li, inputs_n4Li], axis=3)

    n4Li = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n4Li)

    # - Level 5, Li
    n5Li = Conv2DFixed("bilinear", out_channels=32)(n4Li)
    inputs_n5Li = tf.image.resize(input_fixed, [45, 80], antialias=False)
    n5Li = concatenate([n5Li, inputs_n5Li], axis=3)

    n5Li = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n5Li)
    n5M = Conv2D_NA(k_dim=5, output_channel=64, stride=1, padding="SAME", k_reg=l2)(n5Li)

    # - Level 5, Ld
    n5Ld = concatenate([n5Li, n5M], axis=3)
    n5Ld = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n5Ld)
    n5Ld = Conv2DFixed_Transpose("bilinear", [batch, 90, 160, 32])(n5Ld)

    # - Level 4, Ld
    n4Ld = concatenate([n4Li, n5Ld], axis=3)
    n4Ld = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n4Ld)
    n4Ld = Conv2DFixed_Transpose("bilinear", [batch, 180, 320, 32])(n4Ld)

    # - Level 3, Ld
    n3Ld = concatenate([n3Li, n4Ld], axis=3)
    n3Ld = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n3Ld)
    n3Ld = Conv2DFixed_Transpose("bilinear", [batch, 360, 640, 32])(n3Ld)

    # - Level 2, Ld
    n2Ld = concatenate([n2Li, n3Ld], axis=3)
    n2Ld = Conv2D_NA(k_dim=3, output_channel=8, stride=1, padding="SAME", k_reg=l2)(n2Ld)
    n2Ld = Conv2DFixed_Transpose("bilinear", [batch, 720, 1280, 8])(n2Ld)

    # - Level 1, Ld
    n1Ld = concatenate([n1Li, n2Ld])
    n1Ld = Conv2D_NA(k_dim=3, output_channel=5, stride=1, padding="SAME", k_reg=l2)(n1Ld)
    n1Ld = Conv2D(filters=2, kernel_size=(1, 1), kernel_regularizer=l2, padding="SAME", activation="softmax")(n1Ld)

    return n1Ld

def CNet_5L3(inputs, batch, learn_reg=1e-2):
    # Variables
    l2 = L2(learn_reg)

    # Left side = Li, Mid = M y right side = Ld
    # - Level 1, Li
    fixed_n1Li = Conv2DFixed("border", out_channels=3)(inputs)
    input_fixed = concatenate([fixed_n1Li, inputs], axis=3)

    n1Li = Conv2D_NA(k_dim=3, output_channel=8, stride=1, padding="SAME", k_reg=l2)(input_fixed)

    # - Level 2, Li
    n2Li = Conv2DFixed("bilinear", out_channels=8)(n1Li)
    inputs_n2Li = tf.image.resize(input_fixed, [360, 640], antialias=False)
    n2Li = concatenate([n2Li, inputs_n2Li], axis=3)

    n2Li = Conv2D_NA(k_dim=3, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n2Li)

    # - Level 3, Li
    n3Li = Conv2DFixed("bilinear", out_channels=32)(n2Li)
    inputs_n3Li = tf.image.resize(input_fixed, [180, 320], antialias=False)
    n3Li = concatenate([n3Li, inputs_n3Li], axis=3)

    n3Li = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n3Li)

    # - Level 4, Li
    n4Li = Conv2DFixed("bilinear", out_channels=32)(n3Li)
    inputs_n4Li = tf.image.resize(input_fixed, [90, 160], antialias=False)
    n4Li = concatenate([n4Li, inputs_n4Li], axis=3)

    n4Li = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n4Li)

    # - Level 5, Li
    n5Li = Conv2DFixed("bilinear", out_channels=32)(n4Li)
    inputs_n5Li = tf.image.resize(input_fixed, [45, 80], antialias=False)
    n5Li = concatenate([n5Li, inputs_n5Li], axis=3)

    n5Li = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n5Li)
    n5M = Conv2D_NA(k_dim=5, output_channel=64, stride=1, padding="SAME", k_reg=l2)(n5Li)

    # - Level 5, Ld
    n5Ld = concatenate([n5Li, n5M], axis=3)
    n5Ld = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n5Ld)
    n5Ld = Conv2DFixed_Transpose("bilinear", [batch, 90, 160, 32])(n5Ld)

    # - Level 4, Ld
    n4Ld = concatenate([n4Li, n5Ld], axis=3)
    n4Ld = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n4Ld)
    n4Ld = Conv2DFixed_Transpose("bilinear", [batch, 180, 320, 32])(n4Ld)

    # - Level 3, Ld
    n3Ld = concatenate([n3Li, n4Ld], axis=3)
    n3Ld = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n3Ld)
    n3Ld = Conv2DFixed_Transpose("bilinear", [batch, 360, 640, 32])(n3Ld)

    # - Level 2, Ld
    n2Ld = concatenate([n2Li, n3Ld], axis=3)
    n2Ld = Conv2D_NA(k_dim=3, output_channel=8, stride=1, padding="SAME", k_reg=l2)(n2Ld)
    n2Ld = Conv2DFixed_Transpose("bilinear", [batch, 720, 1280, 8])(n2Ld)

    # - Level 1, Ld
    n1Ld = concatenate([n1Li, n2Ld])
    n1Ld = Conv2D_NA(k_dim=3, output_channel=5, stride=1, padding="SAME", k_reg=l2)(n1Ld)
    n1Ld = Conv2D(filters=2, kernel_size=(1, 1), kernel_regularizer=l2, padding="SAME", activation="softmax")(n1Ld)

    return n1Ld

def CNet_4L(inputs, batch, learn_reg=1e-2):
    # Variables
    l2 = L2(learn_reg)

    # Left side = Li, Mid = M y right side = Ld
    # - Level 2, Li
    fixed_n2Li = Conv2DFixed("border", out_channels=3)(inputs)
    input_fixed = concatenate([fixed_n2Li, inputs], axis=3)

    n2Li = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(input_fixed)

    # - Level 3, Li
    n3Li = Conv2DFixed("bilinear", out_channels=32)(n2Li)
    inputs_n3Li = tf.image.resize(input_fixed, [180, 320], antialias=False)
    n3Li = concatenate([n3Li, inputs_n3Li], axis=3)

    n3Li = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n3Li)

    # - Level 4, Li
    n4Li = Conv2DFixed("bilinear", out_channels=32)(n3Li)
    inputs_n4Li = tf.image.resize(input_fixed, [90, 160], antialias=False)
    n4Li = concatenate([n4Li, inputs_n4Li], axis=3)

    n4Li = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n4Li)

    # - Level 5, Li
    n5Li = Conv2DFixed("bilinear", out_channels=32)(n4Li)
    inputs_n5Li = tf.image.resize(input_fixed, [45, 80], antialias=False)
    n5Li = concatenate([n5Li, inputs_n5Li], axis=3)

    n5Li = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n5Li)

    # - Level 6, M
    n6M = Conv2D_NA(k_dim=5, output_channel=64, stride=1, padding="SAME", k_reg=l2)(n5Li)

    # - Level 5, Ld
    n5Ld = concatenate([n5Li, n6M], axis=3)
    n5Ld = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n5Ld)
    n5Ld = Conv2DFixed_Transpose("bilinear", [batch, 90, 160, 32])(n5Ld)

    # - Level 4, Ld
    n4Ld = concatenate([n4Li, n5Ld], axis=3)
    n4Ld = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n4Ld)
    n4Ld = Conv2DFixed_Transpose("bilinear", [batch, 180, 320, 32])(n4Ld)

    # - Level 3, Ld
    n3Ld = concatenate([n3Li, n4Ld], axis=3)
    n3Ld = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n3Ld)
    n3Ld = Conv2DFixed_Transpose("bilinear", [batch, 360, 640, 32])(n3Ld)

    # - Level 2, Ld
    n2Ld = concatenate([n2Li, n3Ld])
    n2Ld = Conv2D_NA(k_dim=5, output_channel=5, stride=1, padding="SAME", k_reg=l2)(n2Ld)
    n2Ld = Conv2D(filters=2, kernel_size=(1, 1), kernel_regularizer=l2, padding="SAME", activation="softmax")(n2Ld)

    return n2Ld

def CNet_3L(inputs, learn_reg=1e-2):
    # Variables
    l2 = L2(learn_reg)

    # Left side = Li, Mid = M y right side = Ld
    # - Level 1, Li
    fixed_n1Li = Conv2DFixed("border", out_channels=3)(inputs)
    input_fixed = concatenate([fixed_n1Li, inputs], axis=3)

    n1Li = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(input_fixed)

    # - Level 2, Li
    n2Li = Conv2DFixed("bilinear", out_channels=32)(n1Li)
    inputs_n2Li = tf.image.resize(input_fixed, [256, 256], antialias=False)
    n2Li = concatenate([n2Li, inputs_n2Li], axis=3)

    n2Li = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n2Li)

    # - Level 3, Li
    n3Li = Conv2DFixed("bilinear", out_channels=32)(n2Li)
    inputs_n3Li = tf.image.resize(input_fixed, [128, 128], antialias=False)
    n3Li = concatenate([n3Li, inputs_n3Li], axis=3)

    n3Li = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n3Li)

    # - Level 4, Li
    n4Li = Conv2DFixed("bilinear", out_channels=32)(n3Li)
    inputs_n4Li = tf.image.resize(input_fixed, [64, 64], antialias=False)
    n4Li = concatenate([n4Li, inputs_n4Li], axis=3)

    n4Li = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n4Li)

    # - Level 5, Li
    n5Li = Conv2DFixed("bilinear", out_channels=32)(n4Li)
    inputs_n5Li = tf.image.resize(input_fixed, [32, 32], antialias=False)
    n5Li = concatenate([n5Li, inputs_n5Li], axis=3)

    n5Li = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n5Li)

    # - Level 6, Li
    n6Li = Conv2DFixed("bilinear", out_channels=32)(n5Li)
    inputs_n6Li = tf.image.resize(input_fixed, [16, 16], antialias=False)
    n6Li = concatenate([n6Li, inputs_n6Li], axis=3)

    n6Li = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n6Li)
    n6M = Conv2D_NA(k_dim=5, output_channel=64, stride=1, padding="SAME", k_reg=l2)(n6Li)

    # - Level 7, Ld
    n7 = concatenate([n6Li, n6M], axis=3)
    n7_1 = Conv2D_NA(k_dim=3, output_channel=128, stride=2, padding="VALID", k_reg=l2)(n7)
    n7_2 = Conv2D_NA(k_dim=3, output_channel=256, stride=1, padding="VALID", k_reg=l2)(n7_1)
    n7_3 = Conv2D_NA(k_dim=3, output_channel=128, stride=1, padding="VALID", k_reg=l2)(n7_2)
    n7_4 = Conv2D(filters=6, kernel_size=(3, 3), kernel_regularizer=l2, strides=(1, 1), padding="VALID",
                  activation=None, use_bias=False)(n7_3)
    n7_4 = BatchNormalization()(n7_4)
    n7_4 = tf.keras.activations.softmax(n7_4)


    return n7_4

def CDNet_3L(inputs, learn_reg=1e-2):
    # Variables
    l2 = L2(learn_reg)

    # Left side = Li, Mid = M y right side = Ld
    # - Level 1, Li
    fixed_n1Li = Conv2DFixed("border", out_channels=3)(inputs)
    input_fixed = concatenate([fixed_n1Li, inputs], axis=3)

    n1Li = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(input_fixed)

    # - Level 2, Li
    n2Li = Conv2DFixed("bilinear", out_channels=32)(n1Li)
    inputs_n2Li = tf.image.resize(input_fixed, [256, 256], antialias=False)
    n2Li = concatenate([n2Li, inputs_n2Li], axis=3)

    n2Li = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n2Li)

    # - Level 3, Li
    n3Li = Conv2DFixed("bilinear", out_channels=32)(n2Li)
    inputs_n3Li = tf.image.resize(input_fixed, [128, 128], antialias=False)
    n3Li = concatenate([n3Li, inputs_n3Li], axis=3)

    n3Li = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n3Li)

    # - Level 4, Li
    n4Li = Conv2DFixed("bilinear", out_channels=32)(n3Li)
    inputs_n4Li = tf.image.resize(input_fixed, [64, 64], antialias=False)
    n4Li = concatenate([n4Li, inputs_n4Li], axis=3)

    n4Li = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n4Li)

    # - Level 5, Li
    n5Li = Conv2DFixed("bilinear", out_channels=32)(n4Li)
    inputs_n5Li = tf.image.resize(input_fixed, [32, 32], antialias=False)
    n5Li = concatenate([n5Li, inputs_n5Li], axis=3)

    n5Li = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n5Li)

    # - Level 6, Li
    n6Li = Conv2DFixed("bilinear", out_channels=32)(n5Li)
    inputs_n6Li = tf.image.resize(input_fixed, [16, 16], antialias=False)
    n6Li = concatenate([n6Li, inputs_n6Li], axis=3)

    n6Li = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n6Li)
    n6M = Conv2D_NA(k_dim=5, output_channel=64, stride=1, padding="SAME", k_reg=l2)(n6Li)

    # - Level 7, Ld
    n7 = concatenate([n6Li, n6M], axis=3)
    n7 = Flatten()(n7)
    n7_1 = Dense(units=24, activation=None, use_bias=False)(n7)
    n7_1 = BatchNormalization()(n7_1)
    n7_1 = tf.keras.activations.relu(n7_1)
    n7_2 = Dense(units=16, activation=None, use_bias=False)(n7_1)
    n7_2 = BatchNormalization()(n7_2)
    n7_2 = tf.keras.activations.relu(n7_2)
    n7_3 = Dense(units=8, activation=None, use_bias=False)(n7_2)
    n7_3 = BatchNormalization()(n7_3)
    n7_3 = tf.keras.activations.relu(n7_3)
    n7_4 = Dense(units=6, activation=None, use_bias=False)(n7_3)
    n7_4 = BatchNormalization()(n7_4)
    n7_4 = tf.keras.activations.softmax(n7_4)

    return n7_4


def CNet_3L_plusplus(inputs, batch, output_type, learn_reg=1e-2):
    # Variables
    l2 = L2(learn_reg)

    # Left side = Li, Mid = M y right side = Ld
    # - Level 3, Li
    fixed_n3Li = Conv2DFixed("border", out_channels=3)(inputs)
    input_fixed = concatenate([fixed_n3Li, inputs], axis=3)

    n3Li = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(input_fixed)

    # - Level 4, Li
    n4Li = Conv2DFixed("bilinear", out_channels=32)(n3Li)
    inputs_n4Li = tf.image.resize(input_fixed, [90, 160], antialias=False)
    n4Li = concatenate([n4Li, inputs_n4Li], axis=3)

    n4Li = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n4Li)

    # - Level 3-4
    n34 = Conv2DFixed_Transpose("bilinear", [batch, 180, 320, 32])(n4Li)
    n34 = concatenate([n3Li, n34, input_fixed], axis=3)
    n34 = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n34)

    # - Level 5, Li
    n5Li = Conv2DFixed("bilinear", out_channels=32)(n4Li)
    inputs_n5Li = tf.image.resize(input_fixed, [45, 80], antialias=False)
    n5Li = concatenate([n5Li, inputs_n5Li], axis=3)

    n5Li = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n5Li)

    # - Level 6, M
    n6M = Conv2D_NA(k_dim=5, output_channel=64, stride=1, padding="SAME", k_reg=l2)(n5Li)

    # - Level 5, Ld
    n5Ld = concatenate([n5Li, n6M], axis=3)
    n5Ld = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n5Ld)
    n5Ld = Conv2DFixed_Transpose("bilinear", [batch, 90, 160, 32])(n5Ld)

    # - Level 4, Ld
    n4Ld = concatenate([n4Li, n5Ld], axis=3)
    n4Ld = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n4Ld)
    n4Ld = Conv2DFixed_Transpose("bilinear", [batch, 180, 320, 32])(n4Ld)

    # - Level 3, Ld
    n3Ld = concatenate([n3Li, n4Ld, n34, input_fixed])
    if output_type == "reg":
        n3Ld = Conv2D(filters=4, kernel_size=(1, 1), kernel_regularizer=l2, padding="SAME", activation="relu")(n3Ld)
    elif output_type == "reg+cls":
        n3Ld_reg = Conv2D(filters=4, kernel_size=(1, 1), kernel_regularizer=l2, padding="SAME", activation="relu")(n3Ld)
        n3Ld_cls = Conv2D(filters=1, kernel_size=(1, 1), kernel_regularizer=l2, padding="SAME", activation="softmax")(
            n3Ld)
        return n3Ld_reg, n3Ld_cls
    else:
        n3Ld = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n3Ld)
        n3Ld = Conv2D(filters=5, kernel_size=(1, 1), kernel_regularizer=l2, padding="SAME", activation="softmax")(n3Ld)
    return n3Ld

def CNet_3L_p(inputs, batch, output_type, learn_reg=1e-2):
    # Variables
    l2 = L2(learn_reg)

    # Left side = Li, Mid = M y right side = Ld
    # - Level 3, Li
    fixed_n3Li = Conv2DFixed("border", out_channels=3)(inputs)
    input_fixed = concatenate([fixed_n3Li, inputs], axis=3)

    n3Li = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(input_fixed)

    # - Level 4, Li
    n4Li = Conv2DFixed("bilinear", out_channels=32)(n3Li)
    inputs_n4Li = tf.image.resize(input_fixed, [90, 160], antialias=False)
    n4Li = concatenate([n4Li, inputs_n4Li], axis=3)

    n4Li = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n4Li)

    # - Level 3-4
    n34 = Conv2DFixed_Transpose("bilinear", [batch, 180, 320, 32])(n4Li)
    n34 = concatenate([n3Li, n34], axis=3)
    n34 = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n34)

    # - Level 5, Li
    n5Li = Conv2DFixed("bilinear", out_channels=32)(n4Li)
    inputs_n5Li = tf.image.resize(input_fixed, [45, 80], antialias=False)
    n5Li = concatenate([n5Li, inputs_n5Li], axis=3)

    n5Li = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n5Li)

    # - Level 6, M
    n6M = Conv2D_NA(k_dim=5, output_channel=64, stride=1, padding="SAME", k_reg=l2)(n5Li)

    # - Level 5, Ld
    n5Ld = concatenate([n5Li, n6M], axis=3)
    n5Ld = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n5Ld)
    n5Ld = Conv2DFixed_Transpose("bilinear", [batch, 90, 160, 32])(n5Ld)

    # - Level 4, Ld
    n4Ld = concatenate([n4Li, n5Ld], axis=3)
    n4Ld = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n4Ld)
    n4Ld = Conv2DFixed_Transpose("bilinear", [batch, 180, 320, 32])(n4Ld)

    # - Level 3, Ld
    n3Ld = concatenate([n3Li, n4Ld, n34])
    if output_type == "reg":
        n3Ld = Conv2D(filters=4, kernel_size=(1, 1), kernel_regularizer=l2, padding="SAME", activation="relu")(n3Ld)
    elif output_type == "reg+cls":
        n3Ld_reg = Conv2D(filters=4, kernel_size=(1, 1), kernel_regularizer=l2, padding="SAME", activation="relu")(n3Ld)
        n3Ld_cls = Conv2D(filters=1, kernel_size=(1, 1), kernel_regularizer=l2, padding="SAME", activation="softmax")(
            n3Ld)
        return n3Ld_reg, n3Ld_cls
    else:
        n3Ld = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n3Ld)
        n3Ld = Conv2D(filters=5, kernel_size=(1, 1), kernel_regularizer=l2, padding="SAME", activation="softmax")(n3Ld)
    return n3Ld

def CNet_3Lite(inputs, batch, output_type, learn_reg=1e-2):
    # Variables
    l2 = L2(learn_reg)

    # Left side = Li, Mid = M y right side = Ld
    # - Level 3, Li
    fixed_n3Li = Conv2DFixed("border", out_channels=3)(inputs)
    input_fixed = concatenate([fixed_n3Li, inputs], axis=3)

    n3Li = Conv2D_NA(k_dim=5, output_channel=8, stride=1, padding="SAME", k_reg=l2)(input_fixed)

    # - Level 4, Li
    n4Li = Conv2DFixed("bilinear", out_channels=8)(n3Li)
    inputs_n4Li = tf.image.resize(input_fixed, [90, 160], antialias=False)
    n4Li = concatenate([n4Li, inputs_n4Li], axis=3)

    n4Li = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n4Li)

    # - Level 5, Li
    n5Li = Conv2DFixed("bilinear", out_channels=32)(n4Li)
    inputs_n5Li = tf.image.resize(input_fixed, [45, 80], antialias=False)
    n5Li = concatenate([n5Li, inputs_n5Li], axis=3)

    n5Li = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n5Li)

    # - Level 6, M
    n6M = Conv2D_NA(k_dim=5, output_channel=64, stride=1, padding="SAME", k_reg=l2)(n5Li)

    # - Level 5, Ld
    n5Ld = concatenate([n5Li, n6M], axis=3)
    n5Ld = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n5Ld)
    n5Ld = Conv2DFixed_Transpose("bilinear", [batch, 90, 160, 32])(n5Ld)

    # - Level 4, Ld
    n4Ld = concatenate([n4Li, n5Ld], axis=3)
    n4Ld = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n4Ld)
    n4Ld = Conv2DFixed_Transpose("bilinear", [batch, 180, 320, 32])(n4Ld)

    # - Level 3, Ld
    n3Ld = concatenate([n3Li, n4Ld])
    if output_type == "reg":
        n3Ld = Conv2D(filters=4, kernel_size=(1, 1), kernel_regularizer=l2, padding="SAME", activation="relu")(n3Ld)
    elif output_type == "reg+cls":
        n3Ld_reg = Conv2D(filters=4, kernel_size=(1, 1), kernel_regularizer=l2, padding="SAME", activation="relu")(n3Ld)
        n3Ld_cls = Conv2D(filters=1, kernel_size=(1, 1), kernel_regularizer=l2, padding="SAME", activation="softmax")(
            n3Ld)
        return n3Ld_reg, n3Ld_cls
    else:
        n3Ld = Conv2D_NA(k_dim=5, output_channel=5, stride=1, padding="SAME", k_reg=l2)(n3Ld)
        n3Ld = Conv2D(filters=2, kernel_size=(1, 1), kernel_regularizer=l2, padding="SAME", activation="softmax")(n3Ld)
    return n3Ld

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    path_img = r"C:\Users\TTe_J\Downloads\MIT_DATASET\Images\vid_2_frame_1708.jpg"
    y_hat = cv2.cvtColor(cv2.imread(path_img), cv2.COLOR_BGR2YUV).astype(np.float32) / 255.
    # model = Net_0(batch=1)
    model = Net_test()
    model.build(input_shape=(1, 720, 1280, 3))
    model.summary()
    x = model(np.expand_dims(y_hat, 0))
    print(x.shape)
    plt.imshow(np.abs(x.numpy()[0, ..., 0]), cmap="gray")
    plt.title("y_border")
    plt.show()
    plt.imshow(np.abs(x.numpy()[0, ..., 1]), cmap="gray")
    plt.title("u_border")
    plt.show()
    plt.imshow(np.abs(x.numpy()[0, ..., 2]), cmap="gray")
    plt.title("v_border")
    plt.show()
    plt.imshow(y_hat[..., 0], cmap="gray")
    plt.title("y")
    plt.show()
    plt.imshow(y_hat[..., 1], cmap="gray")
    plt.title("u")
    plt.show()
    plt.imshow(y_hat[..., 2], cmap="gray")
    plt.title("v")
    plt.show()