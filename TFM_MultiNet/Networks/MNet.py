from tensorflow.keras.layers import Conv2D, concatenate, BatchNormalization, Conv2DTranspose
from tensorflow.keras.regularizers import L2
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
                           padding=padding, kernel_regularizer=k_reg)
        self.bn = BatchNormalization()

    def call(self, inputs):
        channels = self.conv(inputs)
        norm = self.bn(channels)
        return tf.nn.relu(norm + self.conv.bias)

class Net_test(Model):
    def __init__(self):
        super(Net_test, self).__init__()
        # self.l_720 = Conv2DFixed("border", 3)
        # self.l_720 = Conv2DFixed("bilinear", 3)
        # self.d_720 = Conv2DFixed_Transpose("bilinear", [1, 720, 1280, 3])
        # self.l_720 = Conv2DFixed("linear", 3)
        self.l_720 = Conv2DFixed("average", 3)

    def call(self, inputs):
        x = self.l_720(inputs)
        # x = self.d_720(x)
        return x

def MNet_720_down(inputs, learn_reg=1e-2):
    fixed_n1Li = Conv2DFixed("border", out_channels=3)(inputs)
    n1Li = concatenate([fixed_n1Li, inputs], axis=3)

    n1Li = Conv2D_NA(k_dim=3, output_channel=2, stride=1, padding="SAME", k_reg=L2(learn_reg))(n1Li)
    return n1Li

def MNet_360_down(inputs, n1Li, learn_reg=1e-2):
    n2Li = Conv2DFixed("bilinear", out_channels=2)(n1Li)
    inputs_n2Li = tf.image.resize(inputs, [360, 640], antialias=True)
    fixed_n2Li = Conv2DFixed("border", out_channels=3)(inputs_n2Li)
    n2Li = concatenate([fixed_n2Li, n2Li, inputs_n2Li], axis=3)

    n2Li = Conv2D_NA(k_dim=3, output_channel=4, stride=1, padding="SAME", k_reg=L2(learn_reg))(n2Li)
    return n2Li

def MNet_180_down(inputs, n2Li, learn_reg=1e-2):
    n3Li = Conv2DFixed("bilinear", out_channels=4)(n2Li)
    inputs_n3Li = tf.image.resize(inputs, [180, 320], antialias=True)
    fixed_n3Li = Conv2DFixed("border", out_channels=3)(inputs_n3Li)
    n3Li = concatenate([fixed_n3Li, n3Li, inputs_n3Li], axis=3)

    n3Li = Conv2D_NA(k_dim=3, output_channel=8, stride=1, padding="SAME", k_reg=L2(learn_reg))(n3Li)
    return n3Li

def MNet_90_down(inputs, n3Li, learn_reg=1e-2):
    n4Li = Conv2DFixed("bilinear", out_channels=8)(n3Li)
    inputs_n4Li = tf.image.resize(inputs, [90, 160], antialias=True)
    fixed_n4Li = Conv2DFixed("border", out_channels=3)(inputs_n4Li)
    n4Li = concatenate([fixed_n4Li, n4Li, inputs_n4Li], axis=3)

    n4Li = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=L2(learn_reg))(n4Li)
    return n4Li

def MNet_45(inputs, n4Li, learn_reg=1e-2):
    n5Li = Conv2DFixed("bilinear", out_channels=32)(n4Li)
    inputs_n5Li = tf.image.resize(inputs, [45, 80], antialias=True)
    fixed_n5Li = Conv2DFixed("border", out_channels=3)(inputs_n5Li)
    n5Li = concatenate([fixed_n5Li, n5Li, inputs_n5Li], axis=3)

    n5Li = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=L2(learn_reg))(n5Li)

    n6M = Conv2D_NA(k_dim=5, output_channel=64, stride=1, padding="SAME", k_reg=L2(learn_reg))(n5Li)

    n5Ld = concatenate([n5Li, n6M], axis=3)
    n5Ld = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=L2(learn_reg))(n5Ld)
    n5_out = Conv2D(filters=5, kernel_size=(1, 1), kernel_regularizer=L2(learn_reg), padding="SAME",
                    activation="softmax")(n5Ld)
    return n5Ld, n5_out

def MNet_90_up(n4Li, n5Ld, batch, learn_reg=1e-2):
    n4Ld = Conv2DFixed_Transpose("bilinear", [batch, 90, 160, 32])(n5Ld)
    n4Ld = concatenate([n4Li, n4Ld], axis=3)
    n4Ld = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=L2(learn_reg))(n4Ld)
    n4_out = Conv2D(filters=5, kernel_size=(1, 1), kernel_regularizer=L2(learn_reg), padding="SAME",
                    activation="softmax")(n4Ld)
    return n4Ld, n4_out

def MNet_180_up(n3Li, n4Ld, batch, learn_reg=1e-2):
    n3Ld = Conv2DFixed_Transpose("bilinear", [batch, 180, 320, 32])(n4Ld)
    n3Ld = concatenate([n3Li, n3Ld])
    n3Ld = Conv2D_NA(k_dim=3, output_channel=5, stride=1, padding="SAME", k_reg=L2(learn_reg))(n3Ld)
    n3_out = Conv2D(filters=5, kernel_size=(1, 1), kernel_regularizer=L2(learn_reg), padding="SAME",
                    activation="softmax")(n3Ld)

    return n3Ld, n3_out

def MNet_360_up(n2Li, n3Ld, batch, learn_reg=1e-2):
    n2Ld = Conv2DFixed_Transpose("bilinear", [batch, 360, 640, 5])(n3Ld)
    n2Ld = concatenate([n2Li, n2Ld])
    n2Ld = Conv2D_NA(k_dim=3, output_channel=5, stride=1, padding="SAME", k_reg=L2(learn_reg))(n2Ld)
    n2_out = Conv2D(filters=5, kernel_size=(1, 1), kernel_regularizer=L2(learn_reg), padding="SAME",
                  activation="softmax")(n2Ld)
    return n2Ld, n2_out

def MNet_720_up(n1Li, n2Ld, batch, learn_reg=1e-2):
    n1Ld = Conv2DFixed_Transpose("bilinear", [batch, 720, 1280, 5])(n2Ld)
    n1Ld = concatenate([n1Li, n1Ld])
    n1Ld = Conv2D(filters=5, kernel_size=(1, 1), kernel_regularizer=L2(learn_reg), padding="SAME", activation="softmax")(n1Ld)
    return n1Ld

if __name__ == '__main__':
    path_img = glob.glob(r"C:\Users\TTe_J\Downloads\new_RGBs\vid_2_frame_1708.jpg")
    img = cv2.cvtColor(cv2.imread(path_img[0]), cv2.COLOR_BGR2HLS).astype(np.float32) / 255.
    # model = Net_0(batch=1)
    model = Net_test()
    model.build(input_shape=(1, 720, 1280, 3))
    model.summary()
    x = model(np.expand_dims(img, 0))
    print(x.shape)
    cv2.imshow("h_conv", x.numpy()[0, ..., 0])
    cv2.waitKey(0)
    cv2.imshow("s_conv", x.numpy()[0, ..., 1])
    cv2.waitKey(0)
    cv2.imshow("v_conv", x.numpy()[0, ..., 2])
    cv2.waitKey(0)
    cv2.imshow("h", img[...,0])
    cv2.waitKey(0)
    cv2.imshow("s", img[..., 1])
    cv2.waitKey(0)
    cv2.imshow("v", img[..., 2])
    cv2.waitKey(0)
    cv2.destroyAllWindows()