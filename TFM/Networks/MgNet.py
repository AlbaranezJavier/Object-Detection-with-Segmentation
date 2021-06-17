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
            self.strides = [1, 1, 1, 1]
            self.pad = [[0, 0], [1, 1], [1, 1], [0, 0]]
        elif type == "average":
            filter = np.array([[[[1 / 9]], [[1 / 9]], [[1 / 9]]],
                               [[[1 / 9]], [[1 / 9]], [[1 / 9]]],
                               [[[1 / 9]], [[1 / 9]], [[1 / 9]]]])
            self.strides = [1, 1, 1, 1]
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

class Conv2D_W(keras.layers.Layer):
    def __init__(self, kernel_dim=3, input_channel=3, output_channel=16, stride=1, shared_conv=None, padding='SAME'):
        super(Conv2D_W, self).__init__()

        self.stride = stride
        self.padding = padding
        if shared_conv is None:
            w_init = tf.keras.initializers.GlorotUniform()
            self.w = tf.Variable(
                initial_value=w_init(shape=(kernel_dim, kernel_dim, input_channel, output_channel), dtype="float32"),
                trainable=True)

            b_init = tf.keras.initializers.Zeros()
            self.b = tf.Variable(
                initial_value=b_init(shape=(output_channel,), dtype="float32"), trainable=True)
        else:
            self.w = tf.transpose(shared_conv.w, perm=[0, 2, 1, 3])
            self.b = shared_conv.b
        self.bn = BatchNormalization()

    def call(self, inputs):
        # tf.nn.conv2d(   input, filters, strides, padding, data_format='NHWC', dilations=None,   name=None)
        convolucion = tf.nn.conv2d(inputs, self.w, strides=[1, self.stride, self.stride, 1], padding=self.padding)
        convolucion = self.bn(convolucion)
        return tf.nn.relu(convolucion + self.b)

class Mg_DOWN(keras.layers.Layer):
    def __init__(self, k_dim, output_channel, u_shape, j, v, k_reg=None, **kwargs):
        super(Mg_DOWN, self).__init__(**kwargs)
        self.v = v
        self.j = j
        self.u_shape = u_shape

        # Data-feature mapping
        self.a = [Conv2D(filters=3, kernel_size=(k_dim, k_dim), strides=(1, 1), padding='SAME',
                         kernel_regularizer=k_reg, name="a"+str(l)) for l in range(j)]

        # Feature extractor
        self.b = [[Conv2D(filters=5, kernel_size=(k_dim, k_dim), strides=(1, 1), padding='SAME',
                          kernel_regularizer=k_reg, name="b"+str(l)+str(i)) for i in range(v)] for l in range(j)]

        # Interpolator
        self.i = [Conv2D(filters=output_channel, kernel_size=(k_dim, k_dim), strides=(2, 2), padding='SAME',
                         kernel_regularizer=k_reg, name="i"+str(l)) for l in range(1, j)]

        # Restrictor
        self.r = Conv2DFixed("bilinear", 3)

        self.bn_a = BatchNormalization()
        self.bn = BatchNormalization()

    def call(self, f_in):
        u = []
        f = [f_in]
        # Level 1
        # - Smoothing
        a = f_in - self.bn_a(self.a[0](tf.zeros(self.u_shape)))
        u.append(tf.nn.relu(self.bn(self.b[0][0](tf.nn.relu(a)))))
        for i in range(self.v):
            a = f_in - self.bn_a(self.a[0](u[0]))
            b = tf.nn.relu(self.bn(self.b[0][i](tf.nn.relu(a))))
            u[0] = u[0] + b

        # Loop: IaR => Smoothing
        for l in range(1, self.j):
            # - Interpolation and restriction
            u.append(self.bn(self.i[l-1](u[l-1]))) # i[l-1] is correct, is an array
            _a = self.bn_a(self.a[l](u[l]))
            f.append(self.bn_a(self.r(a)) + _a)

            # Next level
            # - Smoothing
            for i in range(self.v):
                a = f[l] - self.bn_a(self.a[l](u[l]))
                b = tf.nn.relu(self.bn(self.b[l][i](tf.nn.relu(a))))
                u[l] = u[l] + b

        return u, f


class Mg_UP(keras.layers.Layer):
    def __init__(self, k_dim, output_size, l, v, k_reg=None):
        super(Mg_UP, self).__init__()
        self.v = v

        # Data-feature mapping
        self.a = Conv2D(filters=3, kernel_size=(k_dim, k_dim), strides=(1, 1), padding='SAME', kernel_regularizer=k_reg)

        # Feature extractor
        self.b = [Conv2D(filters=5, kernel_size=(k_dim, k_dim), strides=(1, 1), padding='SAME',
                         kernel_regularizer=k_reg, name="b_up"+str(l)+str(i)) for i in range(self.v)]

        # Prolongator
        self.p = Conv2DFixed_Transpose("bilinear", output_size)

        self.bn_a = BatchNormalization()
        self.bn = BatchNormalization()

    def call(self, u_prev, u_post, f, u=None):
        # Prolongation and correction
        if u == None:
            _u = u_prev + self.bn(self.p(-u_post))
        else:
            _u = u_prev + self.bn(self.p(u - u_post))
        # Post-smoothing
        for i in range(self.v):
            a = self.bn_a(self.a(_u))
            b = tf.nn.relu(self.bn(self.b[i](tf.nn.relu(f - a))))
            _u = _u + b
        return _u


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


class MgNet_0(Model):
    def __init__(self, batch, learn_reg=1e-2):
        super(MgNet_0, self).__init__()
        # Variables
        l2 = L2(learn_reg)
        self.mg_b = Mg_DOWN(k_dim=3, output_channel=5, u_shape=[batch, 513, 1025, 5], j=4, v=5, k_reg=l2)
        self.mg_c1 = Mg_UP(k_dim=3, output_size=[batch, 129, 257, 5], l=3, v=5, k_reg=l2)
        self.mg_c2 = Mg_UP(k_dim=3, output_size=[batch, 257, 513, 5], l=2, v=5, k_reg=l2)
        self.mg_c3 = Mg_UP(k_dim=3, output_size=[batch, 513, 1025, 5], l=1, v=5, k_reg=l2)
        self.conv_softmax = Conv2D(filters=5, kernel_size=(1, 1), strides=(1, 1), padding='SAME', kernel_regularizer=l2,
                   activation="softmax")

    def call(self, f_in):
        # Block
        _u, _f = self.mg_b(f_in)

        # - Cycles
        u = self.mg_c1(u_prev=_u[2], u_post=_u[3], f=_f[2])
        u = self.mg_c2(u_prev=_u[1], u_post=_u[2], f=_f[1], u=u)
        u = self.mg_c3(u_prev=_u[0], u_post=_u[1], f=_f[0], u=u)

        # Out
        u = self.conv_softmax(u)
        return u


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
    cv2.imshow("h", img[..., 0])
    cv2.waitKey(0)
    cv2.imshow("s", img[..., 1])
    cv2.waitKey(0)
    cv2.imshow("v", img[..., 2])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
