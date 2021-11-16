from tensorflow.keras.layers import Conv2D, Conv2DTranspose, concatenate, BatchNormalization, Layer
from tensorflow.keras.activations import relu
from tensorflow.keras.regularizers import L2


class Conv2D_NA(Layer):
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
        return relu(norm + self.conv.bias)


class Conv2D_NATranspose(Layer):
    def __init__(self, k_dim, output_channel, stride, padding='VALID', k_reg=None):
        super(Conv2D_NATranspose, self).__init__()
        self.stride = stride
        self.padding = padding

        self.conv = Conv2DTranspose(filters=output_channel, kernel_size=(k_dim, k_dim), strides=(stride, stride),
                                    padding=padding, kernel_regularizer=k_reg, output_padding=(1, 1))
        self.bn = BatchNormalization()

    def call(self, inputs):
        channels = self.conv(inputs)
        norm = self.bn(channels)
        return relu(norm + self.conv.bias)


def UNetplusplus_3L(inputs, learn_reg=1e-2):
    # Variables
    l2 = L2(learn_reg)

    # Left side = Li, Mid = M y right side = Ld
    # - Level 3, Li
    n3Li = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(inputs)

    # - Level 4, Li
    n4Li = Conv2D_NA(k_dim=5, output_channel=32, stride=2, padding="VALID", k_reg=l2)(n3Li)
    n4Li = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n4Li)

    # - Level 3-4
    n34 = Conv2D_NATranspose(k_dim=5, output_channel=32, stride=2, padding="VALID", k_reg=l2)(n4Li)
    n34 = concatenate([n3Li, n34], axis=3)
    n34 = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n34)

    # - Level 5, Li
    n5Li = Conv2D_NA(k_dim=5, output_channel=32, stride=2, padding="VALID", k_reg=l2)(n4Li)
    n5Li = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n5Li)

    # - Level 4, Ld
    n4Ld = Conv2D_NATranspose(k_dim=5, output_channel=32, stride=2, padding="VALID", k_reg=l2)(n5Li)
    n4Ld = concatenate([n4Li, n4Ld], axis=3)
    n4Ld = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n4Ld)

    # - Level 3, Ld
    n3Ld = Conv2D_NATranspose(k_dim=5, output_channel=32, stride=2, padding="VALID", k_reg=l2)(n4Ld)
    n3Ld = concatenate([n3Li, n3Ld, n34], axis=3)
    n3Ld = Conv2D_NA(k_dim=5, output_channel=32, stride=1, padding="SAME", k_reg=l2)(n3Ld)
    n3Ld = Conv2D(filters=5, kernel_size=(1, 1), kernel_regularizer=l2, padding="SAME", activation="softmax")(n3Ld)
    return n3Ld
