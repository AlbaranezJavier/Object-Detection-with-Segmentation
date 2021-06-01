from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate, BatchNormalization
from tensorflow.keras.regularizers import L2

'''
This script contains the proposed network structures.
'''

# Small kernel
def HelperNetV1(inputs, learn_reg=1e-2):
  """
  Small kernels
  :param inputs:
  :param learn_reg:
  :return:
  """
  filters=[2, 4, 8, 64, 128, 5]
  kernel_sizes = [(5, 5), (3, 3), (1, 1)]
  downup = (2, 2)
  activation = "relu"
  lreg = L2(learn_reg)

  # Left side = Li, Mid = M y right side = Ld
  # - Level 1, Li
  # n1Li = Dropout(0.1)(inputs)
  n1Li = Conv2D(filters=filters[0], kernel_size=kernel_sizes[1], kernel_regularizer=lreg, padding="same", activation=activation)(inputs)
  n1Li = BatchNormalization()(n1Li)

  # - Level 2, Li
  n2Li = MaxPooling2D(downup)(n1Li)
  n2Li = Conv2D(filters=filters[1], kernel_size=kernel_sizes[1], kernel_regularizer=lreg, padding="same", activation=activation)(n2Li)
  n2Li = BatchNormalization()(n2Li)

  # - Level 3, Li
  n3Li = MaxPooling2D(downup)(n2Li)
  n3Li = Conv2D(filters=filters[2], kernel_size=kernel_sizes[1], kernel_regularizer=lreg, padding="same", activation=activation)(n3Li)
  n3Li = BatchNormalization()(n3Li)

  # - Level 4, Li
  n4Li = MaxPooling2D(downup)(n3Li)
  n4Li = Conv2D(filters=filters[3], kernel_size=kernel_sizes[1], kernel_regularizer=lreg, padding="same", activation=activation)(n4Li)
  n4Li = BatchNormalization()(n4Li)

  # - Level 5, M
  n5M = Conv2D(filters=filters[4], kernel_size=kernel_sizes[0], kernel_regularizer=lreg, padding="same", activation=activation)(n4Li)
  n5M = BatchNormalization()(n5M)

  # - Level 4, Ld
  n4Ld = concatenate([n4Li, n5M], axis=3)
  n4Ld = Conv2D(filters=filters[3], kernel_size=kernel_sizes[1], kernel_regularizer=lreg, padding="same", activation=activation)(n4Ld)
  n4Ld = UpSampling2D(downup)(n4Ld)
  n4Ld = BatchNormalization()(n4Ld)

  # - Level 3, Ld
  n3Ld = concatenate([n3Li, n4Ld])
  n3Ld = Conv2D(filters=filters[5], kernel_size=kernel_sizes[2], kernel_regularizer=lreg, padding="same", activation=activation)(n3Ld)
  n3Ld = UpSampling2D(downup)(n3Ld)
  n3Ld = BatchNormalization()(n3Ld)

  # - Level 2, Ld
  n2Ld = concatenate([n2Li, n3Ld])
  n2Ld = Conv2D(filters=filters[5], kernel_size=kernel_sizes[2], kernel_regularizer=lreg, padding="same", activation=activation)(n2Ld)
  n2Ld = UpSampling2D(downup)(n2Ld)
  n2Ld = BatchNormalization()(n2Ld)

  # - Level 1, Ld
  n1Ld = concatenate([n1Li, n2Ld])
  n1Ld = Conv2D(filters=filters[5], kernel_size=kernel_sizes[2], kernel_regularizer=lreg, padding="same", activation="softmax")(n1Ld)
  return n1Ld

# Big kernel
def HelperNetV2(inputs, learn_reg=1e-2):
  """
  Big kernels
  :param inputs:
  :param learn_reg:
  :return:
  """
  filters=[2, 4, 8, 64, 128, 5]
  kernel_sizes = [(5, 5), (3, 3), (1, 1)]
  downup = (2, 2)
  activation = "relu"
  lreg = L2(learn_reg)

  # Left side = Li, Mid = M y right side = Ld
  # - Level 1, Li
  # n1Li = Dropout(0.1)(inputs)
  n1Li = Conv2D(filters=filters[0], kernel_size=kernel_sizes[1], kernel_regularizer=lreg, padding="same", activation=activation)(inputs)
  n1Li = BatchNormalization()(n1Li)

  # - Level 2, Li
  n2Li = MaxPooling2D(downup)(n1Li)
  n2Li = Conv2D(filters=filters[1], kernel_size=kernel_sizes[1], kernel_regularizer=lreg, padding="same", activation=activation)(n2Li)
  n2Li = BatchNormalization()(n2Li)

  # - Level 3, Li
  n3Li = MaxPooling2D(downup)(n2Li)
  n3Li = Conv2D(filters=filters[2], kernel_size=kernel_sizes[1], kernel_regularizer=lreg, padding="same", activation=activation)(n3Li)
  n3Li = BatchNormalization()(n3Li)

  # - Level 4, Li
  n4Li = MaxPooling2D(downup)(n3Li)
  n4Li = Conv2D(filters=filters[3], kernel_size=kernel_sizes[0], kernel_regularizer=lreg, padding="same", activation=activation)(n4Li)
  n4Li = BatchNormalization()(n4Li)

  # - Level 5, M
  n5M = Conv2D(filters=filters[4], kernel_size=kernel_sizes[0], kernel_regularizer=lreg, padding="same", activation=activation)(n4Li)
  n5M = BatchNormalization()(n5M)

  # - Level 4, Ld
  n4Ld = concatenate([n4Li, n5M], axis=3)
  n4Ld = Conv2D(filters=filters[3], kernel_size=kernel_sizes[0], kernel_regularizer=lreg, padding="same", activation=activation)(n4Ld)
  n4Ld = UpSampling2D(downup)(n4Ld)
  n4Ld = BatchNormalization()(n4Ld)

  # - Level 3, Ld
  n3Ld = concatenate([n3Li, n4Ld])
  n3Ld = Conv2D(filters=filters[5], kernel_size=kernel_sizes[1], kernel_regularizer=lreg, padding="same", activation=activation)(n3Ld)
  n3Ld = UpSampling2D(downup)(n3Ld)
  n3Ld = BatchNormalization()(n3Ld)

  # - Level 2, Ld
  n2Ld = concatenate([n2Li, n3Ld])
  n2Ld = Conv2D(filters=filters[5], kernel_size=kernel_sizes[1], kernel_regularizer=lreg, padding="same", activation=activation)(n2Ld)
  n2Ld = UpSampling2D(downup)(n2Ld)
  n2Ld = BatchNormalization()(n2Ld)

  # - Level 1, Ld
  n1Ld = concatenate([n1Li, n2Ld])
  n1Ld = Conv2D(filters=filters[5], kernel_size=kernel_sizes[2], kernel_regularizer=lreg, padding="same", activation="softmax")(n1Ld)
  return n1Ld

# Only segmentation
def HelperNetV3(inputs, learn_reg=1e-2):
  """
  One mask
  :param inputs:
  :param learn_reg:
  :return:
  """
  filters=[2, 4, 8, 64, 128]
  kernel_sizes = [(5, 5), (3, 3), (1, 1)]
  downup = (2, 2)
  activation = "relu"
  lreg = L2(learn_reg)

  # Left side = Li, Mid = M y right side = Ld
  # - Level 1, Li
  # n1Li = Dropout(0.1)(inputs)
  n1Li = Conv2D(filters=filters[0], kernel_size=kernel_sizes[1], kernel_regularizer=lreg, padding="same", activation=activation)(inputs)
  n1Li = BatchNormalization()(n1Li)

  # - Level 2, Li
  n2Li = MaxPooling2D(downup)(n1Li)
  n2Li = Conv2D(filters=filters[1], kernel_size=kernel_sizes[1], kernel_regularizer=lreg, padding="same", activation=activation)(n2Li)
  n2Li = BatchNormalization()(n2Li)

  # - Level 3, Li
  n3Li = MaxPooling2D(downup)(n2Li)
  n3Li = Conv2D(filters=filters[2], kernel_size=kernel_sizes[1], kernel_regularizer=lreg, padding="same", activation=activation)(n3Li)
  n3Li = BatchNormalization()(n3Li)

  # - Level 4, Li
  n4Li = MaxPooling2D(downup)(n3Li)
  n4Li = Conv2D(filters=filters[3], kernel_size=kernel_sizes[0], kernel_regularizer=lreg, padding="same", activation=activation)(n4Li)
  n4Li = BatchNormalization()(n4Li)

  # - Level 5, M
  n5M = Conv2D(filters=filters[4], kernel_size=kernel_sizes[0], kernel_regularizer=lreg, padding="same", activation=activation)(n4Li)
  n5M = BatchNormalization()(n5M)

  # - Level 4, Ld
  n4Ld = concatenate([n4Li, n5M], axis=3)
  n4Ld = Conv2D(filters=filters[3], kernel_size=kernel_sizes[0], kernel_regularizer=lreg, padding="same", activation=activation)(n4Ld)
  n4Ld = UpSampling2D(downup)(n4Ld)
  n4Ld = BatchNormalization()(n4Ld)

  # - Level 3, Ld
  n3Ld = concatenate([n3Li, n4Ld])
  n3Ld = Conv2D(filters=filters[0], kernel_size=kernel_sizes[1], kernel_regularizer=lreg, padding="same", activation=activation)(n3Ld)
  n3Ld = UpSampling2D(downup)(n3Ld)
  n3Ld = BatchNormalization()(n3Ld)

  # - Level 2, Ld
  n2Ld = concatenate([n2Li, n3Ld])
  n2Ld = Conv2D(filters=filters[0], kernel_size=kernel_sizes[1], kernel_regularizer=lreg, padding="same", activation=activation)(n2Ld)
  n2Ld = UpSampling2D(downup)(n2Ld)
  n2Ld = BatchNormalization()(n2Ld)

  # - Level 1, Ld
  n1Ld = concatenate([n1Li, n2Ld])
  n1Ld = Conv2D(filters=filters[0], kernel_size=kernel_sizes[2], kernel_regularizer=lreg, padding="same", activation="softmax")(n1Ld)
  return n1Ld