import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import CategoricalCrossentropy, MeanAbsoluteError, MeanSquaredError
from tensorflow.keras.metrics import CategoricalAccuracy, Accuracy
from TFM.Networks.Net import *
from TFM.Networks.HNet import *
from TFM.Networks.MgNet import *

import matplotlib.pyplot as plt

'''
This script contains all the necessary methods for training and inference processes.
'''

class ModelManager:
    """
    This class manages the neural models
    """
    def __init__(self, model, dim, path_weights, start_epoch, regresion, learn_reg=1e-3, verbose=1):
        self.model = model
        self.dim = dim
        self.regresion = regresion
        self.path_weights = path_weights
        self.start_epoch = start_epoch
        self.nn = self._load_nn(learn_reg, verbose)

    def _load_weigths(self):
        """
        Load weights for neural network
        :return: None
        """
        if self.start_epoch > 0:
            self.nn.load_weights(f'{self.path_weights}_{self.start_epoch}')
            print(f'Model weights {self.nn}_epoch{self.start_epoch} loaded!')

    def _load_nn(self, learn_reg, verbose=1):
        """
        Load neural network
        :param learn_reg: learning rate of the regularizer
        :param verbose: 1: print summary neural network
        :return: neural network
        """
        inputs = Input(shape=self.dim[1:])
        if self.model == "HelperNetV1":
            self.nn = Model(inputs, HelperNetV1(inputs, learn_reg))
        elif self.model == "HelperNetV2":
            self.nn = Model(inputs, HelperNetV2(inputs, learn_reg))
        elif self.model == "HelperNetV3":
            self.nn = Model(inputs, HelperNetV3(inputs, learn_reg))
        elif self.model == "Net_0":
            self.nn = Model(inputs, Net_0(inputs, self.dim[0], learn_reg))
        elif self.model == "Net_1":
            self.nn = Model(inputs, Net_1(inputs, self.dim[0], learn_reg))
        elif self.model == "Net_2":
            self.nn = Model(inputs, Net_2(inputs, self.dim[0], learn_reg))
        elif self.model == "Net_3":
            self.nn = Model(inputs, Net_3(inputs, self.dim[0], learn_reg))
        elif self.model == "Net_4":
            self.nn = Model(inputs, Net_4(inputs, self.dim[0], learn_reg))
        elif self.model == "Net_5":
            self.nn = Model(inputs, Net_5(inputs, self.dim[0], self.regresion, learn_reg))
        elif self.model == "MgNet_0":
            self.nn = MgNet_0(self.dim[0], learn_reg)
            self.nn.build(input_shape=self.dim)
        else:
            print("ERROR load_mod")
            print(exit)
            exit()
        self.nn.summary() if verbose == 1 else None

        # Load weights
        self._load_weigths()

        return self.nn

class TrainingModel(ModelManager):
    def __init__(self, model, dim, path_weights, start_epoch, learn_opt, learn_reg, regresion, verbose=1):
        super().__init__(model, dim, path_weights, start_epoch, learn_reg, verbose)
        self.optimizer = RMSprop(learn_opt)
        self.worst50 = {}
        if regresion:
            self.loss_fn = MeanSquaredError()
            self.valid_acc_metric = Accuracy()
            self.train_acc_metric = Accuracy()
        else:
            self.loss_fn = CategoricalCrossentropy(from_logits=False)
            self.train_acc_metric = CategoricalAccuracy()
            self.valid_acc_metric = CategoricalAccuracy()

    def _add_worst(self, value, path):
        pass
        # if value < self.worst50[max(self.worst50)]

    # Training and validation steps
    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            logits = self.nn(x, training=True)
            loss_value = self.loss_fn(y, logits)
        grads = tape.gradient(loss_value, self.nn.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.nn.trainable_weights))
        self.train_acc_metric.update_state(y, logits)
        return loss_value

    @tf.function
    def valid_step(self, x, y):
        val_logits = self.nn(x, training=False)
        self.valid_acc_metric.update_state(y, val_logits)
