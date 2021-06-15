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
    def __init__(self, model, dim, path_weights, start_epoch, output_type, learn_reg=1e-3, verbose=1):
        self.model = model
        self.dim = dim
        self.output_type = output_type
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
            self.nn = Model(inputs, Net_5(inputs, self.dim[0], self.output_type, learn_reg))
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
    def __init__(self, model, dim, path_weights, start_epoch, learn_opt, learn_reg, output_type, verbose=1):
        super().__init__(model, dim, path_weights, start_epoch, output_type, learn_reg, verbose)
        self.optimizer = RMSprop(learn_opt)
        self._train_acc_value = 0
        self._valid_acc_value = 0
        self.sets_channels = []
        self.worst50 = {}
        if output_type == "reg":
            self._loss_fn = MeanSquaredError()
            self._valid_acc_metric = Accuracy()
            self._train_acc_metric = Accuracy()
        elif output_type == "reg+cls":
            self.sets_channels = [[0, 4], [4, 5]]
            self._loss_fn = [MeanSquaredError(), CategoricalCrossentropy(from_logits=False)]
            self._valid_acc_metric = [Accuracy(), CategoricalAccuracy()]
            self._train_acc_metric = [Accuracy(), CategoricalAccuracy()]
        else:
            self._loss_fn = CategoricalCrossentropy(from_logits=False)
            self._train_acc_metric = CategoricalAccuracy()
            self._valid_acc_metric = CategoricalAccuracy()

    def _add_worst(self, value, path):
        pass
        # if value < self.worst50[max(self.worst50)]

    # Metrics
    def get_acc(self, type):
        acc_metrics = self._train_acc_metric if type == "train" else self._valid_acc_metric
        acc = 0
        if isinstance(acc_metrics, list):
            for acc_metric in acc_metrics:
                acc += acc_metric.result()
                acc_metric.reset_states()
            return (acc/len(acc_metrics)) * 100.
        else:
            return acc_metrics.result() * 100.

    # Training and validation steps
    def train(self, x, y):
        if self.output_type == "reg" or self.output_type == "reg+cls":
            return self._train_step_rc(x, y)
        else:
            return self._train_step(x, y)

    def valid(self, x, y):
        if self.output_type == "reg" or self.output_type == "reg+cls":
            return self._valid_step_rc(x, y)
        else:
            return self._valid_step(x, y)

    @tf.function
    def _train_step(self, x, y):
        with tf.GradientTape() as tape:
            logits = self.nn(x, training=True)
            loss_value = self._loss_fn(y, logits)
        grads = tape.gradient(loss_value, self.nn.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.nn.trainable_weights))
        self._train_acc_metric.update_state(y, logits)
        return loss_value

    @tf.function
    def _train_step_rc(self, x, y):
        """
        Train step for outputs with regression and classification
        """
        with tf.GradientTape() as tape:
            logits = self.nn(x, training=True)
            loss_value = 0
            for i in range(0, len(self._loss_fn)):
                loss_value += self._loss_fn[i](y[:, :, :, self.sets_channels[i][0]:self.sets_channels[i][1]],
                                               logits[:, :, :, self.sets_channels[i][0]:self.sets_channels[i][1]])
        grads = tape.gradient(loss_value, self.nn.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.nn.trainable_weights))
        for i in range(len(self._train_acc_metric)):
            self._train_acc_metric[i].update_state(y[:, :, :, self.sets_channels[i][0]:self.sets_channels[i][1]],
                                                   logits[:, :, :, self.sets_channels[i][0]:self.sets_channels[i][1]])
        return loss_value

    @tf.function
    def _valid_step(self, x, y):
        val_logits = self.nn(x, training=False)
        self._valid_acc_metric.update_state(y, val_logits)

    @tf.function
    def _valid_step_rc(self, x, y):
        val_logits = self.nn(x, training=False)
        for i in range(len(self._valid_acc_metric)):
            self._valid_acc_metric[i].update_state(y[:, :, :, self.sets_channels[i][0]:self.sets_channels[i][1]],
                                                   val_logits[:, :, :, self.sets_channels[0][0]:self.sets_channels[0][1]])
