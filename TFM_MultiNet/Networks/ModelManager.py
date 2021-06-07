import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from TFM_MultiNet.Networks.MNet import *

import matplotlib.pyplot as plt

'''
This script contains all the necessary methods for training and inference processes.
'''

class ModelManager:
    """
    This class manages the neural models
    """
    def __init__(self, model, dim, path_weights, start_epoch, learn_reg=1e-3, verbose=1):
        self.model = model
        self.dim = dim
        self.out_sizes = []
        self.path_weights = path_weights
        self.start_epoch = start_epoch
        self.nn = self._load_nn(learn_reg, verbose)
        self._load_weigths()

    def _load_weigths(self):
        """
        Load weights for neural network
        :return: None
        """
        if self.start_epoch > 0:
            self.nn[-1].load_weights(f'{self.path_weights}_{self.start_epoch}')
            print(f'Model weights {self.nn[-1]}_epoch{self.start_epoch} loaded!')

    def _load_nn(self, learn_reg, verbose=1):
        """
        Load neural network
        :param learn_reg: learning rate of the regularizer
        :param verbose: 1: print summary neural network
        :return: neural network
        """
        nets = []
        inputs = Input(shape=self.dim[1:])
        if self.model == "MNet_0":
            M720_down = MNet_720_down(inputs)
            M360_down = MNet_360_down(inputs, M720_down)
            M180_down = MNet_180_down(inputs, M360_down)
            M90_down = MNet_90_down(inputs, M180_down)
            M45, M45_out = MNet_45(inputs, M90_down)
            nets.append(Model(inputs, M45_out))
            M90_up, M90_out = MNet_90_up(M90_down, M45, self.dim[0])
            nets.append(Model(inputs, M90_out))
            M180_up, M180_out = MNet_180_up(M180_down, M90_up, self.dim[0])
            nets.append(Model(inputs, M180_out))
            M360_up, M360_out = MNet_360_up(M360_down, M180_up, self.dim[0])
            nets.append(Model(inputs, M360_out))
            M720_up = MNet_720_up(M720_down, M360_up, self.dim[0])
            nets.append(Model(inputs, M720_up))

            self.out_sizes = [[45, 80], [90, 160], [180, 320], [360, 640], [720, 1280]]

        else:
            print("ERROR load_mod")
            print(exit)
            exit()
        nets[-1].summary() if verbose == 1 else None

        return nets

class TrainingModel(ModelManager):
    def __init__(self, model, dim, path_weights, start_epoch, learn_opt, learn_reg, verbose=1):
        super().__init__(model, dim, path_weights, start_epoch, learn_reg, verbose)
        self.optimizer = [RMSprop(learn_opt) for size in self.out_sizes]
        self.loss_fn = [CategoricalCrossentropy(from_logits=False) for size in self.out_sizes]
        self.train_acc_metric = [CategoricalAccuracy() for size in self.out_sizes]
        self.valid_acc_metric = [CategoricalAccuracy() for size in self.out_sizes]
        self.worst50 = {}

    def _add_worst(self, value, path):
        pass
        # if value < self.worst50[max(self.worst50)]

    # Training and validation steps
    @tf.function
    def train_step(self, x, ys):
        losses = []
        for i in range(len(self.out_sizes)):
            y = tf.image.resize(ys, self.out_sizes[i], method='nearest')
            with tf.GradientTape() as tape:
                logits = self.nn[i](x, training=True)
                loss = self.loss_fn[i](y, logits)
            grads = tape.gradient(loss, self.nn[i].trainable_weights)
            losses.append(loss)
            self.optimizer[i].apply_gradients(zip(grads, self.nn[i].trainable_weights))
            self.train_acc_metric[i].update_state(y, logits)
        return losses

    @tf.function
    def valid_step(self, x, ys):
        for i in range(len(self.out_sizes)):
            y = tf.image.resize(ys, self.out_sizes[i], method='nearest')
            val_logits = self.nn[i](x, training=False)
            self.valid_acc_metric[i].update_state(y, val_logits)
