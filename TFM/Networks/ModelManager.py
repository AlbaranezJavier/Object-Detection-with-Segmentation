import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import CategoricalCrossentropy, BinaryCrossentropy, MeanAbsoluteError, MeanSquaredError
from tensorflow.keras.metrics import CategoricalAccuracy, BinaryAccuracy, Accuracy
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
        elif self.model == "Net_6":
            self.nn = Model(inputs, Net_6(inputs, self.dim[0], self.output_type, learn_reg))
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
            self._loss_fn = [MeanSquaredError(), BinaryCrossentropy(from_logits=False)]
            self._valid_acc_metric = [Accuracy(), BinaryAccuracy()]
            self._train_acc_metric = [Accuracy(), BinaryAccuracy()]
        else:
            self._loss_fn = CategoricalCrossentropy(from_logits=False)
            self._train_acc_metric = CategoricalAccuracy()
            self._valid_acc_metric = CategoricalAccuracy()

    def _add_worst(self, value, path):
        pass
        # if value < self.worst50[max(self.worst50)]

    # Save Model
    def save_best(self, best, metric, min_acc, epoch, end_epoch, save_weights, weights_path=None):
        """
        Save the model weights if: it is the best metric so far and exceeds the minimum value, or it is the last
        training epoch. In any case, it is not saved if you have indicated not to save.
        :param best: best metric value to date
        :param metric: value
        :param min_acc: min value to save
        :param epoch: current epoch
        :param end_epoch: last epoch
        :param save_weights: true=save or false=dont save
        :param weights_path: path to store the weights
        :return: true if saved, false if not saved
        """
        current_value = np.sum(metric) / len(metric) if isinstance(metric, list) else metric
        best = np.sum(best) / len(best) if isinstance(best, list) else best
        min_acc = np.sum(min_acc) / len(min_acc) if isinstance(min_acc, list) else min_acc
        if save_weights and ((current_value > min_acc and current_value > best) or epoch == end_epoch):
            self.nn.save_weights(f'{weights_path}_{epoch}')
            return True
        else:
            return False


    # Metrics
    def get_acc(self, type):
        acc_metrics = self._train_acc_metric if type == "train" else self._valid_acc_metric
        if isinstance(acc_metrics, list):
            acc = []
            for acc_metric in acc_metrics:
                acc.append(float(acc_metric.result()*100.))
                acc_metric.reset_states()
            return acc
        else:
            return float(acc_metrics.result() * 100.)

    # Training and validation steps
    def train(self, x, y):
        if self.output_type == "reg+cls":
            return self._train_step_rc(x, y)
        else:
            return self._train_step(x, y)

    def valid(self, x, y):
        if self.output_type == "reg+cls":
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
            reg, cls = self.nn(x, training=True)
            targets = [y[:, :, :, self.sets_channels[0][0]:self.sets_channels[0][1]],
                      y[:, :, :, self.sets_channels[1][0]:self.sets_channels[1][1]]]
            losses = [l(t, o) for l, o, t in zip(self._loss_fn, [reg, cls], targets)]
        grads = tape.gradient(losses, self.nn.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.nn.trainable_weights))
        [m.update_state(t, o) for m, o, t in zip(self._train_acc_metric, [reg, cls], targets)]
        return losses

    @tf.function
    def _valid_step(self, x, y):
        val_logits = self.nn(x, training=False)
        self._valid_acc_metric.update_state(y, val_logits)

    @tf.function
    def _valid_step_rc(self, x, y):
        reg, cls = self.nn(x, training=False)
        self._valid_acc_metric[0].update_state(y[:, :, :, self.sets_channels[0][0]:self.sets_channels[0][1]], reg)
        self._valid_acc_metric[1].update_state(y[:, :, :, self.sets_channels[1][0]:self.sets_channels[1][1]], cls)
