import numpy as np
import tensorflow as tf
import os, random
import tensorflow.keras.backend as K

'''
This script contains all the necessary methods for training and inference processes.
'''

losses = {"categorical_crossentropy": tf.keras.losses.CategoricalCrossentropy(from_logits=False),
          "categorical_crossentropy_true": tf.keras.losses.CategoricalCrossentropy(from_logits=True),
          "mse": tf.keras.losses.MeanSquaredError(),
          "mae": tf.keras.losses.MeanAbsoluteError(),
          "sparse_categorical_crossentropy": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)}
metrics = {"categorical_accuracy": tf.keras.metrics.CategoricalAccuracy,
           "mse": tf.keras.metrics.MeanSquaredError,
           "mae": tf.keras.metrics.MeanAbsoluteError,
           "sparse_categorical_accuracy": tf.keras.metrics.SparseCategoricalAccuracy}


class ModelManager:
    """
    This class manages the neural models
    """

    def __init__(self, nn, weights_path, epoch, verbose=1):
        self.nn = nn
        self.nn.summary() if verbose == 1 else None
        self.weights_path = weights_path
        self.nn.load_weights(f'{weights_path}_{epoch}') if epoch > 0 else None
        print(f'Model {self.nn}_epoch{epoch} ready!')


class TrainingModel(ModelManager):
    def __init__(self, nn, weights_path, start_epoch, optimizer, schedules, loss_func, metric_func, verbose=1):
        super().__init__(nn, weights_path, start_epoch, verbose)
        self.optimizer = optimizer
        self.schedules = schedules
        self._train_acc_value = 0
        self._valid_acc_value = 0
        self._loss_fn = losses[loss_func]
        self._train_acc_metric = metrics[metric_func]()
        self._valid_acc_metric = metrics[metric_func]()

    # Save Model
    def save_best(self, best, metric, min_acc, epoch, end_epoch, save_weights):
        """
        Save the model weights if: it is the best metric so far and exceeds the minimum value, or it is the last
        training epoch. In any case, it is not saved if you have indicated not to save.
        :param best: best metric value to date
        :param metric: value
        :param min_acc: min value to save
        :param epoch: current epoch
        :param end_epoch: last epoch
        :param save_weights: true=save or false=dont save
        :return: true if saved, false if not saved
        """
        current_value = np.sum(metric) / len(metric) if isinstance(metric, list) else metric
        best = np.sum(best) / len(best) if isinstance(best, list) else best
        min_acc = np.sum(min_acc) / len(min_acc) if isinstance(min_acc, list) else min_acc
        if save_weights and ((current_value > min_acc and current_value > best) or epoch == end_epoch):
            self.nn.save_weights(f'{self.weights_path}_{epoch}')
            return True
        else:
            return False

    # Metrics
    def get_acc_categorical(self, type):
        acc_metrics = self._train_acc_metric if type == "train" else self._valid_acc_metric
        return float(acc_metrics.result() * 100.)

    def get_acc_regresion(self, type):
        acc_metrics = self._train_acc_metric if type == "train" else self._valid_acc_metric
        return float((1 - acc_metrics.result()) * 100.)

    # Training and validation steps
    @tf.function
    def train_step(self, x, y, epoch):
        with tf.GradientTape() as tape:
            logits = self.nn(x, training=True)
            loss_value = self._loss_fn(y, logits)

            # learning rate update
            lr = self.schedules["learn_opt"](epoch-1)
            self.optimizer.lr = lr
            if "decay_opt" in self.schedules:
                self.optimizer.weight_decay = self.schedules["decay_opt"](epoch-1)

            grads = tape.gradient(loss_value, self.nn.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.nn.trainable_weights))
            self._train_acc_metric.update_state(y, logits)
        return loss_value, lr

    @tf.function
    def valid_step(self, x, y):
        val_logits = self.nn(x, training=False)
        self._valid_acc_metric.update_state(y, val_logits)


class InferenceModel(ModelManager):
    def __init__(self, model, path_weights, start_epoch, inference_func):
        super().__init__(model, path_weights, start_epoch)
        self.predict = inference_func

    def prob_inference4seg(self, input):
        y_hat = self.nn.predict(input)
        return y_hat

    def mask_inference4seg(self, input):
        y_hat = self.nn.predict(input)
        if len(y_hat.shape) == 2:
            y_hat = np.expand_dims(np.expand_dims(y_hat, 1), 1)
        img = np.ones_like(y_hat, dtype=np.uint8)
        for i in range(y_hat.shape[0]):
            _idx_masks = np.argmax(y_hat[i], axis=2)
            for lab in range(y_hat.shape[3]):
                img[i, ..., lab] = ((_idx_masks == lab) * 1).astype(np.uint8)
        return img


def set_seeds(seed: int = 1234):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
