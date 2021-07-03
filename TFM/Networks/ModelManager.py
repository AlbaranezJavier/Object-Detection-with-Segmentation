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
                acc.append(float(acc_metric.result() * 100.))
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


class InferenceModel(ModelManager):
    def __init__(self, model, dim, path_weights, start_epoch, output_type, inference_type, original_size=None,
                 min_area=None, neighbours=None):
        super().__init__(model, dim, path_weights, start_epoch, output_type)
        self._inference_type = inference_type
        assert self._inference_type == "bbox4seg" or self._inference_type == "bbox4reg" or \
               self._inference_type == "mask4reg" or self._inference_type == "mask4seg", \
            "InferenceModel, init: it must be 'bbox4seg', 'bbox4reg', 'mask4reg' or 'mask4seg'."
        self.predict = self._inference_selector()

        if self._inference_type == "bbox4reg" or self._inference_type == "bbox4seg":
            assert len(original_size) == 3, "InferenceModel, original_size: must be [x,x,x] shape"
            self._scale2original_size = [original_size[0]/dim[1], original_size[1]/dim[2]]
            assert isinstance(min_area, int) and min_area >= 0, "InferenceModel, min_area: must be an int and positive"
            self._min_area = min_area
            assert isinstance(neighbours, int) and neighbours >= 0, \
                "InferenceModel, neightbours: must be an int and positive"
            self._neighbours = neighbours

    def _inference_selector(self):
        if self._inference_type == "bbox4seg":
            return self._bbox_inference4seg
        elif self._inference_type == "bbox4reg":
            return self._bbox_inference4reg
        elif self._inference_type == "mask4seg":
            return self._mask_inference4seg
        elif self._inference_type == "mask4reg":
            return self._mask_inference4reg

    def _bbox_inference4reg(self, input):
        y_hat = self.nn.predict(input)
        bboxs = []
        for i in range(y_hat.shape[0]):
            bboxs_dict = {}
            search_y = [False] * y_hat.shape[2]
            for y in range(1, y_hat.shape[1] - 1):
                search_x = False
                for x in range(1, y_hat.shape[2] - 1):
                    c, find = 0, False
                    while c < y_hat.shape[3] and not find:
                        if y_hat[i, y, x, c] > 0:
                            x_diff = y_hat[i, y, x, c] - y_hat[i, y, x - 1, c]
                            y_diff = y_hat[i, y, x, c] - y_hat[i, y - 1, x, c]
                            if x_diff > 0 and y_diff > 0 and search_x is False and search_y[x] is False:
                                search_x, search_y[x], find = True, True, True
                                bboxs_dict = self._search(y, x, c, y_hat[i, :, :, c], bboxs_dict)

                            if search_x and x_diff <= 0:
                                search_x = False
                            if search_y[x] and y_diff <= 0:
                                search_y[x] = False
                        c += 1
            bboxs.append(list(bboxs_dict.values()))
        return bboxs

    def _search(self, y, x, c, img_c, bboxs):
        center = self._get_max(y, x, img_c)
        if center != [y, x]:
            target_scaled = [round(center[1]*self._scale2original_size[1]),
                             round(center[0]*self._scale2original_size[0]), c]
            target_str = str(target_scaled)
            if bboxs.get(target_str) is None:
                bbox = self._bbox_estimation(center, img_c)
                if np.prod(bbox[2:4]) > self._min_area:
                    bboxs[target_str] = [c, bbox[0:2], [bbox[0]+bbox[2], bbox[1]+bbox[3]]]
        return bboxs

    def _get_max(self, y, x, img_c):
        candidate = [img_c[y, x], y, x]
        max_found = True
        _y, _x = y - 1, x - 1
        while max_found:
            max_found = False
            for f in range(_y, _y + self._neighbours):
                for c in range(_x, _x + self._neighbours):
                    if img_c[f, c] > candidate[0]:
                        max_found = True
                        candidate = [img_c[f, c], f, c]
                        _y, _x = f-1 if f-1 < img_c.shape[0]-3 else img_c.shape[0]-3, c-1
        return candidate[1:3]

    def _bbox_estimation(self, center, img_c):
        y, x = center
        w_mid = round(img_c[y, x] / (img_c[y, x]-img_c[y, x-2]))
        h_mid = round(img_c[y, x] / (img_c[y, x]-img_c[y-2, x]))
        w = w_mid*2 + 1
        h = h_mid*2 + 1

        # Scale to original
        x = round((x-w_mid)*self._scale2original_size[1])
        y = round((y-h_mid)*self._scale2original_size[0])
        w = round(w*self._scale2original_size[1])
        h = round(h*self._scale2original_size[0])

        return [x, y, w, h]

    def _bbox_inference4seg(self, input):
        y_hat = self.nn.predict(input)
        all_bboxs = []
        for i in range(y_hat.shape[0]):
            mask = np.argmax(y_hat[i], axis=2)
            bboxs = []
            for c in range(y_hat.shape[3] - 1):
                countours, _ = cv2.findContours(np.uint8((mask == c) * 1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for countour in countours:
                    x, y, w, h = cv2.boundingRect(countour)
                    if w*h > self._min_area:
                        p1 = [int(x * self._scale2original_size[1]), int(y * self._scale2original_size[0])]
                        p2 = [int((x+w) * self._scale2original_size[1]), int((y+h) * self._scale2original_size[0])]
                        bboxs.append([c, p1, p2])
            all_bboxs.append(bboxs)
        return all_bboxs

    def _mask_inference4seg(self, input):
        y_hat = self.nn.predict(input)
        img = np.ones_like(y_hat, dtype=np.uint8)
        for i in range(y_hat.shape[0]):
            _idx_masks = np.argmax(y_hat[i], axis=2)
            for lab in range(y_hat.shape[3]):
                img[i, ..., lab] = ((_idx_masks == lab) * 1).astype(np.uint8)
        return img

    def _mask_inference4reg(self, input):
        y_hat = self.nn.predict(input)
        img = np.ones_like(y_hat, dtype=np.uint8)
        for i in range(y_hat.shape[0]):
            for c in range(y_hat.shape[3]):
                img[i, ..., c] = ((y_hat[i, ..., c] > 0) * 1).astype(np.uint8)
        return img
