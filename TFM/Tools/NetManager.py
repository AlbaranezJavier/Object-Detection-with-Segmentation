import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
import os, json, cv2, shutil
from TFM.NetStructure.net_0 import Net_0
from TFM.NetStructure.HNet import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

'''
This script contains all the necessary methods for training and inference processes.
'''


class DataManager:
    """
    This class contains all the functionality necessary to control the input/output data to the network.
    """

    def __init__(self, rgb_path, gt_path, labels, label_size, valid_size, batch_size, seed=123, shuffle=True):
        # Managing directories
        self.rgb_paths, self.gt_paths = self._getpaths(gt_path, rgb_path, labels)
        _X_train, _X_valid, _Y_train, _Y_valid = train_test_split(self.rgb_paths, self.gt_paths,
                                                                                  test_size=valid_size,
                                                                                  random_state=seed, shuffle=shuffle)
        self.data_size = {"train":len(_X_train), "valid":len(_X_valid)}
        self.X = {"train":_X_train, "valid":_X_valid}
        self.Y = {"train":_Y_train, "valid":_Y_valid}
        self.label_size = label_size
        # Managing batches
        self.batches = {"valid":self._batch_division(_X_valid, batch_size), "train":self._batch_division(_X_train, batch_size)}
        self.batches_size = {"train":len(self.batches["train"]), "valid":len(self.batches["valid"])}

    # Input data
    def _getpaths(self, json_path, img_path, labels):
        '''
        Obtains the addresses of the input data
        :param json_path: address of the file with the image labels
        :param img_path: address of the folder with all the images
        :param labels: classes to detect
        :return: numpy arrays with the directories and the annotations
        '''
        REG, SATT, RATT, ALLX, ALLY, LAB, NAME = "regions", "shape_attributes", "region_attributes", "all_points_x", "all_points_y", "type", "filename"

        with open(json_path) as i:
            data = json.load(i)
            i.close()

        directories = []
        annotations = []

        for key in data.keys():  # each image
            path = data[key][NAME].split('-')
            if len(path) != 1:
                directories.append(img_path + '/' + path[0] + '/' + data[key][NAME])
            else:
                directories.append(img_path + '/' + path[0])
            regions = [[] for l in range(len(labels))]
            if len(data[key][REG]) > 0:  # could be empty
                for i in range(len(data[key][REG])):  # each region
                    points = np.stack([data[key][REG][i][SATT][ALLX], data[key][REG][i][SATT][ALLY]], axis=1).astype(
                        int)
                    _is_labels_ok = False
                    for l in range(len(labels)):  # depending label
                        if labels[l] == "binary" or data[key][REG][i][RATT][LAB] == labels[l]:
                            _is_labels_ok = True
                            regions[l].append(points)
                            break
                    if _is_labels_ok == False:  # check if label not exist
                        print(f"Error in {key}, review label {data[key][REG][i][RATT][LAB]}!!")
            annotations.append(regions)
        return np.array(directories), np.array(annotations)

    def _doNothing(self):
        return 0

    def _getInto(self, path, dest_path, labeled, unlabeled, names, sizes, limit, function=_doNothing):
        stop = False
        [_, dirnames, filenames] = next(os.walk(path, topdown=True))
        for folder in dirnames:
            unlabeled, names, sizes, stop = self._getInto(path + "/" + folder, dest_path, labeled, unlabeled, names,
                                                          sizes,
                                                          limit, function=function)
            if stop:
                break
        for file in filenames:
            if len(unlabeled) >= limit:
                stop = True
                break
            unlabeled, names, sizes = function(path, dest_path, file, labeled, unlabeled, names, sizes)
        return unlabeled, names, sizes, stop

    def _isUnlabeled(self, path, dest_path, file, labeled, unlabeled, names, sizes):
        if file not in labeled:
            unlabeled.append(path + "/" + file)
            names.append(file)
            sizes.append(os.stat(unlabeled[-1]).st_size)
            shutil.copyfile(unlabeled[-1], dest_path + "/" + file)
        return unlabeled, names, sizes

    @classmethod
    def loadUnlabeled(self, path_json, path_images, path_images_dest, limit):
        with open(path_json) as i:
            data = json.load(i)
            i.close()

        labeled = set([label.split('.')[0] + ".jpg" for label in data.keys()])
        unlabeled = []
        names = []
        sizes = []

        unlabeled, names, sizes, _ = self._getInto(path_images, path_images_dest, labeled, unlabeled, names, sizes,
                                                   limit,
                                                   self._isUnlabeled)

        unlabeled = np.array([cv2.imread(unlab).astype('float32') for unlab in unlabeled]) / 255.

        return unlabeled, names, sizes

    # Management of image batches
    def batch_x(self, idx, step):
        """
        Load batch X
        :param idx: index of batch
        :param step: could be train or valid
        :return: array of images with values [0 - 1]
        """
        x = [cv2.imread(path).astype('float32') for path in self.X[step][self.batches[step][idx]:self.batches[step][idx + 1]]]
        return np.array(x) / 255.

    def batch_y(self, idx, step):
        """
        Load batch Y
        :param idx: index of batch
        :param step: could be train or valid
        :return: array of masks [0-1]
        """
        y = [self._label2mask(label, self.label_size) for label in self.Y[step][self.batches[step][idx]:self.batches[step][idx + 1]]]
        return np.array(y)

    def x_idx(self, idx, step):
        """
        Get one example by index
        :param idx: example index
        :param step: training or valid
        :return: mask
        """
        _img = cv2.imread(self.X[step][idx])
        _rgb = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
        return _img.reshape((1, _img.shape[0], _img.shape[1], _img.shape[2])) / 255., _rgb

    def y_idx(self, idx, step):
        """
        Get one example by index
        :param idx: example index
        :param step: trining or valid
        :return: mask
        """
        return self._label2mask(self.Y[step][idx], self.label_size)

    def _batch_division(self, set, batch_size):
        batch_idx = np.arange(0, len(set), batch_size)
        return batch_idx

    # Annotation management
    def _label2mask(self, data, label_size):
        """
        Generates a mask with the labeling data
        :param data: labeling
        :param label_size: dimensions
        :return: mask
        """
        img = np.ones(label_size, dtype=np.uint8)
        for lab in range(label_size[2] - 1):
            zeros = np.zeros(label_size[0:2], dtype=np.uint8)
            for idx in range(len(data[lab])):
                cv2.fillConvexPoly(zeros, data[lab][idx], 1)
            img[:, :, lab] = zeros.copy()
            img[:, :, label_size[2] - 1] *= np.logical_not(zeros)
        return img

    def prediction2mask(self, prediction, label_size):
        """
        From prediction to mask
        :param prediction: prediction [0,1]
        :param label_size: number of masks
        :return: masks
        """
        img = np.ones(label_size, dtype=np.uint8)
        _idx_masks = np.argmax(prediction, axis=2)
        for lab in range(label_size[2]):
            img[..., lab] = ((_idx_masks == lab)*1).astype(np.uint8)
        return img

    @classmethod
    def mask2vgg(self, masks, labels, names, sizes, save_path=None):
        """
        Generates labeling data from a mask
        :param masks: 1 to N channels
        :param labels: classes
        :param names: file name
        :param sizes: file size
        :param save_path: None = not saving, other case = saving as json
        :return: json format
        """
        file = {}
        for i in range(len(masks)):
            regions = []
            counter = 0
            mask = np.argmax(masks[i], axis=2)
            for m in range(masks[i].shape[2] - 1):
                contours, _ = cv2.findContours(np.uint8((mask == m) * 1), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for c in range(len(contours)):
                    # At least three points to form a polygon
                    countourX = []
                    countourY = []
                    if len(contours[c][:, :, 0]) > 2:
                        countourX = contours[c][:, :, 0][:, 0].tolist()
                        countourY = contours[c][:, :, 1][:, 0].tolist()

                    regions.append({
                        "shape_attributes": {
                            "name": "polygon",
                            "all_points_x": countourX,
                            "all_points_y": countourY
                        },
                        "region_attributes": {
                            "type": labels[m]
                        }
                    })
                    counter += 1
            file[names[i]] = {

                "filename": names[i],
                "size": sizes[i],
                "regions": regions,
                "file_attributes": {}
            }

        if save_path != None:
            json_file = json.dumps(file, separators=(',', ':'))
            with open(save_path, "w") as outfile:
                outfile.write(json_file)
                outfile.close()
        return file


class ModelManager:
    """
    This class manages the neural models
    """

    def load4training(self, model, dim, learn_opt, learn_reg, start_epoch):
        inputs = Input(shape=dim)
        if model == "HelperNetV1":
            mod = Model(inputs, HelperNetV1(inputs, learn_reg))
        elif model == "HelperNetV2":
            mod = Model(inputs, HelperNetV2(inputs, learn_reg))
        elif model == "Net_0":
            mod = Model(inputs, Net_0(inputs, learn_reg))
        else:
            print("ERROR load_mod")
            print(exit)
            exit()
        logdir = f'./Logs/{model}_{start_epoch}/'
        optimizer = RMSprop(learn_opt)
        loss_fn = CategoricalCrossentropy(from_logits=False)
        train_acc_metric = CategoricalAccuracy()
        valid_acc_metric = CategoricalAccuracy()
        mod.summary()
        return mod, optimizer, loss_fn, train_acc_metric, valid_acc_metric, logdir

    def load4inference(self, model, dim):
        inputs = Input(shape=dim)
        if model == "HelperNetV1":
            mod = Model(inputs, HelperNetV1(inputs))
        elif model == "HelperNetV2":
            mod = Model(inputs, HelperNetV2(inputs))
        elif model == "Net_0":
            mod = Model(inputs, Net_0(inputs))
        else:
            print("ERROR load_mod")
            print(exit)
            exit()
        mod.summary()
        return mod

    # Training and validation steps
    @tf.function
    def train_step(self, x, y, model, loss_fn, optimizer, train_acc_metric):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss_value = loss_fn(y, logits)
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        train_acc_metric.update_state(y, logits)
        return loss_value

    @tf.function
    def valid_step(self, x, y, model, valid_acc_metric):
        val_logits = model(x, training=False)
        valid_acc_metric.update_state(y, val_logits)
