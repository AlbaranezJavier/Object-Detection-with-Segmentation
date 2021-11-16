import json, cv2, os
from pprint import pprint
import numpy as np
from sklearn.model_selection import KFold
import tensorflow as tf
from copy import deepcopy
from glob import glob
from tqdm import tqdm
"""
This script manages the data for training
"""

class CellDatasetLoader:
    TRAIN = "/train"
    TEST = "/test"
    XS = "/xs"
    YS = "/ys"
    AUTOTUNE = tf.data.AUTOTUNE

    def __init__(self, data_path: str, k_fold: int, batch: int, resize: tuple = None):
        self.batch = batch
        self.train_path = data_path + f"/{k_fold}_fold" + self.TRAIN + self.XS + "/*.npy"
        self.test_path = data_path + f"/{k_fold}_fold" + self.TEST + self.XS + "/*.npy"
        self.resize = resize

    def read_npy(self, paths):
        data = []
        for path in paths.numpy():
            path_decoded = path.decode("utf-8")
            path_splited = path_decoded.split("\\")
            redirection = ""
            for splits in path_splited[:-4] + ["data"] + path_splited[-2:-1]:
                redirection += f"{splits}/"
            redirection += path_splited[-1]
            data.append(np.load(redirection)/255)
        return np.array(data).astype(np.float32)

    @tf.function
    def _loader(self, xs_path: str) -> tuple:
        image_tf = tf.py_function(self.read_npy, [xs_path], tf.float32)

        ys_path = tf.strings.regex_replace(xs_path, "xs", "ys")
        mask_tf = tf.py_function(self.read_npy, [ys_path], tf.float32)

        return image_tf, mask_tf

    def get_sets(self, seed: int = 123) -> object:
        train = tf.data.Dataset.list_files(self.train_path, seed=seed)
        train = train.batch(self.batch, drop_remainder=True).map(self._loader, num_parallel_calls=self.AUTOTUNE)
        train = train.prefetch(buffer_size=self.AUTOTUNE)

        test = tf.data.Dataset.list_files(self.test_path, seed=seed)
        test = test.batch(self.batch, drop_remainder=True).map(self._loader, num_parallel_calls=self.AUTOTUNE)
        test = test.prefetch(buffer_size=self.AUTOTUNE)
        return train, test

class DataManager:
    # Constants
    README_JSON = "/readme.json"
    K_FOLD = "k_fold"
    XS_FORMAT = "xs_format"
    YS_FORMAT = "ys_format"
    SPLITS = 5
    TRAIN_TEST = ["train", "test"]
    XS_YS = ["xs", "ys"]
    template_kfold = {"train": {"xs": [],
                                "x_dest": "",
                                "ys": [],
                                "y_dest": ""},
                      "test": {"xs": [],
                               "x_dest": "",
                               "ys": [],
                               "y_dest": ""}
                      }

    @classmethod
    def loadDataset(cls, data_path: str, k_fold: int, batch: int, resize: tuple = None) -> CellDatasetLoader:
        with open(data_path + cls.README_JSON, "r") as reader:
            data = json.load(reader)
            reader.close()
        data["cross_validation"][cls.K_FOLD] = f"{data['cross_validation'][cls.K_FOLD]}: {k_fold} <= selected"
        pprint(data)
        return CellDatasetLoader(data_path, k_fold, batch, resize)

    @classmethod
    def _get_files(cls, xs_path: str, ys_regex: list) -> tuple:
        xs = glob(xs_path)
        ys = []
        for x in xs:
            for y in ys_regex:
                x = x.replace(y[0], y[1])
            ys.append(x)
        return np.array(xs), np.array(ys)

