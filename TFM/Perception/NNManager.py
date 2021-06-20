import numpy as np
from TFM.Networks.ModelManager import ModelManager
import os
import cv2

"""
Neural Network Manager
"""


class NNManager:
    def __init__(self, nn_model, epoch, weights_path, original_img_shape):
        self._model = nn_model
        self._original_img_shape = original_img_shape
        self._MODELS = {"Net_5_reg": {"model": "Net_5", "input_dims": (1, 180, 320, 3),
                                      "weights_path": weights_path, "color_space": 82,
                                      "start_epoch": epoch, "output_type": "reg", "verbose": 1}}
        self._nn = self._load_model()

    def _load_model(self):
        return ModelManager(self._MODELS[self._model]["model"], self._MODELS[self._model]["input_dims"],
                            self._MODELS[self._model]["weights_path"], self._MODELS[self._model]["start_epoch"],
                            self._MODELS[self._model]["output_type"], self._MODELS[self._model]["verbose"])

    def _pre_process(self, img):
        _img = cv2.resize(img, (self._MODELS[self._model]["input_dims"][2], self._MODELS[self._model]["input_dims"][1]))
        _img = cv2.cvtColor(_img, self._MODELS[self._model]["color_space"]).astype('float32')
        return np.expand_dims(_img / 255., axis=0)

    def process(self, img):
        assert img is not None, "NNManager, process: img is None"
        _img = self._pre_process(img)
        assert _img.shape[0] == 1 or _img.shape[3] == 4, "NNManager, process(): wrong data shape"

        return self._nn.nn.predict(_img)

    def get_scale2original_size(self):
        input_dims = self._MODELS[self._model]["input_dims"]
        return [self._original_img_shape[0]/input_dims[1], self._original_img_shape[1]/input_dims[2]]

    def get_model_name(self):
        return self._model

    def print_models(self):
        print(self._MODELS)
