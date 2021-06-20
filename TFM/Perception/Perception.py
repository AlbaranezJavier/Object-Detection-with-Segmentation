import numpy as np
import cv2
from TFM.Perception.ImageReader import ImageReader
from TFM.Perception.NNManager import NNManager
from TFM.Perception.PostProcessing import PostProcessing
from TFM.Perception.Visualization import Visualization

"""
Perception system
"""


class Perception:
    def __init__(self, original_img_shape, nn_model, nn_epoch, nn_weights_path, data_saved_format, vs_on, ir_path=None):
        # Variables
        self._original_img_shape = original_img_shape
        self._ir_path = ir_path
        self._nn_model = nn_model
        self._nn_epoch = nn_epoch
        self._nn_weights_path = nn_weights_path
        self._vs_on = vs_on
        # Objects
        self._ir = ImageReader(self._ir_path)
        self._nm = NNManager(self._nn_model, self._nn_epoch, self._nn_weights_path, self._original_img_shape)
        self._scale2original_size = self._nm.get_scale2original_size() # <= Variable

        self._pp = PostProcessing(self._scale2original_size)
        self._vs = Visualization(self._nn_model, self._vs_on, data_saved_format)


    def run(self, frame=None):
        while self._ir.has_next():
            img = self._ir.next(frame)
            out = self._nm.process(img)
            info = self._pp.process(out)
            self._vs.print_rectangles(info, img)

    def resume(self):
        print(f'Variables:\noriginal_img_shape = {self._original_img_shape}\nir_path = {self._ir_path}\n' +
              f'nn_model = {self._nn_model}\nvs_on = {self._vs_on}')
