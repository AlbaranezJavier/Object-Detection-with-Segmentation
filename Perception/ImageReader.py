import cv2
import os
import glob

"""
Image Reader
"""

class ImageReader:
    def __init__(self, ir_path):
        self._ir_path = ir_path
        self._ONLINE = True if self._ir_path is None else False
        self._paths = None if self._ir_path is None else self._get_paths(self._ir_path)
        self._counter = 0

    def _next_offline(self):
        return cv2.imread(self._paths[self._counter])

    def _get_paths(self, path_images):
        return glob.glob(path_images + '/*.jpg') + glob.glob(path_images + '/*.png')

    def next(self, frame):
        assert self._ir_path is not None or frame is not None, \
            "Image Reader, next(): ir_path and frame are None, one of them cannot be None"
        img = frame if self._ONLINE else self._next_offline()
        self._counter += 1
        return img

    def has_next(self):
        return True if self._counter < len(self._paths) else False
