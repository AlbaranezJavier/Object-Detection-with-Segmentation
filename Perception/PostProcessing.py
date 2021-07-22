import numpy as np

"""
Post-Processing System
"""

class PostProcessing:
    def __init__(self, scale2original_size, min_area=25, neighbours=3):
        self._targets = None
        self._min_area = min_area
        self._neighbours = neighbours
        self._scale2original_size = scale2original_size

    def process(self, data):
        assert len(data.shape) == 4, "PostProcessing, process: data dont have 4 dimensions"

        self._targets = {}
        for img in data:
            self._systematic(img)
        return self._targets

    def _systematic(self, img):
        search_y = [False] * img.shape[1]
        for y in range(1, img.shape[0] - 1):
            search_x = False
            for x in range(1, img.shape[1] - 1):
                c, find = 0, False
                while c < img.shape[2] and not find:
                    if img[y, x, c] > 0:
                        x_diff = img[y, x, c] - img[y, x - 1, c]
                        y_diff = img[y, x, c] - img[y - 1, x, c]
                        if x_diff > 0 and y_diff > 0 and search_x is False and search_y[x] is False:
                            search_x, search_y[x], find = True, True, True
                            self._search(y, x, c, img[:, :, c])

                        if search_x and x_diff <= 0:
                            search_x = False
                        if search_y[x] and y_diff <= 0:
                            search_y[x] = False
                    c += 1

    def _search(self, y, x, c, img_c):
        assert self._targets is not None, "PostProcessing, _search: self.targets not initialized"

        center = self._get_max(y, x, img_c)
        if center != [y, x]:
            target_scaled = [round(center[1]*self._scale2original_size[1]),
                             round(center[0]*self._scale2original_size[0]), c]
            target_str = str(target_scaled)
            if self._targets.get(target_str) is None:
                bbox = self._bbox_estimation(center, img_c)
                if np.prod(bbox[2:4]) > self._min_area:
                    self._targets[target_str] = [c, target_scaled[0:2], bbox]



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
                        _y, _x = f-1, c-1
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

