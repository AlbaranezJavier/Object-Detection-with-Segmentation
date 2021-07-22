import cv2

"""
Visualization System
"""

class Visualization:
    def __init__(self, model_name, vs_on, data_saved_format):
        self._vs_on = vs_on
        self._model_name = model_name
        self._print_mode = self._print_mode_selector(data_saved_format)

    def _print_mode_selector(self, data_saved_format):
        assert data_saved_format == dict or data_saved_format == list, "Visualization, data_saved_format: must be dict or list instance."

        if data_saved_format == dict:
            return self._print_from_dict
        elif data_saved_format == list:
            return self._print_from_list

    def _print_from_dict(self, data, img):
        for target in data.keys():
            img = self._print(data[target][2][0], data[target][2][1], data[target][2][2], data[target][2][3],
                              data[target][0], img)
        return img

    def _print_from_list(self, data, img):
        for c in data:
            for target in c:
                img = self._print(target[2][0], target[2][1], target[2][2], target[2][3], target[0], img)
        return img

    def _print(self, x, y, w, h, c, img):
        img = cv2.rectangle(img, (x, y), (x+w, y+h), (36, 255, 12), 1)
        cv2.putText(img, str(c), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        return img

    def print_rectangles(self, data, img):
        if self._vs_on:
            # Print data over image
            img = self._print_mode(data, img)

            # Show
            cv2.imshow(self._model_name, img)
            cv2.waitKey(0)
