import os, json, cv2, shutil, glob, time
import numpy as np
from sklearn.model_selection import train_test_split

"""
This script manages the data for training
"""


class DataManager():
    """
    This class contains all the functionality necessary to control the input/output data to the network.
    """

    def __init__(self, rgb_path, labels, label_size, background, valid_size, batch_size, output_type, seed=123, shuffle=True):
        # Managing directories
        self._constants()
        self.output_type = output_type
        self.rgb_paths, self.gt_paths = self._getpaths(rgb_path, labels)
        _X_train, _X_valid, _Y_train, _Y_valid = train_test_split(self.rgb_paths, self.gt_paths,
                                                                  test_size=valid_size,
                                                                  random_state=seed, shuffle=shuffle)
        self.data_size = {"train": len(_X_train), "valid": len(_X_valid)}
        self.X = {"train": _X_train, "valid": _X_valid}
        self.Y = {"train": _Y_train, "valid": _Y_valid}
        self.background = background
        self.num_classes = label_size[2] + 1 if self.background else label_size[2]
        self.labels = labels
        self.label_size = label_size
        # Managing batches
        self.batches = {"valid": self._batch_division(_X_valid, batch_size),
                        "train": self._batch_division(_X_train, batch_size)}
        self.batches_size = {"train": len(self.batches["train"]), "valid": len(self.batches["valid"])}

        # Print data info
        print(f'Data Info\n - Size: {len(self.rgb_paths)}')
        print(f' - Train: {self.data_size["train"]} y valid: {self.data_size["valid"]}')
        print(f' - Train batches: {self.batches_size["train"]}, valid batches: {self.batches_size["valid"]}')
        print(
            f' - Paths: {rgb_path}, train: {self._get_info(rgb_path, "train")}, valid: {self._get_info(rgb_path, "valid")}')

    # Input data
    @classmethod
    def _constants(self):
        self.JSON_PATH = "jsons"
        self.ERROR_PATH = "errors"
        self.MAIN_JSON = "train.json"
        self.REG = "regions"
        self.SATT = "shape_attributes"
        self.RATT = "region_attributes"
        self.ALLX = "all_points_x"
        self.ALLY = "all_points_y"
        self.LAB = "type"
        self.NAME = "filename"

    @classmethod
    def _is_json_valid(self, data):
        """
        Check if data structure is the expected
        :param data: json
        :return: true or false
        """
        if data == {}:
            return True
        try:
            key = list(data.keys())[0]
            filename = data[key][self.NAME]
            regions = data[key][self.REG]
            if data[key][self.REG] != []:
                all_x = data[key][self.REG][0][self.SATT][self.ALLX]
                all_y = data[key][self.REG][0][self.SATT][self.ALLY]
                type = data[key][self.REG][0][self.RATT][self.LAB]
        except KeyError:
            return False
        return True

    @classmethod
    def _get_json(self, images_path):
        """
        Adapt the root folder for this project and get MAIN_JSON data
        :return: MAIN_JSON data
        """
        self._constants()
        # Get dirnames and filenames
        [_, dirnames, filenames] = next(os.walk(images_path, topdown=True))

        # Generate path to save the jsons
        if self.JSON_PATH not in dirnames:
            os.makedirs(images_path + "/" + self.JSON_PATH)
            with open(images_path + "/" + self.JSON_PATH + "/" + self.MAIN_JSON, "x") as writer:
                json.dump({}, writer)
                writer.close()
            print(f"Path {self.JSON_PATH} not detected, directory and empty {self.MAIN_JSON} generated")

        # Check if json structure is compatible
        with open(images_path + "/" + self.JSON_PATH + "/" + self.MAIN_JSON, "r") as reader:
            data = json.load(reader)
            reader.close()

        if not self._is_json_valid(data):
            raise Exception("Structure from json incompatible")

        return data

    def _getpaths(self, img_paths, labels):
        '''
        Obtains the addresses of the input data
        :param json_path: address of the file with the image labels
        :param img_path: address of the folder with all the images
        :param labels: classes to detect
        :return: numpy arrays with the directories and the annotations
        '''
        # Variables
        directories = []
        annotations = []

        for img_path in img_paths:
            _error_path = img_path + "/" + self.JSON_PATH + "/" + self.ERROR_PATH
            if os.path.exists(_error_path):
                shutil.rmtree(_error_path)
            os.makedirs(_error_path)
            data = self._get_json(img_path)

            for key in data.keys():  # each image
                path = data[key][self.NAME]
                _img_path = img_path + '/' + path
                directories.append(_img_path)
                regions = [[] for l in range(len(labels))]
                if len(data[key][self.REG]) > 0:  # could be empty
                    for i in range(len(data[key][self.REG])):  # each region
                        points = np.stack([data[key][self.REG][i][self.SATT][self.ALLX],
                                           data[key][self.REG][i][self.SATT][self.ALLY]], axis=1).astype(
                            int)
                        _is_labels_ok = False
                        for l in range(len(labels)):  # depending label
                            if labels[l] == "binary" or data[key][self.REG][i][self.RATT][self.LAB] == labels[l]:
                                _is_labels_ok = True
                                regions[l].append(points)
                                break
                        if _is_labels_ok == False:  # check if label not exist
                            shutil.copyfile(_img_path, _error_path + '/' + path)
                            print(f"Error in {key}, review label {data[key][self.REG][i][self.RATT][self.LAB]}!!")
                annotations.append(regions)
        if next(os.walk(_error_path, topdown=True))[2] != []:
            exit()
        return np.array(directories), np.array(annotations)

    @classmethod
    def _isUnlabeled(cls, path, dest_path, file, labeled, unlabeled, names, sizes):
        if file not in labeled:
            unlabeled.append(path + "/" + file)
            names.append(file)
            sizes.append(os.stat(unlabeled[-1]).st_size)
            shutil.copyfile(unlabeled[-1], dest_path + "/" + file)
        return unlabeled, names, sizes

    @classmethod
    def loadUnlabeled(cls, path_images, path_images_dest, limit, color_space, img_size):
        data = cls._get_json(path_images)

        _keys = data.keys()
        _labeled = set([data[k]["filename"] for k in _keys])
        _paths = glob.glob(path_images + '/*.jpg') + glob.glob(path_images + '/*.png')
        _unlabeled = []
        names = []
        sizes = []
        times = []
        for p in _paths:
            if len(_unlabeled) >= limit:
                break
            _filename = p.split('\\')[-1]
            if _filename not in _labeled:
                img = cv2.imread(p)
                start = time.time()
                _img = cv2.resize(img, (img_size[1], img_size[0]))
                _unlabeled.append(cv2.cvtColor(_img, color_space).astype('float32') / 255.)
                times.append(time.time()-start)
                names.append(_filename)
                sizes.append(os.stat(p).st_size)
                shutil.copyfile(p, path_images_dest + "/" + _filename)
        print(times)
        print("Mean time preprocces: ", np.mean(times))
        unlabeled = np.array(_unlabeled)
        return unlabeled, names, sizes

    # Management of image batches
    def batch_x(self, idx, step, color_space):
        """
        Load batch X
        :param idx: index of batch
        :param step: could be train or valid
        :param color_space: format of the image: hsv=40, hsl=52, lab=44, yuv=82 or bgr=None.
        :return: array of images with values [0 - 1]
        """
        x = []
        for path in self.X[step][self.batches[step][idx]:self.batches[step][idx + 1]]:
            _img = cv2.imread(path)
            _img = cv2.resize(_img, (self.label_size[1], self.label_size[0]))
            _img = cv2.cvtColor(_img, color_space).astype('float32') if color_space is not None else _img.astype(
                'float32')
            _img /= 255.
            x.append(_img)
        return np.array(x)

    def batch_y(self, idx, step):
        """
        Load batch Y
        :param idx: index of batch
        :param step: could be train or valid
        :return: array of masks [0-1]
        """
        y = [self._label2mask(label, self.output_type) for label in
             self.Y[step][self.batches[step][idx]:self.batches[step][idx + 1]]]
        return np.array(y)

    def batch_y_bbox(self, idx, step):
        """
        Load batch Y
        :param idx: index of batch
        :param step: could be train or valid
        :return: array of labels
        """
        bboxs = []
        for labels in self.Y[step][self.batches[step][idx]:self.batches[step][idx + 1]]:
            bbox = []
            for c in range(len(labels)):
                for polygon in labels[c]:
                    _x, _y, _w, _h = cv2.boundingRect(polygon)
                    bbox.append([c, [_x, _y], [_x+_w, _y+_h]])
            bboxs.append(bbox)
        return bboxs

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
        return self._label2mask(self.Y[step][idx], self.output_type)

    def _batch_division(self, set, batch_size):
        batch_idx = np.arange(0, len(set), batch_size)
        return batch_idx

    # Annotation management
    def _label2mask(self, data, output_type, original_size=(720, 1280)):
        """
        Generates a mask with the labeling data
        :param data: labeling
        :param output_type: boolean, true = mask between [0, 255] with radial gradient or false = mask between [0,1]
        :param original_size: original size mask labels
        :return: mask
        """
        img = np.ones([self.label_size[0], self.label_size[1], self.num_classes], dtype=np.uint8)
        for lab in range(self.label_size[2]):
            zeros = np.zeros(original_size, dtype=np.uint8)
            for idx in range(len(data[lab])):
                if output_type == "reg" or output_type == "reg+cls":
                    _x, _y, _w, _h = cv2.boundingRect(data[lab][idx])
                    _shape = zeros[_y:_y + _h, _x:_x + _w].shape
                    if _shape[0] // 2 != 0 and _shape[1] // 2 != 0:
                        cv2.fillConvexPoly(zeros, data[lab][idx], 255)
                        _v_grad = np.repeat(1 - np.abs(np.linspace(-0.9, 0.9, _shape[1], dtype=np.float16))[None],
                                            _shape[0], axis=0)
                        _h_grad = np.repeat(1 - np.abs(np.linspace(-0.9, 0.9, _shape[0], dtype=np.float16))[None],
                                            _shape[1], axis=0).T
                        _grad_mask = _v_grad * _h_grad
                        _grad_mask[_shape[0] // 2, _shape[1] // 2] = 1.0
                        zeros[_y:_y + _h, _x:_x + _w] = (zeros[_y:_y + _h, _x:_x + _w] * _grad_mask).astype(np.uint8)
                else:
                    cv2.fillConvexPoly(zeros, data[lab][idx], 1)
            zeros = cv2.resize(zeros, (self.label_size[1], self.label_size[0]), cv2.INTER_NEAREST)
            img[:, :, lab] = zeros.copy()
            if self.background:
                img[:, :, self.num_classes-1] *= np.logical_not(zeros)
        return img

    def prediction2mask(self, prediction):
        """
        From prediction to mask
        :param prediction: prediction [0,1]
        :return: masks
        """

        if self.output_type == "cls":
            img = np.ones([self.label_size[0], self.label_size[1], self.num_classes], dtype=np.uint8)
            _idx_masks = np.argmax(prediction, axis=2)
            for lab in range(self.label_size[2]):
                img[..., lab] = ((_idx_masks == lab) * 1).astype(np.uint8)
            return img
        elif self.output_type == "reg":
            img = (prediction > 0)*1
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

    # Info data for training
    def _get_info(self, directories, step):
        """
        Count the number of images for each directory, differentiating between training and validation.
        :param directories: directories
        :param step: valid or train
        :return: train array, valid array
        """
        counter = [0 for i in directories]
        for path in self.X[step]:
            img_dir = path.split('/')[0]
            for i in range(len(directories)):
                if img_dir == directories[i]:
                    counter[i] += 1
        return counter
