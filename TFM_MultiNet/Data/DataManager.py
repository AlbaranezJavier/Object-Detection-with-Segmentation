import os, json, cv2, shutil, glob
import numpy as np
from sklearn.model_selection import train_test_split

"""
This script manages the data for training
"""

class DataManager():
    """
    This class contains all the functionality necessary to control the input/output data to the network.
    """

    def __init__(self, rgb_path, labels, label_size, valid_size, batch_size, seed=123, shuffle=True):
        # Managing directories
        self._constants()
        self.rgb_paths, self.gt_paths = self._getpaths(rgb_path, labels)
        _X_train, _X_valid, _Y_train, _Y_valid = train_test_split(self.rgb_paths, self.gt_paths,
                                                                                  test_size=valid_size,
                                                                                  random_state=seed, shuffle=shuffle)
        self.data_size = {"train":len(_X_train), "valid":len(_X_valid)}
        self.X = {"train":_X_train, "valid":_X_valid}
        self.Y = {"train":_Y_train, "valid":_Y_valid}
        self.labels = labels
        self.label_size = label_size
        # Managing batches
        self.batches = {"valid":self._batch_division(_X_valid, batch_size), "train":self._batch_division(_X_train, batch_size)}
        self.batches_size = {"train":len(self.batches["train"]), "valid":len(self.batches["valid"])}

        # Print data info
        print(f'Size: {len(self.rgb_paths)}')
        print(f'Train: {self.data_size["train"]} y valid: {self.data_size["valid"]}')
        print(f'Train batches: {self.batches_size["train"]}, valid batches: {self.batches_size["valid"]}')

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
                        points = np.stack([data[key][self.REG][i][self.SATT][self.ALLX], data[key][self.REG][i][self.SATT][self.ALLY]], axis=1).astype(
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
        _paths = glob.glob(path_images+'/*.jpg') + glob.glob(path_images+'/*.png')
        _unlabeled = []
        names = []
        sizes = []
        for p in _paths:
            if len(_unlabeled) >= limit:
                break
            _filename = p.split('\\')[-1]
            if _filename not in _labeled:
                _img = cv2.resize(cv2.imread(p), (img_size[1], img_size[0]))
                _unlabeled.append(cv2.cvtColor(_img, color_space).astype('float32')/255.)
                names.append(_filename)
                sizes.append(os.stat(p).st_size)
                shutil.copyfile(p, path_images_dest + "/" + _filename)
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
            _img = cv2.cvtColor(_img, color_space).astype('float32') if color_space is not None else _img.astype('float32')
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
    def _label2mask(self, data, label_size, original_size=(720, 1280)):
        """
        Generates a mask with the labeling data
        :param data: labeling
        :param label_size: dimensions
        :return: mask
        """
        img = np.ones(label_size, dtype=np.uint8)
        for lab in range(label_size[2] - 1):
            zeros = np.zeros(original_size, dtype=np.uint8)
            for idx in range(len(data[lab])):
                cv2.fillConvexPoly(zeros, data[lab][idx], 1)
            zeros = cv2.resize(zeros, (label_size[1], label_size[0]), cv2.INTER_NEAREST)
            img[:, :, lab] = zeros.copy()
            img[:, :, label_size[2] - 1] *= np.logical_not(zeros)
        return img

    def prediction2mask(self, prediction):
        """
        From prediction to mask
        :param prediction: prediction [0,1]
        :param label_size: number of masks
        :return: masks
        """
        img = np.ones(self.label_size, dtype=np.uint8)
        _idx_masks = np.argmax(prediction, axis=2)
        for lab in range(self.label_size[2]):
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

