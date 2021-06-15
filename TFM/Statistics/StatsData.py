import json, cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class StatsData:
    def __init__(self, paths_json):
        self.jsons = self._get_data(paths_json)
        self.REG = "regions"
        self.SATT = "shape_attributes"
        self.RATT = "region_attributes"
        self.ALLX = "all_points_x"
        self.ALLY = "all_points_y"
        self.LAB = "type"
        self.NAME = "filename"

    def _get_data(self, paths_json):
        """
        Read data from json
        :param paths_json: directories
        :return: dictionary
        """
        data = []
        for path in paths_json:
            with open(path, "r") as reader:
                data.append(json.load(reader))
                reader.close()
        return data

    def _get_area(self, region):
        """
        Get area from countour in data
        :return: value
        """
        _x = np.array(region[self.SATT][self.ALLX])
        _y = np.array(region[self.SATT][self.ALLY])
        _countours = np.zeros((len(_x) + len(_y),), dtype=np.int32)
        _countours[0::2] = _x
        _countours[1::2] = _y
        return cv2.contourArea(_countours.reshape((len(_countours)//2, 1, 2)))

    def analyze(self, verbose=1):
        """
        Get stats of the json: number of images, number of objects, number of instances per class, mean area per class
        :verbose: 0 = print console, 1 = print grafics, 2 = save grafics
        :return: None
        """
        n_images = 0
        n_empty_images = 0
        n_objects = 0
        objects_img = 0
        classes = dict()

        for data in self.jsons:
            for key in data.keys():
                n_images += 1
                _objects = 0
                if not data[key][self.REG]:
                    n_empty_images += 1
                for region in data[key][self.REG]:
                    n_objects += 1
                    _objects += 1
                    _lab = region[self.RATT][self.LAB]
                    if classes.get(_lab) is not None:
                        classes[_lab]["instances"] += 1
                        classes[_lab]["area"] += self._get_area(region)
                    else:
                        classes[_lab] = {"instances": 1, "area": self._get_area(region)}
                objects_img = round((objects_img + _objects)/2, 2) if objects_img != 0 else _objects
        for _lab in classes.keys():
            classes[_lab]["area"] /= classes[_lab]["instances"]

        # Show info
        if verbose == 0:
            print(f'Number of images: {n_images}\n'
                  f'Number of objects: {n_objects}\n'
                  f'Info er class: \n')
            for cls in classes:
                print(f' - Class {cls}: instances = {classes[cls]["instances"]}, area = {classes[cls]["area"]}')

        # Save graffics
        if verbose == 1:
            fig, ax = plt.subplots(2)
            labels = ["N. empty\nimages", "Mean objects\nper image", "N. objects", "N. images"]

            y = np.arange(len(labels))  # the label locations
            x = [n_empty_images, objects_img, n_objects, n_images]
            height = 0.35  # the width of the bars

            info = ax[0].barh(y, x, align='center', height=height)

            # Add some text for labels, title and custom x-axis tick labels, etc.
            [ax[0].text(x[i], info[i].xy[1], x[i], color='black') for i in range(len(x))]
            ax[0].set_xlabel('Counter')
            ax[0].set_xlim((0, np.max(x) * 1.1))
            ax[0].set_title('Dataset information')
            ax[0].set_yticks(y)
            ax[0].set_yticklabels(labels)

            labels = list(classes.keys())
            instances = [classes[cls]["instances"] for cls in classes]
            areas = [classes[cls]["area"] for cls in classes]

            x = np.arange(len(labels))  # the label locations
            width = 0.35  # the width of the bars


            instances_bar = ax[1].bar(x - width / 2, instances, width, label='Instances')
            areas_bar = ax[1].bar(x + width / 2, areas, width, label='Area (px)')

            # Add some text for labels, title and custom x-axis tick labels, etc.
            [ax[1].text(instances_bar[i].xy[0], instances[i], instances[i], color='black') for i in range(len(instances))]
            [ax[1].text(areas_bar[i].xy[0], areas[i], round(areas[i], 2), color='black') for i in range(len(areas))]
            ax[1].set_ylabel('Counter/Mean Pixels')
            ax[1].set_ylim((0, np.max([instances, areas])*1.2))
            ax[1].set_xlabel('Classes')
            ax[1].set_title('Class information')
            ax[1].set_xticks(x)
            ax[1].set_xticklabels(labels)
            ax[1].legend()

            fig.tight_layout()
            # plt.savefig()
            plt.show()

if __name__ == '__main__':
    paths_json = [r"C:\Users\TTe_J\Downloads\17-17-05\jsons\train.json",
                  r"C:\Users\TTe_J\Downloads\SyntheticConeDataset(1005)\RightImages\jsons\train.json"]
    sd = StatsData(paths_json)
    sd.analyze(verbose=1)