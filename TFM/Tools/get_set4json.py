from TFM.Data.DataManager import DataManager
import cv2, shutil, os, json

"""
This script get lables from json and generate train or valid set json
"""

if __name__ == '__main__':
    # Data Variables
    inputs_rgb = [r'C:\Users\TTe_J\Downloads\SyntheticConeDataset(1005)\RightImages',
                  r'C:\Users\TTe_J\Downloads\17-17-05']
    labels_class = ["b", "y", "o_s", "o_b"]
    label_size = (720, 1280, len(labels_class))
    background = False
    batch_size = 8
    valid_size = .10
    output_type = "cls"  # regression = reg, classification = cls, regression + classficiation = reg+cls
    out_type = "valid" # train or valid


    # Data Manager
    dm = DataManager(inputs_rgb, labels_class, label_size, background, valid_size, batch_size, output_type)
    set_type = "valid" if out_type == "train" else "train"

    for i in range(len(inputs_rgb)):
        # Outputs json
        output_dir = f'{inputs_rgb[i]}/jsons/{out_type}_set.json'

        # Get train set
        train_json = inputs_rgb[i] + "/jsons/train.json"
        with open(train_json) as json_file:
            data = json.load(json_file)
        keys = list(data.keys())
        for t in dm.X[set_type]:
            target = t.split("/")[-1]
            find = False
            counter = 0
            while not find and counter < len(keys):
                if data[keys[counter]]["filename"] == target:
                    find = True
                    data.pop(keys[counter])
                    keys.pop(counter)
                counter += 1
        with open(output_dir, 'w') as outfile:
            json.dump(data, outfile)
