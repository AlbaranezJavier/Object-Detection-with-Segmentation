from TFM.Tools.NetManager import ModelManager, DataManager
import sys, time, os, shutil
import numpy as np
import matplotlib.pyplot as plt

'''
This script executes the network inference.
'''

if __name__ == '__main__':
    # Variables
    path_images = r"C:\Users\TTe_J\Downloads\SyntheticConeDataset(1005)\RightImages"
    path_images_destination = r"C:\Users\TTe_J\Downloads\new_RGBs"
    path_json = r'C:\Users\TTe_J\Downloads\train.json'
    path_newlabels = r'C:\Users\TTe_J\Downloads\new_labels.json'
    limit = 50 # <================================================================================ unlabeled image limit
    model = "HelperNetV1" # <=============================================================== models = HelperNetV1, Net_0
    start_epoch = 21 # <================================================================================ trained epochs
    weights_path = f"./Models/{model}/epoch_{start_epoch}"
    labels = ["b", "y", "o_s", "o_b"]
    input_dims = (720, 1280, 3)

    mm = ModelManager()

    # Overwrite control
    if os.path.exists(path_newlabels):
        over = input(f"WARNING!!! Existing labels will be overwritten (overwrite or stop: o/s) => ")
        if over != 'o':
            print("Stoping")
            sys.exit()
    if os.path.exists(path_images_destination):
        shutil.rmtree(path_images_destination)
    os.makedirs(path_images_destination)

    # Load the model and weigths
    mod = mm.load4inference(model, dim=input_dims)
    mod.load_weights(weights_path)

    # Load unlabeled images
    unlabed, names, sizes = DataManager.loadUnlabeled(path_json, path_images, path_images_destination, limit)

    # Predict de una imagen en concreto
    start = time.time()
    y_hat = mod.predict(unlabed)
    print(f'Inference time: {time.time() - start}')

    # Save as json and show names of the modified files
    vgg = DataManager.mask2vgg(np.round(y_hat).astype(np.uint8), labels, names, sizes, save_path=path_newlabels)
    print(names)
