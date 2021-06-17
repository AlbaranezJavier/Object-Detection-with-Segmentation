from TFM.Networks.ModelManager import ModelManager
from TFM.Data.DataManager import DataManager
import sys, time, os, shutil
import numpy as np
import matplotlib.pyplot as plt

'''
This script executes the network inference.
'''

if __name__ == '__main__':
    # Variables
    path_images = r"C:\Users\TTe_J\Downloads\Prueba"
    path_images_destination = r"C:\Users\TTe_J\Downloads\new_RGBs"
    path_newlabels = r'C:\Users\TTe_J\Downloads\new_labels.json'
    limit = 50 # <============================= unlabeled image limit
    model = "Net_5" # <============ models = HelperNetV1, Net_0, Net_1, Net_2
    output_type = "reg+cls"  # regression = reg, classification = cls, regression + classficiation = reg+cls
    background = False
    start_epoch = 1206 # <============== trained epochs
    color_space = 82 # <====== bgr=None, lab=44, yuv=82, hsv=40, hsl=52
    specific_weights = "synthetic_real_yuv_rc"
    weights_path = f'Weights/{model}/{specific_weights}_epoch'
    labels = ["b", "y", "o_s", "o_b"]
    # input_dims = (1, 720, 1280, 3)
    input_dims = (1, 513, 1025, 3)
    # input_dims = (9, 180, 320, 3)

    # Load the model and weigths
    mm = ModelManager(model, input_dims, weights_path, start_epoch, output_type, verbose=1)

    # Overwrite control
    if os.path.exists(path_newlabels):
        over = input(f"WARNING!!! Existing labels will be overwritten (overwrite or stop: o/s) => ")
        if over != 'o':
            print("Stoping")
            sys.exit()
    if os.path.exists(path_images_destination):
        shutil.rmtree(path_images_destination)
    os.makedirs(path_images_destination)


    # Load unlabeled images
    unlabed, names, sizes = DataManager.loadUnlabeled(path_images, path_images_destination, limit, color_space, input_dims[1:3])

    # Predict de una imagen en concreto
    start = time.time()
    y_hat = mm.nn.predict(unlabed)
    print(f'Inference time: {time.time() - start}')

    # Save as json and show names of the modified files
    vgg = DataManager.mask2vgg(np.round(y_hat).astype(np.uint8), labels, names, sizes, save_path=path_newlabels)
    print(names)
