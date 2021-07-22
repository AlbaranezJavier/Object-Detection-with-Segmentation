from TFM.Networks.ModelManager import InferenceModel
from TFM.Data.DataManager import DataManager
import sys, time, os, shutil
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

'''
This script executes the network inference.
'''

if __name__ == '__main__':
    # Variables
    path_images = r"C:\Users\TTe_J\Downloads\Prueba"
    path_images_destination = r"C:\Users\TTe_J\Downloads\new_RGBs"
    path_newlabels = r'C:\Users\TTe_J\Downloads\new_labels.json'
    limit = 12 # <============================= unlabeled image limit
    model = "SNet_3L" # <============ models = HelperNetV1, SNet_5L0, SNet_4L, SNet_3L, SNet_3Lite
    output_type = "cls"  # regression = reg, classification = cls, regression + classficiation = reg+cls
    inference_type = "mask4seg"  # bbox4reg, bbox4seg, mask4reg or mask4seg
    min_area = 3  # <= for bbox4reg
    neighbours = 3  # <= for bbox4reg
    start_epoch = 100 # <============== trained epochs
    color_space = 82 # <====== bgr=None, lab=44, yuv=82, hsv=40, hsl=52
    specific_weights = "synthetic_real_cls_yuv"
    weights_path = f'Weights/{model}/{specific_weights}_epoch'
    labels = ["b", "y", "o_s", "o_b"]
    # input_dims = (1, 720, 1280, 3)
    # input_dims = (1, 513, 1025, 3)
    input_dims = (12, 180, 320, 3)
    # input_dims = (1, 360, 640, 3)
    original_size = (720, 1280, 4)  # <= for bbox4reg

    # Load the model and weigths
    im = InferenceModel(model, input_dims, weights_path, start_epoch, output_type, inference_type, original_size,
                        min_area, neighbours)

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
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # for gpu in gpus:
    #     tf.config.experimental.set_memory_growth(gpu, True)
    #
    # os.environ["CUDA_DEVICE_ORDER"] = '0'
    # from tensorflow.python.saved_model import tag_constants
    # saved_model_loaded = tf.saved_model.load(r"C:\Users\TTe_J\Downloads\exported_fp16", tags=[tag_constants.SERVING])
    # model = saved_model_loaded.signatures['serving_default']
    # for i in range(10):
    #     y_hat = model(tf.constant(unlabed, dtype=tf.uint8))
    #     # y_hat = model(tf.constant(unlabed))
    # for i in range(100):
    #     times = []
    #     start = time.time()
    #     # y_hat = model(tf.constant(unlabed))
    #     y_hat = model(tf.constant(unlabed, dtype=tf.uint8))
    #     times.append(time.time()-start)
    # print("time:", np.mean(times))
    y_hat = im.predict(unlabed)

    # Save as json and show names of the modified files
    vgg = DataManager.mask2vgg(np.round(y_hat).astype(np.uint8), labels, names, sizes, save_path=path_newlabels)
    print(names)
