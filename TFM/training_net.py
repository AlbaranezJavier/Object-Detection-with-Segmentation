import time
from TFM.Networks.ModelManager import TrainingModel
from TFM.Data.DataManager import DataManager
from TFM.Statistics.StatsModel import TrainingStats
import numpy as np

'''
This script executes the training of the network.
'''

if __name__ == '__main__':
    # Data Variables
    inputs_rgb = [r'C:\Users\TTe_J\Downloads\SyntheticConeDataset(1005)\RightImages',
                  r'C:\Users\TTe_J\Downloads\17-17-05']
    # labels_class = ["b", "y", "o_s", "o_b"]
    labels_class = ["binary"]
    # label_size = (720, 1280, len(labels_class))
    label_size = (180, 320, len(labels_class))
    background = True
    # label_size = (513, 1025, len(labels))
    batch_size = 8
    valid_size = .10
    output_type = "cls" # regression = reg, classification = cls, regression + classficiation = reg+cls

    dm = DataManager(inputs_rgb, labels_class, label_size, background, valid_size, batch_size, output_type)

    # Net Variables
    model = "Net_5"  # models = HelperNetV1, ..V2, ..V3, Net_0, .._1, .._2, .._3, .._4, .._5, .._6, MgNet_0
    start_epoch = 102 # <= number epoch trained
    id_copy = "_cls_yuv_binary" # <= logs version? "" => main
    color_space = 82 # <= bgr=None, lab=44, yuv=82, hsv=40, hsl=52
    end_epoch = start_epoch + 100
    learn_opt, learn_reg = 1e-4, 1e-1
    save_weights = True
    min_acc = 99.75
    specific_weights = "synthetic_real"+id_copy
    weights_path = f'Weights/{model}/{specific_weights}_epoch'
    input_dims = (batch_size, label_size[0], label_size[1], 3)

    tm = TrainingModel(model, input_dims, weights_path, start_epoch, learn_opt, learn_reg, output_type)

    ts = TrainingStats(model+id_copy, specific_weights, start_epoch)

    for epoch in range(start_epoch + 1, end_epoch + 1):
        # TRAINING
        start_time = time.time()
        loss_value = np.zeros(2) if output_type == "reg+cls" else 0
        for idx in range(dm.batches_size["train"] - 1):
            data = dm.batch_x(idx, "train", color_space)
            labels = dm.batch_y(idx, "train")
            loss_value += tm.train(data, labels)
            print('\r', f'Train_batch: {idx + 1}/{dm.batches_size["train"] - 1}', end='')
        train_acc = tm.get_acc("train")

        # VALID
        for idx in range(dm.batches_size["valid"] - 1):
            data = dm.batch_x(idx, "valid", color_space)
            labels = dm.batch_y(idx, "valid")
            tm.valid(data, labels)
            print('\r', f'Valid_batch: {idx + 1}/{dm.batches_size["valid"]}         ', end='')
        valid_acc = tm.get_acc("valid")

        # Print and save the metrics
        end_time = round((time.time() - start_time) / 60, 2)

        # Saves the weights of the model if it obtains the best result in validation
        is_saved = tm.save_best(ts.data["best"], valid_acc, min_acc, epoch, end_epoch, save_weights, weights_path)

        ts.update_values(epoch, is_saved, loss_value, train_acc, valid_acc, end_time, verbose=1)

