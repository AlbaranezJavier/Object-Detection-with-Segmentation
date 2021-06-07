import time
from TFM_MultiNet.Networks.ModelManager import TrainingModel
from TFM_MultiNet.Data.DataManager import DataManager
from TFM_MultiNet.Statistics.StatsModel import TrainingStats
import numpy as np

'''
This script executes the training of the network.
'''

if __name__ == '__main__':
    # Data Variables
    inputs_rgb = [r'C:\Users\TTe_J\Downloads\SyntheticConeDataset(1005)\RightImages',
                  r'C:\Users\TTe_J\Downloads\17-17-05']
    labels = ["b", "y", "o_s", "o_b"]
    # labels = ["binary"]
    label_size = (720, 1280, len(labels)+1)
    # label_size = (513, 1025, len(labels)+1)
    batch_size = 8
    valid_size = .10

    dm = DataManager(inputs_rgb, labels, label_size, valid_size, batch_size)

    # Net Variables
    model = "MNet_0"  # models = HelperNetV1, ..V2, ..V3, Net_0, .._1, .._2
    start_epoch = 101 # <= number epoch trained
    id_copy = "_yuv" # <= logs version? "" => main
    color_space = 82 # <= bgr=None, lab=44, yuv=82, hsv=40, hsl=52
    end_epoch = start_epoch + 100
    learn_opt, learn_reg = 1e-5, 1e-2
    save_weights = True
    min_acc = 99.75
    specific_weights = "synthetic_real"+id_copy
    weights_path = f'Weights/{model}/{specific_weights}_epoch'
    input_dims = (batch_size, 720, 1280, 3)

    tm = TrainingModel(model, input_dims, weights_path, start_epoch, learn_opt, learn_reg)

    ts = TrainingStats(model+id_copy, specific_weights, start_epoch)

    for epoch in range(start_epoch + 1, end_epoch + 1):
        # TRAINING
        losses = np.zeros(len(tm.out_sizes))
        start_time = time.time()
        for idx in range(dm.batches_size["train"] - 1):
            data = dm.batch_x(idx, "train", color_space)
            labels = dm.batch_y(idx, "train")
            losses += np.array(tm.train_step(data, labels))
            print('\r', f'Train_batch: {idx + 1}/{dm.batches_size["train"] - 1}', end='')
        losses /= dm.batches_size["train"]
        train_accs = [float(tm.train_acc_metric[i].result().numpy() * 100.) for i in range(len(tm.out_sizes))]
        [tm.train_acc_metric[i].reset_states() for i in range(len(tm.out_sizes))]

        # VALID
        for idx in range(dm.batches_size["valid"] - 1):
            data = dm.batch_x(idx, "valid", color_space)
            labels = dm.batch_y(idx, "valid")
            tm.valid_step(data, labels)
            print('\r', f'Valid_batch: {idx + 1}/{dm.batches_size["valid"]}         ', end='')
        valid_accs = [float(tm.valid_acc_metric[i].result().numpy()) * 100. for i in range(len(tm.out_sizes))]
        [tm.valid_acc_metric[i].reset_states() for i in range(len(tm.out_sizes))]

        # Print and save the metrics
        end_time = round((time.time() - start_time) / 60, 2)

        is_saved = False
        # Saves the weights of the model if it obtains the best result in validation
        if (valid_accs[-1] > min_acc and valid_accs[-1] > ts.data["best"] and save_weights) or epoch == end_epoch:
            is_saved = True
            tm.nn[-1].save_weights(f'{weights_path}_{epoch}')

        ts.update_values(epoch, is_saved, losses.tolist(), train_accs, valid_accs, end_time, verbose=1)

