from sklearn.model_selection import train_test_split
import time
from datetime import datetime
from TFM.Tools.NetManager import *
import matplotlib.pyplot as plt

'''
This script executes the training of the network.
'''

if __name__ == '__main__':
    # Data Variables
    inputs_rgb = r'C:\Users\TTe_J\Downloads\SyntheticConeDataset(1005)\RightImages'
    inputs_json = r'C:\Users\TTe_J\Downloads\train.json'
    labels = ["b", "y", "o_s", "o_b"]
    label_size = (720, 1280, len(labels)+1)
    batch_size = 8
    valid_size = .10

    dm = DataManager(inputs_rgb, inputs_json, labels, label_size, valid_size, batch_size)

    print(f'Size: {len(dm.rgb_paths)}')
    print(f'Train: {dm.data_size["train"]} y valid: {dm.data_size["valid"]}')
    print(f'Train batches: {dm.batches_size["train"]}, valid batches: {dm.batches_size["valid"]}')

    # Net Variables
    model = "HelperNetV1"  # models = HelperNetV1, Net_0
    start_epoch = 449 # <= numero de epocas que ya ha entrenado
    end_epoch = start_epoch + 180
    learn_opt, learn_reg = 1e-5, 1e-2
    save_weights = True
    min_acc = 99
    weights_path = f'./Models/{model}/epoch_{start_epoch}'
    input_dims = (720, 1280, 3)

    mm = ModelManager()

    # Model selection
    mod, optimizer, loss_fn, train_acc_metric, valid_acc_metric, logdir = mm.load4training(model, dim=input_dims,
                                                                                   learn_opt=learn_opt, learn_reg=learn_reg,
                                                                                   start_epoch=start_epoch)

    print(f'Model {model}_epoch{start_epoch} creado!')
    if start_epoch > 0:
        mod.load_weights(weights_path)
        print(f'Model weights {model}_epoch{start_epoch} loaded!')

    # metrics management
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    now = datetime.now()
    writer = open(logdir + now.strftime("%d_%m_%Y__%H_%M_%S") + ".txt", "a")
    writer.write(f'epoch;loss_train;acc_train;acc_valid;min\n')
    writer.close()

    best = 0

    for epoch in range(start_epoch + 1, end_epoch + 1):
        # TRAINING
        start_time = time.time()
        train_loss, train_acc = 0, 0
        loss_value = 0
        for idx in range(dm.batches_size["train"] - 1):
            data = dm.batch_x(idx, "train")
            labels = dm.batch_y(idx, "train")
            loss_value += mm.train_step(data, labels, mod, loss_fn, optimizer, train_acc_metric)
            print('\r', f'Train_batch: {idx + 1}/{dm.batches_size["train"] - 1}', end='')
        train_loss, train_acc = loss_value / (dm.batches_size["train"] - 1), train_acc_metric.result() * 100.
        train_acc_metric.reset_states()

        # VALID
        for idx in range(dm.batches_size["valid"] - 1):
            data = dm.batch_x(idx, "valid")
            labels = dm.batch_y(idx, "valid")
            mm.valid_step(data, labels, mod, valid_acc_metric)
            print('\r', f'Valid_batch: {idx + 1}/{dm.batches_size["valid"]}         ', end='')
        valid_acc = valid_acc_metric.result() * 100.
        valid_acc_metric.reset_states()

        # Print and save the metrics
        end_time = round((time.time() - start_time) / 60, 2)
        writer = open(logdir + now.strftime("%d_%m_%Y__%H_%M_%S") + ".txt", "a")
        writer.write(f'{epoch};{train_loss};{train_acc};{valid_acc};{end_time}\n')
        writer.close()
        print('\r',
              f'Epoch {epoch}, Train_loss: {train_loss}, Train_acc: {train_acc}, Valid_acc: {valid_acc}, Time: {end_time}',
              end='')

        # Saves the weights of the model if it obtains the best result in validation
        if (valid_acc > min_acc and valid_acc > best and save_weights) or epoch == end_epoch:
            best = valid_acc
            mod.save_weights(f'{weights_path}_{epoch}')
            print(f' <= saved', end='')
        print()

