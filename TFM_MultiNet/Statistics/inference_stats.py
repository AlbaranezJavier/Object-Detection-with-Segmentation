from TFM.Networks.ModelManager import ModelManager
from TFM.Data.DataManager import DataManager
from TFM.Statistics.StatsModel import InferenceStats

if __name__ == '__main__':
    # Statistic Variables
    verbose = 1
    example = 2
    p = 0.01
    function = {1: "one_example", 2: "valid_set", 3: "A, IoU, P, R, F1"}
    select_function = 3
    tablefmt = "grid" # grid or latex
    # Net Variables
    # input_dims = (8, 720, 1280, 3)
    input_dims = (8, 513, 1025, 3)
    model = "Net_3"  # <====================================================== models = HelperNetV1, Net_0, Net_1, Net_2
    start_epoch = 73  # <===================================================== numero de épocas que ya ha entrenado
    color_space = 82 # <= bgr=None, lab=44, yuv=82, hsv=40, hsl=52
    specific_weights = f"synthetic_real_yuv_resize"
    weights_path = f'../Weights/{model}/{specific_weights}_epoch'

    # Data Variables
    inputs_rgb = [r'C:\Users\TTe_J\Downloads\SyntheticConeDataset(1005)\RightImages',
                  r'C:\Users\TTe_J\Downloads\17-17-05']
    labels = ["b", "y", "o_s", "o_b"]
    label_size = (513, 1025, len(labels) + 1)
    # label_size = (720, 1280, len(labels) + 1)
    batch_size = 8
    valid_size = .10

    dm = DataManager(inputs_rgb, labels, label_size, valid_size, batch_size)
    mm = ModelManager(model, input_dims, weights_path, start_epoch)

    sm = InferenceStats(mm, dm, p)

    if function[select_function] == function[1]:
        sm.one_example(example, verbose=verbose)
    elif function[select_function] == function[2]:
        sm.set(color_space=color_space, tablefmt=tablefmt)
    elif function[select_function] == function[3]:
        sm.resume(model, color_space=color_space)
    else:
        print("Error selecting function!!")