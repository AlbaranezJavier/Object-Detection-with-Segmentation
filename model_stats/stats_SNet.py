from Data.DataManager import DataManager
from Networks.ModelManager import ModelManager, set_seeds
from tqdm import tqdm
from Statistics.Metrics import Metrics
from Networks.SNet import *
import matplotlib.pyplot as plt

"""
This script generates the metrics of the selected model
"""


if __name__ == '__main__':
    set_seeds(1234)
    # Stats variables
    metric = Metrics(p=1e-2,
                     label=["binary", "binary"])
    # Net Variables
    model = "SNet_3L_overfitting"
    epoch = 500  # <= number epoch trained
    specific_weights = "real_prob_v3"
    input_dims = (8, 180, 320, 3)

    model = ModelManager(nn=locals()[model](input_dims),
                      weights_path=f'../Weights/{model}/{specific_weights}_epoch',
                      epoch=epoch)

    # Data Variables
    train, test = DataManager.loadDataset(
        data_path=r"D:\Datasets\Raabin\segmentation_all_320x180",
        k_fold=0,
        batch=input_dims[0]
    ).get_sets(seed=123)

    # Test
    for batch_x, batch_y in tqdm(test, desc=f'Test_batch: {epoch}'):
        y_hat = model.nn(batch_x, training=False)
        y_hat_npy = y_hat.numpy()
        batch_y_npy = batch_y.numpy()
        for y in range(len(batch_y_npy)):
            metric.cal_probseg_stats(y_hat_npy[y, ..., 0], batch_y_npy[y, ..., 0], threshold=0.1)
    metric.cal_complex_stats()
    metric.print_resume()
