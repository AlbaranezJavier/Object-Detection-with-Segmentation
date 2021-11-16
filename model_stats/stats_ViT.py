from Data.DataManager import DataManager
from Networks.ModelManager import ModelManager, set_seeds
from tqdm import tqdm
from Statistics.Metrics import Metrics
from Networks.ViT import ViT
import tensorflow as tf
import numpy as np
from pprint import pprint

"""
This script generates the metrics of the selected model
"""

if __name__ == '__main__':
    set_seeds(1234)
    # Stats variables
    metrics = [Metrics(p=1e-2, label=["Basophil", 0]),
              Metrics(p=1e-2, label=["Eosinophil", 1]),
              Metrics(p=1e-2, label=["Lymphocyte", 2]),
              Metrics(p=1e-2, label=["Monocyte", 3]),
              Metrics(p=1e-2, label=["Neutrophil", 4]),
              Metrics(p=1e-2, label=["Other", 5])]
    average = Metrics(p=1e-2, label=["Average", -1])
    # Net Variables
    model = "ViT"
    epoch = 506  # <= number epoch trained
    specific_weights = "_cropped_v3_all_512x512"
    input_dims = (32, 72, 72, 3)
    patch_size = 6
    projection_dim = 64

    model = ModelManager(nn=locals()[model](input_dims[1:],
                                            num_classes=6,
                                            patch_size=patch_size,
                                            num_patches=(input_dims[1] // patch_size) ** 2,
                                            projection_dim=projection_dim,
                                            transformer_layers=8,
                                            num_heads=4,
                                            transformer_units=[projection_dim * 2, projection_dim, ],
                                            mlp_head_units=[2048, 1024]),
                         weights_path=f'../Weights/{model}/{specific_weights}_epoch',
                         epoch=epoch)

    # Data Variables
    train, test = DataManager.loadDataset(
        data_path=r"D:\Datasets\Raabin\cropped_v3_all_512x512",
        k_fold=0,
        batch=input_dims[0]
    ).get_sets(seed=123)

    # Test
    labels, predictions = [], []
    for batch_x, batch_y in tqdm(test, desc=f'Test_batch: {epoch}'):
        batch_x = tf.image.resize(batch_x, [72, 72])
        y_hat = model.nn(batch_x, training=False)
        y_hat_npy = y_hat.numpy()
        batch_y_npy = batch_y.numpy()
        for y in range(len(batch_y_npy)):
            gt_idx = int(np.argmax(batch_y_npy[y]))
            y_hat_idx = int(np.argmax(y_hat_npy[y]))
            # Confusion matrix
            labels.append(gt_idx)
            predictions.append(y_hat_idx)
            # Metrix
            for metric in metrics:
                metric.cal_cls_stats(y_hat_idx, gt_idx)

    for metric in metrics:
        metric.cal_complex_stats()
        metric.print_resume()
        average.add4average(metric.basic_stats)

    average.cal_complex_stats()
    average.print_resume()

    # show confusion matrix
    print("\nConfusion matrix")
    pprint(tf.math.confusion_matrix(labels, predictions))
