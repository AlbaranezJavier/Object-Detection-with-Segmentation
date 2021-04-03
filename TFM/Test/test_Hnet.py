import unittest, time, cv2
from ..Tools.NetManager import DataManager, ModelManager
from ..Tools.StatisticalData import Statistics
import matplotlib.pyplot as plt

"""
Tests related to the help network
"""

class tests_Hnet(unittest.TestCase):
    def test_hnet_v1(self):
        """
        Load an example and its groundthruth and display the statistical data.
        If verbose = 0, no print nothing
        If verbose = 1, print statistical table
        If verbose = 2, print predicted mask and gt mask adn statistical table
        :return:check if the difference is not greater than a threshold.
        """
        # Statistic Variables
        _verbose = 2
        _example = 2
        _sd = Statistics(p=0.01)
        _threshold_test = 23
        # Net Variables
        _input_dims = (720, 1280, 3)
        _model = "HelperNetV1" # <========================================================== models = HelperNetV1, Net_0
        _start_epoch = 517 # <===================================================== numero de épocas que ya ha entrenado
        _path_weights = f'../Models/{_model}/epoch_{_start_epoch}'
        _path_weights = f'D:\Work\Repositorios\JaviProject\TFM\Models\{_model}\epoch_{_start_epoch}'

        # Data Variables
        _inputs_rgb = r'C:\Users\TTe_J\Downloads\SyntheticConeDataset(1005)\RightImages'
        _inputs_json = r'C:\Users\TTe_J\Downloads\train.json'
        _labels = ["b", "y", "o_s", "o_b"]
        _label_size = (720, 1280, len(_labels) + 1)
        _batch_size = 8
        _valid_size = .10

        _dm = DataManager(_inputs_rgb, _inputs_json, _labels, _label_size, _valid_size, _batch_size)
        _mm = ModelManager()

        # Print data info
        if _verbose > 0:
            print(f'Size: {len(_dm.rgb_paths)}')
            print(f'Train: {_dm.data_size["train"]} y valid: {_dm.data_size["valid"]}')
            print(f'Train batches: {_dm.batches_size["train"]}, valid batches: {_dm.batches_size["valid"]}')

        # Load example
        _example_x, _rgb = _dm.x_idx(_example, "valid")
        _example_y = _dm.y_idx(_example, "valid")

        # Load the model and weigths
        _mod = _mm.load4inference(_model, dim=_input_dims)
        _mod.load_weights(_path_weights)

        # Get prediction
        start = time.time()
        _y_hat = _mod.predict(_example_x)
        _y_masks = _dm.prediction2mask(_y_hat[0], _dm.label_size)

        if _verbose > 0:
            # Show statistics
            print(f'Inference time: {time.time() - start}')
            for _lab in range(_dm.label_size[2]-1):
                print(f'=================> Statistics for label {_labels[_lab]} <=================')
                _sd.cal_basic_stats(_y_masks[..., _lab], _example_y[..., _lab])
                _sd.cal_complex_stats("basic")
                _sd.print_table("basic")
                _sd.update_cumulative_stats()
                if _verbose > 1:
                    # Show masks
                    fig, axs = plt.subplots(2, 2)
                    axs[0,0].imshow(_y_masks[..., _lab])
                    axs[0,0].set_title("Predicted")
                    axs[0,0].set_xticks([]), axs[0,0].set_yticks([])
                    axs[0,1].imshow(_example_y[..., _lab])
                    axs[0,1].set_title(f'GT')
                    axs[0,1].set_xticks([]), axs[0,1].set_yticks([])
                    axs[1,1].imshow(_rgb)
                    axs[1,1].set_title(f'Original')
                    axs[1,1].set_xticks([]), axs[1,1].set_yticks([])
                    plt.suptitle(f"Label: {_labels[_lab]}")
                    plt.show()
            print(f'=================> Statistics all labels <=================')
            _sd.cal_complex_stats("cumulative")
            _sd.print_table("cumulative")

        # Make test
        self.assertLess(0, _threshold_test)


if __name__ == '__main__':
    unittest.main()
