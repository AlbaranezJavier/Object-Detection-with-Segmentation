import time, os, json
from TFM.Statistics.Metrics import Metrics
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

"""
This script contains the classes related to obtaining statistical data from the neural network.
"""

class InferenceStats():
    """
    Analyze the neural network
    """
    def __init__(self, im, dm, output_type, stats_type, p=0.01):
        """
        :param im: InferenceModel
        :param dm: DataManager
        :param p: margin of credibility
        """
        self.im = im
        self.dm = dm
        self.p = p
        self.stats_type = stats_type
        assert self.stats_type == "det" or self.stats_type == "seg", \
            "InferenceStatas, stats_type: stats_type must be 'det' or 'seg'"
        self.output_type = output_type
        assert output_type == "cls" or output_type == "reg", \
            "InferenceStats (24), output_type: it must be 'cls' or 'reg'"

    def _get_labels4det(self, y, lab):
        counter = 0
        while counter < len(y):
            if y[counter][0] != lab:
                y.pop(counter)
                counter -= 1
            counter += 1
        return y

    def _calc_metrics(self, type_set, color_space):
        # Load stats
        stats = [Metrics(self.p, self.stats_type) for _lab in range(self.dm.num_classes + 1)]

        # Get prediction
        start = time.time()
        counter_images = 0
        for idx in range(self.dm.batches_size[type_set] - 1):
            _example_xs = self.dm.batch_x(idx, type_set, color_space)
            _example_ys = self.dm.batch_y_bbox(idx, type_set) if self.stats_type == "det" else self.dm.batch_y(idx,type_set)

            _ys_hat = self.im.predict(_example_xs)

            for i in range(len(_ys_hat)):
                counter_images += 1
                _y_example = _example_ys[i] if self.output_type == "cls" or self.stats_type == "det" else \
                    self.dm.prediction2mask(_example_ys[i])

                for _lab in range(self.dm.label_size[2]):
                    if self.stats_type == "det":
                        _y_label = self._get_labels4det(_ys_hat[i].copy(), _lab)
                        _y_true = self._get_labels4det(_y_example.copy(), _lab)
                        _tp, _fn, _fp, _tn, _correspondencies = stats[_lab].cal_stats(_y_label, _y_true)
                    else:
                        _tp, _fn, _fp, _tn, _correspondencies = stats[_lab].cal_stats(_ys_hat[i, ..., _lab],
                                                                                      _y_example[..., _lab])
                    stats[_lab].update_cumulative_stats()
                    stats[self.dm.label_size[2]].add_cumulative_stats(_tp, _fn, _fp, _tn, _correspondencies)

        # Show statistics
        print(f'Inference time: {time.time() - start}, counter images: {counter_images}')

        return stats

    def _calc_metrics_4csv(self, type_set, color_space, dir):
        # Load stats
        stats = Metrics(self.p, self.stats_type)

        # Get prediction
        start = time.time()
        counter_images = 0
        for idx in range(self.dm.batches_size[type_set] - 1):
            _example_xs = self.dm.batch_x(idx, type_set, color_space)
            _example_ys = self.dm.batch_y_bbox(idx, type_set) if self.stats_type == "det" else self.dm.batch_y(idx,
                                                                                                               type_set)

            csv = open(dir)
            lines = csv.readlines()

            _ys_hat = []
            counter = counter_images
            while counter < len(self.dm.X["valid"]) and counter < counter_images + len(self.im.predict(_example_xs)):
                name = self.dm.X["valid"][counter].split("/")[-1]
                _y_hat = []
                for l in range(2, len(lines)):
                    if lines[l][0:len(name)] == name or lines[l][0:len(name)] == name[:-3]+"jpg":
                        splits = lines[l].split('"')
                        det, d = splits[1:len(splits):2], 0
                        while d < len(det):
                            if det[d][-4:-1] == "0.0":
                                det.pop(d)
                            else:
                                d += 1
                        for d in range(len(det)):
                            det[d] = det[d][1:-1].split(',')[0:-1]
                            for e in range(len(det[d])):
                                det[d][e] = int(float(det[d][e]))
                        for d in det:
                            _y_hat.append([None, [d[0], d[1]], [d[0]+d[3], d[1]+d[2]]])
                        break
                _ys_hat.append(_y_hat)
                counter += 1

            for i in range(len(_ys_hat)):
                counter_images += 1
                _y_example = _example_ys[i] if self.output_type == "cls" or self.stats_type == "det" else \
                    self.dm.prediction2mask(_example_ys[i])

                _tp, _fn, _fp, _tn, _correspondencies = stats.cal_stats(_ys_hat[i].copy(), _y_example.copy())
                stats.update_cumulative_stats()
                stats.cal_complex_stats("basic")
                print(
                    f'Idx {idx}:{i}: precision {stats.stats["precision"]}, recall {stats.stats["recall"]}')
                stats.add_cumulative_stats(_tp, _fn, _fp, _tn, [])


        # Show statistics
        print(f'Inference time: {time.time() - start}, counter images: {counter_images}')

        return stats

    def one_example(self, example, type_set="valid", verbose=1):
        """
        Get statistics of one example, for each class and for all
        :param type_set: valid or train set
        :param example: index of the example
        :param verbose: 0 print stats, 1 print stats and masks
        :return: None
        """
        # Load example
        example_x, rgb = self.dm.x_idx(example, type_set)
        example_y = self.dm.y_idx(example, type_set)

        # Load the model and weigths
        nn = self.im.load4inference()

        # Load stats
        sd = Metrics(self.p, self.stats_type)

        # Get prediction
        start = time.time()
        y_hat = nn.predict(example_x)
        y_masks = self.dm.prediction2mask(y_hat[0])

        # Show statistics
        print(f'Inference time: {time.time() - start}')
        for _lab in range(self.dm.label_size[2]-1):
            print(f'\n =================> Statistics for label {self.dm.labels[_lab]} <=================')
            sd.cal_stats(y_masks[..., _lab], example_y[..., _lab])
            sd.cal_complex_stats("basic")
            sd.print_table("basic")
            sd.update_cumulative_stats()
            if verbose > 1:
                # Show masks
                fig, axs = plt.subplots(2, 2)
                axs[0,0].imshow(y_masks[..., _lab])
                axs[0,0].set_title("Predicted")
                axs[0,0].set_xticks([]), axs[0,0].set_yticks([])
                axs[0,1].imshow(example_y[..., _lab])
                axs[0,1].set_title(f'GT')
                axs[0,1].set_xticks([]), axs[0,1].set_yticks([])
                axs[1,1].imshow(rgb)
                axs[1,1].set_title(f'Original')
                axs[1,1].set_xticks([]), axs[1,1].set_yticks([])
                plt.suptitle(f"Label: {self.dm.labels[_lab]}")
                plt.show()
        print(f'\n =================> Statistics all labels <=================')
        sd.cal_complex_stats("cumulative")
        sd.print_table("cumulative")

    def set(self, type_set="valid", color_space="hsv", tablefmt="grid"):
        """
        Get stats of the set, for each class and for all
        :param type_set: valid or train
        :return: None
        """
        stats = self._calc_metrics(type_set, color_space)

        # Show statistics
        for _lab in range(self.dm.label_size[2]):
            print(f'\n =================> Statistics for label {self.dm.labels[_lab]} <=================')
            stats[_lab].cal_complex_stats("cumulative")
            stats[_lab].print_table("cumulative", tablefmt)
        print(f'\n =================> Statistics all <=================')
        stats[self.dm.label_size[2]].cal_complex_stats("cumulative")
        stats[self.dm.label_size[2]].print_table("cumulative", tablefmt)

    def resume(self, model, type_set="valid", color_space="hsv", from_csv=False, csv_file=" "):
        """
        Get stats of the set, for each class and for all
        :param type_set: valid or train
        :return: None
        """
        if from_csv:
            stats = self._calc_metrics_4csv(type_set, color_space, csv_file)
        else:
            stats = self._calc_metrics(type_set, color_space)

        # Show statistics
        acc = f' - Accuracy: '
        iou = f'\n - IoU: '
        prec = f'\n - Precision: '
        rec = f'\n - Recall: '
        f1 = f'\n - F1 score: '
        if not from_csv:
            for _lab in range(self.dm.label_size[2]):
                stats[_lab].cal_complex_stats("cumulative")
                acc += f'{stats[_lab].stats["accuracy"]} ({self.dm.labels[_lab]}), '
                iou += f'{stats[_lab].stats["iou"]} ({self.dm.labels[_lab]}), '
                prec += f'{stats[_lab].stats["precision"]} ({self.dm.labels[_lab]}), '
                rec += f'{stats[_lab].stats["recall"]} ({self.dm.labels[_lab]}), '
                f1 += f'{stats[_lab].stats["f1"]} ({self.dm.labels[_lab]}), '
            stats[self.dm.label_size[2]].cal_complex_stats("cumulative")
            acc += f'{stats[self.dm.label_size[2]].stats["accuracy"]} (all) %'
            iou += f'{stats[self.dm.label_size[2]].stats["iou"]} (all) %'
            prec += f'{stats[self.dm.label_size[2]].stats["precision"]} (all) %'
            rec += f'{stats[self.dm.label_size[2]].stats["recall"]} (all) %'
            f1 += f'{stats[self.dm.label_size[2]].stats["f1"]} (all) %'
        else:
            stats.cal_complex_stats("cumulative")
            acc += f'{stats.stats["accuracy"]}, '
            iou += f'{stats.stats["iou"]}, '
            prec += f'{stats.stats["precision"]}, '
            rec += f'{stats.stats["recall"]}, '
            f1 += f'{stats.stats["f1"]}, '
            stats.cal_complex_stats("cumulative")
            acc += f'{stats.stats["accuracy"]} (all) %'
            iou += f'{stats.stats["iou"]} (all) %'
            prec += f'{stats.stats["precision"]} (all) %'
            rec += f'{stats.stats["recall"]} (all) %'
            f1 += f'{stats.stats["f1"]} (all) %'
        print(f'Model {model}:')
        print(acc, iou, prec, rec, f1)

class TrainingStats():
    """
    Manage the metrics in training process
    """
    def __init__(self, model_name, specific_weights, start_epoch=None):
        """
        :param mm: ModelManager
        :param dm: DataManager
        :param model_name: string
        :param specific_weights: string
        """
        self.EPOCHS = "epochs"
        self.ACC_T = "acc_train"
        self.ACC_V = "acc_valid"
        self.LOSS = "loss"
        self.TIME = "time"
        self.DATE = "date"
        self.SAVED = "saved"
        self.MODEL = "model"
        self.BEST = "best"
        self.SPEC_WEIGHTS = "specific_weights"
        self.model_name = model_name
        self.specific_weights = specific_weights
        self.start_epoch = start_epoch
        self.path2save = f'{self._path_logs()}{model_name}.json'
        self.data = self._data_struct()
        self._load()

    def _path_logs(self):
        """
        Generate Logs directory if not exist
        :return: string directory
        """
        path = f'./Logs/'
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def _data_struct(self):
        """
        Init json structure
        :return: dictionary
        """
        data = {self.MODEL: self.model_name,
                self.SPEC_WEIGHTS: self.specific_weights,
                self.EPOCHS: [[]],
                self.SAVED: [[]],
                self.LOSS: [[]],
                self.ACC_T: [[]],
                self.ACC_V: [[]],
                self.BEST: 0,
                self.TIME: [[]],
                self.DATE: datetime.now().strftime("%d_%m_%Y__%H_%M_%S")}
        return data

    def _load(self):
        """
        Loads data from a json to continue training
        :return:None
        """
        # Do I want load the data to print the info?
        if self.start_epoch == None:
            if not os.path.exists(self.path2save):
                self.path2save = '.'+self.path2save
            with open(self.path2save) as f:
                self.data = json.load(f)
                f.close()
        # I want to train a new network from scratch or is it a mistake?
        elif os.path.exists(self.path2save) and self.start_epoch == 0:
            _decision = ""
            while _decision != "y" and _decision != "n":
                _decision = input(f'Do you want overwrite "{self.path2save}" (y/n):')
            if _decision == "n":
                _copy = 1
                _path_split = self.path2save.split('.')
                if len(_path_split) > 2:
                    raise Exception("Log path with more of one '.'")
                while os.path.exists(f'{_path_split[0]}{_copy}.{_path_split[1]}'):
                    _copy += 1
                self.path2save = f'{_path_split[0]}{_copy}.{_path_split[1]}'
                print(f'Logs will be saved as "{self.path2save}"')
        # Do I want to continue training a network?
        elif self.start_epoch > 0:
            with open(self.path2save) as f:
                self.data = json.load(f)
                f.close()
            self.data[self.EPOCHS].append([])
            self.data[self.SAVED].append([])
            self.data[self.LOSS].append([])
            self.data[self.ACC_T].append([])
            self.data[self.ACC_V].append([])
            self.data[self.TIME].append([])

    def _save(self):
        """
        Save data as json
        :return: None
        """
        json_file = json.dumps(self.data, separators=(',', ':'))
        with open(self.path2save, "w") as outfile:
            outfile.write(json_file)
            outfile.close()

    def print_data(self, y_lim_epoch=[99.2, 99.6], x_lim_loss=[0, 25], title="Training progression"):
        _training_tries = len(self.data[self.EPOCHS])
        saved_epochs = []
        _size = 0
        for tt in range(_training_tries):
            saved_epochs.append([])
            for s in self.data[self.SAVED][tt]:
                saved_epochs[-1].append(self.data[self.ACC_V][tt][s-1-_size])
            _size += len(self.data[self.ACC_V][tt])
        xs = [self.data[self.EPOCHS], self.data[self.EPOCHS], self.data[self.SAVED]]
        ys = [self.data[self.ACC_T], self.data[self.ACC_V], saved_epochs]
        colors = ["b", "r", "bo"]
        labels = ["Acc_train", "Acc_valid", "Saved points"]

        fig, ax = plt.subplots(2)
        for j in range(_training_tries):
            ax[0].plot(self.data[self.EPOCHS][j], self.data[self.LOSS][j], 'k')
            ax[0].set_xlim(x_lim_loss)
            ax[0].set_ylabel("Loss")
            ax[0].set_title(title)

        lines = []
        for i in range(len(xs)):
            for j in range(len(ys[i])):
                if j==0:
                    ax[1].plot(xs[i][j], ys[i][j], colors[i], label=labels[i])
                else:
                    ax[1].plot(xs[i][j], ys[i][j], colors[i])
                ax[1].set_ylim(y_lim_epoch)
                ax[1].set_xlabel("Epochs")
                ax[1].set_ylabel("Accuracy")
                ax[1].legend()


        plt.tight_layout()

        plt.show()

    def update_values(self, epoch, saved, loss, acc_train, acc_valid, end_time, verbose=1):
        loss = list(loss) if type(loss).__module__ == np.__name__ else float(loss)
        current_valid = np.sum(acc_valid)/len(acc_valid) if isinstance(acc_valid, list) else acc_valid
        best_valid = np.sum(self.data[self.BEST])/len(self.data[self.BEST]) if isinstance(self.data[self.BEST], list) \
            else self.data[self.BEST]

        self.data[self.EPOCHS][-1].append(epoch)
        self.data[self.SAVED][-1].append(epoch) if saved else None
        self.data[self.LOSS][-1].append(loss)
        self.data[self.ACC_T][-1].append(acc_train)
        self.data[self.ACC_V][-1].append(acc_valid)
        self.data[self.BEST] = acc_valid if best_valid < current_valid else self.data[self.BEST]
        self.data[self.TIME][-1].append(end_time)
        self.data[self.DATE] = datetime.now().strftime("%d_%m_%Y__%H_%M_%S")

        # Show metrics
        if verbose == 1:
            print('\r',
                  f'Epoch {epoch}, Train_loss: {loss}, Train_acc: {acc_train}, Valid_acc: {acc_valid}, Time: {end_time}',
                  end='')
            if saved:
                print(f' <= saved', end='')
            print()
        self._save()


