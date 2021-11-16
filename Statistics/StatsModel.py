import time, os, json
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

"""
This script contains the classes related to obtaining statistical data from the neural network.
"""

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
        path = f'../Logs/'
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

    def update_values(self, epoch, saved, loss, acc_train, acc_valid, end_time, learn_rate, verbose=1):
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
            print(f'\rEpoch {epoch}, Train_loss: {loss}, Learn_rate: {"{:.2E}".format(learn_rate)}, Train_acc: {acc_train}, Valid_acc: {acc_valid}, Time: {end_time}',
                  end='')
            if saved:
                print(f' <= saved')
            print()
        self._save()


