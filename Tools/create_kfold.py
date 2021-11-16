from copy import deepcopy
from glob import glob
from tqdm import tqdm
import json, cv2, os
from pprint import pprint
import numpy as np
from sklearn.model_selection import KFold
import shutil


def createDataset(xs_path: str, ys_replace: list, destination_path: str, k_fold: int) -> tuple:
    template_kfold = {"train": {"xs": [],
                                "x_dest": "",
                                "ys": [],
                                "y_dest": ""},
                      "test": {"xs": [],
                               "x_dest": "",
                               "ys": [],
                               "y_dest": ""}
                      }

    # Creating directories in destination
    if len(os.listdir(path=destination_path)) > 0:
        print("Destination_path is not empty, first delete all")
        exit()
    k_folds = []
    data_xs = f"{destination_path}/data/xs"
    data_ys = f"{destination_path}/data/ys"
    for i in range(k_fold):
        k_folds.append(deepcopy(template_kfold))
        directory: str = f"{destination_path}/{i}_fold"
        os.mkdir(directory)
        if i == 0:
            os.mkdir(f"{destination_path}/data")
            os.mkdir(f"{destination_path}/data/xs")
            os.mkdir(f"{destination_path}/data/ys")
        for j in ["train", "test"]:
            directory_j: str = f"{directory}/{j}"
            os.mkdir(directory_j)
            for k in ["xs", "ys"]:
                directory_k: str = f"{directory_j}/{k}"
                k_folds[i][j][f"{k[0]}_dest"] = directory_k
                os.mkdir(directory_k)

    # Splitting data by k-fold, train-test and xs-ys
    xs, ys = get_files(xs_path, ys_replace)
    skf = KFold(n_splits=k_fold, random_state=123, shuffle=True)
    fold = 0
    for train_index, test_index in skf.split(xs, ys):
        k_folds[fold]["train"]["xs"] = xs[train_index]
        k_folds[fold]["test"]["xs"] = xs[test_index]
        k_folds[fold]["train"]["ys"] = ys[train_index]
        k_folds[fold]["test"]["ys"] = ys[test_index]
        fold += 1

    return k_folds, data_xs, data_ys


def get_files(xs_path: str, ys_regex: list) -> tuple:
    xs = glob(xs_path)
    ys = []
    for x in xs:
        for y in ys_regex:
            x = x.replace(y[0], y[1])
        ys.append(x)
    return np.array(xs), np.array(ys)


def create(destination_path: str, k_folds: list, data_xs: str, data_ys: str, new_size: tuple, name: str,
           summary: str, func_x, func_y):
    # Create readme.json
    cross_validation = {"k_fold": len(k_folds),
                        "n_train_files": len(k_folds[0]["train"]["xs"]),
                        "n_test_files": len(k_folds[0]["test"]["xs"]),
                        "n_files": len(k_folds[0]["train"]["xs"]) + len(k_folds[0]["test"]["xs"])}
    x_values = {"format": "npy",
                "shape": np.load(k_folds[0]["train"]["xs"][0]).shape}
    y_values = {"format": "npy",
                "shape": np.load(k_folds[0]["train"]["ys"][0]).shape}
    add_readme(destination_path, name, summary, cross_validation, x_values, y_values)

    # Processing block
    for k in range(len(k_folds)):
        for tt in ["train", "test"]:
            for xy in ["xs", "ys"]:
                if k == 0:
                    if xy == "xs":
                        [func_x(path, data_xs, new_size) for path in
                         tqdm(k_folds[k][tt][xy], desc=f"Data -> k: {k}, set: {tt}, subset: {xy}")]
                    else:
                        [func_y(path, data_ys, new_size) for path in
                         tqdm(k_folds[k][tt][xy], desc=f"Data -> k: {k}, set: {tt}, subset: {xy}")]
                [create_path_file(path, k_folds[k][tt][f"{xy[0]}_dest"]) for path in
                 tqdm(k_folds[k][tt][xy], desc=f"Paths -> k: {k}, set: {tt}, subset: {xy}")]


def process_img(path: str, destination: str, new_size: tuple) -> None:
    img_name = path.split("\\")[-1].split('.')[0]
    img = np.load(path)
    # img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if destination.split("/")[-1] == "ys":
        new_image = np.zeros(tuple(list(reversed(new_size))+[img.shape[-1]]))
        for channel in range(img.shape[-1]):
            new_image[..., channel] = cv2.resize(img[..., channel], new_size, interpolation=cv2.INTER_NEAREST)
        img = new_image
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        img = cv2.resize(img, new_size)
    np.save(destination + f"/{img_name}", img)


def copy_file(path: str, destination: str, new_size: tuple) -> None:
    name = path.split("\\")[-1].split('.')[0]
    shutil.copy2(path, destination + f"/{name}.npy")


def create_path_file(path: str, destination: str) -> None:
    name = path.split("\\")[-1].split('.')[0]
    f = open(f"{destination}/{name}.npy", "w")
    f.write("")
    f.close()


def add_readme(destination_path: str, name: str, summary: str, cross_validation: dict, x_values: dict, y_values: dict):
    readme = {"name": name,
              "summary": summary,
              "cross_validation": cross_validation,
              "x_values": x_values,
              "y_values": y_values}

    json_file = json.dumps(readme, separators=(',', ':'))
    with open(destination_path + "/readme.json", 'w') as outfile:
        outfile.write(json_file)
        outfile.close()


if __name__ == '__main__':
    destination_path = r"D:\Datasets\Raabin\segmentation_all_320x180"

    k_folds, data_xs, data_ys = createDataset(xs_path=r"D:\Datasets\Raabin\segmentation_all\xs\*.npy",
                                              ys_replace=[["xs", "ys"]],
                                              destination_path=destination_path,
                                              k_fold=5)

    create(destination_path=destination_path,
           k_folds=k_folds,
           data_xs=data_xs,
           data_ys=data_ys,
           new_size=(320, 180),
           name="raabin_prob_seg",
           summary='Dataset for WBCs, containing complete image with a probabilistic segmentation.',
           func_x=process_img,
           func_y=process_img)
