import json, glob, cv2, os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    dir = r"C:\Users\TTe_J\Downloads\BloodSeg\RabbinData\second_v3_all"
    jsons = r"\jsons"
    images = r"\images"
    save_ys = r"D:\Datasets\Raabin\segmentation_all\ys"
    save_xs = r"D:\Datasets\Raabin\segmentation_all\xs"
    new_size = (1920, 1080)

    paths = glob.glob(dir + jsons + "/*.json")
    counter = 0
    for path in tqdm(paths):
        counter += 1
        if counter >= 0:
            with open(path, "r") as reader:
                data = json.load(reader)
                reader.close()

            name_file = path.split("\\")[-1].split('.')[0]
            img = cv2.imread(dir + images + "/" + name_file + ".png")
            # img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            img_shape = img.shape

            mask = np.zeros(tuple(list(img_shape[0:2]) + [2]), dtype="float32")

            for i in range(data["Cell Numbers"]):
                x1 = int(data["Cell_" + str(i)]["x1"])
                x2 = int(data["Cell_" + str(i)]["x2"])
                y1 = int(data["Cell_" + str(i)]["y1"])
                y2 = int(data["Cell_" + str(i)]["y2"])

                if os.path.isfile(save_xs + f"/{name_file}.npy"):
                    raise NameError("path already exist")

                if x1 == 0 and x2 == 0 and y1 == 0 and y2 == 0:
                    print("x1, x2, y1, y2 == 0")
                else:
                    # Control overlapping and negative index
                    x_reg1 = 0 if x1 < 0 else x1
                    y_reg1 = 0 if y1 < 0 else y1
                    region = np.zeros(img_shape[0:2], dtype=np.float32)
                    region[y_reg1:y2, x_reg1:x2] = 1.

                    # Cone generation
                    patch_size = (y2 - y1, x2 - x1)
                    x_axis = np.linspace(-1, 1, patch_size[0])[:, None]
                    y_axis = np.linspace(-1, 1, patch_size[1])[None, :]
                    _grad_mask = 1 - np.sqrt(x_axis ** 2 + y_axis ** 2)
                    _grad_mask = np.clip(_grad_mask, 0., 1.)
                    _grad_mask[patch_size[0] // 2, patch_size[1] // 2] = 1.0

                    # Apply cone inside limits
                    x_hat1 = abs(x1) if x1 < 0 else 0
                    x_hat2 = (patch_size[1]) - (x2 - img_shape[1]) if x2 > img_shape[1] else patch_size[1]
                    y_hat1 = abs(y1) if y1 < 0 else 0
                    y_hat2 = (patch_size[0]) - (y2 - img_shape[0]) if y2 > img_shape[0] else patch_size[0]

                    region[y_reg1:y2, x_reg1:x2] = (
                            region[y_reg1:y2, x_reg1:x2] * _grad_mask[y_hat1:y_hat2, x_hat1:x_hat2]).astype(np.float32)

                    # Add region generated to final mask
                    mask[:, :, 0] = region * (region > mask[:, :, 0]) + mask[:, :, 0] * (region <= mask[:, :, 0])
                    mask[:, :, 1] = np.ones_like(mask[:, :, 0]) - mask[:, :, 0]


            np.save(save_xs + f"/{name_file}", cv2.resize(img, new_size).astype(np.uint8))
            np.save(save_ys + f"/{name_file}", cv2.resize(mask * 255., new_size, interpolation=cv2.INTER_NEAREST).astype(np.uint8))

