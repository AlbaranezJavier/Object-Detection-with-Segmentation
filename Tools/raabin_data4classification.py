import json, glob, cv2, os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

"""
Almacena aquellos ejemplos con etiquetas en las que coinciden ambos expertos
"""

if __name__ == '__main__':
    dir = r"C:\Users\TTe_J\Downloads\BloodSeg\RabbinData\First_microscope_all"
    jsons = r"\jsons"
    images = r"\images"
    save_ys = r"D:\Datasets\Raabin\cropped_v3\ys"
    save_xs = r"D:\Datasets\Raabin\cropped_v3\xs"

    classes_excluded = [None, "Unknn", "Not centered"]

    labels = {"Basophil": np.array([255, 0, 0, 0, 0, 0]), "Eosinophil": np.array([0, 255, 0, 0, 0, 0]),
              "Lymphocyte": np.array([0, 0, 255, 0, 0, 0]), "Monocyte": np.array([0, 0, 0, 255, 0, 0]),
              "Neutrophil": np.array([0, 0, 0, 0, 255, 0]), "Large Lymph": np.array([0, 0, 255, 0, 0, 0]),
              "Burst": np.array([0, 0, 0, 0, 0, 255]), "Band": np.array([0, 0, 0, 0, 255, 0]),
              "Artifact": np.array([0, 0, 0, 0, 0, 255]), "Small Lymph": np.array([0, 0, 255, 0, 0, 0]),
              "Meta": np.array([0, 0, 0, 0, 255, 0]), "NRBC": np.array([0, 0, 0, 0, 0, 255]),
              "Megakar": np.array([0, 0, 0, 0, 0, 255])}

    paths = glob.glob(dir + jsons + "/*.json")
    for path in tqdm(paths):
        with open(path, "r") as reader:
            data = json.load(reader)
            reader.close()

        name_file = path.split("\\")[-1].split('.')[0]
        img = cv2.imread(dir+images + "/"+name_file+".jpg")
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img_shape = img.shape

        for i in range(data["Cell Numbers"]):
            if data[f"Cell_{i}"]["Label1"] == data[f"Cell_{i}"]["Label2"] and data[f"Cell_{i}"]["Label1"] not in classes_excluded:
                x1 = int(data[f"Cell_{i}"]["x1"])
                x2 = int(data[f"Cell_{i}"]["x2"])
                y1 = int(data[f"Cell_{i}"]["y1"])
                y2 = int(data[f"Cell_{i}"]["y2"])

                if x1 == 0 and x2 == 0 and y1 == 0 and y2 == 0:
                    raise NameError("x1, x2, y1, y2 == 0")

                if os.path.isfile(save_xs+f"/{name_file}.npy"):
                    raise NameError("path already exist")

                # Values to get in the region
                x_reg1 = abs(x1) if x1 < 0 else 0
                x_reg2 = (x2-x1) - (x2 - img_shape[1]) if x2 > img_shape[1] else x2-x1
                y_reg1 = abs(y1) if y1 < 0 else 0
                y_reg2 = (y2-y1) - (y2 - img_shape[0]) if y2 > img_shape[0] else y2-y1

                # Values to get in the image
                x_img1 = 0 if x1 < 0 else x1
                x_img2 = img_shape[1] if x2 > img_shape[1] else x2
                y_img1 = 0 if y1 < 0 else y1
                y_img2 = img_shape[0] if y2 > img_shape[0] else y2

                region = np.zeros((y2-y1, x2-x1, 3), dtype=np.float32)
                region[y_reg1:y_reg2, x_reg1:x_reg2, :] = img[y_img1:y_img2, x_img1:x_img2, :]

                np.save(save_xs+f"/{name_file}_{i}", region.astype(np.uint8))
                np.save(save_ys+f"/{name_file}_{i}", labels[data[f"Cell_{i}"]["Label1"]].astype(np.uint8))