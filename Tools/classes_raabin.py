import json, glob
from tqdm import tqdm
import numpy as np

if __name__ == '__main__':
    dir = r"C:\Users\TTe_J\Downloads\BloodSeg\RabbinData\Second_microscope_all\jsons"
    classes = []

    jsons = glob.glob(dir+"/*.json")
    total = len(jsons)
    counter = {}
    total = 0
    for path in tqdm(jsons):
        with open(path, "r") as reader:
            data = json.load(reader)
            reader.close()
        for i in range(data["Cell Numbers"]):
            for l in range(1, 3):
                if data[f"Cell_{i}"][f"Label{l}"] not in classes:
                    counter[data[f"Cell_{i}"][f"Label{l}"]] = 0
                    classes.append(data[f"Cell_{i}"][f"Label{l}"])
            if data[f"Cell_{i}"][f"Label{1}"] == data[f"Cell_{i}"][f"Label{2}"]:
                counter[data[f"Cell_{i}"][f"Label{1}"]] += 1
            total += 1
    print(f"Counter same label: {counter}")
    print(f"Total cells: {np.sum(list(counter.values()))}/{total}")
    print(classes)
