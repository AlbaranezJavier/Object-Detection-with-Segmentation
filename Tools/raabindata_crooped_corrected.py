import json, glob, cv2

"""
Genera el recorte de las imagenes que se encuentran en los conjuntos v2
"""

if __name__ == '__main__':
    dir = r"C:\Users\TTe_J\Downloads\BloodSeg\RabbinData\first_v2_all"
    save_dir = r"C:\Users\TTe_J\Downloads\BloodSeg\RabbinData\cropped_v2"
    jsons = r"\jsons"
    images = r"\images"

    paths = glob.glob(dir + jsons + "/*.json")
    total = len(paths)
    counter = 0
    for path in paths:

        with open(path, "r") as reader:
            data = json.load(reader)
            reader.close()

        for i in range(data["Cell Numbers"]):
            counter += 1
            filename = path.split("\\")[-1][:-5]

            file = {"Label": data[f"Cell_{i}"]["Label1"], "img_id": f"{filename}_{i}"}
            json_file = json.dumps(file, separators=(',', ':'))
            with open(save_dir+jsons+f"/{filename}_{i}.json", "w") as outfile:
                outfile.write(json_file)
                outfile.close()

            x1 = int(data[f"Cell_{i}"]["x1"])
            x2 = int(data[f"Cell_{i}"]["x2"])
            y1 = int(data[f"Cell_{i}"]["y1"])
            y2 = int(data[f"Cell_{i}"]["y2"])
            img = cv2.imread(dir+images+f"/{filename}.jpg")
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)[y1:y2, x1:x2, :]
            cv2.imwrite(save_dir+images+f"/{filename}_{i}.png", img)
            print(f"{counter}/{total}", end="\r")
