import cv2, glob, json
from multiprocessing import Pool

'''
This script pre-processes the input data to speed up the neural network training process.
'''

def task(args):
    # Image processing
    img_name = args[0].split("\\")[-1][:-5]
    img = cv2.imread(args[1] + args[2] + img_name + ".jpg")
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img = cv2.resize(img, args[5])
    cv2.imwrite(args[6] + args[2] + img_name + ".png", img)

    # Jsons processing
    with open(args[0], "r") as reader:
        data = json.load(reader)
        reader.close()
    for i in range(data["Cell Numbers"]):
        data[f"Cell_{i}"]["x1"] = str(int((int(data[f"Cell_{i}"]["x1"]) / args[4][0]) * args[5][0]))
        data[f"Cell_{i}"]["x2"] = str(int((int(data[f"Cell_{i}"]["x2"]) / args[4][0]) * args[5][0]))
        data[f"Cell_{i}"]["y1"] = str(int((int(data[f"Cell_{i}"]["y1"]) / args[4][1]) * args[5][1]))
        data[f"Cell_{i}"]["y2"] = str(int((int(data[f"Cell_{i}"]["y2"]) / args[4][1]) * args[5][1]))
    json_file = json.dumps(data, separators=(',', ':'))
    with open(args[6] + args[3] + img_name + ".json", "w") as outfile:
        outfile.write(json_file)
        outfile.close()

if __name__ == '__main__':
    # original data
    folder = r"C:\Users\TTe_J\Downloads\BloodSeg\RabbinData\first_v3_all"
    images = "/images/"
    jsons = "/jsons/"
    original_resolution = (5312, 2988)
    # new data
    new_resolution = (320, 180)
    new_destination = r"C:\Users\TTe_J\Downloads\BloodSeg\RabbinData\first_v3_all_320x180"
    # multiprocessing
    pool = Pool(None)

    # paths
    args = []
    json_paths = glob.glob(folder+jsons+"*.json")
    [args.append([path, folder, images, jsons, original_resolution, new_resolution, new_destination]) for path in json_paths]
    pool.map(task, args)