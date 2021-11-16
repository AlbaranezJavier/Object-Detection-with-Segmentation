from Data.DataManager import DataManager
import cv2, shutil, os

"""
This script transform segmentation labels to bbox and save them in a csv file.
"""

if __name__ == '__main__':
    # Data Variables
    inputs_rgb = [r'C:\Users\TTe_J\Downloads\SyntheticConeDataset(1005)\RightImages',
                  r'C:\Users\TTe_J\Downloads\17-17-05']
    labels_class = ["b", "y", "o_s", "o_b"]
    label_size = (720, 1280, len(labels_class))
    background = False
    batch_size = 8
    valid_size = .10
    output_type = "cls"  # regression = reg, classification = cls, regression + classficiation = reg+cls

    # CSV directory
    csv_dir = r"C:\Users\TTe_J\Downloads"
    train_csv = "train_set.txt"
    valid_csv = "valid_set.txt"

    # Img directory
    img_dir = r"C:\Users\TTe_J\Downloads"
    train_imgs = "train_imgs"
    valid_imgs = "valid_imgs"

    # Data Manager
    dm = DataManager(inputs_rgb, labels_class, label_size, background, valid_size, batch_size, output_type)

    # Train set
    writer = open(f'{csv_dir}/{train_csv}', 'w')
    os.makedirs(f'{img_dir}/{train_imgs}')

    counter = 0
    head = ["Unknown information\n", 'Name,URL,Width,Height,Scale,"X0, Y0, H0, W0","X1, Y1, H1, W1",etc,"\n',
            ",,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,\n"]
    for h in head:
        writer.write(h)
    for idx in range(dm.batches_size["train"] - 1):
        x_data = dm.X["train"][dm.batches["train"][idx]:dm.batches["train"][idx + 1]]
        y_data = dm.Y["train"][dm.batches["train"][idx]:dm.batches["train"][idx + 1]]
        for i in range(len(x_data)):
            counter += 1
            shutil.copyfile(x_data[i], f'{img_dir}/{train_imgs}/{x_data[i].split("/")[1]}')
            row = f'{x_data[i].split("/")[1]},N/A,{label_size[1]},{label_size[0]},0.1,'
            for cls in y_data[i]:
                for polygon in cls:
                    x, y, w, h = cv2.boundingRect(polygon)
                    row += f'"[{x},{y},{h},{w}]",'
            row += ",\n"
            writer.write(row)
    writer.close()
    print("Train counter: ", counter)

    # Valid set
    writer = open(f'{csv_dir}/{valid_csv}', 'w')
    os.makedirs(f'{img_dir}/{valid_imgs}')

    counter = 0
    head = ["Unknown information\n", 'Name,URL,Width,Height,Scale,"X0, Y0, H0, W0","X1, Y1, H1, W1",etc,"\n',
            ",,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,\n"]
    for h in head:
        writer.write(h)
    for idx in range(dm.batches_size["valid"] - 1):
        x_data = dm.X["valid"][dm.batches["valid"][idx]:dm.batches["valid"][idx + 1]]
        y_data = dm.Y["valid"][dm.batches["valid"][idx]:dm.batches["valid"][idx + 1]]
        for i in range(len(x_data)):
            counter += 1
            shutil.copyfile(x_data[i], f'{img_dir}/{valid_imgs}/{x_data[i].split("/")[1]}')
            row = f'{x_data[i].split("/")[1]},N/A,{label_size[1]},{label_size[0]},0.1,'
            for cls in y_data[i]:
                for polygon in cls:
                    x, y, w, h = cv2.boundingRect(polygon)
                    row += f'"[{x},{y},{h},{w}]",'
            row += ",\n"
            writer.write(row)
    writer.close()
    print("Valid counter: ", counter)