import cv2, glob, json


def click_event(event, x, y, flags, params):
    global ix, iy, img, click

    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        ix, iy = x, y
        # marco la célula
        cv2.rectangle(img, (ix - 2, iy - 2), (ix + 2, iy + 2), COLOR, THICKNESS)
        click = True


if __name__ == '__main__':
    PATH = r"C:\Users\TTe_J\Downloads\BloodSeg\RabbinData\Second_microscope_all"
    IMAGES = "/images/"
    JSONS = "/jsons/"
    SAVE_PATH = r"C:\Users\TTe_J\Downloads\BloodSeg\RabbinData\second_v3_all"
    start_in = 2004
    CELL_NUMBERS = "Cell Numbers"
    COLOR = (0, 0, 255)
    THICKNESS = 3
    CELL = "Cell_"
    LABEL = None
    ix, iy = -1, -1
    data = {}
    k = None
    FOCUSED_LABELS = ['Neutrophil', 'Large Lymph', 'Band', 'Small Lymph', 'Monocyte', 'Eosinophil', 'Meta', 'Basophil', None]

    display_size = (1920, 1080)
    new_size = (320, 180)

    paths = glob.glob(PATH + JSONS + "*.json")
    i = start_in
    while i < len(paths):
        # Cargo el json
        with open(paths[i], "r") as reader:
            data = json.load(reader)
            reader.close()
        # Cargo la imagen
        img_name = paths[i].split("\\")[-1][:-5]
        ori = cv2.imread(PATH + IMAGES + img_name + ".jpg")
        original_size = ori.shape[0:2]
        ori = cv2.rotate(ori, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img = cv2.resize(ori, display_size)
        # Almaceno las células de interés
        data2keep = []
        for j in range(data[CELL_NUMBERS]):
            if data[f"{CELL}{j}"]["Label1"] in FOCUSED_LABELS or \
                    data[f"{CELL}{j}"]["Label2"] in FOCUSED_LABELS:
                data2keep.append(data[f"{CELL}{j}"])
            data[CELL_NUMBERS] -= 1
            data.pop(f"{CELL}{j}")
        # renombro las celulas que se mantienen en el conjunto, menos si se ha pulsado "d"
        if k != ord('d'):
            for j in range(len(data2keep)):
                data[f"{CELL}{j}"] = data2keep[j]
                data[CELL_NUMBERS] += 1
                x1 = (int(data[f"{CELL}{j}"]["x1"]) / original_size[0]) * display_size[0]
                x2 = (int(data[f"{CELL}{j}"]["x2"]) / original_size[0]) * display_size[0]
                y1 = (int(data[f"{CELL}{j}"]["y1"]) / original_size[1]) * display_size[1]
                y2 = (int(data[f"{CELL}{j}"]["y2"]) / original_size[1]) * display_size[1]
                p1 = (int(x1), int(y1))
                p2 = (int(x2), int(y2))
                cv2.rectangle(img, p1, p2, COLOR, THICKNESS)
                if x1 == 0 and x2 == 0 and y1 == 0 and y2 == 0:
                    print("Revisar: ", paths[i])
        # Bucle para marcar las nuevas células
        click = False
        k = ord('c')
        while k == ord('c'):
            # Muestro la imagen
            cv2.imshow("image", img)
            # Click event
            cv2.setMouseCallback("image", click_event)
            if click:
                # Agrego la nueva célula
                real_x = int((ix / display_size[0]) * original_size[0])
                real_y = int((iy / display_size[1]) * original_size[1])
                data[f"{CELL}{data[CELL_NUMBERS]}"] = {"Label1": None,
                                                       "Label2": None,
                                                       "x1": str(real_x - 287),
                                                       "x2": str(real_x + 288),
                                                       "y1": str(real_y - 287),
                                                       "y2": str(real_y + 288),
                                                       "Cell_ID": None}
                data[CELL_NUMBERS] += 1
                click = False
                print(ix, ' ', iy)

            # Espero respuesta usuario
            k = cv2.waitKey(0)
        # Cierra el programa
        if k == 27:
            cv2.destroyAllWindows()
            exit()
        # vuelve una imagen hacia atrás
        elif k == ord('a'):
            i = 0 if i <= 0 else i-2
        elif k == ord('d'):
            i -= 1
        else:
            # Almaceno la info en un json
            json_file = json.dumps(data, separators=(',', ':'))
            with open(SAVE_PATH + JSONS + img_name + ".json", "w") as outfile:
                outfile.write(json_file)
                outfile.close()

            # Almaceno la imagen
            cv2.imwrite(SAVE_PATH + IMAGES + img_name + ".png", ori)
            # Doy la información por consola
            print(f"{i}/{len(paths)} - {img_name} - {data[CELL_NUMBERS]}")
        i += 1

    cv2.destroyAllWindows()
