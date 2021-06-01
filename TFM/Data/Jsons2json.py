import json
import argparse
import os

'''
this script joins several jsons in one
'''

# Input management
ap = argparse.ArgumentParser()
ap.add_argument("-fj", "--directory_jsons", type=str, required=False, default=r"C:\Users\TTe_J\Downloads\17-17-05\jsons")
ap.add_argument("-o", "--output", type=str, required=False, default=r"C:\Users\TTe_J\Downloads\17-17-05\jsons\train.json")

if __name__ == "__main__":
    args = vars(ap.parse_args())
    jsons_path = args['directory_jsons']
    output_path = args['output']

    data = {}

    jsons = os.listdir(jsons_path)
    for j in jsons:
        with open(os.path.join(jsons_path, j)) as j:
            data = {**data, **json.load(j)}
            j.close()

    with open(output_path, "w") as o:
        json.dump(data, o)
        o.close()