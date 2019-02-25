#Script for censoring images/videos

import os
import json
import cv2
import configparser
import subprocess

# TODO for detect.py
# load img
# load json
# draw boundig box
# save img

def load_json(path, name):
    """for file in os.listdir(path):
        # load json file
        if file.endswith(".json"):
            p = os.path.join(path, file)
            f = open(p, "r")
            data = json.loads(f.read())"""
    with open(os.path.join(path, name + ".json"), "r") as fh:
        data = json.load(fh)

    if data:
        return data


def censor(img, json_data):
    new_img = np.copy(img)
    for item in json_data:
        top_x = int(item["topleft"]["x"])
        top_y = int(item["topleft"]["y"])

        btm_x = int(item["bottomright"]["x"])
        btm_y = int(item["bottomright"]["y"])

        confidence = item["confidence"]

        if confidence > 0.55:
            print(confidence)
            new_img = cv2.rectangle(img, (top_x, top_y), (btm_x, btm_y), (0, 0, 0), -1)
    return new_img


def main():
    default_img_path = "censor_data/images/"
    default_json_path = "censor_data/images/out"

    config = configparser.ConfigParser()
    config.read('./config.cfg')
    detection_command = config.get('commands', 'run_detecting')

    os.chdir("model/darkflow")
    subprocess.call([detection_command], shell=True)
    os.chdir("../../")

    for file in os.listdir(default_img_path):
        # load jpg file
        if file.endswith(".JPG") or file.endswith(".jpg"):
            img = cv2.imread(default_img_path + file, 1)
            # load json
            json_data = load_json(default_json_path, os.path.splitext(file)[0])

            # detected jpgs
            img = censor(img, json_data)

            cv2.imwrite(default_img_path + "/" + file, img)

if __name__ == "__main__":
    main()