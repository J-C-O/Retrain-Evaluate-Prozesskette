# Script for drawing bouding boxes on pictures

import os
import json
import cv2
import configparser
import subprocess
import numpy as np


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


def detect(img, json_data):
    new_img = np.copy(img)

    if json_data is None:
        return new_img
    else:
        for item in json_data:
            top_x = int(item["topleft"]["x"])
            top_y = int(item["topleft"]["y"])

            btm_x = int(item["bottomright"]["x"])
            btm_y = int(item["bottomright"]["y"])

            confidence = item["confidence"]
            label = item['label'] + " " + str(round(confidence, 3))

            if confidence > 0.5:
                #print(confidence)
                new_img = cv2.rectangle(img, (top_x, top_y), (btm_x, btm_y), (0, 255, 0), 4)
                new_img = cv2.putText(new_img, label, (top_x, top_y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 255, 0),
                                      1, cv2.LINE_AA)
        return new_img


def censor(img, json_data):
    for item in json_data:
        img = cv2.rectangle(img, (int(item["topleft"]["x"]), int(item["topleft"]["y"])),
                            (int(item["bottomright"]["x"]), int(item["bottomright"]["y"])), (0, 0, 0), -1)
    return img


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
            img = detect(img, json_data)

            cv2.imwrite(default_img_path + "/" + file, img)


if __name__ == "__main__":
    main()
