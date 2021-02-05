import os, errno
import xml.etree.ElementTree as ET
import cv2
import json

def create_folders(path):
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def store_xml(xml, path):
    create_folders(path)
    with open(path, "wb") as xml_file:
        xml_file.write(ET.tostring(xml))


def store_json(json_data, path):
    create_folders(path)
    with open(path, 'w') as json_file:
        json.dump(json_data, json_file)


def store_img(img, path):
    create_folders(path)
    cv2.imwrite(path, img)
