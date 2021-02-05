import errno
import json
import os
import xml.etree.ElementTree as ET


def create_folders(path):
    if not os.path.exists(os.path.dirname(path)):
        try:
            if len(os.path.dirname(path)) > 0:
                os.makedirs(os.path.dirname(path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def get_files_in(folder_path):
    return [os.path.join(folder_path, filename) for filename in os.listdir(folder_path)]


def read_json(json_path):
    with open(json_path) as f:
        return json.load(f)


def save_xml(xml, path):
    create_folders(path)
    with open(path, "wb") as xml_file:
        xml_file.write(ET.tostring(xml))
    return path


def save_to_json(json_data, path):
    create_folders(path)
    with open(path, 'w') as json_file:
        json.dump(json_data, json_file, indent=True, sort_keys=True)
