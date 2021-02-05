import json

def load_file_list(file_list_path):
    with open(file_list_path) as f:
        data = json.load(f)
    for d in data:
        json_path = d[0]
        img_path = d[1]
        d[0] = json_path
        d[1] = img_path
    return data
