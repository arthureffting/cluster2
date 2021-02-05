import sys
import json
from collections import defaultdict

def load_char_set(char_set_path):
    with open(char_set_path) as f:
        char_set = json.load(f)

    idx_to_char = {}
    for k, v in char_set['idx_to_char'].items():
        idx_to_char[int(k)] = v

    return idx_to_char, char_set['char_to_idx']

def create_char_set(target_json_path, origin_json_paths=[]):
    out_char_to_idx = {}
    out_idx_to_char = {}
    char_freq = defaultdict(int)
    for data_file in origin_json_paths:
        with open(data_file) as f:
            paths = json.load(f)
        for json_path, image_path in paths:
            with open(json_path) as f:
                data = json.load(f)

            cnt = 1  # this is important that this starts at 1 not 0
            for data_item in data:
                for c in data_item.get('gt', None):
                    if c is None:
                        print("There was a None GT")
                        continue
                    if c not in out_char_to_idx:
                        out_char_to_idx[c] = cnt
                        out_idx_to_char[cnt] = c
                        cnt += 1
                    char_freq[c] += 1

    out_char_to_idx2 = {}
    out_idx_to_char2 = {}

    for i, c in enumerate(sorted(out_char_to_idx.keys())):
        out_char_to_idx2[c] = i + 1
        out_idx_to_char2[i + 1] = c

    output_data = {
        "char_to_idx": out_char_to_idx2,
        "idx_to_char": out_idx_to_char2
    }

    print("[Character set generated] Size: ", len(output_data['char_to_idx']))

    with open(target_json_path, 'w') as outfile:
        json.dump(output_data, outfile)

