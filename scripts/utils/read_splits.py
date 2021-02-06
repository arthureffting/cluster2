import os
from pathlib import Path


class Split:

    def __init__(self, set_map):
        self.set_map = set_map
        self.sets = {

        }

    @staticmethod
    def read_iam_data(folder):
        split = Split({
            "training": "tr.txt",
            "testing": "te.txt",
            "validation": "va.txt",
        })
        for set in split.set_map:
            splitSet = SplitSet(set, split.set_map[set])
            with open(os.path.join(folder, split.set_map[set])) as file:
                lines = file.readlines()
                for line in lines:
                    split_by_space = line.split(" ")
                    line_index = split_by_space[0]
                    line_index = Path(line_index).stem.rstrip()
                    splitSet.items.append(SplitLine(set, line_index))
            split.sets[set] = splitSet
        return split

    @staticmethod
    def read_orcas_data(folder):
        split = Split({
            "training": "tr.txt",
            "testing": "te.txt",
            "validation": "va.txt",
        })
        for set in split.set_map:
            splitSet = SplitSet(set, split.set_map[set])
            with open(os.path.join(folder, split.set_map[set])) as file:
                lines = file.readlines()
                for line in lines:
                    split_by_space = line.split(" ")
                    line_index = split_by_space[0]
                    line_index = Path(line_index).stem.rstrip()
                    splitSet.items.append(SplitImage(set, line_index))
            split.sets[set] = splitSet
        return split

class SplitSet:

    def __init__(self, name, filename):
        self.name = name
        self.filename = filename
        self.items = []


class SplitImage:

    def __init__(self, set, filename):
        self.set = set
        self.page_index = filename


class SplitLine:
    def __init__(self, set, index):
        self.set = set
        # line index like c03-534-123 or f07-019a-00
        split = index.split("-")
        self.page_index = split[0] + "-" + split[1]
        self.line_index = split[2]
