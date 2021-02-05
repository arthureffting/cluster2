import os


class Split:

    def __init__(self, set_map):
        self.set_map = set_map
        self.sets = {

        }

    @staticmethod
    def read_from(folder):
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
                    transcription = " ".join(split_by_space[1:])
                    splitSet.lines.append(SplitLine(set, line_index, transcription))
            split.sets[set] = splitSet
        return split


class SplitSet:

    def __init__(self, name, filename):
        self.name = name
        self.filename = filename
        self.lines = []


class SplitLine:
    def __init__(self, set, index, transcription):
        self.set = set
        # line index like c03-534-123 or f07-019a-00
        split = index.split("-")
        self.page_index = split[0] + "-" + split[1]
        self.line_index = split[2]
        self.transcription = transcription
