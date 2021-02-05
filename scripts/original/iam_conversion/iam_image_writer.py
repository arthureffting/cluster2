import os, cv2, sys
import numpy as np


class IamImageWriter:

    def __init__(self, doc_pair):
        self.img = cv2.imread(doc_pair.img)

    def draw_ground_truth(self, ground_truth_data):
        for line in ground_truth_data["lines"]:
            gt = line["gt"]
            convex_hull = line["convex_hull"]
            baseline = (line["baseline"])
            avg_height = line["avg_height"]
            avg_y = line["avg_y"]
            polygon = np.asarray([convex_hull], )

            self.img = cv2.line(self.img, baseline[0], baseline[1], (255, 0, 0), 3, lineType=cv2.LINE_AA)
            self.img = cv2.polylines(self.img, np.int32(polygon), True, (0, 0, 255), 3)

        return self.img
