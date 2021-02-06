import xml.etree.ElementTree
from tkinter import *
from PIL import Image, ImageTk
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import itertools
import math

class IamDataLoader:

    def __init__(self, doc_pair):
        self.doc_pair = doc_pair

    def original_components(self):
        root = xml.etree.ElementTree.parse(self.doc_pair.xml).getroot()
        components = []
        for child in root:
            if child.tag == "handwritten-part":
                for line_index, line in enumerate(child):
                    gt = line.get('text')
                    for component in line:
                        if component.tag == "word":
                            for i, cmp in enumerate(component):
                                components.append({
                                    "index": line_index,
                                    "x": int(cmp.get("x")),
                                    "y": int(cmp.get("y")),
                                    "width": int(cmp.get("width")),
                                    "height": int(cmp.get("height"))
                                })
        return components

    def get_ground_truth_data(self):

        root = xml.etree.ElementTree.parse(self.doc_pair.xml).getroot()
        lines = []

        for child in root:
            if child.tag == "handwritten-part":

                text_region_points = []
                max_x = 0
                max_y = 0
                min_x = 0
                min_y = 0

                for line in child:

                    gt = line.get('text')

                    all_points = []
                    bottom_points = []
                    top_points = []

                    left_vertical = None
                    right_vertical = None

                    # Used for normalizing component heights
                    component_count = 0
                    component_height_sum = 0
                    component_height_square_sum = 0
                    y_sum = 0

                    for component in line:
                        if component.tag == "word":
                            for i, cmp in enumerate(component):
                                offset = 0
                                x = int(cmp.get("x"))
                                y = int(cmp.get("y"))
                                width = int(cmp.get("width"))
                                height = int(cmp.get("height"))

                                # Only account for component statistics if the component has a minimum size
                                if width * height > 100:
                                    component_count += 1
                                    component_height_sum += height
                                    component_height_square_sum += (height * height)
                                    y_sum += (y + (height / 2))

                                top_left = (x - offset, y - offset)
                                min_x = min_x if top_left[0] > min_x else top_left[0]
                                min_y = min_y if top_left[1] > min_y else top_left[1]
                                bottom_left = (x - offset, y + height + offset)
                                min_x = min_x if bottom_left[0] > min_x else bottom_left[0]
                                max_y = max_y if bottom_left[1] < max_y else bottom_left[1]

                                top_right = (x + width + offset, y - offset)
                                max_x = max_x if top_right[0] < max_x else top_right[0]
                                min_y = min_y if top_right[1] > min_y else top_right[1]
                                bottom_right = (x + width + offset, y + height + offset)
                                max_x = max_x if bottom_right[0] < max_x else bottom_right[0]
                                max_y = max_y if bottom_right[1] < max_y else bottom_right[1]

                                if i == 0:
                                    left_vertical = (bottom_left, top_left)
                                if i == len(component) - 1:
                                    right_vertical = (bottom_right, top_right)

                                all_points.append(top_right)
                                all_points.append(top_left)
                                all_points.append(bottom_left)
                                all_points.append(bottom_right)
                                bottom_points.append(bottom_right)
                                bottom_points.append(bottom_left)
                                top_points.append(top_right)
                                top_points.append(top_left)

                    # All components have been processed

                    # Normalize average heights for baseline
                    avg_y = y_sum / component_count
                    avg_height = component_height_sum / component_count
                    variance_height = (
                                              component_count * component_height_square_sum - component_height_sum * component_height_sum) / (
                                              component_count * component_count)
                    height_std_dev = int(math.sqrt(variance_height))

                    # Calculate convex hull
                    hull = ConvexHull(all_points)
                    hull_points = []
                    underline_points = []
                    overscore_points = []

                    for vertice in hull.vertices:
                        point = all_points[vertice]
                        hull_points.append(point)
                        if point in bottom_points:
                            underline_points.append(point)
                        if point in top_points:
                            overscore_points.append(point)

                    def order_underline(point):
                        return point[0]

                    underline_points.sort(key=order_underline)
                    underline_points = [underline_points[0], underline_points[-1]]

                    overscore_points.sort(key=order_underline)
                    overscore_points = [overscore_points[0], overscore_points[-1]]

                    left_vertical_distance = underline_points[0][1] - overscore_points[0][1]
                    right_vertical_distance = underline_points[1][1] - overscore_points[1][1]

                    max_height = (avg_height + (height_std_dev))
                    min_height = avg_height

                    if (left_vertical_distance > max_height) or (left_vertical_distance < min_height):
                        underline_points = [(underline_points[0][0], avg_y + (avg_height / 2)), underline_points[1]]
                    if (right_vertical_distance > max_height) or (right_vertical_distance < min_height):
                        underline_points = [underline_points[0], (underline_points[1][0], underline_points[0][1])]

                    underline_points = [(int(underline_points[0][0]), int(underline_points[0][1])),
                                        (int(underline_points[1][0]), int(underline_points[1][1]))]

                    lines.append({
                        "gt": gt,
                        "convex_hull": hull_points,
                        "baseline": underline_points,
                        "overscore": overscore_points,
                        "avg_height": avg_height,
                        "avg_y": avg_y
                    })

                region_top_left = (min_x, min_y)
                region_bottom_right = (max_x, max_y)
                region_top_right = (max_x, min_y)
                region_bottom_left = (min_x, max_y)

                region = {
                    "coords": [
                        region_bottom_left, region_top_left, region_top_right, region_bottom_right
                    ],
                    "lines": lines,
                }

                return region


class Window(Frame):

    def __init__(self, master, data):
        Frame.__init__(self, master)
        self.master = master
        self.pack(fill=BOTH, expand=YES)

        image = Image.open(data.get("img"))
        render = ImageTk.PhotoImage(image)

        self.canvas = Canvas(self, bg="black")
        self.canvas.create_image(0, 0, image=render, anchor=NW)

        for line in data.get("region").get("lines"):
            flatten = itertools.chain.from_iterable
            self.canvas.create_polygon(tuple(flatten(line.get("convex_hull"))), fill="green", stipple="gray50")
            self.canvas.create_line(tuple(flatten(line.get("baseline"))), fill="red", stipple="gray50", width=5)
            self.canvas.create_line(tuple(flatten(line.get("overscore"))), fill="red", stipple="gray50", width=5,
                                    dash=(4, 2))
            self.canvas.create_line(0, line.get("avg_y"), 5000, line.get("avg_y"), fill="yellow", stipple="gray50",
                                    width=5, dash=(4, 2))

        self.canvas.pack(fill=BOTH, expand=YES)
        self.canvas.bind('<ButtonPress-1>', lambda event: self.canvas.scan_mark(event.x, event.y))
        self.canvas.bind("<B1-Motion>", lambda event: self.canvas.scan_dragto(event.x, event.y, gain=1))
        self.master.mainloop()

