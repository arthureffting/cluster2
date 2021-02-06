import cv2
from shapely.geometry import Point

from utils.geometry import angle_between_points, get_new_point
from utils.image_handling import subimage


class ViewingWindow:

    def __init__(self, parameters, image, origin, size, angle):
        self.parameters = parameters
        self.source_image = image
        self.origin = origin
        self.focus = get_new_point(origin,
                                   angle,
                                   self.parameters.patch_ratio * size / 2)
        self.size = size
        self.angle = angle
        self.ratio = self.parameters.patch_ratio * self.size / self.parameters.patch_size

    def image(self, img_data):
        width = height = self.parameters.patch_ratio * self.size
        image = subimage(img_data,
                         (self.focus.x, self.focus.y),
                         self.angle,
                         width,
                         height)
        if image.shape[0] != image.shape[1] or image.shape[0] == 0 or image.shape[1] == 0:
            raise Exception("Viewing window outside of bounds", image.shape)
        return image

    def relative(self, point):
        actual_distance = self.origin.distance(point)
        actual_angle = angle_between_points(self.origin, point)
        scaled_down_distance = actual_distance * (1 / self.ratio)
        rotated_angle = actual_angle - self.angle
        return get_new_point(Point(0, 0), rotated_angle, scaled_down_distance)

    def absolute(self, point):
        relative_distance_to_point = Point(0, 0).distance(point)
        relative_angle_to_point = angle_between_points(Point(0, 0), point)
        actual_distance_to_point = relative_distance_to_point * self.ratio
        actual_angle_to_point = relative_angle_to_point + self.angle
        predicted_point = get_new_point(self.origin, actual_angle_to_point, actual_distance_to_point)
        return predicted_point
