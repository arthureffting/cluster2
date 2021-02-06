
import os

from scripts.original.iam_conversion.IamXmlDataConverter import IamXmlDataConverter
from scripts.original.iam_conversion.iam_data_loader import IamDataLoader
from scripts.original.iam_conversion.iam_image_writer import IamImageWriter


class ImageXmlPair:

    def __init__(self, index, img_filename, xml_filename):
        self.index = index
        self.img = img_filename
        self.xml = xml_filename
        self.processed_xml = xml_filename

    def convert(self):
        data_loader = IamDataLoader(self)
        xml_converter = IamXmlDataConverter(self)
        image_writer = IamImageWriter(self)

        ground_truth_data = data_loader.get_ground_truth_data()
        new_xml = xml_converter.convert(ground_truth_data)
        new_img = image_writer.draw_ground_truth(ground_truth_data)

        return new_img, new_xml

    def set_processed_xml_path(self, processed_xml):
        self.processed_xml = processed_xml

    def is_converted(self, p_xml_path):
        is_true = os.path.exists(p_xml_path)
        if is_true:
            self.processed_xml = p_xml_path
        return self.processed_xml is not None or is_true
