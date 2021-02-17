import os
from pathlib import Path

from PIL import Image

from scripts.new.iam_conversion.iam_data_loader import IamDataLoader
from scripts.new.iam_conversion.iam_xml_data_converter import IamXmlDataConverter
from scripts.new.iam_conversion.page_xml_to_gt import process_xml_custom
from scripts.utils.files import save_xml


class ImageXmlPair:

    def __init__(self, index, base_folder, img_filename, xml_filename, convert=True, config=None):
        self.index = index
        self.base = base_folder
        self.config = config

        img_extension = Path(img_filename).suffix

        if str(img_extension).lower() in [".jpg", ".jpeg"]:
            im1 = Image.open(img_filename)
            png_filename = img_filename[0:len(img_filename) - len(img_extension)] + ".png"
            im1.save(png_filename)
            self.img = png_filename
            os.remove(img_filename)
        else:
            self.img = img_filename
        self.xml = xml_filename

        self.height_threshold = 0.8
        self.extracted_ground_truth = None

        self.data_loader = IamDataLoader(self)

        if convert:
            self.xml_converter = IamXmlDataConverter(self)
            self.transformation = self.data_loader.get_ground_truth_data()
            self.transformation_xml = self.xml_converter.convert(self.get_transformation())
            self.transformation_xml_path = os.path.join(self.base, "transformed_xml", self.stem(extension=".xml"))
            save_xml(self.transformation_xml, self.transformation_xml_path)
        else:
            self.transformation_xml_path = xml_filename

    def set_height_threshold(self, threshold):
        self.height_threshold = threshold

    def stem(self, extension=""):
        return Path(self.img).stem + str(extension)

    def get_components(self):
        return self.data_loader.original_components()

    def get_transformation(self):
        return self.transformation

    def get_transformation_xml_path(self):
        return self.transformation_xml_path

    def extract_ground_truth(self):
        if self.extracted_ground_truth is None:
            self.extracted_ground_truth = process_xml_custom(self)
        return self.extracted_ground_truth

    def is_converted(self, p_img_path, p_xml_path):
        is_true = os.path.exists(p_img_path) and os.path.exists(p_xml_path)
        if is_true:
            self.transformation_xml_path = p_xml_path
        return self.transformation_xml_path is not None or is_true
