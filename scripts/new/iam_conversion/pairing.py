import os
import sys
from pathlib import Path

from iam_to_steps_conversion.img_xml_pair import ImageXmlPair


def pair(folder, img_folder="img", xml_folder="xml", img_extensions=None, start_index=0, convert=False, config=None):
    if img_extensions is None:
        img_extensions = [".jpg", ".jpeg", ".png"]
    pairs = []
    set_img = os.path.join(folder, "pages", img_folder)
    set_xml = os.path.join(folder, "pages", xml_folder)
    i = start_index
    for subdir, dirs, files in os.walk(set_img):
        for file in files:
            # print os.path.join(subdir, file)
            split_img_name = file.split(".")
            img_name = split_img_name[0]
            filepath = subdir + os.sep + file

            extension = Path(filepath).suffix

            if extension in img_extensions:
                for xml_subdir, xml_dirs, xml_files in os.walk(set_xml):
                    for xml_file in xml_files:
                        # print os.path.join(subdir, file)
                        split_xml_name = xml_file.split(".")
                        xml_name = split_xml_name[0]
                        xml_filepath = xml_subdir + os.sep + xml_file
                        if xml_filepath.endswith(".xml"):
                            if img_name == xml_name:
                                given_pair = ImageXmlPair(i, folder, filepath, xml_filepath, convert=convert,
                                                          config=config)
                                pairs.append(given_pair)
                                i = i + 1
    return pairs


def pair_files(base, img_filenames, xml_filenames, config=None):
    pairs = []

    # Creates pairs of images and XML data
    for i, image_filename in enumerate(img_filenames):

        img_path = Path(image_filename)

        basename = img_path.stem

        img_number = basename.split("-")[1]
        leading_path = img_path.parts[0:-2]
        xml_path = os.path.join(*leading_path, "xml", "xml-" + str(img_number) + ".xml")

        if os.path.exists(xml_path):
            pairs.append(ImageXmlPair(img_number, base, image_filename, xml_path, config=config))

        sys.stdout.write("\r" + "[Paring documents] " + str(round(100 * (i + 1) / len(img_filenames), 2)) + "%")

    return pairs
