import os
from pathlib import Path

from scripts.original.iam_conversion.document_pair import ImageXmlPair

folder = "data/orcas"
img = "data/orcas/pages"
xml = "data/orcas/xml"

img_filenames = os.listdir(img)
xml_filenames = os.listdir(xml)

pairs = []

for img_filename in img_filenames:
    img_stem = Path(img_filename).stem
    for xml_filename in xml_filenames:
        xml_stem = Path(xml_filename).stem
        if img_stem == xml_stem:
            pairs.append(ImageXmlPair(img_stem, img_filename, xml_filename))

print(str(len(pairs)), "pairs found")

split = [0.65, 0.25, 0.10]

training_limit = int(split[0] * len(pairs))
testing_limit = int(training_limit + split[1] * len(pairs))

training_pairs = pairs[0:training_limit]
testing_pairs = pairs[training_limit:testing_limit]
validation_pairs = pairs[testing_limit:]

print(str(len(training_pairs)), "for training")
print(str(len(testing_pairs)), "for testing")
print(str(len(validation_pairs)), "for validation")
