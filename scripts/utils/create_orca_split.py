import os
import random
from pathlib import Path

from scripts.original.iam_conversion.document_pair import ImageXmlPair
from scripts.original.iam_conversion.prepare_train_data import prepare

folder = "data/original/orcas"
img = "data/original/orcas/pages"
xml = "data/original/orcas/xml"
output = "data/split/orcas"

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

random.shuffle(pairs)

split = [0.60, 0.25, 0.15]

training_limit = int(split[0] * len(pairs))
testing_limit = int(training_limit + split[1] * len(pairs))

training_pairs = pairs[0:training_limit]
testing_pairs = pairs[training_limit:testing_limit]
validation_pairs = pairs[testing_limit:]

print(str(len(training_pairs)), "for training")
print(str(len(testing_pairs)), "for testing")
print(str(len(validation_pairs)), "for validation")

with open(os.path.join(output, "tr.txt"), "w") as file:
    file.writelines([t.index + "\n" for t in training_pairs])

with open(os.path.join(output, "te.txt"), "w") as file:
    file.writelines([t.index + "\n"  for t in testing_pairs])

with open(os.path.join(output, "va.txt"), "w") as file:
    file.writelines([t.index + "\n"  for t in validation_pairs])
