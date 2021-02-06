import os
import sys

from scripts.new.iam_conversion.concave import run_transformation_approach
from scripts.new.iam_conversion.iam_original import store_iam_original
from scripts.new.iam_conversion.pairing import pair_files
from scripts.new.iam_conversion.stepper import to_steps
from scripts.utils.files import get_files_in

if __name__ == '__main__':

    folder = sys.argv[0]  # Folder containing the IAM image files and XML data in /img and /xml, respectively
    img_folder = os.path.join(folder, "img")
    xml_folder = os.path.join(folder, "xml")

    # Useful for testing with a smaller subset
    use_subset = False
    use_filter = False
    subset_size = 10
    filter = ["816"]

    n = subset_size if use_subset else len(get_files_in(img_folder))
    iam_img_files = get_files_in(img_folder)[:n]
    iam_xml_files = get_files_in(xml_folder)[:n]
    iam_pairs = pair_files("dataset", iam_img_files, iam_xml_files)

    # For each IAM pair, visualizations are generated for:
    # 1. The original XML data
    # 2. The XML data transformed into the "Page" format
    # 3. The custom transformation
    # 4. The steps generated

    ground_truth = {}
    used_pairs = [p for p in iam_pairs if not use_filter or p.index in filter]

    for i, pair in enumerate(used_pairs):
        # Since the IAM dataset is transformed to have a baseline composed of a single segment
        # the height threshold has to be significantly lower
        pair.set_height_threshold(0.1)
        store_iam_original(pair)
        data = run_transformation_approach(pair, alpha=0.0025)
        ground_truth[pair.index] = data

    steps = to_steps(ground_truth, used_pairs)
