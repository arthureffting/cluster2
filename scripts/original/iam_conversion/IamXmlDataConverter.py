from xml.etree.ElementTree import Element, SubElement, Comment, tostring
import cv2


class IamXmlDataConverter:

    def __init__(self, doc_pair):
        self.doc_pair = doc_pair

    def convert(self, ground_truth_data):
        # read image
        img = cv2.imread(self.doc_pair.img, cv2.IMREAD_UNCHANGED)
        height = img.shape[0]
        width = img.shape[1]

        document = Element("PcGts")
        page = SubElement(document, "Page", {
            "imageFileName": str(self.doc_pair.img),
            "imageWidth": str(width),
            "imageHeight": str(height)
        })
        reading_order = SubElement(page, "ReadingOrder")
        ordered_group = SubElement(reading_order, "OrderedGroup")
        region_ref_index = SubElement(ordered_group, "RegionRefIndexed", {
            "index": "0",
            "regionRef": "single-region"
        })

        text_region = SubElement(page, "TextRegion", {
            "orientation": "0.0",
            "id": "single-region",
            "custom": "readingOrder {index:0;}"
        })

        region_coords_points = ground_truth_data.get("coords")

        def array_to_string(target):

            string = ""

            for tuple in target:
                string += str(tuple[0]) + "," + str(tuple[1]) + " "
            return string[:-1]

        region_coords = SubElement(text_region, "Coords", {
            "points": array_to_string(region_coords_points)
        })

        for index, line_data in enumerate(ground_truth_data.get("lines")):
            text_line = SubElement(text_region, "TextLine", {
                "id": "line-" + str(index + 1),
                "custom": "readingOrder {index:" + str(index + 1) + ";}"
            })

            # Text GROUND TRUTH
            gt = line_data.get("gt")
            text_equiv = SubElement(text_line, "TextEquiv")
            unicode = SubElement(text_equiv, "Unicode")
            unicode.text = gt

            # Coordinates of the CONVEX HULL
            convex_hull = line_data.get("convex_hull")
            line_coords = SubElement(text_line, "Coords", {
                "points": array_to_string(convex_hull)
            })

            # Coordinates of the BASELINE
            baseline = line_data.get("baseline")
            baseline_coords = SubElement(text_line, "Baseline", {
                "points": array_to_string(baseline)
            })

        return document
