import re
import xml
from svgpathtools import Path, Line
import cv2
import numpy as np
from scipy.interpolate import griddata
import sys

from scripts.new.iam_conversion.shapely_preprocessing import GtLine


def shapely(line, height_threshold):
    gt_line = GtLine(line["bounding_poly"], line["baseline"], line["ground_truth"], height_threshold)

    return {
        "lf": gt_line.steps(),
        "bounding_poly": line["bounding_poly"],
        "baseline": line['baseline'],
        "ground_truth": line["ground_truth"]
    }


def process_xml_custom(pair):
    total_result = []
    with open(pair.get_transformation_xml_path()) as f:
        xml_string_data = f.read()
    # Parse invalid xml data
    xml_string_data = xml_string_data.replace("&amp;", "&")
    xml_string_data = xml_string_data.replace("&", "&amp;")
    xml_data = readXMLFile(xml_string_data)

    print("[Pair] " + pair.stem())
    for i, line in enumerate(xml_data[0]["lines"]):
        sys.stdout.write("\r[Processing lines] " + str(round(100 * (i + 1) / len(xml_data[0]["lines"]), 2)) + "% ")
        # total_result.append(sympy_line(line))
        total_result.append(shapely(line, pair.height_threshold))
    print("\n[Finished]", pair.stem(), "\n")
    return total_result


# Extract ground truth from XML as described in the original paper
def process_xml(pair):
    result = []

    img = cv2.imread(pair.img)

    with open(pair.get_transformation_xml_path()) as f:
        xml_string_data = f.read()
    # Parse invalid xml data
    xml_string_data = xml_string_data.replace("&amp;", "&")
    xml_string_data = xml_string_data.replace("&", "&amp;")
    xml_data = readXMLFile(xml_string_data)

    for region_index, region in enumerate(xml_data[0]["regions"]):
        for line_index, line in enumerate(xml_data[0]["lines"]):

            line_mask = extract_region_mask(img, line['bounding_poly'])

            masked_img = img.copy()
            masked_img[line_mask == 0] = 0

            summed_axis0 = (masked_img.astype(float) / 255).sum(axis=0)
            summed_axis1 = (masked_img.astype(float) / 255).sum(axis=1)

            non_zero_cnt0 = np.count_nonzero(summed_axis0) / float(len(summed_axis0))
            non_zero_cnt1 = np.count_nonzero(summed_axis1) / float(len(summed_axis1))

            avg_height0 = np.median(summed_axis0[summed_axis0 != 0])
            avg_height1 = np.median(summed_axis1[summed_axis1 != 0])

            avg_height = min(avg_height0, avg_height1)
            if non_zero_cnt0 > non_zero_cnt1:
                target_step_size = avg_height0
            else:
                target_step_size = avg_height1

            paths = []
            for i in range(len(line['baseline']) - 1):
                i_1 = i + 1

                p1 = line['baseline'][i]
                p2 = line['baseline'][i_1]

                p1_c = complex(*p1)
                p2_c = complex(*p2)

                paths.append(Line(p1_c, p2_c))

            # Add a bit on the end
            tan = paths[-1].unit_tangent(1.0)
            p3_c = p2_c + target_step_size * tan
            paths.append(Line(p2_c, p3_c))

            path = Path(*paths)

            ts = find_t_spacing(path, target_step_size)

            # Changing this causes issues in pretraining - not sure why
            target_height = 32

            rectified_to_warped_x, rectified_to_warped_y, warped_to_rectified_x, warped_to_rectified_y, max_min = generate_offset_mapping(
                masked_img, ts, path, 0, -2 * target_step_size, cube_size=target_height)
            warped_above = cv2.remap(line_mask, rectified_to_warped_x, rectified_to_warped_y, cv2.INTER_CUBIC,
                                     borderValue=(0, 0, 0))

            rectified_to_warped_x, rectified_to_warped_y, warped_to_rectified_x, warped_to_rectified_y, max_min = generate_offset_mapping(
                masked_img, ts, path, 2 * target_step_size, 0, cube_size=target_height)
            warped_below = cv2.remap(line_mask, rectified_to_warped_x, rectified_to_warped_y, cv2.INTER_CUBIC,
                                     borderValue=(0, 0, 0))

            above_scale = np.max((warped_above.astype(float) / 255).sum(axis=0))
            below_scale = np.max((warped_below.astype(float) / 255).sum(axis=0))

            ab_sum = above_scale + below_scale
            above = target_step_size * (above_scale / ab_sum)
            below = target_step_size * (below_scale / ab_sum)

            above = target_step_size * (above_scale / (target_height / 2.0))
            below = target_step_size * (below_scale / (target_height / 2.0))
            target_step_size = above + below
            ts = find_t_spacing(path, target_step_size)

            rectified_to_warped_x, rectified_to_warped_y, warped_to_rectified_x, warped_to_rectified_y, max_min = generate_offset_mapping(
                masked_img, ts, path, below, -above, cube_size=target_height)

            rectified_to_warped_x = rectified_to_warped_x[::-1, ::-1]
            rectified_to_warped_y = rectified_to_warped_y[::-1, ::-1]
            warped_to_rectified_x = warped_to_rectified_x[::-1, ::-1]
            warped_to_rectified_y = warped_to_rectified_y[::-1, ::-1]

            warped = cv2.remap(img, rectified_to_warped_x, rectified_to_warped_y, cv2.INTER_CUBIC,
                               borderValue=(255, 255, 255))

            mapping = np.stack([rectified_to_warped_y, rectified_to_warped_x], axis=2)

            top_left = mapping[0, 0, :] / np.array(img.shape[:2]).astype(np.float32)
            btm_right = mapping[min(mapping.shape[0] - 1, target_height - 1),
                        min(mapping.shape[1] - 1, target_height - 1), :] / np.array(img.shape[:2]).astype(
                np.float32)

            line_points = []
            for i in range(0, mapping.shape[1], target_height):
                x0 = float(rectified_to_warped_x[0, i])
                x1 = float(rectified_to_warped_x[-1, i])

                y0 = float(rectified_to_warped_y[0, i])
                y1 = float(rectified_to_warped_y[-1, i])

                line_points.append({
                    "x0": x0,
                    "x1": x1,
                    "y0": y0,
                    "y1": y1
                })

            result.append({
                "line_index": line_index,
                # "line_image": warped_image_path,
                "bounding_poly": line['bounding_poly'],
                "original_image": pair.img,
                "region_index": region_index,
                "gt": line.get("ground_truth", ""),
                "sol": line_points[0],
                "lf": line_points,
            })

    return result


def get_namespace(element):
    m = re.match('\{.*\}', element.tag)
    return m.group(0) if m else ''


def readXMLFile(xml_string):
    root = xml.etree.ElementTree.fromstring(xml_string)
    namespace = get_namespace(root)
    return processXML(root, namespace)


def processXML(root, namespace):
    pages = []
    for page in root.findall(namespace + 'Page'):
        pages.append(process_page(page, namespace))
    return pages


def extract_points(data_string):
    return [tuple(int(x) for x in v.split(',')) for v in data_string.split()]


def process_region(region, namespace):
    region_out = {}

    coords = region.find(namespace + 'Coords')
    region_out['bounding_poly'] = extract_points(coords.attrib['points'])
    region_out['id'] = region.attrib['id']

    lines = []
    for line in region.findall(namespace + 'TextLine'):
        line_out = process_line(line, namespace)
        line_out['region_id'] = region.attrib['id']
        lines.append(line_out)

    ground_truth = None
    text_equiv = region.find(namespace + 'TextEquiv')
    if text_equiv is not None:
        ground_truth = text_equiv.find(namespace + 'Unicode').text

    region_out['ground_truth'] = ground_truth

    return region_out, lines


def process_page(page, namespace):
    page_out = {}
    regions = []
    lines = []
    for region in page.findall(namespace + 'TextRegion'):
        region_out, region_lines = process_region(region, namespace)

        regions.append(region_out)
        lines += region_lines

    graphic_regions = []
    for region in page.findall(namespace + 'GraphicRegion'):
        region_out, region_lines = process_region(region, namespace)
        graphic_regions.append(region_out)

    page_out['regions'] = regions
    page_out['lines'] = lines
    page_out['graphic_regions'] = graphic_regions

    return page_out


def process_line(line, namespace):
    errors = []
    line_out = {}

    if 'custom' in line.attrib:
        custom = line.attrib['custom']
        custom = custom.split(" ")
        if "readingOrder" in custom:
            roIdx = custom.index("readingOrder")
            ro = int("".join([v for v in custom[roIdx + 1] if v.isdigit()]))
            line_out['read_order'] = ro

    if 'id' in line.attrib:
        line_out['id'] = line.attrib['id']

    baseline = line.find(namespace + 'Baseline')

    if baseline is not None:
        line_out['baseline'] = extract_points(baseline.attrib['points'])
    else:
        errors.append('No baseline')

    coords = line.find(namespace + 'Coords')
    line_out['bounding_poly'] = extract_points(coords.attrib['points'])

    ground_truth = None
    text_equiv = line.find(namespace + 'TextEquiv')
    if text_equiv is not None:
        ground_truth = text_equiv.find(namespace + 'Unicode').text

    if ground_truth == None or len(ground_truth) == 0:
        errors.append("No ground truth")
        ground_truth = ""

    line_out['ground_truth'] = ground_truth
    if len(errors) > 0:
        line_out['errors'] = errors

    return line_out


def extract_region_mask(img, bounding_poly):
    pts = np.array(bounding_poly, np.int32)

    # http://stackoverflow.com/a/15343106/3479446
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    roi_corners = np.array([pts], dtype=np.int32)

    ignore_mask_color = (255,)
    cv2.fillPoly(mask, roi_corners, ignore_mask_color, lineType=cv2.LINE_8)
    return mask


def extract_baseline(img, pts):
    new_pts = []
    for i in range(len(pts) - 1):
        new_pts.append([pts[i], pts[i + 1]])
    pts = np.array(new_pts, np.int32)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)

    cv2.polylines(mask, pts, False, 255)
    return mask


def dis(pt1, pt2):
    a = (pt1.real - pt2.real) ** 2
    b = (pt1.imag - pt2.imag) ** 2
    return np.sqrt(a + b)


def complexToNpPt(pt):
    return np.array([pt.real, pt.imag], dtype=np.float32)


def normal(pt1, pt2):
    dif = pt1 - pt2
    return complex(-dif.imag, dif.real)


def find_t_spacing(path, cube_size):
    l = path.length()
    error = 0.01
    init_step_size = cube_size / l

    last_t = 0
    cur_t = 0
    pts = []
    ts = [0]
    pts.append(complexToNpPt(path.point(cur_t)))
    path_lookup = {}
    for target in np.arange(cube_size, int(l), cube_size):
        step_size = init_step_size
        for i in range(1000):
            cur_length = dis(path.point(last_t), path.point(cur_t))
            if np.abs(cur_length - cube_size) < error:
                break

            step_t = min(cur_t + step_size, 1.0)
            step_l = dis(path.point(last_t), path.point(step_t))

            if np.abs(step_l - cube_size) < np.abs(cur_length - cube_size):
                cur_t = step_t
                continue

            step_t = max(cur_t - step_size, 0.0)
            step_t = max(step_t, last_t)
            step_t = max(step_t, 1.0)

            step_l = dis(path.point(last_t), path.point(step_t))

            if np.abs(step_l - cube_size) < np.abs(cur_length - cube_size):
                cur_t = step_t
                continue

            step_size = step_size / 2.0

        last_t = cur_t

        ts.append(cur_t)
        pts.append(complexToNpPt(path.point(cur_t)))

    pts = np.array(pts)

    return ts


def generate_offset_mapping(img, ts, path, offset_1, offset_2, max_min=None, cube_size=None):
    # cube_size = 80

    offset_1_pts = []
    offset_2_pts = []
    # for t in ts:
    for i in range(len(ts)):
        t = ts[i]
        pt = path.point(t)

        norm = None
        if i == 0:
            norm = normal(pt, path.point(ts[i + 1]))
            norm = norm / dis(complex(0, 0), norm)
        elif i == len(ts) - 1:
            norm = normal(path.point(ts[i - 1]), pt)
            norm = norm / dis(complex(0, 0), norm)
        else:
            norm1 = normal(path.point(ts[i - 1]), pt)
            norm1 = norm1 / dis(complex(0, 0), norm1)
            norm2 = normal(pt, path.point(ts[i + 1]))
            norm2 = norm2 / dis(complex(0, 0), norm2)

            norm = (norm1 + norm2) / 2
            norm = norm / dis(complex(0, 0), norm)

        offset_vector1 = offset_1 * norm
        offset_vector2 = offset_2 * norm

        pt1 = pt + offset_vector1
        pt2 = pt + offset_vector2

        offset_1_pts.append(complexToNpPt(pt1))
        offset_2_pts.append(complexToNpPt(pt2))

    offset_1_pts = np.array(offset_1_pts)
    offset_2_pts = np.array(offset_2_pts)

    h, w = img.shape[:2]

    offset_source2 = np.array([(cube_size * i, 0) for i in range(len(offset_1_pts))], dtype=np.float32)
    offset_source1 = np.array([(cube_size * i, cube_size) for i in range(len(offset_2_pts))], dtype=np.float32)

    offset_source1 = offset_source1[::-1]
    offset_source2 = offset_source2[::-1]

    source = np.concatenate([offset_source1, offset_source2])
    destination = np.concatenate([offset_1_pts, offset_2_pts])

    source = source[:, ::-1]
    destination = destination[:, ::-1]

    n_w = int(offset_source2[:, 0].max())
    n_h = int(cube_size)

    grid_x, grid_y = np.mgrid[0:n_h, 0:n_w]

    grid_z = griddata(source, destination, (grid_x, grid_y), method='cubic')
    map_x = np.append([], [ar[:, 1] for ar in grid_z]).reshape(n_h, n_w)
    map_y = np.append([], [ar[:, 0] for ar in grid_z]).reshape(n_h, n_w)
    map_x_32 = map_x.astype('float32')
    map_y_32 = map_y.astype('float32')

    rectified_to_warped_x = map_x_32
    rectified_to_warped_y = map_y_32

    grid_x, grid_y = np.mgrid[0:h, 0:w]
    grid_z = griddata(source, destination, (grid_x, grid_y), method='cubic')
    map_x = np.append([], [ar[:, 1] for ar in grid_z]).reshape(h, w)
    map_y = np.append([], [ar[:, 0] for ar in grid_z]).reshape(h, w)
    map_x_32 = map_x.astype('float32')
    map_y_32 = map_y.astype('float32')

    warped_to_rectified_x = map_x_32
    warped_to_rectified_y = map_y_32

    return rectified_to_warped_x, rectified_to_warped_y, warped_to_rectified_x, warped_to_rectified_y, max_min
