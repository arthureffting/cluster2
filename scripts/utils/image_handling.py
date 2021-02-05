import os
import struct

import cv2
import numpy as np
from PIL import Image
from scipy.interpolate import griddata
import sys

INTERPOLATION = {
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC
}



def rescale(src_path, scale_percent=50):
    # percent by which the image is resized
    scale_percent = 50
    src = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
    # calculate the 50 percent of original dimensions
    width = int(src.shape[1] * scale_percent / 100)
    height = int(src.shape[0] * scale_percent / 100)
    # dsize
    dsize = (width, height)
    # resize image
    output = cv2.resize(src, dsize)
    cv2.imwrite(src_path, output)


def extract_line_section(img, upper_point, lower_point):
    # the order of the box points: bottom left, top left, top right,
    # bottom right
    box = np.int0(np.array([
        [int(upper_point.x) - 1, int(upper_point.y) - 1],
        [int(upper_point.x), int(upper_point.y)],
        [int(lower_point.x), int(lower_point.y)],
        [int(lower_point.x) - 1, int(lower_point.y) - 1]
    ]))
    cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
    # get width and height of the detected rectangle
    width = 1
    height = int(upper_point.distance(lower_point))
    src_pts = box.astype("float32")
    # coordinate of the points in box points after the rectangle has been
    # straightened
    dst_pts = np.array([[0, height - 1],
                        [0, 0],
                        [width - 1, 0],
                        [width - 1, height - 1]], dtype="float32")
    # the perspective transformation matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    # directly warp the rotated rectangle to get the straightened rectangle
    warped = cv2.warpPerspective(img, M, (width, height))
    # cv2.imwrite("crop_img.jpg", warped)
    return warped


def subimage(image, center, theta, width, height):
    '''
    Rotates OpenCV image around center with angle theta (in deg)
    then crops the image according to width and height.
    '''

    # Uncomment for theta in radians
    # theta *= 180/np.pi
    image = image[..., ::-1]
    shape = (image.shape[1], image.shape[0])  # cv2.warpAffine expects shape in (length, height)

    matrix = cv2.getRotationMatrix2D(center=center, angle=theta, scale=1)
    image = cv2.warpAffine(src=image, M=matrix, dsize=shape)

    x = int(center[0] - width / 2)
    y = int(center[1] - height / 2)

    left_border = 0 if x > 0 else -x
    top_border = 0 if y > 0 else -y

    x = 0 if x < 0 else x
    y = 0 if y < 0 else y

    max_x = x + int(width) - left_border
    max_y = y + int(height) - top_border

    bottom_border = 0 if max_y < image.shape[0] else max_y - image.shape[0]
    right_border = 0 if max_x < image.shape[1] else max_x - image.shape[1]

    max_x = image.shape[1] if max_x > image.shape[1] else max_x
    max_y = image.shape[0] if max_y > image.shape[0] else max_y

    image = image[y:max_y, x:max_x]

    make_borders = left_border != 0 or top_border != 0 or right_border != 0 or bottom_border != 0

    if make_borders:
        image = cv2.copyMakeBorder(
            image,
            top=top_border,
            bottom=bottom_border,
            left=left_border,
            right=right_border,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )

    return image


def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = max(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)

def vconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    v_min = max(im.shape[1] for im in im_list)
    im_list_resize = [cv2.resize(im, (v_min, int(im.shape[0] * v_min / im.shape[1])), interpolation=interpolation)
                      for im in im_list]
    return cv2.vconcat(im_list_resize)


def vertical_concat_resize(imgs, color=(255, 255, 255)):
    max_width = max(img.width for img in imgs)
    resized = [img.resize((max_width, img.height * int(max_width / img.width))) for img in imgs]
    total_height = sum(img.height for img in imgs)
    dst = Image.new('RGB', (max_width, total_height), color)
    current_height = 0
    for img in resized:
        dst.paste(img, (0, current_height))
        current_height = img.height + current_height
    return dst


def get_concat_h_blank(im1, im2, color=(255, 255, 255), padding=4):
    target_height = max(im1.height, im2.height)
    resized = [img.resize((img.width * int(target_height / img.height), target_height)) for img in [im1, im2]]
    total_width = sum(img.width for img in resized) + padding
    dst = Image.new('RGB', (total_width, target_height), color)

    current_width = 0
    for img in resized:
        dst.paste(img, (current_width, 0))
        current_width = img.width + current_width + padding
    return dst


def get_concat_v_blank(im1, im2, color=(255, 255, 255), padding=4):
    dst = Image.new('RGB', (max(im1.width, im2.width), im1.height + im2.height + padding), color)
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height + padding))
    return dst


def add_padding(im1, padding=4, color=(255, 255, 255)):
    dst = Image.new('RGB', (im1.width + 2 * padding, im1.height + 2 * padding), color)
    dst.paste(im1, (padding, padding))
    return dst


def add_horizontal_padding(im1, padding=4, color=(255, 255, 255)):
    dst = Image.new('RGB', (im1.width + 2 * int(padding), im1.height), color)
    dst.paste(im1, (int(padding), 0))
    return dst


def get_concat_h_multi_blank(im_list):
    _im = im_list.pop(0)
    for im in im_list:
        _im = get_concat_h_blank(_im, im)
    return _im


def tensmeyer_brightness(img, foreground=0, background=0):
    img_bgr = img.astype(dtype=np.uint8)
    ret, th = cv2.threshold(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = (th.astype(np.float32) / 255)[..., None]
    img = img.astype(np.float32)
    img = img + (1.0 - th) * foreground
    img = img + th * background
    img[img > 255] = 255
    img[img < 0] = 0
    return img.astype(np.uint8)


def apply_tensmeyer_brightness(img, sigma=30, **kwargs):
    random_state = np.random.RandomState(kwargs.get("random_seed", None))
    foreground = random_state.normal(0, sigma)
    background = random_state.normal(0, sigma)
    img = tensmeyer_brightness(img, foreground, background)
    return img


def increase_brightness(img, brightness=0, contrast=1):
    img = img.astype(np.float32)
    img = img * contrast + brightness
    img[img > 255] = 255
    img[img < 0] = 0

    return img.astype(np.uint8)


def apply_random_brightness(img, b_range=[-50, 51], **kwargs):
    random_state = np.random.RandomState(kwargs.get("random_seed", None))
    brightness = random_state.randint(b_range[0], b_range[1])
    img = increase_brightness(img, brightness)
    return img


def apply_random_color_rotation(img, **kwargs):
    random_state = np.random.RandomState(kwargs.get("random_seed", None))
    shift = random_state.randint(0, 255)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[..., 0] = hsv[..., 0] + shift
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return img


class UnknownImageFormat(Exception):
    pass


def get_image_size(file_path):
    """
    Return (width, height) for a given img file content - no external
    dependencies except the os and struct modules from core
    """
    size = os.path.getsize(file_path)

    with open(file_path) as input:
        height = -1
        width = -1
        data = input.read(25)

        if (size >= 10) and data[:6] in ('GIF87a', 'GIF89a'):
            # GIFs
            w, h = struct.unpack("<HH", data[6:10])
            width = int(w)
            height = int(h)
        elif ((size >= 24) and data.startswith('\211PNG\r\n\032\n')
              and (data[12:16] == 'IHDR')):
            # PNGs
            w, h = struct.unpack(">LL", data[16:24])
            width = int(w)
            height = int(h)
        elif (size >= 16) and data.startswith('\211PNG\r\n\032\n'):
            # older PNGs?
            w, h = struct.unpack(">LL", data[8:16])
            width = int(w)
            height = int(h)
        elif (size >= 2) and data.startswith('\377\330'):
            # JPEG
            msg = " raised while trying to decode as JPEG."
            input.seek(0)
            input.read(2)
            b = input.read(1)
            try:
                while (b and ord(b) != 0xDA):
                    while (ord(b) != 0xFF): b = input.read(1)
                    while (ord(b) == 0xFF): b = input.read(1)
                    if (ord(b) >= 0xC0 and ord(b) <= 0xC3):
                        input.read(3)
                        h, w = struct.unpack(">HH", input.read(4))
                        break
                    else:
                        input.read(int(struct.unpack(">H", input.read(2))[0]) - 2)
                    b = input.read(1)
                width = int(w)
                height = int(h)
            except struct.error:
                raise UnknownImageFormat("StructError" + msg)
            except ValueError:
                raise UnknownImageFormat("ValueError" + msg)
            except Exception as e:
                raise UnknownImageFormat(e.__class__.__name__ + msg)
        else:
            raise UnknownImageFormat(
                "Sorry, don't know how to get information from this file."
            )

    return width, height


def warp_image(img, random_state=None, **kwargs):
    if random_state is None:
        random_state = np.random.RandomState()

    w_mesh_interval = kwargs.get('w_mesh_interval', 12)
    w_mesh_std = kwargs.get('w_mesh_std', 1.5)

    h_mesh_interval = kwargs.get('h_mesh_interval', 12)
    h_mesh_std = kwargs.get('h_mesh_std', 1.5)

    interpolation_method = kwargs.get('interpolation', 'linear')

    h, w = img.shape[:2]

    if kwargs.get("fit_interval_to_image", True):
        # Change interval so it fits the image size
        w_ratio = w / float(w_mesh_interval)
        h_ratio = h / float(h_mesh_interval)

        w_ratio = max(1, round(w_ratio))
        h_ratio = max(1, round(h_ratio))

        w_mesh_interval = w / w_ratio
        h_mesh_interval = h / h_ratio
        ############################################

    # Get control points
    source = np.mgrid[0:h+h_mesh_interval:h_mesh_interval, 0:w+w_mesh_interval:w_mesh_interval]
    source = source.transpose(1,2,0).reshape(-1,2)

    if kwargs.get("draw_grid_lines", False):
        if len(img.shape) == 2:
            color = 0
        else:
            color = np.array([0,0,255])
        for s in source:
            img[int(s[0]):int(s[0])+1,:] = color
            img[:,int(s[1]):int(s[1])+1] = color

    # Perturb source control points
    destination = source.copy()
    source_shape = source.shape[:1]
    destination[:,0] = destination[:,0] + random_state.normal(0.0, h_mesh_std, size=source_shape)
    destination[:,1] = destination[:,1] + random_state.normal(0.0, w_mesh_std, size=source_shape)

    # Warp image
    grid_x, grid_y = np.mgrid[0:h, 0:w]
    grid_z = griddata(destination, source, (grid_x, grid_y), method=interpolation_method).astype(np.float32)
    map_x = grid_z[:,:,1]
    map_y = grid_z[:,:,0]
    warped = cv2.remap(img, map_x, map_y, INTERPOLATION[interpolation_method], borderValue=(255,255,255))

    return warped