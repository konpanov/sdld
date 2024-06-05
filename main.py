import itertools
import os
from itertools import product

import cv2
import numpy as np
from cv2.typing import MatLike

from utils import ccl
from utils import downsample
from utils import draw_obj_roi
from utils import grey_scale
from utils import is_logo
from utils import pole_check
from utils import rate_arcs
from utils import rate_dots
from utils import recolor_mask
from utils import Segment
from utils import select_color


def median_cut_quantize(img, img_arr):
    # when it reaches the end, color quantize
    avg = np.mean(img_arr[:, :3], axis=0)
    img[img_arr[:, 3], img_arr[:, 4]] = avg


def split_into_buckets(img, img_arr, depth):
    if len(img_arr) == 0:
        return

    if depth == 0:
        median_cut_quantize(img, img_arr)
        return

    r_range = np.max(img_arr[:, 0]) - np.min(img_arr[:, 0])
    g_range = np.max(img_arr[:, 1]) - np.min(img_arr[:, 1])
    b_range = np.max(img_arr[:, 2]) - np.min(img_arr[:, 2])

    space_with_highest_range = 0

    if g_range >= r_range and g_range >= b_range:
        space_with_highest_range = 1
    elif b_range >= r_range and b_range >= g_range:
        space_with_highest_range = 2
    elif r_range >= b_range and r_range >= g_range:
        space_with_highest_range = 0

    # sort the image pixels by color space with highest range
    # and find the median and divide the array.
    img_arr = img_arr[img_arr[:, space_with_highest_range].argsort()]
    median_index = int((len(img_arr) + 1) / 2)

    # split the array into two blocks
    split_into_buckets(img, img_arr[0:median_index], depth - 1)
    split_into_buckets(img, img_arr[median_index:], depth - 1)


def median_cut(img: MatLike, num_colors):
    h, w = img.shape[:2]
    bucket = itertools.product(range(h), range(w))
    bucket = np.array([[*img[y, x], y, x] for y, x in bucket])
    split_into_buckets(img, bucket, num_colors)
    return img


def find_segments(img: MatLike):
    # cv2.imshow("img", img)
    img = median_cut(img.copy(), 10)
    img = grey_scale(img)
    N = 4
    img = np.floor(img.astype(np.float16) * (N / 255)) * (255 // N)
    img = img.astype(np.uint8)
    cv2.imshow("median_cut", img)
    mask, counts, rois, ms = ccl(img)
    recolor_mask(len(counts), mask)
    w, h = img.shape[:2]
    segments = [
        Segment(i, select_color(mask, i), counts[i], rois[i], ms[i])
        for i in range(len(counts))
        if pole_check(w, h, counts[i])
    ]
    return segments


if __name__ == "__main__":
    # get_dot_pattern()
    dir = "images"
    for i, infile in enumerate(os.listdir(dir)):
        # if i != 2:
        #     continue
        infile = os.path.join(dir, infile)
        img = cv2.imread(infile)
        img = downsample(img)
        segments = find_segments(img)
        dots = zip(rate_dots(segments), segments)
        arcs = zip(rate_arcs(segments), segments)
        dots = sorted(dots, key=lambda x: x[0])
        arcs = sorted(arcs, key=lambda x: x[0])
        # dots = dots[: min(len(dots), 100)]
        # arcs = dots[: min(len(dots), 100)]
        logos = sorted(product(dots, arcs), key=lambda x: x[0][0] * x[1][0])
        for dot, arc in logos:
            if dot[0] + arc[0] >= 0.3:
                continue
            if is_logo(dot[1], arc[1]):
                img = draw_obj_roi(img, dot[1], color=(255, 0, 0))
                img = draw_obj_roi(img, arc[1], color=(255, 0, 0))
            # tmp = img.copy()
            # tmp = draw_obj_roi(tmp, dot[1], color=(255, 0, 0))
            # tmp = draw_obj_roi(tmp, arc[1], color=(255, 0, 0))
            # cv2.imshow(f"{dot[0]:.4f}/{arc[0]:.4f}/{is_logo(dot[1], arc[1])}", tmp)
            # cv2.waitKey(0)

        cv2.imshow(f"{i}/{infile}", img)
        cv2.waitKey(1)

    # cv2.waitKey(0)
