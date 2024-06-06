import itertools
import os
import pickle
import sys
from typing import List
from typing import Tuple

import cv2
import numpy as np
from cv2.typing import MatLike
from numpy.typing import ArrayLike

eps = sys.float_info.epsilon


def downsample(img: MatLike):
    h, w = img.shape[:2]
    while w * h > 1000000:
        img = img[::2, ::2, :].astype(np.uint8)
        h, w = h // 2, w // 2
    return img


def find_segments(img: MatLike):
    img = median_cut(img.copy(), 10)
    img = grey_scale(img)
    img = quant(img, 4)
    mask, counts, rois, ms = ccl(img)
    recolor_mask(len(counts), mask)
    w, h = img.shape[:2]
    segments = [
        Segment(i, select_color(mask, i), counts[i], rois[i], ms[i])
        for i in range(len(counts))
        if pole_check(w, h, counts[i])
    ]
    return segments


def median_cut(img: MatLike, num_colors):

    def step(colors, yx, depth):
        if len(colors) == 0:
            return

        y = yx[:, 0]
        x = yx[:, 1]

        if depth == 0:
            img[y, x] = np.mean(colors, axis=0)
            return

        ranges = np.max(colors, axis=0) - np.min(colors, axis=0)
        poses = colors[:, ranges.argmax()].argsort()
        colors = colors[poses]
        yx = yx[poses]
        median = len(colors) // 2
        step(colors[:median], yx[:median], depth - 1)
        step(colors[median:], yx[median:], depth - 1)

    h, w = img.shape[:2]
    yx = np.array(list(itertools.product(range(h), range(w))))
    color = img.reshape(-1, 3)
    step(color, yx, num_colors)
    return img


def grey_scale(img: MatLike):
    return np.array(np.sum(img, axis=2) // 3, dtype=np.uint8)


def quant(img: MatLike, to: int):
    img = np.floor(img.astype(np.float16) * (to / 255)) * (255 // to)
    img = img.astype(np.uint8)
    return img


class Segment:
    id: int
    img: MatLike
    count: int
    roi: Tuple[int, int, int, int]
    m: np.ndarray
    desc: ArrayLike

    def __init__(self, id, img, count, roi, moments) -> None:
        self.id = id
        self.img = img
        self.count = count
        self.roi = roi
        self.m = moments


def draw_obj_roi(img: MatLike, obj: Segment, color=(255, 0, 0)):
    x1, x2, y1, y2 = obj.roi
    h, w = img.shape[:2]
    for y in range(h):
        for x in range(w):
            if (x == x1 or x == x2) and y >= y1 and y <= y2:
                img[y, x] = color
            if (y == y1 or y == y2) and x >= x1 and x <= x2:
                img[y, x] = color
    return img


def M(obj: MatLike):
    h, w = obj.shape[:2]
    obj = obj.astype(np.uint32)
    ys, xs = np.arange(h), np.arange(w)
    xsum, ysum = np.sum(obj, axis=1), np.sum(obj, axis=0)
    m00 = np.sum(obj)
    m10 = np.sum(ysum * xs)
    m01 = np.sum(xsum * ys)
    m11 = np.sum(obj * np.outer(ys, xs))
    m20 = np.sum(ysum * xs**2)
    m02 = np.sum(xsum * ys**2)
    m12 = np.sum(obj * np.outer(ys**2, xs))
    m21 = np.sum(obj * np.outer(ys, xs**2))
    m30 = np.sum(ysum * xs**3)
    m03 = np.sum(xsum * ys**3)
    return m00, m10, m01, m11, m20, m02, m12, m21, m30, m03


def is_logo(dot: Segment, arc: Segment):
    proportion = dot.count / arc.count
    good_proportion = proportion < 1.5 and proportion > 0.5
    if not good_proportion:
        return False
    m00, m10, m01 = dot.m[:3]
    dotcX = m10 / m00
    dotcY = m01 / m00
    m00, m10, m01 = arc.m[:3]
    arccX = m10 / m00
    arccY = m01 / m00

    odleg = odleglosc([dotcX, dotcY], [arccX, arccY])
    if odleg < eps:
        return False

    d = np.sqrt(dot.count) / odleg
    return d > 0.5 and d < 2.5


def odleglosc(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.sqrt(((vec1 - vec2) ** 2).sum())


def Hu(moments):
    m00, m10, m01, m11, m20, m02, m12, m21, m30, m03 = moments

    cx = m10 / m00
    cy = m01 / m00
    mu11 = m11 - cy * m10
    mu20 = m20 - cx * m10
    mu02 = m02 - cy * m01
    mu30 = m30 - 3 * cx * m20 + 2 * cx**2 * m10
    mu03 = m03 - 3 * cy * m02 + 2 * cy**2 * m01
    mu12 = m12 - 2 * cy * m11 + -cx * m02 + 2 * cy**2 * m10
    mu21 = m21 - 2 * cx * m11 - cy * m20 + 2 * cx**2 * m01

    nu11 = mu11 / (m00 ** (1 + (1 + 1) / 2))
    nu20 = mu20 / (m00 ** (1 + (2 + 0) / 2))
    nu02 = mu02 / (m00 ** (1 + (0 + 2) / 2))
    nu30 = mu30 / (m00 ** (1 + (3 + 0) / 2))
    nu03 = mu03 / (m00 ** (1 + (0 + 3) / 2))
    nu12 = mu12 / (m00 ** (1 + (1 + 2) / 2))
    nu21 = mu21 / (m00 ** (1 + (2 + 1) / 2))

    t0 = nu30 + nu12
    t1 = nu21 + nu03
    q0 = t0 * t0
    q1 = t1 * t1
    n4 = 4 * nu11
    s = nu20 + nu02
    d = nu20 - nu02
    h0 = s
    h1 = d * d + n4 * nu11
    h3 = q0 + q1
    h5 = d * (q0 - q1) + n4 * t0 * t1
    t0 *= q0 - 3 * q1
    t1 *= 3 * q0 - q1
    q0 = nu30 - 3 * nu12
    q1 = 3 * nu21 - nu03
    h2 = q0 * q0 + q1 * q1
    h4 = q0 * t0 + q1 * t1
    h6 = q1 * t0 - q0 * t1
    hu = np.array([h0, h1, h2, h3, h4, h5, h6])
    return hu


def load_hus_from_dir(dir):
    fnames = [os.path.join(dir, fname) for fname in os.listdir(dir)]
    imgs = [cv2.imread(fname) for fname in fnames]
    return [Hu(img) for img in imgs]


def rate_arcs(segments: List[Segment]):
    patterns = load_hus_from_dir("arcs")
    return [np.mean([match_shapes(seg, pat) for pat in patterns]) for seg in segments]


def rate_dots(segments: List[Segment]):
    patterns = load_hus_from_dir("dots")
    return [
        np.min([match_shapes(seg, pat) for pat in patterns], axis=0) for seg in segments
    ]


def match_shapes(seg: Segment, pattern_hu):
    huA = Hu(seg.m)
    huB = pattern_hu
    eps = 1.0e-5
    nonzero = np.logical_and(np.abs(huA) > eps, np.abs(huB) > eps)
    huA = huA[nonzero]
    huB = huB[nonzero]
    mA = np.sign(huA) * np.log(huA)
    mB = np.sign(huB) * np.log(huB)
    return np.sum(np.abs(mA - mB))


def select_color(img: MatLike, value):
    return 255 * (img == np.intp(value))


def ccl_firstpass(img: MatLike):
    h, w = img.shape[:2]
    labels = np.zeros(img.shape[:2], dtype=np.uint32)
    roots = []
    ret = 0
    for y in range(h):
        for x in range(w):
            left = x != 0 and (img[y, x] == img[y, x - 1]).all()
            top = y != 0 and (img[y, x] == img[y - 1, x]).all()
            if left and top:
                leftl = labels[y, x - 1]
                topl = labels[y - 1, x]
                label = min(roots[leftl], roots[topl])
                labels[y, x] = label
                roots[topl] = label
                roots[leftl] = label
            elif left:
                label = labels[y, x - 1]
                labels[y, x] = label
            elif top:
                label = labels[y - 1, x]
                labels[y, x] = label
            else:
                labels[y, x] = ret
                roots.append(ret)
                ret += 1
    return labels, roots


def ccl_second_pass(labels, roots):
    h, w = labels.shape[:2]
    relabel = []
    ret = 0
    for i, root in enumerate(roots):
        if root == i:
            relabel.append(ret)
            ret += 1
        else:
            roots[i] = roots[root]
            relabel.append(relabel[roots[i]])
    ms = np.zeros((ret, 10))
    counts = np.zeros(ret)
    rois = [[w, 0, h, 0] for _ in range(ret)]

    def calc_m(m, x, y):
        m[0] += 255  # m00
        m[1] += 255 * x  # m10
        m[2] += 255 * y  # m01
        m[3] += 255 * x * y  # m11
        m[4] += 255 * x * x  # m20
        m[5] += 255 * y * y  # m02
        m[6] += 255 * y * y * x  # m12
        m[7] += 255 * x * x * y  # m21
        m[8] += 255 * x * x * x  # m30
        m[9] += 255 * y * y * y  # m03

    for y in range(h):
        for x in range(w):
            label = relabel[labels[y, x]]
            labels[y, x] = label
            counts[label] += 1
    for y in range(h):
        for x in range(w):
            label = labels[y, x]
            if not pole_check(w, h, counts[label]):
                continue
            rois[label][0] = min(rois[label][0], x)
            rois[label][1] = max(rois[label][1], x)
            rois[label][2] = min(rois[label][2], y)
            rois[label][3] = max(rois[label][3], y)
            calc_m(ms[label], x, y)
    return labels, counts, rois, ms


def ccl(img: MatLike):
    return ccl_second_pass(*ccl_firstpass(img))


def recolor_mask(label_num, mask: MatLike):
    h, w = mask.shape[:2]
    img = np.zeros((h, w, 3))
    colors = [
        np.array((np.random.random(), np.random.random(), np.random.random()))
        for _ in range(label_num)
    ]
    for y in range(h):
        for x in range(w):
            img[y, x] = colors[mask[y, x]]
    cv2.imshow(f"{label_num}, {np.max(mask)}", img)
    cv2.waitKey(1)


def pole_check(w, h, count):
    stosunek = count / (w * h)
    return stosunek >= 0.0001 and stosunek < 0.25
