from typing import List
import numpy as np
import cv2
import sys

from cv2.typing import MatLike
from segment import Segment
from utils import *
eps = sys.float_info.epsilon


def cutout(obj: MatLike):
    x1, y1, x2, y2 = find_roi(obj)
    return obj[y1:y2+2,x1:x2+2]

# def M(obj: MatLike, p, q):
#     if p == 0 and q == 0:
#         return np.sum(obj)
#     elif p == 0:
#         return np.sum(np.arange(obj.shape[0])**q * np.sum(obj, axis=1))
#     elif q == 0:
#         return np.sum(np.arange(obj.shape[1])**p * np.sum(obj, axis=0))
#     else:
#         indices_x = np.arange(obj.shape[1])
#         indices_y = np.arange(obj.shape[0])
#         return np.sum((indices_x**p) * (indices_y**q) * obj)
#


def central_moments(obj: MatLike):
    xs = np.arange(obj.shape[1])
    ys = np.arange(obj.shape[0])
    xsum = np.sum(obj, axis=0)
    ysum = np.sum(obj, axis=1)
    m00 = np.sum(obj)
    m10 = np.sum(xs * xsum)
    m01 = np.sum(ys * ysum)
    return {'m00': m00, 'm10': m10, 'm01': m01}


def is_logo(dot: Segment, arc: Segment):
    proportion = (dot.count / arc.count)
    good_proportion = proportion < 1.25 and proportion > .75
    if not good_proportion:
        return False
    m00, m10, m01, *_ = M(dot.img)
    dotcX = m10 / m00
    dotcY = m01 / m00
    m00, m10, m01, *_ = M(arc.img)
    arccX = m10 / m00
    arccY = m01 / m00

    odleg = odleglosc([dotcX, dotcY], [arccX, arccY])
    if odleg < eps:
        return False

    d = np.sqrt(dot.count) / odleg
    return good_proportion and d > 1 and d < 2

def Hu(obj: MatLike):
    m00, m10, m01, m11, m20, m02, m12, m21, m30, m03 = M(obj)

    cx = m10 / m00
    cy = m01 / m00
    mu11 = m11 - cy * m10
    mu20 = m20 - cx * m10
    mu02 = m02 - cy * m01
    mu30 = m30 - 3 * cx * m20 + 2 * cx**2 * m10
    mu03 = m03 - 3 * cy * m02 + 2 * cy**2 * m01
    mu12 = m12 - 2 * cy * m11 + - cx * m02 + 2 * cy**2 * m10 
    mu21 = m21 - 2 * cx * m11 - cy * m20 + 2 * cx**2 * m01

    nu11 = mu11 / (m00**(1 + (1 + 1)/2))
    nu20 = mu20 / (m00**(1 + (2 + 0)/2))
    nu02 = mu02 / (m00**(1 + (0 + 2)/2))
    nu30 = mu30 / (m00**(1 + (3 + 0)/2))
    nu03 = mu03 / (m00**(1 + (0 + 3)/2))
    nu12 = mu12 / (m00**(1 + (1 + 2)/2))
    nu21 = mu21 / (m00**(1 + (2 + 1)/2))

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
    hu = np.array([h0, h1, h2, h3, h4 ,h5 ,h6])
    # print(hu, cv2.HuMoments(cv2.moments(img)))
    return hu


dots_pattern1 = Hu(cutout(cv2.imread("dots/3-52.jpg", cv2.IMREAD_GRAYSCALE)))
dots_pattern2 = Hu(cutout(cv2.imread("dots/1-7.jpg", cv2.IMREAD_GRAYSCALE)))

def is_dot(seg : Segment):
    return match_shapes(seg.img, dots_pattern1) < .1 or match_shapes(seg.img, dots_pattern2) < .1

arcs_pattern1 = Hu(cutout(cv2.imread("arcs/3-8.jpg", cv2.IMREAD_GRAYSCALE)))
arcs_pattern2 = Hu(cutout(cv2.imread("arcs/7-444.jpg", cv2.IMREAD_GRAYSCALE)))

def is_arc(seg : Segment):
    return match_shapes(seg.img, arcs_pattern1) < .3 or match_shapes(seg.img, arcs_pattern2) < .1

def find_dots(segments: List[Segment]):
    return [seg for seg in segments if is_dot(seg)]

def find_arcs(segments: List[Segment]):
    return [seg for seg in segments if is_arc(seg)]

def rate_arcs(segments: List[Segment]):
    patterns = [arcs_pattern1]
    return [np.mean([match_shapes(seg.img, pat) for pat in patterns]) for seg in segments]

def rate_dots(segments: List[Segment]):
    patterns = [dots_pattern1]
    return [np.mean([match_shapes(seg.img, pat) for pat in patterns]) for seg in segments]

def M(obj: MatLike):
    h, w = obj.shape[:2]
    obj = obj.astype(np.uint32)
    ys, xs = np.arange(h), np.arange(w)
    xsum, ysum = np.sum(obj, axis=1), np.sum(obj, axis=0)
    m00 = np.sum(obj)
    m10 = np.sum(ysum * xs)
    m01 = np.sum(xsum * ys)
    m11 = np.sum(obj * np.outer(ys, xs))
    m20 = np.sum(ysum * xs ** 2)
    m02 = np.sum(xsum * ys ** 2)
    m12 = np.sum(obj * np.outer(ys ** 2, xs))
    m21 = np.sum(obj * np.outer(ys, xs ** 2))
    m30 = np.sum(ysum * xs ** 3)
    m03 = np.sum(xsum * ys ** 3)
    return m00, m10, m01, m11, m20, m02, m12, m21, m30, m03

def match_shapes(A: MatLike, B):
    # A = cutout(A)
    # B = cutout(B)
    huA = Hu(A)
    huB = B
    eps = 1.e-5;
    nonzero = np.logical_and(np.abs(huA) > eps, np.abs(huB) > eps)
    huA = huA[nonzero]
    huB = huB[nonzero]
    mA = np.sign(huA) * np.log(huA)
    mB = np.sign(huB) * np.log(huB)
    return np.sum(np.abs(mA - mB))

