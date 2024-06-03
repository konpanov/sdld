from typing import List
import numpy as np
import cv2

from cv2.typing import MatLike
from segment import Segment
from utils import *


def cutout(obj: MatLike):
    x1, y1, x2, y2 = find_roi(obj)
    return obj[y1:y2+2,x1:x2+2]

def M(obj: MatLike, p, q):
    if p == 0 and q == 0:
        return np.sum(obj)
    elif p == 0:
        return np.sum(np.arange(obj.shape[0])**q * np.sum(obj, axis=1))
    elif q == 0:
        return np.sum(np.arange(obj.shape[1])**p * np.sum(obj, axis=0))
    else:
        indices_x = np.arange(obj.shape[1])
        indices_y = np.arange(obj.shape[0])
        return np.sum((indices_x**p) * (indices_y**q) * obj)



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
    M = central_moments(dot.img)
    dotcX = int(M["m10"] / M["m00"])
    dotcX = int(M["m10"] / M["m00"])
    dotcY = int(M["m01"] / M["m00"])
    M = central_moments(arc.img)
    arccX = int(M["m10"] / M["m00"])
    arccY = int(M["m01"] / M["m00"])

    d = np.sqrt(dot.count)/ odleglosc([dotcX, dotcY], [arccX, arccY])
    return good_proportion and d > 1 and d < 2

dots_pattern1 = cv2.imread("dots/3-52.jpg", cv2.IMREAD_GRAYSCALE)
dots_pattern2 = cv2.imread("dots/1-7.jpg", cv2.IMREAD_GRAYSCALE)
def is_dot(seg : Segment):
    return match_shapes(seg.img, dots_pattern1) < .1 or match_shapes(seg.img, dots_pattern2) < .1

arcs_pattern1 = cv2.imread("arcs/3-8.jpg", cv2.IMREAD_GRAYSCALE)
arcs_pattern2 = cv2.imread("arcs/7-444.jpg", cv2.IMREAD_GRAYSCALE)
def is_arc(seg : Segment):
    return match_shapes(seg.img, arcs_pattern1) < .3 or match_shapes(seg.img, arcs_pattern2) < .1

def find_dots(segments: List[Segment]):
    return [seg for seg in segments if is_dot(seg)]

def find_arcs(segments: List[Segment]):
    return [seg for seg in segments if is_arc(seg)]


def Hu(obj: MatLike):
    return cv2.HuMoments(cv2.moments(obj))

def match_shapes(A: MatLike, B: MatLike):
    A = cutout(A)
    B = cutout(B)
    huA = Hu(A)
    huB = Hu(B)
    eps = 1.e-5;
    nonzero = np.logical_and(np.abs(huA) > eps, np.abs(huB) > eps)
    huA = huA[nonzero]
    huB = huB[nonzero]
    mA = np.sign(huA) * np.log(huA)
    mB = np.sign(huB) * np.log(huB)
    return np.sum(np.abs(mA - mB))

