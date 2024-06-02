import numpy as np
import cv2

from cv2.typing import MatLike
from utils import *


def pole(img:MatLike):
    return np.sum(img)

def obwod(img: MatLike):
    krawedz = cv2.Laplacian(img, cv2.CV_8U) # TODO: Rewrite
    return pole(krawedz)

def cutout(obj: MatLike):
    x1, y1, x2, y2 = find_roi(obj)
    return obj[y1:y2+2,x1:x2+2]

def Hu(obj: MatLike, txt=""):
    # obj = cv2.threshold(obj, 128)
    # obj = cv2.Laplacian(obj, cv2.CV_8U)
    # obj = cutout(obj)
    # obj = cv2.resize(obj, (50, 50))
    # cv2.imshow(txt, obj)
    # cv2.waitKey(0)
    moments = cv2.moments(obj)
    huMoments = cv2.HuMoments(moments)
    return huMoments


def findHuMoments(obj: MatLike, name=""):
    obj = cutout(obj)
    # obj = cv2.resize(obj, (50, 50))
    # obj = cv2.Laplacian(obj, cv2.CV_8U)
    # cv2.imshow(f"{name}", obj)
    # cv2.waitKey(0)
    moments = cv2.moments(obj)
    huMoments = cv2.HuMoments(moments)
    return huMoments

def detect(obj: MatLike, pattern_hus):
    hu = findHuMoments(obj, "end")
    dists = [odleglosc(hu, pattern) for pattern in pattern_hus]
    d = np.mean(dists)
    # dists = odleglosc(hu, pattern_hus)
    # print(hu)
    # print(pattern_hus)
    # return True, dists
    return any(d <= .026 for d in dists), dists
    # return any(d <= .00001 for d in dists), dists
    for pattern_hu in pattern_hus:
        d = odleglosc(hu, pattern_hu) 
        is_detected = d <= .01
        if is_detected:
            return is_detected, d
    return False, 1



