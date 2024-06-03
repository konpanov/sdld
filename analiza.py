from typing import List
import numpy as np
import cv2

from cv2.typing import MatLike
from segment import Segment
from utils import *


def pole(img:MatLike):
    return np.sum(img)//np.max(img)

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
    moments = cv2.moments(obj, binaryImage=True)
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

# def detect(obj: MatLike, pattern_hus):
#     hu = findHuMoments(obj, "end")
#     dists = [odleglosc(hu, pattern) for pattern in pattern_hus]
#     d = np.mean(dists)
#     # dists = odleglosc(hu, pattern_hus)
#     # print(hu)
#     # print(pattern_hus)
#     # return True, dists
#     return any(d <= .026 for d in dists), dists
#     # return any(d <= .00001 for d in dists), dists
#     for pattern_hu in pattern_hus:
#         d = odleglosc(hu, pattern_hu) 
#         is_detected = d <= .01
#         if is_detected:
#             return is_detected, d
#     return False, 1

dots_pattern = cv2.imread("dots/3-52.jpg", cv2.IMREAD_GRAYSCALE)

def is_logo(dot: Segment, arc: Segment):
    proportion = (dot.count / arc.count)
    good_proportion = proportion < 1.5
    M = cv2.moments(dot.img)
    dotcX = int(M["m10"] / M["m00"])
    dotcY = int(M["m01"] / M["m00"])
    M = cv2.moments(arc.img)
    arccX = int(M["m10"] / M["m00"])
    arccY = int(M["m01"] / M["m00"])

    d = np.sqrt(dot.count)/ odleglosc([dotcX, dotcY], [arccX, arccY])
    # print(proportion, d)
    return good_proportion and d > 1 and d < 2

def is_dot(seg : Segment):
    return match_shapes(seg.img, dots_pattern) < .2

arcs_pattern1 = cv2.imread("arcs/3-8.jpg", cv2.IMREAD_GRAYSCALE)
arcs_pattern2 = cv2.imread("arcs/7-444.jpg", cv2.IMREAD_GRAYSCALE)
def is_arc(seg : Segment):
    return match_shapes(seg.img, arcs_pattern1) < .3 or match_shapes(seg.img, arcs_pattern2) < .1

def find_dots(segments: List[Segment]):
    return [seg for seg in segments if is_dot(seg)]

def find_arcs(segments: List[Segment]):
    return [seg for seg in segments if is_arc(seg)]



def match_shapes(A: MatLike, B: MatLike):
    A = cutout(A)
    B = cutout(B)
    # A = resize(cutout(A), 300)
    # B = resize(cutout(B), 300)
    return cv2.matchShapes(A, B, cv2.CONTOURS_MATCH_I2, 0.)

def detect_arc(obj: MatLike):
    return pole_check(obj) and (detect_arc_straight(obj) or detect_arc_angle_left(obj) or detect_arc_angle_bottom(obj))

def pole_check(obj: Segment):
    w, h = obj.img.shape[:2]
    return obj.count/(w*h) >= 0.0005

def detect_arc_straight(obj: MatLike):
    pattern = np.array([[ 4.07952677e-01],
                        [ 6.96002152e-02],
                        [ 2.32427969e-02],
                        [ 3.32455031e-03],
                        [-2.87032991e-05],
                        [-8.67187617e-04],
                        [-5.49346259e-06]]
                       ) 

    # margin = np.array([[ .1], [.7], [.3], [.3], [.5], [.1], [.1] ])
    # margin = np.array([[ .1], [.07], [.03], [.003], [.05], [.1], [.1] ])
    margin = np.array([[ .1], [.07], [.3], [.004], [.05], [.01], [.00002] ])
    hu = Hu(obj)
    diff = np.abs(hu-pattern)
    # print(np.abs(hu-pattern) < margin)
    return [(diff < margin).all(), diff]

def detect_arc_angle_left(obj: MatLike):
    pattern = np.array([[ 5.55317908e-01],
               [ 2.10306547e-01],
               [ 3.90654734e-02],
               [ 5.74158070e-03],
               [-4.24527008e-05],
               [-1.78680894e-03],
               [ 7.47790447e-05]])
    margin = np.array([[ .1], [.1], [.3], [.3], [.5], [.1], [.1] ])
    hu = Hu(obj)
    diff = np.abs(hu-pattern)
    # print(np.abs(hu-pattern) < margin)
    return [(diff < margin).all(), diff]


def detect_arc_angle_bottom(obj: MatLike):
    pattern = np.array([[ 3.85286172e-01],
                        [ 4.04302080e-02],
                        [ 2.47167399e-02],
                        [ 4.07824088e-03],
                        [-3.41615071e-05],
                        [-7.65565240e-04],
                        [-2.25724513e-05]])
    # margin = np.array([[ .1], [.1], [.3], [.3], [.5], [.1], [.1] ])
    margin = np.array([[ .1], [.07], [.3], [.004], [.05], [.01], [.00002] ])
    hu = Hu(obj)
    diff = np.abs(hu-pattern)
    # print(np.abs(hu-pattern) < margin)
    return [(diff < margin).all(), diff]
