import cv2
import pickle
import numpy as np
from segment import *
from analiza import *
from utils import *

huMomentsDots = []
huMomentsArcs = []
fnames = [
    "20240423_103203.jpg", # dot - 1, arc = 1
    "20240423_103255.jpg",
    "20240423_103317.jpg",
    "20240423_103340.jpg",
    "20240423_103425.jpg",
    "20240423_103500.jpg",
    "20240423_103512.jpg",
]

og_size = np.array([4624, 3468])
resize = tuple(og_size//5)

for fname in fnames:
    img = cv2.imread(fname)
    img = cv2.resize(img, resize)
    img = grey_scale(img)
    objs = find_objects(img)
    for i, o in enumerate(objs):
        roi = find_roi(o)
        if abs(roi[0] - roi[2]) > 10 and abs(roi[1] - roi[3]) > 10:
            cv2.imwrite(f"detected_objects/{i}-{fname}", o)
