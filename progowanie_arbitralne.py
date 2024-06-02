import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment import *
from utils import *


fname = "20240423_103500.jpg"
img = cv2.imread(fname)
size = np.array([4624, 3468])
img = cv2.resize(img, tuple(size//5))
img = grey_scale(img)
arc = progowanie(img, 180, 255)
dot = progowanie(img, 100, 130)
cv2.imshow("arc", arc)
cv2.imshow("dot", dot)
cv2.waitKey(0)
cv2.destroyAllWindows()
