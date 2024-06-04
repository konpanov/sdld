import argparse
import os
from itertools import product

import cv2

from analiza import *
from segment import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", help="Output file")
parser.add_argument("-i", "--input", help="Input file")
args = parser.parse_args()
outfile = args.output
infile = args.input

for infile in os.listdir():
    ...
img = cv2.imread(infile)

img = down_scale(img, 480)
segments = find_segments(img)
dots = zip(rate_dots(segments), segments)
arcs = zip(rate_arcs(segments), segments)
dots = sorted(dots, key=lambda x: x[0])
arcs = sorted(arcs, key=lambda x: x[0])

dots = dots[: min(len(dots), 100)]
arcs = dots[: min(len(dots), 100)]

logos = sorted(product(dots, arcs), key=lambda x: x[0][0] * x[1][0])

for dot, arc in logos:
    if dot[0] * arc[0] >= 0.001:
        continue
    if is_logo(dot[1], arc[1]):
        img = draw_obj_roi(img, dot[1].img, color=(255, 0, 0))
        img = draw_obj_roi(img, arc[1].img, color=(255, 0, 0))

cv2.imshow("test", img)
cv2.waitKey(0)
# cv2.imwrite(outfile, img)
