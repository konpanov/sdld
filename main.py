import cv2
import argparse
from segment import *
from analiza import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", help = "Output file")
parser.add_argument("-i", "--input", help = "Input file")
args = parser.parse_args()
outfile = args.output
infile = args.input

img = cv2.imread(infile)
img = down_scale(img, 480)
segments = find_segments(img)
dots = find_dots(segments)
arcs = find_arcs(segments)
for dot in dots:
    for arc in arcs:
        if is_logo(dot, arc):
            img = draw_obj_roi(img, dot.img, color=(255, 0, 0))
            img = draw_obj_roi(img, arc.img, color=(255, 0, 0))
cv2.imshow('test', img)
cv2.waitKey(0)
# cv2.imwrite(outfile, img)
