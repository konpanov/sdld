import argparse
import os
from itertools import product

import cv2

from utils import downsample
from utils import draw_obj_roi
from utils import find_segments
from utils import is_logo
from utils import rate_arcs
from utils import rate_dots


parser = argparse.ArgumentParser()
boolopt = argparse.BooleanOptionalAction
parser.add_argument("-o", "--output", help="Save output", action=boolopt)
parser.add_argument("-r", "--remap", help="Show remapped colors", action=boolopt)
parser.add_argument("-p", "--pause", help="Pause between images", action=boolopt)
parser.add_argument("-q", "--quit", help="Quit after last image", action=boolopt)
args = parser.parse_args()


if __name__ == "__main__":
    dir = "images"
    for i, fname in enumerate(os.listdir(dir)):
        dirfname = os.path.join(dir, fname)
        img = cv2.imread(dirfname)
        img = downsample(img)
        segments = find_segments(img, args.remap)
        dots = zip(rate_dots(segments), segments)
        arcs = zip(rate_arcs(segments), segments)
        dots = sorted(dots, key=lambda x: x[0])
        arcs = sorted(arcs, key=lambda x: x[0])
        dots = dots[: min(len(dots), 50)]
        arcs = arcs[: min(len(arcs), 50)]

        d = a = 0
        for dot, arc in product(dots, arcs):
            if dot[0] + arc[0] >= 0.3:
                continue
            if is_logo(dot[1], arc[1]):
                d = dot[0]
                a = arc[0]
                img = draw_obj_roi(img, dot[1], color=(255, 0, 0))
                img = draw_obj_roi(img, arc[1], color=(255, 0, 0))

        if args.output:
            cv2.imwrite(os.path.join("out", fname), img)
        cv2.imshow(f"{i}", img)
        cv2.waitKey(0 if args.pause else 1)
    if not args.quit:
        cv2.waitKey(0)
