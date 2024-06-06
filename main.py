import os
from itertools import product

import cv2

from utils import downsample
from utils import draw_obj_roi
from utils import find_segments
from utils import is_logo
from utils import rate_arcs
from utils import rate_dots


if __name__ == "__main__":
    dir = "images"
    for i, infile in enumerate(os.listdir(dir)):
        infile = os.path.join(dir, infile)
        img = cv2.imread(infile)
        img = downsample(img)
        segments = find_segments(img)
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

        cv2.imshow(f"{i}/{d:.5f}/{a:.5f}", img)
        cv2.waitKey(1)
    cv2.waitKey(0)
