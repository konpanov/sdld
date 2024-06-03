import cv2
from cv2.typing import MatLike
import numpy as np
from numpy.typing import ArrayLike

from utils import find_roi, smooth, grey_scale


def progowanie(img:MatLike, bottom = 0, top = 255):
    return np.array(255 * np.logical_and(img >= np.intp(bottom), img <= np.intp(top)),dtype=np.uint8)

def select_color(img:MatLike, value):
    return np.array(255 * (img == np.intp(value)),dtype=np.uint8)


def prog_erode(img:MatLike):
    img = progowanie(img, 50, 255)
    img = cv2.erode(img, np.ones((3, 3), np.uint8)) # TODO: Rewrite
    return img

def quantize(img:MatLike, to=4):
    div = 255//to
    img = img // div * div + div//2
    return img
    # return np.round(np.round(img*(to/255))*(255/to));
    # return (img // div) * div


def ccl(img: MatLike):
    ret = 0
    lables = np.zeros(img.shape, dtype=np.uint32)
    h, w = img.shape
    queue = []
    counts = []
    for y in range(h):
        for x in range(w):
            if lables[y][x] != 0:
                continue
            ret += 1
            count = 1
            v = img[y][x]
            # print(v)
            lables[y][x] = ret
            queue.append((x,y))
            while len(queue):
                x, y = queue.pop()
                neigh = [(x-1, y), (x+1,y), (x, y-1), (x, y+1)]
                neigh = [(x, y) for x, y in neigh if x >= 0 and x <w and y >= 0 and y< h]
                for x, y in neigh:
                    if lables[y][x] == 0 and img[y][x] == v:
                        lables[y][x] = ret
                        count += 1
                        queue.append((x, y))
            counts.append(count)
    return ret, lables, counts


class Segment:
    id: int
    img: MatLike
    count: int
    desc: ArrayLike

    def __init__(self, id, img, count) -> None:
        self.id = id
        self.img = img
        self.count = count

def find_segments(img: MatLike):
    img = grey_scale(img)
    img = smooth(img)
    # img = cv2.equalizeHist(img)
    img = quantize(img, 4)
    cv2.imshow("quant", img)
    ret, mask, counts = ccl(img)
    w, h = img.shape[:2]
    return [Segment(i, select_color(mask, i+1), counts[i]) for i in range(ret) if counts[i] >= w * h * 0.0001 ]


