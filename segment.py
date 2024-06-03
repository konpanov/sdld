from cv2.typing import MatLike
import numpy as np
from numpy.typing import ArrayLike

from utils import find_roi, smooth, grey_scale


def progowanie(img:MatLike, bottom = 0, top = 255):
    return np.array(255 * np.logical_and(img >= np.intp(bottom), img <= np.intp(top)),dtype=np.uint8)

def select_color(img:MatLike, value):
    return np.array(255 * (img == np.intp(value)),dtype=np.uint8)


def quantize(img:MatLike, to=4):
    div = 255//to
    img = img // div * div + div//2
    return img


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
    img = quantize(img, 4)
    ret, mask, counts = ccl(img)
    w, h = img.shape[:2]
    return [Segment(i, select_color(mask, i+1), counts[i]) for i in range(ret) if pole_check(w, h, counts[i]) ]


def pole_check(w, h, count):
    stosunek = count/(w*h) 
    return stosunek >= 0.0001 and stosunek < .25
