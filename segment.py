import cv2
from cv2.typing import MatLike
import numpy as np

from utils import find_roi, gauss


def progowanie(img:MatLike, bottom = 0, top = 255):
    return np.array(255 * np.logical_and(img >= np.intp(bottom), img <= np.intp(top)),dtype=np.uint8)

def progowanie_rowne(img:MatLike, value):
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

def find_objects(img: MatLike):
    # cv2.imshow("before", img)
    img = gauss(img)
    # cv2.imshow("gauss", img)
    img = quantize(img, 4)
    # cv2.imshow("quant", img)
    # img = cv2.Laplacian(img, cv2.CV_8U)
    # cv2.imshow("laplacian", img)
    ret, mask, counts = ccl(img)
    # cv2.imshow("cll", mask.astype(np.uint8)*20)
    labels = [i for i in range(ret) if counts[i] > 100]
    objs = [progowanie_rowne(mask, label+1) for label in labels]
    return objs

