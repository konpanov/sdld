import numpy as np
from cv2.typing import MatLike
import cv2

def down_scale(img: MatLike, height: int):
    w, h = img.shape[:2]
    factor_h = h // height
    factor_w = w // (height * w//h)
    return img[::factor_h, ::factor_w, :].astype(np.uint8)

def grey_scale(img: MatLike):
    return np.array(np.sum(img, axis=2)//3, dtype=np.uint8)

def find_roi(img: MatLike):
    Y, X = np.nonzero(img)
    return (min(X), min(Y), max(X), max(Y))

def draw_obj_roi(img: MatLike, obj: MatLike, color = (255, 0, 0)):
    Y, X = np.nonzero(obj)
    p1, p2 = ((min(X), min(Y)), (max(X), max(Y)))
    img = cv2.rectangle(img,p1, p2, color, 2)
    return img


def draw_obj_label(img: MatLike, obj: MatLike, text: str):
    x, y = srodek(obj)
    img = cv2.putText(img, text, (x-15,y), cv2.FONT_ITALIC, .5, (0, 255, 0), 2)
    return img

def srodek(img: MatLike):
    Y, X = np.nonzero(img)
    return (int(np.mean(X)), int(np.mean(Y)))

def odleglosc(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.sqrt(((vec1 - vec2)**2).sum())

def neighbours(x, y, diag=False):
    neigh = [(x-1, y), (x+1,y), (x, y-1), (x, y+1)] 
    if diag:
        neigh.extend([(x-1,y-1),(x-1,y+1),(x+1, y+1),(x+1,y-1)])
    return neigh

def smooth(img: MatLike):
    h, w = img.shape
    out = np.zeros((h-2, w-2))
    k = np.array([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ])
    for y in range(h-2):
        for x in range(w-2):
            roi = img[y:y+3,x:x+3]
            out[y][x] = np.sum(np.multiply(roi, k))//np.sum(k)
    out = np.array(out, dtype=np.uint8)
    return out

def M(obj: MatLike):
    h, w = obj.shape[:2]
    obj = obj.astype(np.uint32)
    ys, xs = np.arange(h), np.arange(w)
    xsum, ysum = np.sum(obj, axis=1), np.sum(obj, axis=0)
    m00 = np.sum(obj)
    m10 = np.sum(ysum * xs)
    m01 = np.sum(xsum * ys)
    m11 = np.sum(obj * np.outer(ys, xs))
    m20 = np.sum(ysum * xs ** 2)
    m02 = np.sum(xsum * ys ** 2)
    m12 = np.sum(obj * np.outer(ys ** 2, xs))
    m21 = np.sum(obj * np.outer(ys, xs ** 2))
    m30 = np.sum(ysum * xs ** 3)
    m03 = np.sum(xsum * ys ** 3)
    return m00, m10, m01, m11, m20, m02, m12, m21, m30, m03

