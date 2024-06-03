import numpy as np
from cv2.typing import MatLike
import cv2

def down_scale(img: MatLike, height: int):
    w, h = img.shape[:2]
    factor_h = h // height
    factor_w = w // (height * w//h)
    return img[::factor_h, ::factor_w, :].astype(np.uint8)

def grey_scale(img: MatLike):
    return np.array(1 + np.sum(img, axis=2)//3, dtype=np.uint8)

def histogram(img: MatLike):
    shape = img.shape
    if len(shape) > 2:
        raise Exception("histogram supports only grey scale")
    histo = np.zeros(255)
    w, h = img.shape
    for x in range(w):
        for y in range(h):
            histo[img[x][y]] += 1
    return histo


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
