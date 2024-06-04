import os
import cv2
from segment import *
from analiza import *
from utils import *


fnames = [
 os.path.join("train", "20240423_103157.jpg"),
 os.path.join("train", "20240423_103203.jpg"),
 os.path.join("train", "20240423_103255.jpg"),
 os.path.join("train", "20240423_103317.jpg"),
 os.path.join("train", "20240423_103340.jpg"),
 os.path.join("train", "20240423_103425.jpg"),
 os.path.join("train", "20240423_103500.jpg"),
 os.path.join("train", "20240423_103512.jpg"),
 os.path.join("test", "273q6x8k3ddc1.jpeg"),
 os.path.join("test", "9zp9si73gs2d1.jpeg"),
 os.path.join("test", "my-house-burned-down-yesterday-but-my-steam-deck-v0-gv4sn5ykfyja1.webp"),
 os.path.join("test", "photo_2024-05-28_15-54-24.jpg"),
 os.path.join("test", "photo_2024-05-28_15-58-57.jpg"),
]
segments = []
for imgnum, fname in enumerate(fnames):
    img = cv2.imread(fname)
    img = down_scale(img, 720)
    for i, seg in enumerate(find_segments(img)):
        if seg.count > 10:
            cv2.imwrite(f"segments/{imgnum}-{i}.jpg", seg.img)
