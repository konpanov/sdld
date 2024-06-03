import os
import cv2
from segment import *
from analiza import *
from utils import *

# img = cv2.imread("arcs/2-466.jpg", cv2.IMREAD_GRAYSCALE)
# print(Hu(img))
# exit(0)
img = cv2.imread("test/273q6x8k3ddc1.jpeg")
# img = cv2.imread("train/20240423_103157.jpg")
# img = cv2.imread("train/20240423_103203.jpg")
# img = cv2.imread("train/20240423_103255.jpg")
# img = cv2.imread("train/20240423_103317.jpg")
# img = cv2.imread("train/20240423_103340.jpg")
# img = cv2.imread("test/my-house-burned-down-yesterday-but-my-steam-deck-v0-gv4sn5ykfyja1.webp")
# img = cv2.imread("test/photo_2024-05-28_15-54-24.jpg")
# img = cv2.imread("test/photo_2024-05-28_15-58-57.jpg")
# img = cv2.imread("test/9zp9si73gs2d1.jpeg")
# img = cv2.imread("test/til-the-steam-deck-oled-case-has-a-built-in-travel-case-v0-g5gf82uqcurc1.webp")

img = resize(img, 480)
segments = find_segments(img)
dots = find_dots(segments)
arcs = find_arcs(segments)

for dot in dots:
    img = draw_obj_roi(img, dot.img, color=(0, 255, 0))
for arc in arcs:
    img = draw_obj_roi(img, arc.img, color=(0, 0, 255))

for dot in dots:
    for arc in arcs:
        if is_logo(dot, arc):
            img = draw_obj_roi(img, dot.img, color=(255, 0, 0))
            img = draw_obj_roi(img, arc.img, color=(255, 0, 0))



cv2.imshow("test", img)
cv2.waitKey(0)
