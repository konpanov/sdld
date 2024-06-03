import os
import cv2
from segment import *
from analiza import *
from utils import *


fnames = [
    "test/273q6x8k3ddc1.jpeg",
    "test/9zp9si73gs2d1.jpeg",
    "test/photo_2024-05-28_15-54-24.jpg",
    "train/20240423_103157.jpg",
    "train/20240423_103203.jpg",
    "train/20240423_103255.jpg",
    "train/20240423_103317.jpg",
    "train/20240423_103340.jpg",
    "train/20240423_103500.jpg",
]


for fname in fnames:
    img = cv2.imread(fname)
    img = resize(img, 480)
    segments = find_segments(img)
    segments = [seg for seg in segments if pole_check(seg.img)]
    segments = [(seg, match_shapes(seg.img, arcs_pattern)) for seg in segments]
    segments = sorted(segments, key=lambda x:x[1])
    segments = [seg for seg in segments if seg[1] < .02]
    for seg, d in segments:
        img = draw_obj_roi(img, seg.img)
    cv2.imshow(fname, img)
cv2.waitKey(0)




# pattern_name = os.path.join("arcs", "3-8.jpg")
# pattern_img = cv2.imread(pattern_name, cv2.IMREAD_GRAYSCALE)
# pattern_img = resize(pattern_img, 480)
# pattern = Hu(pattern_img)
# print(pattern)
#
# dir = "arcs"
# fnames = [os.path.join(dir, file) for file in os.listdir(dir)]
# hus = np.array([Hu(cv2.imread(fname, cv2.IMREAD_GRAYSCALE)) for fname in fnames ])
# min, mean, max = np.min(hus, axis=0), np.mean(hus, axis=0), np.max(hus, axis=0)
# # print(min)
# # print(mean)
# # print(max)
# # print(max - min)
# # print(np.var(hus, axis=0))
#
#
#
#
# dir = "arcs"
# fnames = [os.path.join(dir, file) for file in os.listdir(dir)]
# # hus = np.array([Hu(cv2.imread(fname, cv2.IMREAD_GRAYSCALE)) for fname in fnames ])
# imgs = [cv2.imread(fname, cv2.IMREAD_GRAYSCALE) for fname in fnames]
# imgs = [resize(img, 480) for img in imgs]
# hus = [Hu(img) for img in imgs]
# # hus = np.array(hus)[:,:,0]
# # detected = [check(hu, pattern) for hu in hus]
# # print(hus)
# # print(np.min(hus))
# # print(np.mean(hus))
# # print(np.max(hus))
# # print(np.max(hus) - np.min(hus))
# # print(f"{sum(detected)}/15")
#
# # imgs = sorted(imgs, key=lambda img: odleglosc(Hu(img), pattern))
# # count = 0
# # for img in imgs:
# #     hu = Hu(img)
# #     # print(hu)
# #     if check(hu, pattern):
# #         count+=1
# #     cv2.imshow(f"{distance(hu, pattern):.2f}/{["{0:.02f}".format(h) for h in hu[:, 0]]}", img)
# #     # cv2.imshow(f"{distance(hu, pattern):.2f}", img)
# #     if cv2.waitKey(0) == 27:
# #         break
# # print(f"{count}/15")
#
# dir = "segments"
# fnames = [os.path.join(dir, file) for file in os.listdir(dir)]
# for fn in fnames:
#     img = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
#     img = resize(img, 480)
#     hu = Hu(img)
#     if check(hu, pattern):
#     # cv2.imshow(f"{["{}".format(h) for h in hu[:, 0]]}", img)
#         cv2.imshow(f"{distance(hu, pattern):.2f}/{["{0:.09f}".format(h) for h in hu[:, 0]]}", img)
#         # cv2.imshow(f"{distance(hu, pattern)}", img)
#         if cv2.waitKey(0) == 27:
#             exit(0)
#
# #
# #
# #
# # #
# # # print(f"{count}/15")
# while cv2.waitKey(0) != 27:
#     exit(0)
# # #
