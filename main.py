import os
import cv2
from segment import *
from analiza import *
from utils import *

#

def distance(hu, pattern):
    return odleglosc(hu, pattern)
    return np.abs(hu[0] - pattern[0])
    # return True
    return (
        # hu[2] > 0.01
    )
def check(hu, pattern):
    return  distance(hu, pattern) < 1




pattern_name = os.path.join("arcs", "3-8.jpg")
pattern_img = cv2.imread(pattern_name, cv2.IMREAD_GRAYSCALE)
pattern_img = resize(pattern_img, 480)
pattern = Hu(pattern_img)

dir = "arcs"
fnames = [os.path.join(dir, file) for file in os.listdir(dir)]
hus = np.array([Hu(cv2.imread(fname, cv2.IMREAD_GRAYSCALE)) for fname in fnames ])
min, mean, max = np.min(hus, axis=0), np.mean(hus, axis=0), np.max(hus, axis=0)
print(min)
print(mean)
print(max)
print(max - min)
print(np.var(hus, axis=0))




dir = "arcs"
fnames = [os.path.join(dir, file) for file in os.listdir(dir)]
# hus = np.array([Hu(cv2.imread(fname, cv2.IMREAD_GRAYSCALE)) for fname in fnames ])
imgs = [cv2.imread(fname, cv2.IMREAD_GRAYSCALE) for fname in fnames]
imgs = [resize(img, 480) for img in imgs]
# imgs = sorted(imgs, key=lambda img: odleglosc(Hu(img), pattern))
# count = 0
# for img in imgs:
#     hu = Hu(img)
#     cv2.imshow(f"{distance(hu, pattern):.2f}/{["{0:.02f}".format(h) for h in hu[:, 0]]}", img)
#     # cv2.imshow(f"{distance(hu, pattern):.2f}", img)
#     if cv2.waitKey(0) == 27:
#         exit(0)

dir = "segments"
fnames = [os.path.join(dir, file) for file in os.listdir(dir)]
for fn in fnames:
    img = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
    img = resize(img, 480)
    hu = Hu(img)
    if np.sum(img)//255 > 100 and check(hu, pattern):
    # cv2.imshow(f"{["{}".format(h) for h in hu[:, 0]]}", img)
        cv2.imshow(f"{distance(hu, pattern):.2f}/{["{0:.02f}".format(h) for h in hu[:, 0]]}", img)
        # cv2.imshow(f"{distance(hu, pattern)}", img)
        if cv2.waitKey(0) == 27:
            exit(0)



#
# print(f"{count}/15")
# while cv2.waitKey(0) != 27:
#     exit(0)
#
