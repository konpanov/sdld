import numpy as np
import cv2
from segment import *
from analiza import *
from utils import *
from os import listdir
from os.path import join

arc_path = "luki"
# arcfiles = ["base.jpg", "angle45.jpg", "weird_angle.jpg"]
arcfiles = ["base.jpg"]
# arcfiles = [join(arc_path, f) for f in listdir(arc_path) ]
arcfiles = [join(arc_path, file) for file in arcfiles]
objs = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in arcfiles]
# [cv2.imshow(str(i), o) for i, o in enumerate(objs)]
pattern_hus = np.array([findHuMoments(o) for o in objs], dtype=np.float_)

fname = "20240423_103317.jpg"
fname = "20240423_103157.jpg"
fname = "20240423_103203.jpg"
# fname = "20240423_103255.jpg"
# fname = "20240423_103340.jpg" # BAD
# fname = "20240423_103425.jpg"
# fname = "20240423_103500.jpg"
# fname = "20240423_103512.jpg"

fname = join("train", fname)
# fname = join("test", "9zp9si73gs2d1.jpeg")
# fname = join("test", "my-house-burned-down-yesterday-but-my-steam-deck-v0-gv4sn5ykfyja1.webp")
# fname = join("test", "photo_2024-05-28_15-54-24.jpg")
# fname = join("test", "photo_2024-05-28_15-58-57.jpg")
fname = join("test", "273q6x8k3ddc1.jpeg")
# fname = join("test", "rotated.jpg")


img = cv2.imread(fname)
size = np.array(img.shape[:2])
wsp = size[0]/size[1]
width = 480
height = int(width*wsp)
size = (width, height)
img = cv2.resize(img, size)
img[:,:] = 0
img = cv2.circle(img, (width//4, height//4), 40, 255, -1)
img = cv2.circle(img, (width//2, height//2), 40, 255, -1)
img[:height//2,width:] = 0
img = cv2.circle(img, (width//2, height//2), 20, 0, -1)
og = img
cv2.imshow("before", og)

img = grey_scale(img)
objs = find_objects(img)
objs = [(obj, *detect(obj, pattern_hus), i) for i, obj in enumerate(objs)]
for obj, detected, dist, i in objs:
    if detected:
        og = draw_obj_roi(og, obj)
        og = draw_obj_label(og, obj, "arc")

cv2.imshow("og", og)
cv2.waitKey(0)

# objs = [objs[387]]
objs = sorted(objs, key=lambda o: np.mean(o[2]))
for j, (obj, detected, dist, i) in enumerate(objs):
    # if j < 470:
    #     continue
    cv2.imshow(f"{j}:{["{0:0.3f}".format(d) for d in dist]},{i}", obj)
    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()
        exit(0)

