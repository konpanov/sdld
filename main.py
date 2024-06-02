import os
import cv2
from segment import *
from analiza import *
from utils import *






arc1 = cv2.imread(os.path.join("luki", "base.jpg"), cv2.IMREAD_GRAYSCALE)
arcs = [arc1]
arcsHu = [findHuMoments(arc) for arc in arcs]


# Load image
fnames = [
 os.path.join("train", "20240423_103157.jpg"),
 os.path.join("test", "9zp9si73gs2d1.jpeg")
]
for fname in fnames:
    # fname = os.path.join("test", "273q6x8k3ddc1.jpeg")
    test = cv2.imread(fname)
    hieght = 720
    size = test.shape[:2]
    test = cv2.resize(test, (hieght, hieght * (size[0]//size[1])))
    test_bgr = test
    test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)

    # Segments objects
    objs = find_objects(test)
    objs = [(o, *detect(o, arcsHu)) for o in objs]
    [draw_obj_roi(test_bgr, o) for o , detected, _ in objs if detected]

    cv2.imshow("test", test_bgr)
    objs = sorted(objs, key=lambda o: np.mean(o[2]))
    # objs = [objs[111]]
    # cv2.imwrite(os.path.join("luki", "base2.jpg"), objs[51][0])
    # o, detected, distance = objs[0]
    # print(arcsHu)

    # cv2.imshow(f"{["{0:0.3f}".format(d) for d in findHuMoments(arc2)[:, 0]]}", arc2)
    # cv2.imshow(f"{distance}, {["{0:0.3f}".format(d) for d in findHuMoments(o)[:, 0]]}", o)
    for i, (o, detected, distance) in enumerate(objs):
        # print(findHuMoments(o))
        cv2.imshow(f"{i}:{distance}", o)
        if cv2.waitKey(0) == 27:
            break
    while cv2.waitKey(0) != 27:
        ...
