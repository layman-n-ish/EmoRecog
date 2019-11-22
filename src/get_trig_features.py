import cv2
import numpy as np
import dlib
import glob
import imutils
import math

from imutils import face_utils
from utils import * # utility file containing commonly used functions.

F = [i for i in range(0, 9)]
Fp = [i for i in range (9, 17)]
A = 49
B = 55
U = [52, 63 ]
D = [58, 67]

def cos (first, second):
    costheta = (first[0]*second[0] + first[1]*second[1]) \
            / ((first[0]*first[0] + first[1]*first[1]) * (second[0]*second[0] + second[1]*second[1]))
    return costheta

def extract_trig_features (image, l_coor):
    cts = []
    for f in F:
        first = (l_coor[A][0] - l_coor[f][0], l_coor[A][1] - l_coor[f][1])
        second= (l_coor[B][0] - l_coor[A][0], l_coor[B][1] - l_coor[A][1])
        costheta = (first[0]*second[0] + first[1]*second[1]) \
            / ((first[0]*first[0] + first[1]*first[1]) * (second[0]*second[0] + second[1]*second[1]))
        # cv2.line(image, (first[0], first[1]), (second[0], second[1]), (0,255,0), 2)
        cts.append (costheta)
    
    for f in Fp:
        first = (l_coor[B][0] - l_coor[f][0],l_coor[B][1] - l_coor[f][1])
        second = (l_coor[A][0] - l_coor[B][0], l_coor[A][1] - l_coor[B][1])
        costheta = (first[0]*second[0] + first[1]*second[1]) \
            / ((first[0]*first[0] + first[1]*first[1]) * (second[0]*second[0] + second[1]*second[1]))
        cts.append (costheta)

    junc = line_intersection ( ((l_coor[A][0], l_coor[A][1]), (l_coor[B][0], l_coor[B][1])), \
                        ( (l_coor[U[0]][0], l_coor[U[0]][1]), (l_coor[D[0]][0], l_coor[D[0]][1])  ) )
    
    for m in range (48, 68):
        costheta = cos ( \
            (junc[0] - l_coor[m][0] , junc[1], l_coor[m][1]), \
            (l_coor[A][0] - l_coor[B][0], l_coor[A][1] - l_coor[B][1]) \
        )
        cts.append (costheta)
    return cts




def getLandmarks1(path):
    image = cv2.imread(path)

    rects = detector(image, 1)
    t = []
    for (i, rect) in enumerate(rects):
        l_coor = predictor(image, rect)
        l_coor = face_utils.shape_to_np(l_coor) 

        t = extract_trig_features (image, l_coor)
    return t

if __name__ == "__main__":
    img_path = "../Data/categorized_neat_new/anger/10.jpg"
    image = cv2.imread(img_path)
    image = imutils.resize(image, width=350)  

    getLandmarks1(image)

    cv2.imshow("Ellipse", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()