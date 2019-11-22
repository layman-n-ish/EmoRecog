from imutils import face_utils
import numpy as np
import imutils 
import dlib
import cv2
import math 
import glob

emotions = ["neutral", "anger", "disgust", "fear", "joy", "sadness", "surprise"]

detector = dlib.get_frontal_face_detector() 
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") 

def rotateImage(image, shape, angle):
    abt_pt = tuple(shape[30])
    if(shape[27][0] < shape[30][0]):
        angle = -angle
    rot_mat = cv2.getRotationMatrix2D(abt_pt, angle, 1.0)
    rotated_img = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

    print("Rotated the image!\n")

    return rotated_img

def getAngle(shape):
    dist_axis = shape[30][1]-shape[27][1]
    dist_rot = math.sqrt(((shape[30][0]-shape[27][0])**2) + ((shape[30][1]-shape[27][1])**2))
    angle = math.degrees(math.acos(dist_axis/dist_rot))

    print("Got the angle! %f\n" %angle)

    return angle

def makeRotatedImages(emotion):
    files = glob.glob('../Data/google_set/%s/*'%emotion)

    fNum = 1

    for file in files:
        img = cv2.imread(file)
        print("Dealing with: %s\n"%(file))
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        rects = detector(gray_img, 1)

        if(len(rects) == 0):
            print("No face found!\n")

        elif(len(rects) == 1):
            for rect in rects:
                l_coor = predictor(gray_img, rect)
                l_coor = face_utils.shape_to_np(l_coor)

                angle = getAngle(l_coor)

                rot_img = rotateImage(gray_img, l_coor, angle)

                cv2.imwrite("../Data/google_set_new/%s/%s.jpg"%(emotion, fNum), rot_img)
                print("Written!\n")

        else:
            print("Too many faces!\n")

        fNum += 1           

if __name__ == "__main__":
    for emotion in emotions:
        makeRotatedImages(emotion)