import cv2
import numpy as np
import dlib
import glob
import imutils
from imutils import face_utils
import math
import pandas as pd
from utils import *

emotions = ["neutral", "anger", "disgust", "fear", "joy", "sadness", "surprise"]

ellipse_1 = [48, 51, 54] #upper mouth
ellipse_2 = [48, 57, 54] #lower mouth
ellipse_7 = [17, 19, 21] #left eyebrow
ellipse_8 = [22, 24, 26] #right eyebrow

ellipses = [ellipse_1, ellipse_2, ellipse_7, ellipse_8]
 
feature_vec = []
labels = []

def extractFeatures(image, l_coor):
	features_per_image = []
	for ellipse_region in ellipses:
		center = (l_coor[ellipse_region[1]][0], int((l_coor[ellipse_region[0]][1]+l_coor[ellipse_region[2]][1])/2))
		a = int(abs((l_coor[ellipse_region[2]][0]-l_coor[ellipse_region[0]][0])/2))
		b = int(abs(int((l_coor[ellipse_region[0]][1]+l_coor[ellipse_region[2]][1])/2)-l_coor[ellipse_region[1]][1]))  

		l = max(a, b)
		r = min(a, b)
		a = l
		b = r

		e = math.sqrt(1-((b**2)/(a**2)))

		#axes = (a, b)
        #angle = -14.036243
        #print(axes)

        #for indices in ellipse_region:
        #    cv2.circle(image, tuple(l_coor[indices]), 2, (0, 0, 255), -1)
        #    cv2.ellipse(image, center, axes, 0, 0, 360, (0, 0, 255), 2)
		
		features_per_image.append(a)
		features_per_image.append(b)
		features_per_image.append(e)

	# print("\t"+str(features_per_image))
	return features_per_image
        
def getLandmarks(img_path):
	image = cv2.imread(img_path)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	rects = detector(image, 1)
	if (len(rects) == 0):	
		print("No face found!\n")
		features_per_image = []
		
	elif(len(rects) == 1):
		for rect in rects:	
			l_coor = predictor(image, rect)
			l_coor = face_utils.shape_to_np(l_coor) 

			features_per_image = extractFeatures(image, l_coor)
			
	else:
		print("Too many faces!\n")  
		features_per_image = []

	return features_per_image

def getFeatures(emotion):
	files = glob.glob("../Data/emotions/%s/*"%(emotion))

	for file in files:
		image = cv2.imread(file)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		print("Working with %s"%file)

		indi_features = getLandmarks(image)
		if (indi_features != "Nil"):
			print("\tGot features\n")
			feature_vec.append(indi_features)
			labels.append(emotions.index(emotion))
		else:
			print("Nope\n")

if __name__ == "__main__":

	for emotion in emotions:
		getFeatures(emotion)
	
	print(len(feature_vec))
	print(feature_vec[0])
	print(labels[0])

	print(feature_vec[1])
	print(labels[1])

	df = pd.DataFrame(feature_vec)
	print(df.head())

	lab = pd.DataFrame(labels)
	print(df.head())

	df.to_csv('ellipse_features.csv')
	lab.to_csv('labels.csv')