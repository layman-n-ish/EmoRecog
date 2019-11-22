import pandas as pd
import cv2
import numpy as np
import imutils 
import dlib
import cv2
import math
import glob

from imutils import face_utils
from get_linear_features import *
from get_elliptical_features import *
from get_trig_features import *
from utils import *


def make_dataframe(path):
	data = {}
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
	emotions = ["neutral", "anger", "disgust", "fear", "happy", "sadness", "surprise"]
	emotion_map = {"neutral" : 0, "anger": 1, "disgust": 2, "fear": 3, "happy": 4, "sadness": 5, "surprise": 6}

	i = 0
	for emotion in emotions:
		files = glob.glob(path+'%s/*' %(emotion))

		for file in files:
			linear_features = get_linear_features(file)
			eccentricity_features = getLandmarks(file)
			trig_features = getLandmarks1(file)
			features = linear_features
			features.extend(eccentricity_features)
			features.extend(trig_features)
			if(len(features) == 0):
				continue
			# visualize_feature_extraction(file,i+1)
			i+=1
			features.append(emotion_map[emotion])
			data[i] = features
			print("done" + file)
	df = pd.DataFrame.from_dict(data=data, orient='index')
	return df

df = make_dataframe("google_set_new/")
print(df.head(20))
print(df.shape)
df.to_csv("rorated_test.csv")