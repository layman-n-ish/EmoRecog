import numpy as np
import imutils 
import dlib
import cv2
import math

from imutils import face_utils
from utils import *

def get_linear_features(img_path):
	image = cv2.imread(img_path)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	#image = imutils.resize(image, width=350)
	l = []
	rects = detector(image, 1) 
	for (i, rect) in enumerate(rects): 
		shape = predictor(image, rect)
		shape = face_utils.shape_to_np(shape)
		# l = []
		major_axis = dist(tuple(shape[48]), tuple(shape[54]))
		C = line_intersection((tuple(shape[48]), tuple(shape[54])), (tuple(shape[51]), tuple(shape[57])))
		for j in range(1,10):
			d = dist(tuple(shape[j-1]), tuple(shape[48]))
			d/=major_axis
			l.append(d)
		for j in range(9,18):
			d = dist(tuple(shape[j-1]), tuple(shape[54]))
			d/=major_axis
			l.append(d)
		for j in range(49,69):
			d = dist(tuple(shape[j-1]), C)
			d/=major_axis
			l.append(d)

		mouth_open = dist(tuple(shape[51]), tuple(shape[57]))
		l.append(mouth_open/major_axis)

		eye_major_left = dist(tuple(shape[36]), tuple(shape[39]))
		eye_open_left_1 = dist(tuple(shape[37]), tuple(shape[41]))
		l.append(eye_open_left_1/eye_major_left)
		eye_open_left_2 = dist(tuple(shape[38]), tuple(shape[40]))
		l.append(eye_open_left_2/eye_major_left)

		eye_major_right = dist(tuple(shape[42]), tuple(shape[45]))
		eye_open_right_1 = dist(tuple(shape[43]), tuple(shape[47]))
		l.append(eye_open_right_1/eye_major_right)
		eye_open_right_2 = dist(tuple(shape[44]), tuple(shape[46]))
		l.append(eye_open_right_2/eye_major_right)


		eye_left_up = mid_point(tuple(shape[37]), tuple(shape[38]))
		eye_left_down = mid_point(tuple(shape[41]), tuple(shape[40]))
		eye_left_centre = line_intersection((tuple(shape[36]), tuple(shape[39])), (eye_left_up, eye_left_down))
		eyebrow_left_mid = mid_point(tuple(shape[17]), tuple(shape[21]))

		eyebrow_major_left = dist(tuple(shape[17]), tuple(shape[21]))
		eyecentre_to_eyebrow_left = dist(eyebrow_left_mid, eye_left_centre)

		for j in range(18,23):
			d = dist(tuple(shape[j-1]), eye_left_centre)
			l.append(d/eyebrow_major_left)
			l.append(d/eyecentre_to_eyebrow_left)

		eye_right_up = mid_point(tuple(shape[43]), tuple(shape[44]))
		eye_right_down = mid_point(tuple(shape[47]), tuple(shape[46]))
		eye_right_centre = line_intersection((tuple(shape[42]), tuple(shape[45])), (eye_right_up, eye_right_down))
		eyebrow_right_mid = mid_point(tuple(shape[22]), tuple(shape[26]))

		eyebrow_major_right = dist(tuple(shape[22]), tuple(shape[26]))
		eyecentre_to_eyebrow_right = dist(eyebrow_right_mid, eye_right_centre)
		for j in range(23,28):
			d = dist(tuple(shape[j-1]), eye_right_centre)
			l.append(d/eyecentre_to_eyebrow_right) 

	return l

def visualize_feature_extraction(img_path, ind):
	image = cv2.imread(img_path)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	#image = imutils.resize(image, width=350)
	l = []
	rects = detector(image, 1) 
	for (i, rect) in enumerate(rects):
		shape = predictor(image, rect)
		shape = face_utils.shape_to_np(shape)
		C = line_intersection((tuple(shape[48]), tuple(shape[54])), (tuple(shape[51]), tuple(shape[57])))

		for j in range(1,10):
			cv2.line(image, tuple(shape[j-1]), tuple(shape[48]), (255, 0, 0), 2)
		for j in range(9,18):
			cv2.line(image, tuple(shape[j-1]), tuple(shape[54]), (255, 0, 0), 2)
		for j in range(49,69):
			cv2.line(image, tuple(shape[j-1]), C, (255, 0, 0), 2)

		cv2.line(image, tuple(shape[48]), tuple(shape[54]), (255, 0, 0), 2)
		cv2.line(image, tuple(shape[51]), tuple(shape[57]), (255, 0, 0), 2)

		cv2.line(image, tuple(shape[36]), tuple(shape[39]), (255, 0, 0), 2)
		cv2.line(image, tuple(shape[37]), tuple(shape[41]), (255, 0, 0), 2)
		cv2.line(image, tuple(shape[38]), tuple(shape[40]), (255, 0, 0), 2)
		cv2.line(image, tuple(shape[42]), tuple(shape[45]), (255, 0, 0), 2)
		cv2.line(image, tuple(shape[43]), tuple(shape[47]), (255, 0, 0), 2)
		cv2.line(image, tuple(shape[44]), tuple(shape[46]), (255, 0, 0), 2) 

		eye_left_up = mid_point(tuple(shape[37]), tuple(shape[38]))
		eye_left_down = mid_point(tuple(shape[41]), tuple(shape[40]))
		eye_left_centre = line_intersection((tuple(shape[36]), tuple(shape[39])), (eye_left_up, eye_left_down))

		for j in range(18,23):
			cv2.line(image, tuple(shape[j-1]), eye_left_centre, (255, 0, 0), 2) 

		eye_right_up = mid_point(tuple(shape[43]), tuple(shape[44]))
		eye_right_down = mid_point(tuple(shape[47]), tuple(shape[46]))
		eye_right_centre = line_intersection((tuple(shape[42]), tuple(shape[45])), (eye_right_up, eye_right_down))

		for j in range(23,28):
			cv2.line(image, tuple(shape[j-1]), eye_right_centre, (255, 0, 0), 2) 

	file_path = "Pictures/"+str(ind) + ".jpg"

	cv2.imwrite(file_path, image)

if __name__ == '__main__':
	
	img_path = "21.jpg"
	image = cv2.imread(img_path)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	#image = imutils.resize(image, width=350)

	rects = detector(image, 1) 
	#'1'-number of image pyramid layers to apply when upscaling the image prior to applying the detector 
	#(this is the equivalent of computing "cv2.pyrUp" N number of times on the image)
	##### Alternate - The 1 in the second argument indicates that we should upsample the image 1 time.  
	# This will make everything bigger and allow us to detect more
	# faces.

	for (i, rect) in enumerate(rects):
		shape = predictor(image, rect)
		shape = face_utils.shape_to_np(shape)
		C = line_intersection((tuple(shape[48]), tuple(shape[54])), (tuple(shape[51]), tuple(shape[57])))

		for j in range(1,10):
			cv2.line(image, tuple(shape[j-1]), tuple(shape[48]), (255, 0, 0), 2)
		for j in range(9,18):
			cv2.line(image, tuple(shape[j-1]), tuple(shape[54]), (255, 0, 0), 2)
		for j in range(49,69):
			cv2.line(image, tuple(shape[j-1]), C, (255, 0, 0), 2)

		cv2.line(image, tuple(shape[48]), tuple(shape[54]), (255, 0, 0), 2)
		cv2.line(image, tuple(shape[51]), tuple(shape[57]), (255, 0, 0), 2)

		cv2.line(image, tuple(shape[36]), tuple(shape[39]), (255, 0, 0), 2)
		cv2.line(image, tuple(shape[37]), tuple(shape[41]), (255, 0, 0), 2)
		cv2.line(image, tuple(shape[38]), tuple(shape[40]), (255, 0, 0), 2)
		cv2.line(image, tuple(shape[42]), tuple(shape[45]), (255, 0, 0), 2)
		cv2.line(image, tuple(shape[43]), tuple(shape[47]), (255, 0, 0), 2)
		cv2.line(image, tuple(shape[44]), tuple(shape[46]), (255, 0, 0), 2) 

		eye_left_up = mid_point(tuple(shape[37]), tuple(shape[38]))
		eye_left_down = mid_point(tuple(shape[41]), tuple(shape[40]))
		eye_left_centre = line_intersection((tuple(shape[36]), tuple(shape[39])), (eye_left_up, eye_left_down))

		for j in range(18,23):
			cv2.line(image, tuple(shape[j-1]), eye_left_centre, (255, 0, 0), 2) 

		eye_right_up = mid_point(tuple(shape[43]), tuple(shape[44]))
		eye_right_down = mid_point(tuple(shape[47]), tuple(shape[46]))
		eye_right_centre = line_intersection((tuple(shape[42]), tuple(shape[45])), (eye_right_up, eye_right_down))

		for j in range(23,28):
			cv2.line(image, tuple(shape[j-1]), eye_right_centre, (255, 0, 0), 2) 


	cv2.imwrite("Pictures/1_linear_feat.jpg", image)

	l = get_linear_features(img_path)
	print(len(l))
	for i in range(len(l)):
		print(l[i])
		if(l[i] == 0.0):
			print(i)
