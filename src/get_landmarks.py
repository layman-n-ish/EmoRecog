from imutils import face_utils
import numpy as np
import imutils 
import dlib
import cv2
import math

#http://dlib.net/face_detector.py.html

def rotateImage(image, shape, angle):
	abt_pt = tuple(shape[30])
	rot_mat = cv2.getRotationMatrix2D(abt_pt, -angle, 1.0)
	rotated_img = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
	return rotated_img

detector = dlib.get_frontal_face_detector() 
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") 

img_path = "../Data/categorized_neat_new/anger/10.jpg"
image = cv2.imread(img_path)
image = imutils.resize(image, width=350)

rects = detector(image, 1) 
#'1'-number of image pyramid layers to apply when upscaling the image prior to applying the detector 
#(this is the equivalent of computing "cv2.pyrUp" N number of times on the image)
##### Alternate - The 1 in the second argument indicates that we should upsample the image 1 time.  
# This will make everything bigger and allow us to detect more
# faces.

for (i, rect) in enumerate(rects): 
	shape = predictor(image, rect)
	shape = face_utils.shape_to_np(shape)

	#(x, y, w, h) = face_utils.rect_to_bb(rect)
	#cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

	index = [27, 28, 29, 30]

	#for i in index:
	#	cv2.circle(image, tuple(shape[i]), 2, (0, 0, 255), -1)

	cv2.line(image, tuple(shape[27]), tuple(shape[30]), (255, 0, 0), 2)
	cv2.line(image, tuple(shape[30]), (shape[30][0], shape[27][1]), (255, 0, 0), 2)

	dist_axis = shape[30][1]-shape[27][1]
	dist_rot = math.sqrt(((shape[30][0]-shape[27][0])**2) + ((shape[30][1]-shape[27][1])**2))
	angle = math.degrees(math.acos(dist_axis/dist_rot))
	print("\nAngle: %f"%angle)

	rot_img = rotateImage(image, shape, angle)
 
cv2.imshow("Facial Landmarks Detector", image)
cv2.imshow("Rotated Image", rot_img)
#cv2.imwrite("Pictures/anger_10_angle.jpg", image)
cv2.imwrite("Pictures/anger_10_rotated.jpg", rot_img)
cv2.waitKey(0)

cv2.destroyAllWindows()

