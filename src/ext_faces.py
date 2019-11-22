import cv2
import glob

face_cascade_1 = cv2.CascadeClassifier('OpenCV_HaarCascades/haarcascade_frontalface_default.xml') 
face_cascade_2 = cv2.CascadeClassifier('OpenCV_HaarCascades/haarcascade_frontalface_alt2.xml')
face_cascade_3 = cv2.CascadeClassifier('OpenCV_HaarCascades/haarcascade_frontalface_alt.xml')
face_cascade_4 = cv2.CascadeClassifier('OpenCV_HaarCascades/haarcascade_frontalface_alt_tree.xml')

emotions = ["neutral", "anger", "disgust", "fear", "joy", "sadness", "surprise"]

def face_detector(emotion):
    files = glob.glob('../Data/categorized_new/%s/*' %(emotion))

    fNum = 1

    for file in files:
        img = cv2.imread(file)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces_1 = face_cascade_1.detectMultiScale(gray_img, scaleFactor = 1.1, minNeighbors = 10, minSize = (5, 5), flags = cv2.CASCADE_SCALE_IMAGE)
        faces_2 = face_cascade_2.detectMultiScale(gray_img, scaleFactor = 1.1, minNeighbors = 10, minSize = (5, 5), flags = cv2.CASCADE_SCALE_IMAGE)
        faces_3 = face_cascade_3.detectMultiScale(gray_img, scaleFactor = 1.1, minNeighbors = 10, minSize = (5, 5), flags = cv2.CASCADE_SCALE_IMAGE)
        faces_4 = face_cascade_4.detectMultiScale(gray_img, scaleFactor = 1.1, minNeighbors = 10, minSize = (5, 5), flags = cv2.CASCADE_SCALE_IMAGE)

        if len(faces_1) == 1:
            face = faces_1
        elif len(faces_2) == 1:
            face = faces_2
        elif len(faces_3) == 1:
            face = faces_3
        elif len(faces_4) == 1:
            face = faces_4
        else:
            face = ""
            print("Alas! Face not found in img: %s" %(file))
        
        for (x, y, w, h) in face:
            print("Yay! Face found in img: %s" %(file))
            gray_img = gray_img[y:y+h, x:x+w]

            try: 
                resized_img = cv2.resize(gray_img, (350, 350))
                cv2.imwrite("../Data/categorized_neat_new/%s/%s.jpg" %(emotion, fNum), resized_img)
                print("Yay! Written successfully.\n")

            except Exception as e:
                print(str(e))  

        fNum += 1

if __name__ == "__main__":    
    for emotion in emotions:
        face_detector(emotion)