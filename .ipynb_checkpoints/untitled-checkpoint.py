import cv2 as cv
from keras.models import model_from_json

import numpy as np 
json_file  = open("emotiondetectore.json" ,"r")
modle_json = json_file.read()
json_file.close()

model = model_from_json(json_file)

model.load_weights("emotiondetectore.h5")

haar_file = cv.data.haarcascades + 'haarcascade_frontalface_defaul'
face_cascade=cv2.CascadeClassifier(haar_file)

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48 ,48,1)
    return feature/255.0

webcam = cv.VideoCapture(0)
labels = {0 : 'angry' , 1 : 'disgust' ,2 : 'fear' ,3 : 'happy' ,4 : 'neutral' ,5 : 'sad' , 6 : 'surprise'}
while True :
    i , im =webcam.read()
    gray =cv.cvtColor(im , cv.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(im ,1.3,5)
    try:
        for p ,q , r, s in faces:
            image = gray[q: q+s , p: p+r]
            cv.rectangle(im , (p,q) ,(p+r ,q+s) ,(255,0,0),2)
            image =cv.resize(image , (48, 48))
            img = extract_features(image)
            prod =model.predict(img)
            pro_label = labels[prod.argmax()]

            cv.putText(im , '% s' %(pro_label) ,(p-10 ,q-10) cv.FONT_HERSHEY_COMPLEX_SMALL, 2 ,(0,0,255))
        cv.imshow("OUTPUT" ,im)
        cv.waitKey(27)
    except cv.error:
        pass