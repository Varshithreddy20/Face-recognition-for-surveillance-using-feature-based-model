import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import random




path = 'Training_images'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


def findEncodings(images):
    encodeList = []


    for img in images:
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        except:
            pass
    return encodeList




encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
# img = captureScreen()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
# print(faceDis)
        matchIndex = np.argmin(faceDis)
        y1, x2, y2, x1 = faceLoc
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
# print(name)
            
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            #cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            #markAttendance(name)
        else:
            #img2 = img[x1:y1, x2:y2];cv2.imwrite('E:\face\Training_images\1.jpg', img2)
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            imgg=img
            crop_img = imgg[y1-50:y2+50, x1-50:x2+50]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            #cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (255, 0, 0), cv2.FILLED)
            
            #cv2.imshow("cropped", crop_img)
            os.chdir(r"D:\face\Training_images")
            
 
            # prints a random value from the list
            list1 = list(range(1,101))
            st=str(random.choice(list1))
            cv2.imwrite(st+'.jpg', crop_img)
            os.chdir(r"D:\face")
                
    
                
 

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)