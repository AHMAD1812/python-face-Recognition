import cv2
import face_recognition
import os
import numpy as np
from random import randrange
from datetime import datetime

path="attendence"
images=[]
className=[]
myList=os.listdir(path)
print(myList)

for cl in myList:
    curImg= cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    className.append(os.path.splitext(cl)[0])

print(className)    

def findEncodings(images):
    encodelist=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist

def markAttendence(name):
    with open('attendences.csv','r+') as f:
        myDatalist=f.readlines()
        nameList=[]
        for line in myDatalist:
            entry=line.split(',')
            nameList.append(entry)
        if name not in nameList:
            now = datetime.now()    
            dtstring= now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtstring}')

         

list=findEncodings(images)
print('encodings completes')

cap=cv2.VideoCapture(0)

while True:
    success,img = cap.read()
    imgS= cv2.resize(img,(0,0),None,0.25,0.25)
    imgS=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
    
    faceCurFrame = face_recognition.face_locations(imgS)
    encodeVideo = face_recognition.face_encodings(imgS,faceCurFrame)

    for encodeF,FaceL in zip(encodeVideo,faceCurFrame):
        matches=face_recognition.compare_faces(list,encodeF)
        faceDis = face_recognition.face_distance(list,encodeF)
        print(faceDis)
        match=np.argmin(faceDis)

        if matches[match]:
            name=className[match].upper()
            print(name)
            y1,x1,y2,x2=FaceL
            y1,x1,y2,x2=y1*4,x1*4,y2*4,x2*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(randrange(256),randrange(256),randrange(256)),4)
            cv2.putText(img,name,(x1+6,y1-6),cv2.FONT_HERSHEY_COMPLEX,1,(randrange(256),randrange(256),randrange(256)),2)
            markAttendence(name)  
        

    cv2.imshow('webcam',img)    
    key = cv2.waitKey(1)

    if key==81 or key==113:
        break