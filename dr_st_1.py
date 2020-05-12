import cv2
import numpy as np

hand_cascade = cv2.CascadeClassifier('hand1.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

fireball=cv2.imread('fireball.png',-1)
fireeye=cv2.imread('fireeye.png',-1)

cap = cv2.VideoCapture(0)

while(True):
    fireball=cv2.imread('fireball.png',-1)
    fireeye=cv2.imread('fireeye.png',-1)

    ret, frame = cap.read()
    gray = cv2. cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    hands = hand_cascade.detectMultiScale(gray, 1.3, 5)

    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2BGRA)


    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        #cv2.rectangle(frame, (x,y),(x+w, y+h),(255,0,255),2)

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            roi_eyes = roi_gray[ey:ey+h, ex:ex+w]
            #cv2.rectangle(roi_color, (ex,ey),(ex+ew, ey+eh),(255,0,0),2)
            fireeye = cv2.resize(fireeye,(ew,eh))

            gw, gh, gc = fireeye.shape
            for i in range(0, gw):
                for j in range(0 ,gh):
                    if fireeye[i,j][3] != 0:
                        roi_color[ey+i,ex+j] = fireeye[i,j] 

    for (hx,hy,hw,hh) in hands:
        roi_gray = gray[hy:hy+hh, hx:hx+hw]
        roi_color_1 = frame[hy:hy+hh, hx:hx+hw]
        #cv2.rectangle(frame, (hx,hy),(hx+hw, hy+hh),(255,0,255),2)
        fireball = cv2.resize(fireball,(hw,hh))

        kw,kh,kc = fireball.shape
        for p in range(0, kw):
            for q in range(0, kh):
                if fireball[p,q][3] != 0:
                    frame[hy+p][hy+q] = fireball[p,q]


    frame = cv2.cvtColor(frame,cv2.COLOR_BGRA2BGR)
    cv2.imshow('frame',frame)
    
    

    k = cv2.waitKey(30) & 0xff
    if k==27:
        break

cap.release()
cv2.destroyAllWindows()