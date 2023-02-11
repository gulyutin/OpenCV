import cv2

face_cascade_db = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
eye_cascade_db = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_eye.xml")

myrtmp_addr = "rtmp://ovsu.mycdn.me/input/4728756120437_3756447238773_iwly3g3icu"
cap = cv2.VideoCapture(myrtmp_addr)


# cap = cv2.VideoCapture(0)

while True: 
    success, img = cap.read()

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade_db.detectMultiScale(img_gray, 1.1, 19)

    for (x,y,w,h) in faces: 
        cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)
        img_gray_face = img_gray[y:y+h,x:x+w]
        eyes = eye_cascade_db.detectMultiScale(img_gray_face, 1.1, 19)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(img, (x+ex,y+ey), (x+ex+ew, y+ey+eh), (255, 0, 0), 2)

    cv2.imshow('rez', img)
    if cv2.waitKey() & 0xff == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
