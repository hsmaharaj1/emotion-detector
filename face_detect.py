import cv2

face_detection_model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video = cv2.VideoCapture(0)

while True:
    chk, frame = video.read()
    grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_detection_model.detectMultiScale(grey_frame, scaleFactor=1.1, minNeighbors=6)

    for (x,y,width,height) in faces:
        cv2.rectangle(frame, (x,y), (x+width, y+height), (255, 255, 255), 3)

    cv2.imshow('video', frame)

    key = cv2.waitKey(30) & 0xff
    if key == 27:
        break

video.release()
cv2.destroyAllWindows()
