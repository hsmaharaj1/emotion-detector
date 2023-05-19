import cv2
import tensorflow as tf
import numpy as np

emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
face_detection_model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
emotion_model = tf.keras.models.load_model("fer2013_mini_XCEPTION.102-0.66.hdf5")

video = cv2.VideoCapture(0)

while True:
    chk, frame = video.read()
    grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_detection_model.detectMultiScale(grey_frame, 1.2, 6)

    for (x,y,width,height) in faces:
        face_image = grey_frame[y:y+height, x:x+width]
        # cv2.rectangle(frame, (x,y), (x+width, y+height), (255, 255, 255), 3)

        #Preprocessing
        face_image = cv2.resize(face_image, (64,64))
        face_image = np.expand_dims(face_image, axis=0)
        face_image = np.expand_dims(face_image, axis=-1)
        face_image = face_image/255

        emotion_predict = emotion_model.predict(face_image)[0]
        max_index = np.argmax(emotion_predict)
        emotion_label = np.argmax(emotion_predict)
        preicted_data=emotions[emotion_label]
        emotion_probab = emotion_predict[max_index]
        probability_percentages = f"{emotion_probab*100:.2f}%"

        cv2.rectangle(frame, (x,y), (x+width, y+height), (1,255,23), 3)
        cv2.putText(frame, preicted_data, (x,y-10), cv2.FONT_HERSHEY_DUPLEX, 0.9, (36,255,1), 2)
        cv2.putText(frame, probability_percentages, (x, y+height+25), cv2.FONT_ITALIC, 0.6, (255, 255, 255), 2)


    cv2.imshow('CAM', frame)

    key = cv2.waitKey(1)
    if key == 27 or cv2.getWindowProperty('CAM', cv2.WND_PROP_VISIBLE)<1:
        break

video.release()
cv2.destroyAllWindows()
