import cv2
from keras.models import load_model
from keras.preprocessing.image import img_to_array
import numpy as np
import os

# Ensure current directory is where script resides
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Absolute path to haarcascade file
face_cascade_path = 'C:/path_to_files/haarcascade_frontalface_default.xml'
model_path = './Emotion_Detection.h5'

# Load the pre-trained models
face_classifier = cv2.CascadeClassifier(face_cascade_path)
classifier = load_model(model_path)

class_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Open the webcam (adjust the index if you have multiple cameras)
cap = cv2.VideoCapture(0)

while True:
    # Grab a single frame of video
    ret, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # make a prediction on the ROI, then lookup the class
            preds = classifier.predict(roi)[0]
            label = class_labels[preds.argmax()]
            label_position = (x, y)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, 'No Face Found', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    cv2.imshow('Emotion Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
