# inferface.py
import os
import cv2
import csv
import datetime
import numpy as np
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('face_recognition_model.h5')

# Step 6: Inference and Attendance System
def detect_faces_and_mark_attendance(model, dataset_path, threshold=0.7):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)
    attendees = []

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (64, 64))
            face_img = np.expand_dims(face_img, axis=-1)
            face_img = np.expand_dims(face_img, axis=0)
            prediction = model.predict(face_img)
            confidence = np.max(prediction)
            label = np.argmax(prediction)

            if confidence > threshold:
                person = os.listdir(dataset_path)[label]
                if person not in attendees:
                    attendees.append(person)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, person, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    mark_attendance(person)
                else:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(frame, f"Already Marked {person}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, "Unrecognized Person", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def mark_attendance(person):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open('attendance.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([person, 'Present', timestamp])

# Set the threshold value for detection confidence
detection_threshold = 0.7

# Detect faces and mark attendance
dataset_path = "data"
detect_faces_and_mark_attendance(model, dataset_path, threshold=detection_threshold)
