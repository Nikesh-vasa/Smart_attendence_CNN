#  Face Recognition Attendance System

This project is a **Face Recognition-based Attendance System** that uses a webcam to capture real-time photos, detects faces using OpenCV, and identifies individuals using a CNN model built with TensorFlow. It logs attendance automatically by writing recognized names and timestamps to a CSV file.

---

##  Project Overview

The system works in 3 stages:

1. **Data Collection** (`capture.py`) – Collects face images from the webcam and saves them in a structured folder based on the user's roll number.
2. **Model Training** (`train_model.py`) – Preprocesses the collected images and trains a Convolutional Neural Network (CNN) to classify faces.
3. **Real-time Attendance** (`interface.py`) – Uses the trained model to recognize faces from the webcam feed and logs attendance into a CSV file.

---

##  Directory Structure

FaceRecognitionAttendance/
│
├── capture.py # Face image collection script<br>
├── train_model.py # CNN training script<br>
├── interface.py # Face recognition + attendance logging<br>
├── face_recognition_model.h5 # Trained CNN model (auto-generated)<br>
├── attendance.csv # CSV file logging attendance (auto-generated)<br>
├── requirements.txt # List of required Python libraries<br>
├── README.md # Project documentation<br>
└── data/ # Dataset directory<br>
├── 123/ # Folder for user with roll number 123<br>
......
