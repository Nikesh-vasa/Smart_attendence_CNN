import cv2
import os
import time
import uuid

# Directory to save the collected face images
output_dir = input('Enter your roll number: ')

# Base path for saving images
base_path = 'C:/Users/Nikesh/OneDrive/Desktop/mini p/Data'
output_path = os.path.join(base_path, output_dir)

# Create the output directory if it doesn't exist
if not os.path.exists(output_path):
    os.makedirs(output_path)

# Load the pre-trained face detector from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the webcam
video_capture = cv2.VideoCapture(0)

time.sleep(2)  # Give some time for the camera to warm up

# Counter for the number of collected face images
image_count = 0

# Set the maximum number of images to collect
max_images = 150

# Flag to indicate when to start capturing images
capture_started = False

# Instructions for the user
print("Press 's' to start capturing images. Press 'q' to quit.")

while image_count < max_images:
    # Read a frame from the webcam
    ret, frame = video_capture.read()

    if not ret:
        print("Failed to grab frame")
        break

    if not capture_started:
        # Show instructions to the user
        cv2.putText(frame, 'Press "s" to start capturing images', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around the detected faces and save the images
        for (x, y, w, h) in faces:
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Increment the image count
            image_count += 1

            # Save the face image to the output directory
            face_image_path = os.path.join(output_path, f'{output_dir}_{str(uuid.uuid4())}.jpg')
            cv2.imwrite(face_image_path, frame[y:y+h, x:x+w])

            # Break the loop if the maximum number of images is reached
            if image_count >= max_images:
                break

        # Display the resulting frame with the image count overlay
        cv2.putText(frame, f'Images Taken: {image_count}/{max_images}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the video feed
    cv2.imshow('Video', frame)

    # Wait for a key press
    key = cv2.waitKey(1) & 0xFF

    # Start capturing images if 's' is pressed
    if key == ord('s'):
        capture_started = True

    # Break the loop when the 'q' key is pressed
    if key == ord('q'):
        break

# Release the webcam and close all windows
video_capture.release()
cv2.destroyAllWindows()

