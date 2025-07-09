# train_model.py
import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers

# Step 1: Dataset Collection
dataset_path = "data"

# Step 2: Data Preprocessing
def preprocess_dataset(dataset_path):
    faces = []
    labels = []

    for label, person in enumerate(os.listdir(dataset_path)):
        person_folder = os.path.join(dataset_path, person)
        for image_name in os.listdir(person_folder):
            image_path = os.path.join(person_folder, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (64, 64))
            faces.append(image)
            labels.append(label)

    faces = np.array(faces)
    labels = np.array(labels)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(faces, labels, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = preprocess_dataset(dataset_path)

# Reshape the input data for ImageDataGenerator
X_train = X_train.reshape(-1, 64, 64, 1)

# Step 3: Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest'
)

datagen.fit(X_train)

# Step 4: CNN Model Training
input_shape = (64, 64, 1)

# Functional API for model definition
inputs = layers.Input(shape=input_shape)

x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Dropout(0.25)(x)
x = layers.Conv2D(64, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Dropout(0.25)(x)
x = layers.Conv2D(128, (3, 3), activation='relu')(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Dropout(0.25)(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)

# Branch for person recognition
person_branch = layers.Dense(len(os.listdir(dataset_path)), activation='softmax')(x)

# Final model
model = models.Model(inputs=inputs, outputs=person_branch)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

# Step 5: Model Evaluation
loss, accuracy = model.evaluate(X_test, y_test)
print("Accuracy:", accuracy)

# Save the model
model.save('face_recognition_model.h5')
