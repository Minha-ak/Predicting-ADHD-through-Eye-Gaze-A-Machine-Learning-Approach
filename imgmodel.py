import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from google.oauth2 import service_account
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

# Authenticate and create a Google Drive service
def authenticate_drive():
    SCOPES = ['https://www.googleapis.com/auth/drive']
    SERVICE_ACCOUNT_FILE = 'path/to/your/credentials.json'  # Update with your credentials file path

    creds = None
    creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SCOPES)

    return build('drive', 'v3', credentials=creds)

# Function to download files from Google Drive
def download_file_from_drive(file_id, file_name):
    drive_service = authenticate_drive()

    request = drive_service.files().get_media(fileId=file_id)
    fh = open(file_name, 'wb')
    downloader = MediaIoBaseDownload(fh, request)

    done = False
    while not done:
        status, done = downloader.next_chunk()

    fh.close()

# Example usage to download a file
# download_file_from_drive('your_file_id', 'local_file_name')

# Paths to training and testing data
train_dir = '/path/to/your/training/data'
test_dir = '/path/to/your/testing/data'

# Parameters
batch_size = 4
image_size = (150, 150)
epochs = 20

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Flow training images in batches of batch_size
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary'
)

# Flow validation images in batches of batch_size
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='binary'
)

# Build simplified CNN model with dropout
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),  # Add dropout with dropout rate of 0.5
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps=test_generator.samples // batch_size
)

# Save the trained model
model.save("keras_Model.h5")

# Save the labels
class_names = train_generator.class_indices
with open("labels.txt", "w") as f:
    for class_name in class_names:
        f.write(class_name + '\n')

# Load the model
model = load_model("keras_Model.h5")

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Example prediction with a local image
from PIL import Image, ImageOps

def predict_local_image(image_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    size = (150, 150)
    image = ImageOps.fit(image, size, Image.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 255.0)
    data = np.ndarray(shape=(1, 150, 150, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Predict using the loaded model
    prediction = model.predict(data)
    class_name = class_names[int(prediction[0][0])]
    confidence_score = prediction[0][0]

    return class_name.strip(), confidence_score

# Example usage to predict a local image
# result = predict_local_image('/path/to/your/image.jpg')
# print("Class:", result[0])
# print("Confidence Score:", result[1])


















# Path to your service account JSON file
SERVICE_ACCOUNT_FILE = r"C:\Users\hp\Downloads\image-checker-423607-292291835303.json"

# Function to authenticate and build Google Drive service


def create_drive_service():
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE,
        scopes=['https://www.googleapis.com/auth/drive']
    )
    return build('drive', 'v3', credentials=credentials)
