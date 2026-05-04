import os
import cv2
import joblib
import numpy as np

# Load the saved best model
model = joblib.load("best_fire_model.pkl")

# Image settings must match training
image_size = 64

# Choose class folder:
# 1 = Fire
# 0 = Normal / No Fire
test_folder = "datafire/dataset/1"

# Get first image automatically from the folder
image_files = os.listdir(test_folder)
image_files = [f for f in image_files if f.lower().endswith((".jpg", ".jpeg", ".png"))]

if len(image_files) == 0:
    print("Error: No image files found in the folder.")
else:
    image_path = os.path.join(test_folder, image_files[0])
    print("Testing image:", image_path)

    img = cv2.imread(image_path)

    if img is None:
        print("Error: Image not found or cannot be opened.")
    else:
        img = cv2.resize(img, (image_size, image_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.flatten()
        img = img / 255.0

        img = np.array([img])

        prediction = model.predict(img)[0]

        if prediction == 1:
            print("Prediction: Fire")
        else:
            print("Prediction: Normal / No Fire")