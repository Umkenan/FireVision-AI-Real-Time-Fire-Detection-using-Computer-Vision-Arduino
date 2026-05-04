import cv2
import joblib
import numpy as np
import serial
import time

# Connect to Arduino
arduino = serial.Serial("COM3", 9600)
time.sleep(2)

# Load saved model
model = joblib.load("best_fire_model.pkl")

# Image settings same as training
image_size = 64

# Open laptop camera
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Could not open laptop camera.")
else:
    print("Camera opened successfully.")
    print("Put the phone image inside the green box.")
    print("Press P to predict.")
    print("Press Q to quit.")

    last_prediction_text = "No prediction yet"

    while True:
        ret, frame = camera.read()

        if not ret:
            print("Error: Could not read frame from camera.")
            break

        h, w, _ = frame.shape

        # Center green box
        box_size = 300
        x1 = w // 2 - box_size // 2
        y1 = h // 2 - box_size // 2
        x2 = x1 + box_size
        y2 = y1 + box_size

        # Draw green box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Crop only the region inside the green box
        roi = frame[y1:y2, x1:x2]

        # Show prediction text on camera window
        cv2.putText(
            frame,
            last_prediction_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Fire Detection Camera", frame)

        key = cv2.waitKey(1) & 0xFF

        # Press P to predict
        if key == ord("p"):
            img = cv2.resize(roi, (image_size, image_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.flatten()
            img = img / 255.0
            img = np.array([img])

            prediction = model.predict(img)[0]

            if prediction == 1:
                last_prediction_text = "Prediction: Fire"
                print("Prediction: Fire")
                arduino.write(b'1')   # Red LED ON

            else:
                last_prediction_text = "Prediction: Normal / No Fire"
                print("Prediction: Normal / No Fire")
                arduino.write(b'0')   # Green LED ON

        # Press Q to quit
        elif key == ord("q"):
            print("Camera closed.")
            break

    camera.release()
    arduino.close()
    cv2.destroyAllWindows()