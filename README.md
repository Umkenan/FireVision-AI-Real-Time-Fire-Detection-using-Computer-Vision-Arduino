# FireVision-AI-Real-Time-Fire-Detection-using-Computer-Vision-Arduino
Real-time fire detection system using Machine Learning (KNN, SVM, Logistic Regression) + OpenCV + Arduino.  Classifies images as Fire or Normal, then runs on live camera and triggers LED alerts.
# FireVision AI – Fire Detection System

## Overview
This project uses **Machine Learning** (KNN, SVM, Logistic Regression) to classify images into **Fire** (1) or **Normal/No Fire** (0).  
The best model is then deployed in a **real-time camera system** that sends a signal to an **Arduino** to light up a red LED when fire is detected.

## Features
- Trains 3 models (KNN, SVM, Logistic Regression) on balanced fire/normal images
- Compares accuracy and selects the best model automatically
- Real-time detection from laptop camera (green box guides the user)
- Arduino integration: Red LED for fire, Green LED for normal state
- Works on **live video stream** with press "P" to predict

## Technologies Used
- Python
- OpenCV (image processing)
- Scikit-learn (KNN, SVM, Logistic Regression)
- Joblib (model saving/loading)
- PySerial (Arduino communication)
- Arduino Uno + LEDs

## How to Run
1. Train the model:
2. Test on a single image:
3. Run live camera detection:

## Folder Structure
FireVision/
├── datafire/dataset/0/ (normal images)
├── datafire/dataset/1/ (fire images)
├── main.py (training)
├── predict.py (single image test)
├── live.py (camera + arduino)
├── best_fire_model.pkl (saved best model)
├── requirements.txt
└── README.md

## Arduino Connection
- Connect Arduino to **COM3** (or update in code)
- Red LED → pin 13
- Green LED → pin 12 (example, adjust as needed)

