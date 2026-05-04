import os
import cv2
import numpy as np
import random

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Dataset path
dataset_path = "datafire/dataset"

# Image settings
image_size = 64

# Lists to store images and labels
images = []
labels = []

# Class folders
class_folders = ["0", "1"]

# Count images in each class
class_counts = {}

for label in class_folders:
    folder_path = os.path.join(dataset_path, label)
    file_names = os.listdir(folder_path)
    class_counts[label] = len(file_names)

print("Original dataset count:")
print("Normal / No Fire:", class_counts["0"])
print("Fire:", class_counts["1"])

# Balance the dataset by taking the same number from each class
samples_per_class = min(class_counts.values())

print("\nBalanced samples per class:", samples_per_class)

# Load images
for label in class_folders:
    folder_path = os.path.join(dataset_path, label)

    file_names = os.listdir(folder_path)

    # Shuffle images so we do not always take the first files only
    random.seed(42)
    random.shuffle(file_names)

    # Take equal number from each class
    file_names = file_names[:samples_per_class]

    for file_name in file_names:
        image_path = os.path.join(folder_path, file_name)

        img = cv2.imread(image_path)

        if img is not None:
           img = cv2.resize(img, (image_size, image_size))
           img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
           img = img.flatten()
           img = img / 255.0

           images.append(img)
           labels.append(int(label))

# Convert to numpy arrays
X = np.array(images)
y = np.array(labels)

print("\nImages loaded successfully")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Labels:", np.unique(y, return_counts=True))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nData split completed")
print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# =========================
# KNN Model
# =========================
knn_model = KNeighborsClassifier(n_neighbors=3)

knn_model.fit(X_train, y_train)

knn_predictions = knn_model.predict(X_test)

knn_accuracy = accuracy_score(y_test, knn_predictions)

print("\nKNN Model Training Completed")
print("KNN Accuracy:", knn_accuracy)

# =========================
# SVM Model
# =========================
svm_model = SVC(kernel="linear")

svm_model.fit(X_train, y_train)

svm_predictions = svm_model.predict(X_test)

svm_accuracy = accuracy_score(y_test, svm_predictions)

print("\nSVM Model Training Completed")
print("SVM Accuracy:", svm_accuracy)

# =========================
# Logistic Regression Model
# =========================
logistic_model = LogisticRegression(max_iter=1000)

logistic_model.fit(X_train, y_train)

logistic_predictions = logistic_model.predict(X_test)

logistic_accuracy = accuracy_score(y_test, logistic_predictions)

print("\nLogistic Regression Model Training Completed")
print("Logistic Regression Accuracy:", logistic_accuracy)

# =========================
# Evaluation Reports
# =========================
print("\n==============================")
print("KNN Evaluation")
print("==============================")
print("Confusion Matrix:")
print(confusion_matrix(y_test, knn_predictions))
print("\nClassification Report:")
print(classification_report(y_test, knn_predictions, target_names=["Normal", "Fire"]))

print("\n==============================")
print("SVM Evaluation")
print("==============================")
print("Confusion Matrix:")
print(confusion_matrix(y_test, svm_predictions))
print("\nClassification Report:")
print(classification_report(y_test, svm_predictions, target_names=["Normal", "Fire"]))

print("\n==============================")
print("Logistic Regression Evaluation")
print("==============================")
print("Confusion Matrix:")
print(confusion_matrix(y_test, logistic_predictions))
print("\nClassification Report:")
print(classification_report(y_test, logistic_predictions, target_names=["Normal", "Fire"]))

# =========================
# Best Model Selection
# =========================
accuracies = {
    "KNN": knn_accuracy,
    "SVM": svm_accuracy,
    "Logistic Regression": logistic_accuracy
}

best_model_name = max(accuracies, key=accuracies.get)
best_accuracy = accuracies[best_model_name]

print("\n==============================")
print("Model Comparison")
print("==============================")
print("KNN Accuracy:", knn_accuracy)
print("SVM Accuracy:", svm_accuracy)
print("Logistic Regression Accuracy:", logistic_accuracy)

print("\nBest Model:", best_model_name)
print("Best Accuracy:", best_accuracy)

import joblib

joblib.dump(logistic_model, "best_fire_model.pkl")

print("\nBest model saved as best_fire_model.pkl")