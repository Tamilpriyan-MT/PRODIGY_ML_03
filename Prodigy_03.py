import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Function to load images and labels
def load_images_and_labels(dataset_path):
    images = []
    labels = []
    for label, category in enumerate(['cat', 'dog']):
        category_path = os.path.join(dataset_path, category)
        for filename in os.listdir(category_path):
            if filename.endswith('.jpg'):
                img = cv2.imread(os.path.join(category_path, filename))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                img = cv2.resize(img, (64, 64))  # Resize to a standard size
                images.append(img.flatten())  # Flatten to use as feature vector
                labels.append(label)
    return np.array(images), np.array(labels)

# Load dataset
dataset_path = '/path/to/your/dataset'
X, y = load_images_and_labels(dataset_path)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create SVM model
svm_model = SVC(kernel='linear', C=1.0, random_state=42)

# Train the model
svm_model.fit(X_train, y_train)

# Predictions on test set
y_pred = svm_model.predict(X_test)

# Evaluate performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['cat', 'dog']))
