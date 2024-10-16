import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from skimage.feature import hog
from sklearn.model_selection import train_test_split, cross_val_score
import os
import pickle
from sklearn.utils import shuffle
from tqdm import tqdm  # For progress tracking

# Parameters
image_size = (64, 64)  # Resize all images to 64x64 pixels
allowed_extensions = ['.png', '.jpg', '.jpeg', '.bmp']  # Supported image formats

# Function to load images and extract HOG features
def load_images_from_folder(folder):
    images = []
    labels = []
    class_names = []
    
    for label, object_name in enumerate(os.listdir(folder)):
        object_folder = os.path.join(folder, object_name)
        if not os.path.isdir(object_folder):
            continue
        class_names.append(object_name)  # Store the object name for later use

        # Iterate through all files in the folder
        for filename in tqdm(os.listdir(object_folder), desc=f'Loading {object_name}'):
            img_path = os.path.join(object_folder, filename)
            ext = os.path.splitext(filename)[-1].lower()
            if ext not in allowed_extensions:
                continue  # Skip files that are not images

            img = cv2.imread(img_path)
            if img is not None:
                # Resize image and convert to grayscale
                img_resized = cv2.resize(img, image_size)
                gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
                
                # Feature extraction using HOG
                feature_vector = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                     cells_per_block=(2, 2), visualize=False)
                
                images.append(feature_vector)
                labels.append(label)
    
    return np.array(images), np.array(labels), class_names

# Load the dataset
dataset_folder = 'your dataset location'  # Update to your dataset path
X, y, class_names = load_images_from_folder(dataset_folder)

# Shuffle the dataset
X, y = shuffle(X, y, random_state=42)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the KNN model
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)  # Use all available cores to speed up training
knn.fit(X_train, y_train)

# Evaluate the model using cross-validation
cv_scores = cross_val_score(knn, X, y, cv=5, n_jobs=-1)  # 5-fold cross-validation
print(f'Cross-validation accuracy: {cv_scores.mean() * 100:.2f}%')

# Evaluate the model on the test set
accuracy = knn.score(X_test, y_test)
print(f'Test set accuracy: {accuracy * 100:.2f}%')

# Save the trained model and class names to a file for later use
with open('knn_model.pkl', 'wb') as model_file:
    pickle.dump((knn, class_names), model_file)
