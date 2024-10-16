import cv2
import numpy as np
from skimage.feature import hog
import pickle
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Load the trained KNN model and class names
with open('knn_model.pkl', 'rb') as model_file:
    knn, class_names = pickle.load(model_file)

# Parameters
image_size = (64, 64)  # Resize all images to 64x64 pixels

# Function to preprocess the image and extract HOG features
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray, image_size)
    feature_vector = hog(resized_image, orientations=9, pixels_per_cell=(8, 8),
                         cells_per_block=(2, 2), visualize=False)
    return feature_vector

# Function to wrap or truncate text if it's too long
def fit_text(text, max_length=20):
    if len(text) > max_length:
        return text[:max_length] + "..."  # Truncate and add ellipsis
    return text

# Function to load and predict the class of an uploaded image
def upload_and_predict():
    # Open file dialog to upload an image
    Tk().withdraw()  # We don't want a full GUI, so keep the root window from appearing
    file_path = askopenfilename(title="Select an image", filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")])
    
    if not file_path:
        print("No file selected.")
        return

    # Load and preprocess the uploaded image
    image = cv2.imread(file_path)
    if image is None:
        print("Failed to load the image.")
        return

    feature_vector = preprocess_image(image)
    
    # Predict the class of the image
    prediction = knn.predict([feature_vector])
    label = class_names[prediction[0]]
    label = fit_text(label)  # Ensure the text fits

    # Display the prediction
    print(f"Predicted class: {label}")

    # Adjust font scale, color, and text position
    font_scale = 0.8  # Reduce the font size
    text_position = (10, 50)  # Lower the position
    text_color = (255, 0, 0)  # Red text
    thickness = 2  # Thickness of the text
    
    # Show the image with the predicted label
    cv2.putText(image, f'Predicted: {label}', text_position, cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, text_color, thickness, cv2.LINE_AA)
    cv2.imshow('Prediction', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Call the function to upload and predict the image class
upload_and_predict()
