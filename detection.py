import cv2
import numpy as np
from skimage.feature import hog
import pickle

# Load the trained KNN model and class names
with open('knn_model.pkl', 'rb') as model_file: #load your model, keep the model in the same folder
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

# Start capturing video from the webcam
def live_video_predict():
    cap = cv2.VideoCapture(0)  # Use '0' for webcam

    if not cap.isOpened():
        print("Failed to open the webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Sobel operator to detect edges
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)  # Gradient in x-direction
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)  # Gradient in y-direction
        sobel_combined = cv2.magnitude(sobel_x, sobel_y)  # Combine the gradients

        # Normalize the result to the range 0-255
        sobel_combined = np.uint8(255 * sobel_combined / np.max(sobel_combined))

        # Threshold the Sobel result to get binary image
        _, thresh = cv2.threshold(sobel_combined, 70, 255, cv2.THRESH_BINARY)

        # Find contours in the Sobel edge map
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Loop over the contours
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filter small contours
                # Get the bounding box for the contour
                (x, y, w, h) = cv2.boundingRect(contour)

                # Extract the region of interest (ROI) for prediction
                roi = frame[y:y+h, x:x+w]
                feature_vector = preprocess_image(roi)

                # Predict the class of the ROI
                prediction = knn.predict([feature_vector])
                label = class_names[prediction[0]]
                label = fit_text(label)

                # Draw a bounding box around the object
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw the predicted label on the bounding box
                cv2.putText(frame, f'Predicted: {label}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.8, (255, 0, 0), 2, cv2.LINE_AA)

        # Display the original frame with the bounding box
        cv2.imshow('Live Prediction with Bounding Box', frame)

        # Display the Sobel effect frame
        cv2.imshow('Sobel Effect', sobel_combined)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Call the function to start live video prediction
live_video_predict()
