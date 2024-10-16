<h1 align="center">KNN Object Detection</h1>

<p>This project implements a K-Nearest Neighbors (KNN) algorithm for object detection, designed to classify objects within an image or video stream by comparing pixel feature vectors to a labeled dataset. The system scans an image using sliding windows, extracting features from each region and determining the object's class based on its K-nearest neighbors from the training set. The model only labels detected objects, ensuring minimal interference with non-object regions. Customizable parameters like the number of neighbors and window size allow for fine-tuning the detection accuracy. The project supports real-time video and image-based classification, making it versatile for various applications.</p>

<h2>Key Features</h2>
<ul>
  <li>Uses KNN for object detection.</li>
  <li>Supports both image-based and live video classification.</li>
  <li>Labels only the detected objects, leaving other areas untouched.</li>
  <li>Customizable parameters like the number of neighbors and window size for detection accuracy.</li>
</ul>

<h2>Dataset Used</h2>
<p>The dataset used for this project consists of images that represent different object classes. Below are some sample images from the dataset:</p>

<div style="display: flex; justify-content: space-around;">
  <img src="dataset_image1.png" alt="Dataset Image 1" width="400" height="300">
  <img src="dataset_image2.png" alt="Dataset Image 2" width="400" height="300">
</div>

<p>Each class has been carefully labeled to ensure accurate training for object detection.</p>

<h2>Installation</h2>
<ul>
  <li>Clone the repository: <code>git clone https://github.com/yourusername/knn-object-detection.git</code></li>
  <li>Install dependencies: <code>pip install -r requirements.txt</code></li>
  <li>Run the detection: <code>python knn_detection.py</code></li>
</ul>

<h2>Example Output</h2>
<p>Below is an example of the output after running the KNN object detection:</p>

<img src="output_image.png" alt="Output Image" width="600">
