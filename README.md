<h1 align="center">KNN Object Detection</h1>

<p style="text-align: justify;">This project implements a K-Nearest Neighbors (KNN) algorithm for object detection, designed to classify objects within an image or video stream by comparing pixel feature vectors to a labeled dataset. The system scans an image using sliding windows, extracting features from each region and determining the object's class based on its K-nearest neighbors from the training set. The model only labels detected objects, ensuring minimal interference with non-object regions. Customizable parameters like the number of neighbors and window size allow for fine-tuning the detection accuracy. The project supports real-time video and image-based classification, making it versatile for various applications.</p>

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
  <img src="images/Dataset Define.png" alt="Dataset Image 1" width="400" height="300">
  <img src="images/KNN Example Data.png" alt="Dataset Image 2" width="400" height="300">
</div>

<p>Each class has been carefully labeled to ensure accurate training for object detection.</p>

<h2>Installation</h2>
<ul>
  <li>Clone the repository: <code>git clone https://github.com/kartheek1104/KNN-object-detection.git</code></li>
  <li>Change the location of the dataset in the "training.py"</li>
  <li>After training the model open "detection.py" and make sure to change model location before executing</li>
</ul>

<h2>Example Output</h2>
<p>Below is an example of the output after running the KNN object detection:</p>

<img src="images/Output Result.png" alt="Output Image" width="600">
