# Drowsiness-Detection-system

## Overview
This Project is About the Drowsiness Detection System for detecting sleepy drivers. it uses OpenCV Library to capture the video from a webcam.  After that, Haar Cascade or Viola Jones Algorithm is used to detect A face and eyes. The system detects sleepy drivers by predicting the condition of the eyes, whether open or closed. it can be done by using the CNN model which is already trained before. 
The main goal of this project is to test the system's performance using NVIDIA Jetson Nano in real-world conditions inside the cars. The testing results showed that the system performs well as long as the camera is installed parallel to the face and the video captured is not blurry because of the high speed of the car and shaking due to uneven roads. The Performances measured are Accuracy, FPS, and Inference Speed of the system. 

## Dataset
This Project uses the MRL Eye Dataset to train the models to predict eye states. This is a large dataset consisting of open eyes and closed eyes. Because the dataset is too big, we only use 15.000 of its images. The explanation below shows the configuration of the dataset we used for training the model :
  - Training Data:
      - Open Eyes: 4000
      - Closed Eyes: 8000
  - Validation Data :
      - Open Eyes: 1000
      - Closed Eyes: 2000

## Method 
The Project consists of two main stages, training and testing. The training stage focuses on training the model to predict eye states and the testing stage focuses on building a system to detect drowsiness on a Windows laptop. 
### Training 
In the training state, there are 4 models used. All model architecture can be accessed in the CNN Graphs folder inside the Training Result Folder. All Models are trained in different epochs, which are 10,25, 50, and 75. The models were then evaluated using the Confusion Matrix and ROC Graph. Besides that, The SVM Models with different kernels and validation methods are also used to compare their performances with the CNN Models. All training results can be accessed in the Training Results folder. 
The Program used to train the model can be accessed on modelTraining.ipynb
### Testing 
After the Best model was obtained, it was used in the drowsiness detection system. The drowsiness Systems Mechanism will be explained below:
  1. Take a video input from a webcam
  2. Detect the face using Haar Cascade
  3. Detect the left eye and right using Haar Cascade
  4. Eye States predicted using the Best Trained Model
  5. If the eyes are closed for 3 seconds in a row, the alarm is triggered
The Program used for testing the system can be accessed on sistemDeteksi.ipynb

## Jetson Nano Deployment
The system was first developed on a Windows PC and deployed on NVIDIA Jetson Nano for real-world Testing. There are some developments in the program used in Jetson Nano which are :
  1. The way the camera is accessed: In Jetson Nano, we can't directly access the external camera using OpenCV. We have to use GStreamer pipelines first.
  2. Video Resolution: to maintain the system performance, the video resolution is lowered
  3. Conversion of the CNN Model: Jetson Nano can't handle the .h5 file format well. So the model is conversed into ONNX format
The Program can be accessed at sistemDeteksiJN.py


