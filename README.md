# Drowsiness-Detection-system

## Overview
This Project is aiming to detect sleepy drivers. it uses OpenCV Library to capture the video from a webcam.  After that, Haar Cascade or Viola Jones Algorithm is used to detect A face and eyes. The system detects sleepy drivers by predicting the condition of the eyes, whether open or closed. it can be done by using the CNN model which is already trained before. 
## Dataset
This Project uses the MRL Eye Dataset. This is a large dataset consisting of open eyes and closed eyes. Because the dataset is too big, we only use 15.000 of its images and can be accessed [here](https://github.com/irsyadnrzn/Drowsiness-Detection-system/tree/main/Dataset). The explanation below shows the configuration of the dataset we used for training the model : <br>      
| **Eyes State** | **Training Data** | **Validation Data** |
| ---- | ---- | ---- |
| **Open Eyes** |  4000  | 1000 |
| **Closed Eyes** |  8000  | 2000 |
## Method 
The Project consists of two main stages, training and testing. The training stage focuses on training the model to predict eye states and the testing stage focuses on building a system to detect drowsiness on a Windows laptop. 
### Training 
Training done using 4 models and all the model architecture can be accessed in the [CNN Graphs folder](). All Models are trained in different epochs, 10,25, 50, and 75. The models were evaluated using Confusion Matrix and ROC Graph. Besides that, The SVM Models with different kernels and validation methods are also used to compare their performances with the CNN Models. All training results can be accessed in the [Training Results folder](https://github.com/irsyadnrzn/Drowsiness-Detection-system/tree/main/Training%20Results). 
The Program used to train the model can be accessed [here](https://github.com/irsyadnrzn/Drowsiness-Detection-system/blob/main/modelTraining.ipynb).
### Testing 
After the Best model was obtained, it was used in the drowsiness detection system. The drowsiness Systems Mechanism will be explained below:
  1. Take a video input from a webcam
  2. Detect the face using Haar Cascade
  3. Detect the left eye and right using Haar Cascade
  4. Eye States predicted using the Best Trained Model
  5. If the eyes are closed for 3 seconds in a row, the alarm is triggered
The Program used for testing the system can be accessed on [sistemDeteksi.ipynb](https://github.com/irsyadnrzn/Drowsiness-Detection-system/blob/main/sistemDeteksi.ipynb)
