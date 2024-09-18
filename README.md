# Drowsiness Detection System

## Overview

Traffic accidents are one of the leading causes of death in the world. Based on publications by the World Health Organization (WHO), in 2018, traffic accidents caused at least 1.35 million deaths every year, making it the eighth most significant cause of death in the world. One way to overcome this is by developing a drowsiness detection system for drivers. Some research has been done and got excellent results. Nevertheless, most earlier studies did not test the system in real-world scenarios in a moving car. This research aims to test the drowsiness detection system in a moving vehicle. The system uses the Haar cascade algorithm to detect faces and eyes. The convolutional Neural Network (CNN) model predicted open and closed eyes. The research results showed that the best model performance was obtained by CNN model-2, with accuracy, precision, recall, and f1-score values on test data of 92%. In the real-time detection, the detection system obtained an accuracy of 94.6% in bright light and 86.4% in dim light. The system was also tested in a moving car using Nvidia Jetson Nano as a computational device and got an overall accuracy of 87%.

## Dataset

This project utilizes the MRL Eye Dataset, a large dataset containing images of both open and closed eyes. Due to the dataset's size, we use a subset of 15,000 images as shown in the table bellow.

| **Eyes State** | **Training Data** | **Validation Data** |
| -------------- | ----------------- | ------------------- |
| **Open Eyes**  | 4000              | 1000                |
| **Closed Eyes**| 8000              | 2000                |

The images bellow show the example of images used for training the model. To access the wholes images, you can click [here](https://github.com/irsyadnrzn/Drowsiness-Detection-system/tree/main/Dataset).

<p align="center">
  <img src="https://github.com/irsyadnrzn/Drowsiness-Detection-system/blob/main/closed-open%20eyes.png" width="40%" height="40%">
</p>

## Method

The project consists of two main stages: training and testing. The training stage focuses on developing the model to predict eye states, while the testing stage involves building a system to detect drowsiness on a Windows laptop.

### Training

The training process was conducted using four different models, and the architectures for these models can be found in the [Model Architectures folder](https://github.com/irsyadnrzn/Drowsiness-Detection-system/tree/main/Model%20Architectures). Each model was trained for various epochs, ranging from 25 to 75. The models' performance was evaluated using a Confusion Matrix and ROC curve.

Additionally, Support Vector Machine (SVM) models with different kernels and validation methods were tested to compare their performance with the CNN models.

- All training results are available in the [Training Results folder](https://github.com/irsyadnrzn/Drowsiness-Detection-system/tree/main/Training%20Results).
- The script used to train the models can be accessed [here](https://github.com/irsyadnrzn/Drowsiness-Detection-system/blob/main/modelTraining.ipynb).

### Testing

Once the best model was identified, it was integrated into the drowsiness detection system. The system's mechanism is described below:

1. Capture video input from a webcam.
2. Detect the face using Haar Cascade.
3. Detect the left and right eyes using Haar Cascade.
4. Predict eye states (open or closed) using the best-trained model.
5. Trigger an alarm if the eyes remain closed for 3 consecutive seconds.

The script for testing the system can be found in [sistemDeteksi.ipynb](https://github.com/irsyadnrzn/Drowsiness-Detection-system/blob/main/sistemDeteksi.ipynb).

## Future Work

- **Real-Time Optimization**: Improve the real-time performance of the system for use in real-world scenarios.
- **Integration with Vehicles**: Develop a prototype for integrating the system into vehicles.
- **Dataset Expansion**: Use a larger and more diverse dataset to improve the robustness of the model.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any bugs or feature suggestions.
