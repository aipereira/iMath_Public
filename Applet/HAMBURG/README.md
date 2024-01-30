## Image Recognition Web Application with TensorFlow and Streamlit

Welcome to the Picture Recognition App! This README provides an overview of the app and guides you through its functionalities.
This web application enables image recognition using neural networks for various datasets such as Fashion MNIST, CIFAR-10, and MNIST. The project was created using TensorFlow and Streamlit.

## Overview

The web application provides various actions, including retraining the model, testing images, and guidelines for usage. It is based on TensorFlow for the neural network and Streamlit for the user interface.

## How to Use the Application

1. **Select Action**

   Choose one of the following actions in the sidebar: "Test Image," "Retrain Model," "Guide Lines."

3. **Retrain Model**
- Click on "Retrain Model" in the sidebar.
- Select the desired dataset (Fashion MNIST, CIFAR-10, or MNIST).
- Adjust retraining options, such as the number of epochs and the size of the training set.
- Click the "Retrain Model" button and observe the training progress.
- After training completion, accuracy and loss graphs will be displayed.

3. **Test Image:**
- Click on "Test Image" in the sidebar.
- Choose the dataset on which the model was trained.
- Upload an image using the file upload tool.
- Click the "Predict" button to view the model's prediction.

4. **General Tips:**
- Wait for the model to complete training before expecting accurate predictions.
- When testing an image, choose the appropriate dataset for the trained model.
  
## Instructions

### Retrain Model:
- Various options are provided for retraining the model, including the number of epochs and the size of the training set. The progress indicator informs about the training status.

### Test Image 
- Upload an image and get a prediction from the trained model. The prediction is displayed directly on the webpage.
### General Tips:
- The sidebar offers easy navigation options for the different actions.
- Accuracy and loss graphs provide insights into the training progress.

## Dependencies
- Python==3.10 or 3.11
- TensorFlow==2.15
- Streamlit==1.29
- NumPy==1.24.3
- scikit-learn==1.3.0
- Matplotlib==3.7.2
- PIL (Pillow)==9.4.0

## Installation and Execution 

1- **Install the required dependencies with the command:**
```
   conda create -n app_env python=3.10
   conda activate app_env
   pip install tensorflow==2.15
   pip install streamlit==1.29
   pip install numpy==1.24.3
   pip install scikit-learn==1.3.0
   pip install matplotlib==3.7.2
   pip install Pillow==9.4.0
```

2- **Start the application with the command:** 
```
   streamlit run app.py
```
The application will then be available at localhost:8501.
   
**Enjoy using the Picture Recognition App!**
