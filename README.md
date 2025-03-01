# ENGR 1ML - Winter 2025 🧠💻  
Team: LA Hackeronie 🎭✨  

## MNIST Digit Prediction Web Application

This web application allows users to upload an image of a handwritten digit, and the application predicts the digit using a pre-trained Convolutional Neural Network (CNN) model. The app runs on Flask and uses PyTorch for the model inference.

### Table of Contents
1. [Prerequisites](#prerequisites)
2. [Setting Up the Environment](#setting-up-the-environment)
3. [Running the Application](#running-the-application)
4. [How to Use](#how-to-use)
5. [Troubleshooting](#troubleshooting)

### Prerequisites

Before running this application, ensure you have the following installed:

- **Python 3.x** (preferably Python 3.7 or higher)
- **Pip** (Python package installer)

#### Required Python Packages:
You can install all the required dependencies using the following commands:
```bash
pip install Flask torch torchvision numpy Pillow
```

### Setting up the environment
1. Clone the repository (copy the repository url into the command below)

```git clone <repository-url>```

2. Model file: the pre-trained model (`mnist_model.pth`) should eb placed in the root directory of the project, or specify the path in the Flask app (`app.py`) where the model is loaded. 

3. Download MNIST model: If you don't have the model, you can run the training loop in `image_classifier.ipynb`

4. Ensure your directory looks like this:
```
    /project-directory
        /app.py  # Your Flask app
        /templates/index.html  # Frontend HTML file
        /mnist_model.pth  # Your trained model file

```

## Running the Application
1. Start the Flask Application: run the following command in the terminal
`python app.py`

2. Access the application by navigating to `http://127.0.0.1:5000/`

## How to Use
Upload and image to the webpage, submit the image by clicking on the 'predict' button. 

## How it works
- **Frontend:** User interface built using HTML and JavaScript
- **Backend:** Built using Flask
- **Image Processing:** Uploaded image is received by Flask and passed to `transform_image` function for preprocessing. The image is converted to grayscale (since MNIST images are single-channel), resized to 28x28, which is the input size expected by the model. Pixel values are then normalized to the range [0,1] to help with model accuracy. The image is then converted into a NumPy array, which is a format that can be processed by the model. This processed image is then ready for the mdoel inference.
- **Model Inference:** The pretrained model (`mnist_mdoel.pth`) is then loaded in the backend. This model was trained on the MNIST dataset and is capable of recognizing handwritten digits (0-9). The processed image is passed to the model as a PyTorch tensor. The model performs inference and outputs a prediction in the form of logits. The predicted digit is obtained by applying `torch.argmax()`, which returns the class with the highest score. 
- **Returning the Result:** Once the mdoel predicts the digit, the result is sent back to the frontend (browser) as a JSON response. The prediction is displayed on the page under the "Predicted Digit" label, so the user can see the model's guess. 
