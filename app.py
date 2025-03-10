from flask import Flask, request, jsonify, render_template, send_file
import torch
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from cnn import CNN  # Your CNN model
from app_util import transform_image, predict_digit  # Your image processing functions
from torchvision import datasets

# Initialize Flask app
app = Flask(__name__)

# Load the trained MNIST model
model = CNN()
model.load_state_dict(torch.load("mnist_cnn.pth"))
model.eval()

UPLOAD_FOLDER = "static/uploads"
PROCESSED_FOLDER = "static/processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles image upload and prediction."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Save uploaded image
        file_path = os.path.join(UPLOAD_FOLDER, "uploaded.png")
        file.save(file_path)

        # Process image
        processed_image_path = os.path.join(PROCESSED_FOLDER, "processed.png")
        transformed_image = transform_image(file_path)  # Shape: (28, 28)
        
        # Save processed image for display
        plt.imsave(processed_image_path, transformed_image.squeeze(), cmap="gray")

        # Predict digit
        predicted_digit = predict_digit(model, transformed_image)

        return jsonify({
            "predicted_digit": predicted_digit,
            "uploaded_image_url": "/" + file_path.replace("\\", "/"),
            "processed_image_url": "/" + processed_image_path.replace("\\", "/")
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict_random', methods=['POST'])
def predict_random():
    """Selects a random MNIST image, processes it, and predicts."""
    mnist_data = datasets.MNIST(root='./data', train=True, download=True)
    idx = random.randint(0, len(mnist_data) - 1)
    image, label = mnist_data[idx]

    # Save original MNIST image
    mnist_original_path = os.path.join(UPLOAD_FOLDER, "mnist_sample.png")
    image.save(mnist_original_path)

    # Process image
    processed_image_path = os.path.join(PROCESSED_FOLDER, "processed_mnist.png")
    transformed_image = transform_image(mnist_original_path)
    
    # Save processed image
    plt.imsave(processed_image_path, transformed_image.squeeze(), cmap="gray")

    # Predict digit
    predicted_digit = predict_digit(model, transformed_image)

    return jsonify({
        "predicted_digit": predicted_digit,
        "processed_image_url": "/" + processed_image_path.replace("\\", "/")
    })

if __name__ == '__main__':
    app.run(debug=True)
