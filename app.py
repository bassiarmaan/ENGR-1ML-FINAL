from flask import Flask, request, jsonify, render_template
import torch
from cnn import CNN  # Assuming your model is in CNN_model.py
from app_util import transform_image, predict_digit  # Assuming this function is in transform_image.py

# Initialize the Flask app
app = Flask(__name__)

# Load the trained MNIST model
model = CNN()
model.load_state_dict(torch.load("mnist_cnn.pth"))  # Load model weights
model.eval()  # Set model to evaluation mode

@app.route('/')
def index():
    """Render the homepage with an image upload form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle the image upload and prediction."""
    # Check if the image file is part of the request
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Save the uploaded image temporarily
        file_path = "temp_image.png"
        file.save(file_path)

        # Transform the image and predict the digit
        transformed_image = transform_image(file_path)  # Shape: (28, 28)
        predicted_digit = predict_digit(model, transformed_image)

        # Return the prediction as a JSON response
        return jsonify({"predicted_digit": predicted_digit})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
