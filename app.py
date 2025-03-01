import torch
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
from flask import Flask, render_template, request, jsonify
import io
import base64
import binascii
from cnn import CNN

# Load the trained model
file_path = 'mnist_cnn.pth'
model = CNN()
model.load_state_dict(torch.load(file_path))
model.eval()

# Define the transformation for the image
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # Resize to match MNIST dimensions
    transforms.Grayscale(num_output_channels=1),  # Ensure it's grayscale
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize (for MNIST)
])

# Function to convert PIL image to base64
def pil_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

# Initialize the Flask app
app = Flask(__name__)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle image processing
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if the image was uploaded via file or captured from camera
        if 'image' in request.form:
            # Image from form upload (Base64)
            img_data = request.form['image']
            # Process the base64 data
            img_data = img_data.split(',')[1]  # Remove the "data:image/png;base64," part

            # Ensure proper padding for the base64 string
            padding = len(img_data) % 4
            if padding != 0:
                img_data += '=' * (4 - padding)  # Add the necessary padding

            img_bytes = io.BytesIO(base64.b64decode(img_data))
            img = Image.open(img_bytes)

        elif 'file' in request.files:
            # Image uploaded via file input
            img = Image.open(request.files['file'])
        
        else:
            return jsonify({'error': 'No image data provided'}), 400

    except (UnidentifiedImageError, OSError, binascii.Error) as e:
        return jsonify({'error': f'Invalid image data: {str(e)}'}), 400

    # Save the original image as base64
    original_img_base64 = pil_to_base64(img)

    # Apply the transformations to the image (convert to grayscale)
    grayscale_img = transforms.Grayscale(num_output_channels=1)(img)  # Convert to grayscale
    grayscale_img = transform(grayscale_img).unsqueeze(0)  # Add batch dimension

    # Convert the grayscale image to base64
    grayscale_img_pil = transforms.ToPILImage()(grayscale_img.squeeze(0))  # Convert tensor to PIL image
    grayscale_img_base64 = pil_to_base64(grayscale_img_pil)

    # Make a prediction
    with torch.no_grad():
        output = model(grayscale_img)
        _, predicted = torch.max(output.data, 1)
        predicted_label = predicted.item()

    # Return the prediction along with the base64 images
    return jsonify({
        'predicted_label': predicted_label,
        'original_img': original_img_base64,
        'grayscale_img': grayscale_img_base64
    })

if __name__ == '__main__':
    app.run(debug=True)
