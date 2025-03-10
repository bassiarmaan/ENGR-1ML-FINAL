from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
import torch

def transform_image(image_path, display=False):
    # Open the image file
    img = Image.open(image_path)

    # Convert the image to grayscale
    img_gray = img.convert('L')

    # Apply sharpening filter to enhance edges
    img_sharpened = img_gray.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

    # Increase contrast to make the digits clearer
    enhancer = ImageEnhance.Contrast(img_sharpened)
    img_contrast = enhancer.enhance(2.0)  # Adjust contrast factor (higher = more contrast)

    # Resize using high-quality resampling to reduce blurring
    img_resized = img_contrast.resize((28, 28), Image.Resampling.LANCZOS)

    # Convert to NumPy array and normalize to [0,1]
    img_array = np.array(img_resized) / 255.0

    # Invert colors (assuming dark background, light digits)
    img_array = 1 - img_array

    if display:
        # Display the processed image
        plt.imshow(img_array, cmap='gray')
        plt.axis('off')
        plt.show()

    return img_array

def predict_digit(model, image_array):
    # Convert NumPy array to PyTorch tensor, add batch & channel dimensions
    image_tensor = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 28, 28)

    # Ensure model is on the same device as tensor
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)

    # Set model to evaluation mode
    model.eval()

    # Perform inference
    with torch.no_grad():
        output = model(image_tensor)

    # Get the predicted class (digit)
    predicted_digit = torch.argmax(output, dim=1).item()

    return predicted_digit