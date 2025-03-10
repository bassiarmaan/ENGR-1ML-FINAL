from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch

def transform_image(image_path, display=False):
    # Open the image file
    img = Image.open(image_path)
    
    # Convert the image to grayscale
    img_gray = img.convert('L')
    
    # Resize the image to 28x28 pixels
    img_resized = img_gray.resize((28, 28))
    
    # Convert the image to a numpy array and normalize pixel values to [0, 1]
    img_array = np.array(img_resized) / 255.0
    
    # Invert colors
    img_array = 1 - img_array

    if display:
        # Display the inverted image using matplotlib
        plt.imshow(img_array, cmap='gray')
        plt.axis('off')  # Turn off axis labels
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
