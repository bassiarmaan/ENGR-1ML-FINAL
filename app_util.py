from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def transform_image(image_path, display=False):
    # Open the image file
    img = Image.open(image_path)
    
    # Convert the image to grayscale
    img_gray = img.convert('L')
    
    # Resize the image to 32x32 pixels
    img_resized = img_gray.resize((32, 32))
    
    # Convert the image to a numpy array and normalize pixel values to [0, 1]
    img_array = np.array(img_resized) / 255.0

    if display:
        # Display the image using matplotlib
        plt.imshow(img_array, cmap='gray')
        plt.axis('off')  # Turn off axis labels
        plt.show()
        
    return img_array

