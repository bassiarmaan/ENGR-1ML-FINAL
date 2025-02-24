import torch.nn as nn
import matplotlib.pyplot as plt

def training_loop(images, labels, model, criterion, train_loader, optimizer, epochs):
    for epoch in range(epochs):
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} completed.")
    print("Pytorch model training completed.")

def show_images(dataset, num_images=5):
    fig, axes = plt.subplots(1, num_images, figsize=(10,3))
    for i in range(num_images):
        image, label = dataset[i]
        axes[i].imshow(image.squeeze(), cmap='grey')
        axes[i].set_title(f"Label:{label}")
        axes[i].axis("off")
    plt.show()