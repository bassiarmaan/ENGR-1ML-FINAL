import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

def train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=5, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Train the CNN model with data augmentation, regularization, and learning rate scheduling."""
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()  # Reset gradients
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()  # Backpropagation
            optimizer.step()  # Update weights

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        scheduler.step()  # Adjust learning rate

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100 * correct/total:.2f}%')

    print("Training complete!")

def create_optimizer(model, learning_rate=0.001, weight_decay=1e-4):
    """Returns an Adam optimizer with weight decay for regularization."""
    return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

def create_scheduler(optimizer, step_size=2, gamma=0.5):
    """Returns a learning rate scheduler that reduces LR by `gamma` every `step_size` epochs."""
    return StepLR(optimizer, step_size=step_size, gamma=gamma)
