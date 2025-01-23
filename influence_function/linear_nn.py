import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import json
import random
from torchinfo import summary
import os
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Set the random seed for reproducibility
seed = 42
random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def load_data(store_mnist='../data'):
    # Define the transformation to flatten the images
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)), transforms.Lambda(lambda x: x.view(-1))])

    # Download MNIST dataset and create DataLoader
    train_dataset = datasets.MNIST(root=store_mnist, train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root=store_mnist, train=False, transform=transform, download=True)

    # Split the training dataset into training and validation sets
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=128, shuffle=False)
    return train_loader, val_loader, test_loader

# Define a neural network model with an output layer of 10 nodes
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)  # Flattened input size is 28*28
        self.fc2 = nn.Linear(256, 10)  # Output layer with 10 nodes

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate the model, loss function, and optimizer
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
#mse_criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop with checkpoints
def train(model, train_loader, val_loader, criterion, optimizer, epochs=10):
    checkpoint_dir = '../models/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            
            # Compute cross-entropy loss
            try:
                loss = criterion(outputs, labels)
            except:
                # Compute mean squared error
                loss = criterion(outputs, torch.nn.functional.one_hot(labels, num_classes=10).float())
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Compute accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total

        # Print validation loss during training
        val_loss, val_acc = validate(model, val_loader, criterion)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy * 100:.2f}%, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc * 100:.2f}%")
        
        if epoch % 5 == 0:
            # Save checkpoint
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{epoch + 1}_linear_trained_model.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

    # Save the trained model
    torch.save(model.state_dict(), '../models/linear_trained_model.pth')
    print("Trained model saved.")

# Validation loop
def validate(model, val_loader, criterion):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            
            # Compute cross-entropy loss
            try:
                loss = criterion(outputs, labels)
            except:
                # Compute mean squared error
                loss = criterion(outputs, torch.nn.functional.one_hot(labels, num_classes=10).float())
            
            total_loss += loss.item()
            
            # Compute accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    
    return avg_loss, accuracy

# Testing loop
def test(model, test_loader, criterion, get_preds_only = False, train_loader = None):

    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not get_preds_only:
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)

                # Compute cross-entropy loss
                try:
                    loss = criterion(outputs, labels)
                except:
                    # Compute mean squared error
                    loss = criterion(outputs, torch.nn.functional.one_hot(labels, num_classes=10).float())

                # Combine the two losses
                #loss = loss_ce #+ loss_mse

                total_loss += loss.item()

                # Compute accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(test_loader)
        accuracy = correct / total

        print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy * 100:.2f}%")
        return avg_loss, accuracy
    else:
        all_preds_array = []
        output_grads = []
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            all_preds_array.append(outputs)
            outputs.retain_grad()

            # Compute cross-entropy loss
            try:
                loss = criterion(outputs, labels)
            except:
                # Compute mean squared error
                loss = criterion(outputs, torch.nn.functional.one_hot(labels, num_classes=10).float())

            loss.backward()

            output_grads.append(outputs.grad)

        return all_preds_array, output_grads

def get_model():
    # Instantiate the model, loss function, and optimizer
    model = SimpleNN()
    criterion = nn.CrossEntropyLoss()
    #mse_criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    return model, criterion, optimizer


# Function to load the model from a saved state
def load_model(model, filepath='../models/linear_trained_model.pth'):
    model.load_state_dict(torch.load(filepath, map_location=torch.device(device)))
    model.eval()
    return model.to(device)


if __name__ == '__main__':
    train_loader, val_loader, test_loader = load_data()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    summary(model, (128, 784))
    # Train the model
    train(model, train_loader, val_loader, criterion, optimizer, epochs=20)

    # Test the model
    test_loss, test_accuracy = test(model, test_loader, criterion)
    print(f"Final Test Loss: {test_loss:.4f}, Final Test Accuracy: {test_accuracy * 100:.2f}%")
