import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
from config import epochs, learning_rate, checkpoint_path, data_dir
from model import SeaAnimalCNN
from dataset import create_dataloaders
from tqdm import tqdm

def train_model():
    # Initialize model, loss, optimizer
    model = SeaAnimalCNN()
    criterion = nn.CrossEntropyLoss(label_smoothing = 0.1)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 1.0e-05)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = 10,T_mult = 1)
    #leastLR = 0.00001
    # Get dataloaders
    train_loader, val_loader, _ = create_dataloaders()
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    best_val_accuracy = 0.0
    patience = 10
    trigger_times = 0
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Training phase
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        
        # Validation phase
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)
        
        scheduler.step()
        
        print(f'Epoch [{epoch+1}/{epochs}], '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.3f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.3f}%')
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            trigger_times = 0
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), checkpoint_path)
            print(f'New best model saved with val accuracy: {val_accuracy:.3f}%')
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f'early stopping at epoch {epoch +1}')
                break
    
    print('Training complete')

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    loss = running_loss / len(dataloader)
    accuracy = 100 * correct / total
    
    return loss, accuracy
