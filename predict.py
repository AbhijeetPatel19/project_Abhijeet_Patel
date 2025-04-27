import torch
from torchvision import transforms
from PIL import Image
import os
from config import resize_x, resize_y
from config import data_dir, checkpoint_path
from model import SeaAnimalCNN

def predict_single_image(image_path, model, device):
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((resize_x, resize_y)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)
    
    # Predict
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
    
    return predicted.item()

def classify_sea_animals(list_of_image_paths):
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = SeaAnimalCNN()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Get class names (assuming you have them)
    class_names = sorted(os.listdir(os.path.join(data_dir, 'Training')))
    
    # Process each image
    predictions = []
    for img_path in list_of_image_paths:
        class_idx = predict_single_image(img_path, model, device)
        predictions.append(class_names[class_idx])
    
    return predictions
