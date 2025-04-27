import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from config import resize_x, resize_y
from config import data_dir, batch_size
from config import checkpoint_path

class SeaAnimalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.images = self._load_images()
        
    def _load_images(self):
        images = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_name)
                    images.append((img_path, self.class_to_idx[class_name]))
        return images
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def create_dataloaders():
    # Define transformations
    train_transform = transforms.Compose([
        transforms.Resize((resize_x, resize_y)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(size=resize_x, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3089,0.3920,0.3956],std=[0.1821,0.1888,0.1836])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((resize_x, resize_y)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3089,0.3920,0.3956],std=[0.1821,0.1888,0.1836])
    ])
    
    # Create datasets
    train_dataset = SeaAnimalDataset(
        root_dir=os.path.join(data_dir, 'Training'),
        transform=train_transform
    )
    
    val_dataset = SeaAnimalDataset(
        root_dir=os.path.join(data_dir, 'Validation'),
        transform=val_transform
    )
    
    test_dataset = SeaAnimalDataset(
        root_dir=os.path.join(data_dir, 'Testing'),
        transform=val_transform
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader
