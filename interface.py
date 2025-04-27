# Import your model
from model import SeaAnimalCNN as TheModel

# Import training function
from train import train_model as the_trainer

# Import prediction function
from predict import classify_sea_animals as the_predictor

# Import dataset and dataloader
from dataset import SeaAnimalDataset as TheDataset
from dataset import create_dataloaders as the_dataloader

# Import config parameters
from config import batch_size as the_batch_size
from config import epochs as total_epochs
