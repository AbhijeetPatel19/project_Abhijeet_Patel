
# Training hyperparameters
batch_size = 32
epochs = 100
learning_rate = 0.001

# Image processing
resize_x = 150
resize_y = 150
input_channels = 3

# Model architecture
num_classes = 10  # for 10 sea animal species

# Paths (Kaggle specific)
#data_dir = '/kaggle/input/sea-animals-ds/DatasetsCNN'

# Update with your dataset path

checkpoint_path = '/kaggle/working/project_Abhijeet_Patel/checkpoints/final_weights.pth'
test_dir = './data'