import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm  # Import tqdm for progress bars
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from typing import List
from PIL import Image
from Classify.FineTune import CompactVisionNetWithTransfer
from Classify.TrainModel import Trainer
from collections import Counter
import torch
# Define transformations for the new 5-class dataset
transform = transforms.Compose([
    transforms.Resize((512, 512)), # Resize to 512x512 pixels This is the size of the images in the dataset that we are using to train the model
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load Your dataset
data_dir = 'DATASETS'
train_dataset = datasets.ImageFolder(root=data_dir + '/train', transform=transform)
val_dataset = datasets.ImageFolder(root=data_dir + '/val', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
# Count the occurrences of each class in the dataset
class_counts = Counter([label for _, label in train_dataset.samples])

print(class_counts)
# Create label_list based on the class counts
label_list = [class_counts[i] for i in range(len(train_dataset.classes))]
# Define class names for the new 5 classes
new_class_names = train_dataset.classes


# Initialize the model with the modified classifier
model = CompactVisionNetWithTransfer(num_classes=len(new_class_names))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Freeze the backbone layers
model.freeze_backbone()

# Define the loss function and optimizer (only for parameters that require gradients)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

# Example usage
def main():
    # Configuration
    config = {
        'project_name': 'image_classification_256_class',
        'model_name': 'EnhancedClassifier',
        'learning_rate': 0.0001, # Learning rate for the optimizer [0.0001 is a good starting point]
        'weight_decay': 1e-4, # L2 regularization strength [1e-4 is a good starting point]
        'batch_size': 38, # Batch size for training [32 is a good starting point]
        'early_stopping_patience': 6, # Patience for early stopping [6 is a good starting point]
        'save_dir': './Heavy_models', # Directory to save the model
        'num_epochs': 100, # Number of epochs to train [100 is a good starting point]
        'use_wandb': False  # Set to True if you want to use WandB logging
    }
    
    # Initialize model and trainer
    model = CompactVisionNetWithTransfer(num_classes=len(label_list), in_channels=3)  # Your model class  in_channels=3 means 3 channels for RGB images
    class_weights = [1 / count for count in label_list]
    class_weights = torch.FloatTensor(class_weights)
    device = torch.device("cuda")

    # Define loss function with class weights to handle class imbalance
    # class_weights gives higher importance to underrepresented classes
    # Move criterion to same device (GPU/CPU) as model
    criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)

    # Initialize trainer with model, loss function, config and data loaders
    # model: The CompactVisionNetWithTransfer model instance
    # criterion: CrossEntropyLoss with class weights for imbalanced data
    # config: Training configuration parameters like learning rate, batch size etc
    # train_loader: DataLoader for training data
    # val_loader: DataLoader for validation data
    trainer = Trainer(
        model=model,
        criterion=criterion,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader
    )

    # Train model
    trainer.train(epochs=config['num_epochs'])

if __name__ == "__main__":
    main()

