import torch
from torchvision import transforms
from PIL import Image
import os
from Classify.Classify import CompactVisionNet
from Classify.Test import predict_single_image
model = CompactVisionNet(num_classes=15,in_channels=3)
# Load best model
checkpoint = torch.load('./Heavy_models/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
device = torch.device("cuda")
model = model.to(device)

# Define the transformations you applied during training
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # resize to match input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # assuming normalization values
])
class_names = [
    "Class1", "Class2", "Class3", "Class4", "Class5",
    "Class6", "Class7", "Class8", "Class9", "Class10",
    "Class11", "Class12", "Class13", "Class14", "Class15"
]

test_image_path = './DATASETS/Class1/1.jpg'
predicted_class, raw_output = predict_single_image(model, device, transform, test_image_path, class_names)
print(f"Predicted Class: {predicted_class}")
print(f"Raw Model Output: {raw_output}")
