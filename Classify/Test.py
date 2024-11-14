import torch
from PIL import Image
from typing import List, Tuple
from torchvision import transforms


def predict_single_image(model, device, transform, image_path: str, class_names: List[str]) -> Tuple[str, torch.Tensor]:
    """
    Function to test model on a single image.
    
    Args:
        model: Trained PyTorch model.
        image_path (str): Path to the input image.
        class_names (List[str]): List of class names to decode predictions.
    
    Returns:
        Tuple containing the predicted class name and the raw prediction tensor.
    """
    # Load the image
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  # add batch dimension

    # Move image to the same device as the model
    image = image.to(device)

    # Set model to evaluation mode
    model.eval()
    with torch.no_grad():
        # Run the model and get predictions
        output = model(image)
        _, predicted = torch.max(output, 1)
        
        # Decode the prediction
        predicted_class = class_names[predicted.item()]
    
    return predicted_class, output

