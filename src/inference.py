import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as torch_nn
import argparse
import os
import json

def load_model(model_path, num_classes=102, device='cuda'):
    """
    Loads the trained model from a checkpoint.
    """
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = torch_nn.Linear(num_ftrs, num_classes)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        print(f"Model loaded from {model_path}")
    else:
        raise FileNotFoundError(f"No model found at {model_path}")

    model = model.to(device)
    model.eval()
    return model

def process_image(image_path):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns an Numpy array
    """
    # Define transformations
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Load image
    if image_path.startswith('http'):
        import requests
        from io import BytesIO
        response = requests.get(image_path)
        img = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        img = Image.open(image_path).convert('RGB')
        
    img_tensor = preprocess(img)
    return img_tensor.unsqueeze(0) # Add batch dimension

def predict(image_path, model, top_k=3, device='cuda'):
    """
    Predict the class (or classes) of an image using a trained deep learning model.
    """
    img_tensor = process_image(image_path).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        
        top_probs, top_indices = probs.topk(top_k)
        
    return top_probs.cpu().numpy()[0], top_indices.cpu().numpy()[0]

def main():
    parser = argparse.ArgumentParser(description='Inference for Flowers102 Classifier')
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--top_k', type=int, default=3, help='Number of top classes to return')
    args = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        model = load_model(args.model_path, device=device)
        probs, classes = predict(args.image_path, model, args.top_k, device)
        
        print("\nPredictions:")
        for i in range(len(probs)):
            print(f"{i+1}: Class {classes[i]} (Probability: {probs[i]:.4f})")
            
    except Exception as e:
        print(f"Error during inference: {e}")

if __name__ == '__main__':
    main()
