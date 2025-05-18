import torch
from torchvision import transforms
from PIL import Image
import json
import os
from models.multilabel_model import MultiLabelResNet
from utils.label_encoder import LabelEncoder
import numpy as np
import argparse

SCHEMA_PATH = 'data/feature_schema.json'
MODEL_PATH = 'checkpoints/model_90acc.pth'
IMG_SIZE = 224

# Image transforms
transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

def load_model(encoder):
    # Load the model
    model = MultiLabelResNet(output_dim=encoder.total_dim)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

def predict_image(image_path, model, encoder):
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    tensor = transforms(image).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        probs = model(tensor)[0].numpy()

    output = {
        "image_id": os.path.basename(image_path),
        "features": {},
        "overall_confidence": float(np.mean(probs)),
        "improvement_suggestions": [
            "Increase data augmentation for rare 'Bohemian' examples",
            "Add hard-negative mining between 'Straight' and 'Angular' shapes",
        ]
    }

    for feature, meta in encoder.feature_classes.items():
        start = meta["offset"]
        end = start + meta["size"]
        sub_probs = probs[start:end]
        max_idx = int(np.argmax(sub_probs))
        value = meta["values"][max_idx]
        confidence = float(sub_probs[max_idx])
        output["features"][feature] = [{
            "value": value,
            "confidence": confidence,
        }]
    
    return output

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='Path to the image file')
    parser.add_argument('--output', default='output.json', help='Path to save JSOSN output')
    args = parser.parse_args()

    encoder = LabelEncoder(SCHEMA_PATH)
    model = load_model(encoder)
    result = predict_image(args.image, model, encoder)

    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Prediction saved to {args.output}")