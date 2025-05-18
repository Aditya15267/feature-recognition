import os
import json
import random

with open('data/feature_schema.json', 'r') as f:
    SCHEMA = json.load(f)

IMAGE_DIR = 'data/images'
OUTPUT_FILE = 'data/annotations.json'

def generate_labels(schema):
    """
    Generate fake labels for the data based on the schema.
    
    Args:
        schema (dict): The schema containing feature categories and their classes.
    
    Returns:
        dict: A dictionary containing image paths and their corresponding labels.
    """
    labels = {}
    for feature, values in schema.items():
        # Randomly select a class for each feature
        selected_class = random.choice(values)
        labels[feature] = selected_class

    return labels

def main():
    annotations = {}
    image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith('.jpg')]
    print(f"Found {len(image_files)} images in {IMAGE_DIR}")
    for img in image_files:
        annotations[img] = generate_labels(SCHEMA)

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(annotations, f, indent=2)
    print(f"Saved labels for {len(image_files)} images to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
