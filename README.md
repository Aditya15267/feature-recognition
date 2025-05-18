# Multi-Label Furniture Feature Recognition

This project implements a deep learning pipeline that identifies multiple visual attributes of **furniture** (e.g. sofas, chairs, tables) from images. The model outputs predictions in JSON format with confidence scores and improvement suggestions.

---

## Features Detected

The model recognizes multiple categories per furniture item, including:

- **Product Type** (Sofa, Armchair, etc.)
- **Style** (Modern, Bohemian, etc.)
- **Shape** (L Shape, Circular, etc.)
- **Usage**, **Base**, **Arms**, **Legs**
- And more: Upholstery, Seat, Piping, Back, Structure

---

## Model Architecture

- **Backbone**: ResNet-50 pretrained on ImageNet
- **Multi-task Head**: One classification layer per feature category
- **Loss**: Combined multi-label cross-entropy loss
- **Framework**: PyTorch

---

## Dataset

This repo includes a **data generator** that:
- Lists of search items + assigned feature labels
- Generates `.json` files and stores images locally

To generate data:

```bash
python script/download_images.py
```

## Setup & Training

1. Clone this repository:
    ```sh
    git clone https://github.com/Aditya15267/feature-recognition.git
    cd furniture-query-parse
    ```

2. Install dependencies
    ```sh
    pip install -r requirements.txt
    ```

3. Train model
    ```sh
    python train.py
    ```

Model checkpoints are saved to `checkpoints/`

## Inference

Run the inference on a folder of images:
```sh
python inference.py --image data/images/image
```

Sample output:
```sh
{
  "image_id": "sofa_001.jpg",
  "features": {
    "Product Type": [{"value": "Sofa", "confidence": 0.92}],
    "Style": [{"value": "Modern", "confidence": 0.89}],
    ...
  },
  "overall_confidence": 0.91,
  "improvement_suggestions": [
    "Increase data augmentation for rare 'Bohemian' examples",
    "Add hard-negative mining between 'Straight' and 'Angular' shapes"
  ]
}
```

## Design Notes

This project is designed for modularity, scalability, and extensibility. Key architectural decisions include:

1. Multi-Label Classification Setup

- Each image is mapped to multiple independent feature categories.
- A shared CNN backbone (ResNet-50) is used for feature extraction.
- Each feature category has its own classification head.
- Allows independent learning signals per label group.

2. Label Handling

- Feature categories and label options are loaded from a JSON schema (feature_schema.json).
- Dynamic label encoders are built at runtime to support any number of categories.

3. Data Generator

- Since no real dataset was provided, a data generator builds fake image-label pairs.
- Images are pulled from online sources.
- Features are randomly assigned from the schema to simulate label diversity.

4. Prediction Output Format

- Inference returns structured JSON output:
    - Per-feature prediction
    - Confidence scores
    - Overall average confidence