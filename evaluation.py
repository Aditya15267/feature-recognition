import json
import os
from sklearn.metrics import precision_score, recall_score, average_precision_score
from tqdm import tqdm

from inference import predict_image
from utils.label_encoder import LabelEncoder
from models.multilabel_model import MultiLabelResNet

import torch

GT_PATH = "data/annotations.json"
IMAGE_DIR = "data/images"
SCHEMA_PATH = "data/feature_schema.json"
MODEL_PATH = "checkpoints/model_90acc.pth"

def load_ground_truth(path):
    with open(path, "r") as f:
        return json.load(f)

def evaluate():
    encoder = LabelEncoder(SCHEMA_PATH)
    model = MultiLabelResNet(output_dim=encoder.total_dim)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    gt = load_ground_truth(GT_PATH)

    all_preds = {feature: [] for feature in encoder.feature_classes}
    all_true = {feature: [] for feature in encoder.feature_classes}

    for img_id, sample in tqdm(gt.items()):
        image_path = os.path.join(IMAGE_DIR, img_id)
        y_true = sample
        result = predict_image(image_path, model, encoder)

        for feature in encoder.feature_classes:
            pred_value = result["features"][feature][0]["value"]
            true_value = y_true[feature]

            pred_vector = [1 if v == pred_value else 0 for v in encoder.feature_classes[feature]["values"]]
            true_vector = [1 if v == true_value else 0 for v in encoder.feature_classes[feature]["values"]]

            all_preds[feature].append(pred_vector)
            all_true[feature].append(true_vector)

    print("\n🔍 Evaluation Results:")
    all_mAP = []
    all_accuracy = []

    for feature in encoder.feature_classes:
        y_true = all_true[feature]
        y_pred = all_preds[feature]

        y_true_flat = [label.index(1) for label in y_true]
        y_pred_flat = [label.index(1) for label in y_pred]

        acc = sum([t == p for t, p in zip(y_true_flat, y_pred_flat)]) / len(y_true)
        all_accuracy.append(acc)

        ap = average_precision_score(y_true, y_pred, average="macro")
        all_mAP.append(ap)

        print(f"🪪 {feature}: Accuracy={acc:.2%}, mAP={ap:.2%}")

    print(f"\n✅ Overall Accuracy: {sum(all_accuracy)/len(all_accuracy):.2%}")
    print(f"✅ Overall mAP: {sum(all_mAP)/len(all_mAP):.2%}")

    if all(x >= 0.90 for x in all_accuracy):
        print("🎯 Passed: ≥90% accuracy per feature")
    else:
        print("❌ Failed: Some features below 90% accuracy")

if __name__ == "__main__":
    evaluate()