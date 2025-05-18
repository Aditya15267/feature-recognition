import torch
from PIL import Image
from torchvision import transforms
import os
import json
from utils.label_encoder import LabelEncoder
from torch.utils.data import Dataset

class Furnituredata(Dataset):
    """
        data for furniture feature recognition
    """

    def __init__(self, img_dir, label_path, schema_path, transform=None):
        self.img_dir = img_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.encoder = LabelEncoder(schema_path)
        with open(label_path, 'r') as f:
            self.labels = json.load(f)
        self.image_files = list(self.labels.keys())

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label_dict = self.labels[img_name]
        label_vec = torch.FloatTensor(self.encoder.encode(label_dict))
        return image, label_vec