import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
from utils.dct import extract_dct_features

class CASIADatasetDCT(Dataset):
    def __init__(self, image_paths, labels, rgb_tf):
        self.image_paths = image_paths
        self.labels = labels
        self.rgb_tf = rgb_tf

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        rgb = self.rgb_tf(img)
        dct = extract_dct_features(img)

        return (
            rgb,
            torch.tensor(dct, dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32)
        )

    def __len__(self):
        return len(self.image_paths)


def load_casia_data(au_dir, tp_dir):
    image_paths, labels = [], []

    for img in os.listdir(au_dir):
        image_paths.append(os.path.join(au_dir, img))
        labels.append(0)

    for img in os.listdir(tp_dir):
        image_paths.append(os.path.join(tp_dir, img))
        labels.append(1)

    combined = list(zip(image_paths, labels))
    random.shuffle(combined)
    image_paths, labels = zip(*combined)

    total = len(image_paths)
    train_end = int(0.8 * total)
    val_end = int(0.9 * total)

    return (
        image_paths[:train_end], labels[:train_end],
        image_paths[train_end:val_end], labels[train_end:val_end],
        image_paths[val_end:], labels[val_end:]
    )
