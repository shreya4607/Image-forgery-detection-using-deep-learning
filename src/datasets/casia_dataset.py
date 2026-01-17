from torch.utils.data import Dataset
from PIL import Image
import os

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
