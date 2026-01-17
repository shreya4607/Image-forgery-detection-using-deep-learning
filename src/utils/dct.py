import cv2
import numpy as np
from PIL import Image

def extract_dct_features(img, block=8):
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    gray = np.float32(gray) / 255.0

    h, w = gray.shape
    feats = []

    for i in range(0, h - block, block):
        for j in range(0, w - block, block):
            dct = cv2.dct(gray[i:i+block, j:j+block])
            feats.append(np.abs(dct.flatten()[1:16]))

    feats = np.array(feats)
    return feats.mean(axis=0)
