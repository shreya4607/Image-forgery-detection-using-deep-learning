## CNN Baseline (RGB only)

- Model: DenseNet121 (ImageNet pretrained)
- Input: 224x224 RGB
- Loss: BCEWithLogits
- Best Accuracy: ~0.83
- Best AUC: ~0.88

### Observations
- Validation loss plateaus after ~10 epochs
- More epochs did not improve accuracy
