# Image Forgery Detection on CASIA v2

## Problem
Deep learning–based image forgery detection using the CASIA v2 dataset, with CNN and frequency-domain feature analysis.


## Why this is hard
- Heavy class overlap
- Compression artifacts dominate signal
- Post-processing noise hides manipulation cues

## Experiments
| Method | Accuracy | AUC |
|------|---------|-----|
| CNN (RGB) | ~0.83 | ~0.88 |
| CNN + ELA | ~0.83 | ~0.89 |
| CNN + DCT | ~0.78 | ~0.85 |

## Key Takeaways
- Accuracy saturates quickly
- Forensic cues alone are weak
- Dataset limitations dominate performance

## What this project demonstrates
- Controlled experimentation
- Failure analysis
- ML debugging mindset

