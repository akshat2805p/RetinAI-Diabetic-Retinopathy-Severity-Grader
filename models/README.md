# Model Checkpoints

This directory stores trained model weights.

## Expected files:
- `best_fold0.pth` — Best checkpoint from Fold 0 of 5-Fold Stratified CV

## How to generate:
```bash
python src/train.py --fold 0 --epochs 30 --img_size 512 --batch_size 16
```

## Note:
Model files are excluded from git (too large). Download or train locally.
