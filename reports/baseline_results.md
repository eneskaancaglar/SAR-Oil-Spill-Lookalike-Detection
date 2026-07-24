# Full Baseline U-Net Results

## Dataset

- Dataset: Refined SOS
- Train samples: 6455
- Validation samples: 1615
- Oil-containing samples: 7745
- Completely oil-free samples: 325
- Input size: 256 x 256
- Sensors: Sentinel-1 and PALSAR

## Model

- Architecture: U-Net
- Input channels: 1
- Output channels: 1
- Base channels: 16
- Loss: 0.5 BCE + 0.5 Dice Loss
- Optimizer: Adam
- Learning rate: 1e-3
- Epochs: 10
- Batch size: 4

## Best Checkpoint

- Best epoch according to validation Dice at threshold 0.80: Epoch 5
- Checkpoint: checkpoints/full_baseline/best.pth

## Validation Results at Threshold 0.60

- Dice: 0.8334
- IoU: 0.7143
- Precision: 0.7959
- Recall: 0.8745
- Oil-free image alarm rate: 14.47%

## Validation Results at Threshold 0.80

- Dice: 0.8264
- IoU: 0.7041
- Precision: 0.8418
- Recall: 0.8115
- Oil-free image alarm rate: 10.53%
- False-positive oil pixels in oil-free images: 0.0087%

## Validation Results at Threshold 0.90

- Dice: 0.7955
- IoU: 0.6605
- Precision: 0.8801
- Recall: 0.7258
- Oil-free image alarm rate: 2.63%

## Current Limitation

The model has not yet been evaluated on an independent dataset
containing explicit SAR look-alike phenomena. Refined SOS validation
performance alone does not demonstrate real-world look-alike robustness.
