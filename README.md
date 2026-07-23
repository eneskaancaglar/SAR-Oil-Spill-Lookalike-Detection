# SAR Oil Spill Segmentation

Sentinel-1 SAR görüntülerinden deniz yüzeyindeki petrol sýzýntýlarýnýn
tespiti ve piksel seviyesinde segmentasyonu üzerine geliþtirilen
baðýmsýz görüntü iþleme ve derin öðrenme projesidir.

## Project Goal

Input:
- SAR image

Output:
- Oil spill segmentation mask

## Project Stages

- [x] Project environment created
- [x] Initial dependencies installed
- [ ] Dataset selected and downloaded
- [ ] Image-mask pairs inspected
- [ ] Dataset pipeline implemented
- [ ] Baseline U-Net trained
- [ ] Segmentation metrics calculated
- [ ] Lookalike errors analyzed

## Project Structure

- data/raw/images: Original SAR images
- data/raw/masks: Ground-truth segmentation masks
- data/processed: Prepared patches and processed data
- src: Reusable source code
- scripts: Executable project scripts
- checkpoints: Trained model files
- outputs/figures: Generated figures
- reports: Technical reports
- configs: Experiment configurations
