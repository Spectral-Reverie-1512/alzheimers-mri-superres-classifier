# Alzheimer's MRI Super-Resolution GAN & Classifier
This project implements a Generative Adversarial Network (GAN) for super-resolution and denoising of Alzheimer's MRI images, followed by a ResNet18-based classifier to detect dementia stages. It aims to improve diagnostic accuracy on low-quality medical scans by enhancing image clarity and leveraging AI-based classification.

## Project Structure

├── Preprocessing.py                # Generates noisy MRI dataset from clean images

├── GAN_Def-Train-Eval-Vis.py       # Defines and trains Super-Resolution GAN

├── Classifier_Def-Train-Val.py     # Defines and trains the dementia stage classifier

├── Inference-UI.py                 # Gradio UI for testing SR & classification pipeline

├── Data_Balanced/                   # Original balanced dataset (zipped)

├── Data_Noisy/                      # Generated noisy dataset (zipped)

├── Data_SR/                         # Super-resolved images used for classifier training

├── dementia_classifier_best.pt      # Trained Classifier Model

├── generator_superres_denoise_final.pt # Trained Generator Model

└── Test Image/                      # Sample MRI image for testing

## Features
1. GAN with Residual and Squeeze-Excitation Blocks for MRI super-resolution & denoising.
2. Custom PyTorch Dataset class to load images directly from .zip files.
3. ResNet18 classifier trained to classify MRI images into:
    i.    Non Demented
    ii.   Very Mild Dementia
    iii.  Mild Dementia
    iv.   Moderate Dementia
4. Gradio-powered web interface for easy demo.

## Methodology
1. Preprocessing
  Adds Gaussian noise to balanced MRI images.
  Saves noisy dataset in ZIP format without extraction.

2. Super-Resolution GAN
  Generator: Residual + Squeeze-Excitation blocks, PixelShuffle upsampling.
  Discriminator: PatchGAN-like CNN.
  Loss: Combination of L1, MSE, and adversarial losses.

3. Classifier
  Fine-tunes ResNet18 on the generated super-resolved MRI images.
  Achieves classification into 4 dementia progression stages.

4. Inference
  Input: Low-quality MRI image
  Output: Super-resolved MRI + Predicted Dementia Stage
  Built with Gradio for easy demo.


## Setup
1. Prerequisites
  Python 3.8+
  PyTorch
  torchvision
  albumentations
  gradio
  tqdm
  pillow
  matplotlib

## Results
1. GAN:

   Evaluation Results:

     PSNR : 37.56 dB

     SSIM : 0.9541
3. Classifier:

     Train accuracy: ~97%

     Validation accuracy: ~85%

## Dataset
Balanced MRI Dataset: OASIS MRI

https://www.kaggle.com/datasets/ninadaithal/imagesoasis

## 📜 License

⚠️ This repository and its contents are the intellectual property of Shreya Khadse. Unauthorized use, reproduction, or distribution is strictly prohibited.
