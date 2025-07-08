#Creating a noisy dataset from a balanced dataset of Alzheimer's MRI images

#This script reads images from a balanced dataset zip file, resizes and applies Gaussian noise to them, and saves the noisy images into a new zip file.
#The images are organized by class, and the script also prints the number of images per class in both the original and noisy datasets.
# This script also visualizes a few noisy images from each class.

import os
import io
import zipfile
import numpy as np
from tqdm import tqdm
import albumentations as A
from PIL import Image
import matplotlib.pyplot as plt
import random

balanced_zip_path = r"C:\Users\Shrey\OneDrive\Desktop\Alzeimer's_MRI_Superres\Data_Balanced.zip"
noisy_zip_path = r"C:\Users\Shrey\OneDrive\Desktop\Alzeimer's_MRI_Superres\Data_Noisy.zip"

noise_transform = A.Compose([
    A.Resize(128, 128),
    A.GaussNoise(p=0.8)
])

with zipfile.ZipFile(balanced_zip_path, 'r') as balanced_zip, \
     zipfile.ZipFile(noisy_zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as noisy_zip:

    all_files = [f for f in balanced_zip.namelist() if f.lower().endswith('.jpg')]
    class_names = sorted({f.split('/')[0] for f in all_files})

    print("Balanced Dir:")
    for cls in class_names:
        count = sum(1 for f in all_files if f.startswith(cls + '/'))
        print(f"Class '{cls}': {count} images")

    for cls in class_names:
        class_files = [f for f in all_files if f.startswith(cls + '/')]

        for img_path in tqdm(class_files, desc=f"Processing {cls}"):
            with balanced_zip.open(img_path) as file:
                image = Image.open(file).convert('RGB')
                image_np = np.array(image)

            augmented = noise_transform(image=image_np)
            noisy_image = np.clip(augmented['image'], 0, 255).astype(np.uint8)

            buffer = io.BytesIO()
            Image.fromarray(noisy_image).save(buffer, format='JPEG')

            noisy_zip_path_in_zip = os.path.join(cls, os.path.basename(img_path))
            noisy_zip.writestr(noisy_zip_path_in_zip, buffer.getvalue())

print("\n✅ Noisy ZIP created at:", noisy_zip_path)

with zipfile.ZipFile(noisy_zip_path, 'r') as noisy_zip:
    noisy_files = noisy_zip.namelist()
    class_names_noisy = sorted({f.split('/')[0] for f in noisy_files})

    print("\nNoisy Dir:")
    for cls in class_names_noisy:
        count = sum(1 for f in noisy_files if f.startswith(cls + '/'))
        print(f"Class '{cls}': {count} images")

num_images = 5
with zipfile.ZipFile(noisy_zip_path, 'r') as noisy_zip:
    for cls in class_names_noisy:
        class_files = [f for f in noisy_zip.namelist() if f.startswith(cls + '/')]
        sample_files = random.sample(class_files, min(num_images, len(class_files)))

        plt.figure(figsize=(15, 3))
        plt.suptitle(f"Class: {cls}", fontsize=16)

        for i, file_name in enumerate(sample_files):
            with noisy_zip.open(file_name) as file:
                img = Image.open(file).convert('RGB')

            plt.subplot(1, num_images, i + 1)
            plt.imshow(img)
            plt.axis('off')

        plt.show()