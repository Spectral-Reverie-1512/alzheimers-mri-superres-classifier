import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import gradio as gr
import os
import zipfile
import random
from io import BytesIO
from torchvision.utils import save_image

# ======================= Generator Model =======================
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.block = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1), nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.block(x)

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
        )
        self.se = SEBlock(channels)

    def forward(self, x):
        return x + self.se(self.block(x))

class Generator(nn.Module):
    def __init__(self, num_blocks=16, channels=64, scale=4):
        super().__init__()
        self.entry = nn.Conv2d(3, channels, 9, 1, 4)
        self.res_blocks = nn.Sequential(*[ResBlock(channels) for _ in range(num_blocks)])
        self.upsample = nn.Sequential(
            nn.Conv2d(channels, channels * scale ** 2, 3, 1, 1),
            nn.PixelShuffle(scale),
            nn.Conv2d(channels, 3, 9, 1, 4)
        )

    def forward(self, x):
        features = self.entry(x)
        res = self.res_blocks(features)
        return self.upsample(features + res)

# ======================= Classifier Model =======================
class DementiaClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        base = models.resnet18(pretrained=False)
        base.fc = nn.Linear(base.fc.in_features, num_classes)
        self.model = base

    def forward(self, x):
        return self.model(x)

# ==================== Load Saved Models ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = Generator(scale=4).to(DEVICE)
generator.load_state_dict(torch.load(
    r"C:\Users\Shrey\OneDrive\Desktop\Alzeimer's_MRI_Superres\generator_superres_denoise_final.pt",
    map_location=DEVICE
))
generator.eval()

classifier = DementiaClassifier(num_classes=4).to(DEVICE)
classifier.load_state_dict(torch.load(
    r"C:\Users\Shrey\OneDrive\Desktop\Alzeimer's_MRI_Superres\dementia_classifier_best.pt",
    map_location=DEVICE
))
classifier.eval()

print("✅ Models loaded successfully.")

# ==================== Prediction Function ====================
class_labels = ['Mild Dementia', 'Moderate Dementia', 'Non Demented', 'Very Mild Dementia']

def predict(image):
    # Preprocess for Generator
    lr_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    # Preprocess for Classifier
    classifier_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Super Resolution
    lr_tensor = lr_transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        sr_tensor = generator(lr_tensor).squeeze(0).clamp(0, 1)
    sr_image = transforms.ToPILImage()(sr_tensor.cpu())

    # Classification
    input_tensor = classifier_transform(sr_image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = classifier(input_tensor)
        _, predicted = outputs.max(1)
    result = class_labels[predicted.item()]

    return sr_image, result

# ==================== Gradio UI ====================
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload MRI Image"),
    outputs=[
        gr.Image(type="pil", label="Super-Resolved Image"),
        gr.Text(label="Predicted Dementia Stage")
    ],
    title="Alzheimer's MRI Super-Resolution & Classification",
)

if __name__ == "__main__":
    demo.launch(share=True)  # Set share=True if you want public link