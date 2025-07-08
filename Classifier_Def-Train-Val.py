#Alzeimer's MRI Super-resolution Classifier

import os
import zipfile
from PIL import Image
import torch
from torchvision import transforms, models
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import shutil
import torch.nn as nn

# ======================== Paths ========================
noisy_zip_path = r"C:\Users\Shrey\OneDrive\Desktop\Alzeimer's_MRI_Superres\Data_Noisy.zip"
sr_output_dir = r"C:\Users\Shrey\OneDrive\Desktop\Alzeimer's_MRI_Superres\Data_SR"
os.makedirs(sr_output_dir, exist_ok=True)

# ==================== Load Generator ====================
generator = Generator(scale=4)
generator.load_state_dict(torch.load(r"C:\Users\Shrey\OneDrive\Desktop\Alzeimer's_MRI_Superres\generator_superres_denoise_final.pt"))
generator.eval().cpu()

lr_transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
hr_transform = transforms.Resize((224, 224))

# =============== Process Noisy Images from ZIP ==================
with zipfile.ZipFile(noisy_zip_path, 'r') as zip_ref:
    image_files = [f for f in zip_ref.namelist() if f.lower().endswith(('.jpg', '.png'))]

    for file in tqdm(image_files, desc="Processing Noisy Images"):
        with zip_ref.open(file) as image_file:
            image = Image.open(image_file).convert("RGB")

        lr_tensor = lr_transform(image).unsqueeze(0)

        with torch.no_grad():
            sr_tensor = generator(lr_tensor).squeeze(0).clamp(0, 1)

        sr_image = transforms.ToPILImage()(sr_tensor)
        sr_image = hr_transform(sr_image)

        class_name = os.path.dirname(file)
        class_dir = os.path.join(sr_output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        save_path = os.path.join(class_dir, os.path.basename(file))
        sr_image.save(save_path)

print("Super-resolved images saved in:", sr_output_dir)

# Create ZIP of SR images
shutil.make_archive(r"C:\Users\Shrey\OneDrive\Desktop\Alzeimer's_MRI_Superres\Data_SR", 'zip', sr_output_dir)

# ===================== Classifier ======================
transform_for_classifier = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

dataset = ImageFolder(root=sr_output_dir, transform=transform_for_classifier)
print(f"📂 Loaded {len(dataset)} images from SR directory.")

subset, _ = random_split(dataset, [5000, len(dataset)-5000])
train_size = int(0.8 * len(subset))
val_size = len(subset) - train_size
train_dataset, val_dataset = random_split(subset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

# ==================== Model ==========================
class DementiaClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        base = models.resnet18(pretrained=True)
        base.fc = nn.Linear(base.fc.in_features, num_classes)
        self.model = base

    def forward(self, x):
        return self.model(x)

model = DementiaClassifier().to('cuda')
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

# ==================== Training =======================
for epoch in range(15):
    model.train()
    correct, total, total_loss = 0, 0, 0

    print(f"\nEpoch {epoch+1}/15")
    train_bar = tqdm(train_loader, desc="Training", leave=False)

    for images, labels in train_bar:
        images, labels = images.to('cuda'), labels.to('cuda')
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, preds = outputs.max(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        train_bar.set_postfix({
            'Loss': f"{total_loss / (total // labels.size(0) + 1):.4f}",
            'Acc': f"{100 * correct / total:.2f}%"
        })

    train_acc = 100 * correct / total

    # Validation
    model.eval()
    val_correct, val_total = 0, 0
    val_bar = tqdm(val_loader, desc="Validation", leave=False)

    with torch.no_grad():
        for val_images, val_labels in val_bar:
            val_images, val_labels = val_images.to('cuda'), val_labels.to('cuda')
            val_outputs = model(val_images)
            _, val_preds = val_outputs.max(1)
            val_correct += (val_preds == val_labels).sum().item()
            val_total += val_labels.size(0)

    val_acc = 100 * val_correct / val_total
    print(f"Epoch {epoch+1}: Train Acc = {train_acc:.2f}%, Val Acc = {val_acc:.2f}%")

# ==================== Save Model ======================
torch.save(model.state_dict(), r"C:\Users\Shrey\OneDrive\Desktop\Alzeimer's_MRI_Superres\dementia_classifier_best.pt")

print("Classifier model saved.")