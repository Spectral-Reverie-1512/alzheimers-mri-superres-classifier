# ======================== Imports ========================
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import zipfile
import matplotlib.pyplot as plt
import numpy as np
import random
import albumentations as A
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# ===================== Dataset Class =====================
class SuperResZipDataset(Dataset):
    def __init__(self, zip_path, transform_lr=None, transform_hr=None):
        self.zip = zipfile.ZipFile(zip_path, 'r')
        self.img_list = [f for f in self.zip.namelist() if f.lower().endswith('.jpg')]
        self.transform_lr = transform_lr
        self.transform_hr = transform_hr

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        with self.zip.open(self.img_list[idx]) as file:
            hr = Image.open(BytesIO(file.read())).convert('RGB')
        lr = hr.resize((128, 128), resample=Image.BICUBIC)
        if self.transform_hr: hr = self.transform_hr(hr)
        if self.transform_lr: lr = self.transform_lr(lr)
        return lr, hr

    def __del__(self):
        self.zip.close()

# ======================= Models ==========================
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

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        def block(in_c, out_c): return [
            nn.Conv2d(in_c, out_c, 3, 2, 1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        layers, in_c = [], 3
        for out_c in [64, 128, 256, 512]:
            layers += block(in_c, out_c)
            in_c = out_c
        self.model = nn.Sequential(*layers)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(self.model(x))

# ===================== Loss & Training ====================
def fusion_loss(fake, real, fake_logits, w1=1.0, wRG=0.001, wMSE=0.5):
    return (w1 * F.l1_loss(fake, real) +
            wRG * torch.mean(F.softplus(1 - fake_logits)) +
            wMSE * F.mse_loss(fake, real))

def train_loop(generator, discriminator, loader, opt_G, opt_D, device, epochs, warmup=3):
    generator.train(); discriminator.train()
    for epoch in range(epochs):
        print(f"\n[Epoch {epoch + 1}/{epochs}]")
        pbar = tqdm(loader)
        for lr, hr in pbar:
            lr, hr = lr.to(device), hr.to(device)

            if epoch >= warmup:
                with torch.no_grad():
                    fake = generator(lr)
                real_logits = discriminator(hr)
                fake_logits = discriminator(fake.detach())
                d_loss = -torch.mean(torch.log(real_logits + 1e-8) + torch.log(1 - fake_logits + 1e-8))
                opt_D.zero_grad(); d_loss.backward(); opt_D.step()
            else:
                d_loss = torch.tensor(0.0)

            fake = generator(lr)
            if epoch < warmup:
                g_loss = F.mse_loss(fake, hr) + F.l1_loss(fake, hr)
            else:
                fake_logits = discriminator(fake)
                g_loss = fusion_loss(fake, hr, fake_logits)

            opt_G.zero_grad(); g_loss.backward(); opt_G.step()
            pbar.set_description(f"G Loss: {g_loss.item():.2f} | D Loss: {d_loss.item():.2f}")

# ==================== Evaluation ========================
def evaluate_generator(generator, dataset, device='cuda', batch_size=1, max_batches=None):
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    generator.eval(); psnr_metric.reset(); ssim_metric.reset()

    with torch.no_grad():
        for i, (lr, hr) in enumerate(loader):
            lr, hr = lr.to(device), hr.to(device)
            sr = generator(lr).clamp(0, 1)
            psnr_metric.update(sr, hr)
            ssim_metric.update(sr, hr)
            if max_batches and i >= max_batches:
                break

    print(f"\n Evaluation Results: PSNR: {psnr_metric.compute():.2f} dB, SSIM: {ssim_metric.compute():.4f}")

# ================= Visualization ========================
def show_sample(generator, dataset, idx=0, device='cuda'):
    generator.eval()
    lr_img, hr_img = dataset[idx]
    with torch.no_grad():
        sr_img = generator(lr_img.unsqueeze(0).to(device)).squeeze(0).cpu().clamp(0, 1)

    titles = ["Low-Resolution (Input)", "Super-Resolved (Output)", "High-Resolution (Ground Truth)"]
    images = [lr_img, sr_img, hr_img]
    plt.figure(figsize=(15, 5))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(1, 3, i + 1)
        plt.title(title); plt.imshow(TF.to_pil_image(img)); plt.axis("off")
    plt.show()

# ================ Unseen Noisy Data Test =================
def test_on_unseen(generator, image_paths, device='cuda'):
    noise_transform = A.Compose([A.Resize(128, 128), A.GaussNoise(p=0.2)])
    preprocess = T.Compose([T.ToTensor()])
    generator.eval()

    for path in image_paths:
        image = cv2.imread(path)
        if image is None: continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        noisy = noise_transform(image=image)['image']
        hr = cv2.resize(image, (512, 512))
        noisy_tensor = preprocess(Image.fromarray(noisy)).unsqueeze(0).to(device)
        hr_tensor = preprocess(Image.fromarray(hr))

        with torch.no_grad():
            sr_tensor = generator(noisy_tensor).squeeze(0).cpu().clamp(0, 1)

        psnr_score = psnr(hr_tensor.permute(1, 2, 0).numpy(), sr_tensor.permute(1, 2, 0).numpy(), data_range=1.0)
        ssim_score = ssim(hr_tensor.permute(1, 2, 0).numpy(), sr_tensor.permute(1, 2, 0).numpy(), channel_axis=2, data_range=1.0)

        plt.figure(figsize=(12, 4))
        plt.suptitle(f"PSNR: {psnr_score:.2f} dB | SSIM: {ssim_score:.4f}", fontsize=13)

        for i, img in enumerate([noisy_tensor.squeeze(0).cpu(), sr_tensor, hr_tensor]):
            plt.subplot(1, 3, i + 1)
            plt.imshow(TF.to_pil_image(img)); plt.axis("off")
        plt.show()

# =================== Hyperparameters ====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS, SCALE, LR, WARMUP = 3, 4, 1e-4, 3

transform_hr = T.Compose([T.Resize((512, 512)), T.ToTensor()])
transform_lr = T.Compose([T.Resize((128, 128)), T.ToTensor()])

dataset = SuperResZipDataset(
    zip_path=r"C:\Users\Shrey\OneDrive\Desktop\Alzeimer's_MRI_Superres\Data_Noisy.zip",
    transform_lr=transform_lr, transform_hr=transform_hr
)

loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
gen = Generator(scale=SCALE).to(DEVICE)
disc = Discriminator().to(DEVICE)
opt_G = torch.optim.Adam(gen.parameters(), lr=LR)
opt_D = torch.optim.Adam(disc.parameters(), lr=LR)

# ======================= Train ==========================
train_loop(gen, disc, loader, opt_G, opt_D, device=DEVICE, epochs=EPOCHS, warmup=WARMUP)

# =================== Evaluation & Test =================
show_sample(gen, dataset, idx=3, device=DEVICE)
evaluate_generator(gen, dataset, device=DEVICE, max_batches=100)

# Test on unseen noisy images
original_dir = r"C:\Users\Shrey\OneDrive\Desktop\Alzeimer's_MRI_Superres\Data_Balanced"
all_image_paths = [os.path.join(dp, f) for dp, _, filenames in os.walk(original_dir) for f in filenames if f.lower().endswith('.jpg')]
test_on_unseen(gen, random.sample(all_image_paths, 5), device=DEVICE)

# ==================== Save Model ======================
save_path = r"C:\Users\Shrey\OneDrive\Desktop\Alzeimer's_MRI_Superres\generator_superres_denoise_final.pt"
torch.save(gen.state_dict(), save_path)

print(f"\nGenerator model saved to: {save_path}")