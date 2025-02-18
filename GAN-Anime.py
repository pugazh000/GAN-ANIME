from google.colab import drive
import tarfile
import os
import time
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader

# Mount Google Drive to access dataset
drive.mount('/content/drive')

# Extract dataset from tar.gz file
dataset_path = '/content/drive/MyDrive/GAN/anime-faces.tar.gz'
if not os.path.exists('./anime-faces'):
    tar = tarfile.open(dataset_path, "r:gz")
    tar.extractall()
    tar.close()
print("Dataset extracted successfully!")

# Function to check if an image is valid
def is_valid_image(filepath):
    try:
        with Image.open(filepath) as img:
            img.verify()  # Verify image integrity
        return not os.path.basename(filepath).startswith("._")  # Ignore hidden files
    except:
        return False  # If invalid, return False

# Custom dataset class to filter out bad images
class FilteredImageFolder(dset.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform, is_valid_file=is_valid_image)

# Define dataset parameters
data_root = "./anime-faces"  # Path to extracted dataset
image_size = 64  # Resize images to 64x64
batch_size = 64  # Batch size for training

# Define data transformations
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to range [-1,1]
])

# Load dataset and filter invalid images
dataset = FilteredImageFolder(root=data_root, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
print(f"Total valid images: {len(dataset)}")

# Display sample images
real_batch = next(iter(dataloader))
plt.figure(figsize=(10, 10))
plt.axis("off")
plt.title("Sample Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0][:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
plt.show()

# Initialize model parameters
nz = 100  # Latent vector size
ngf = ndf = 64  # Number of feature maps
nc = 3  # Number of channels (RGB)
lr = 0.0002  # Learning rate
beta1 = 0.5  # Beta1 hyperparameter for Adam optimizer
num_epochs = 30  # Training epochs

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize weights function
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# Define Generator class
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, x):
        return self.main(x)

# Define Discriminator class
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.main(x)

# Create models
netG = Generator().to(device)
netD = Discriminator().to(device)

# Apply weights
netG.apply(weights_init)
netD.apply(weights_init)

# Loss function and optimizers
criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

print("GAN Model Ready!")
