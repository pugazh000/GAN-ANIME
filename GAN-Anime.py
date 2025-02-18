# Mount Google Drive to access dataset
from google.colab import drive
drive.mount('/content/drive')

#libraies import
from __future__ import print_function
import os
import time
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from matplotlib import pyplot as plt
import numpy as np
import datetime

print('SETUP DONE')
# Extract dataset from tar.gz file
import tarfile
tar = tarfile.open('/content/drive/MyDrive/GAN/anime-faces.tar.gz', "r:gz") #change the loction here
tar.extractall()
tar.close()

# Set parameters for training
data_root = './anime-faces'  # Dataset directory
nc = 3  # Number of channels (RGB images)
batch_size = 512  # Number of images per batch
images_size = 64  # Image size
nz = 100  # Size of the noise vector
ngf = ndf = 64  # Generator & Discriminator feature maps
num_epochs = 30  # Number of training epochs
lr = 0.0002  # Learning rate
beta1 = 0.5  # Adam optimizer beta1 value
ngpu = 1  # Number of GPUs
print_every = 250  # Print loss after every 250 batches
device = torch.device('cuda:0')  # Use GPU if available

# Function to check if an image is valid
def is_valid_image(filepath):
    try:
        with Image.open(filepath) as img:
            img.verify()  # Verify image integrity
        return not os.path.basename(filepath).startswith("._")  # Ignore hidden files
    except:
        return False  # If invalid, return False

# Custom dataset class to filter out bad images
class FilteredImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform, is_valid_file=is_valid_image)

# Dataset parameters
data_root = "./anime-faces"  # Change if needed
image_size = 64  # Resize images to this size

# Define transformations
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load dataset with filtering applied
dataset = FilteredImageFolder(root=data_root, transform=transform)

# DataLoader to handle batch loading
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

# Check if the dataset is properly loaded
print(f"Total valid images: {len(dataset)}")

# Display some images
real_batch = next(iter(dataloader))
plt.figure(figsize=(15, 15))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0][:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
plt.show()

#This generates total valid images

# Function to initialize weights of model
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
# Define the Generator model
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
        
# Define the Discriminator model
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

# Create instances of Generator and Discriminator
netG = Generator().to(device)
netD = Discriminator().to(device)

# Apply weight initialization
netG.apply(weights_init)
netD.apply(weights_init)

# Define loss function
criterion = nn.BCELoss()

# Generate fixed noise for evaluation
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Labels for real and fake images
real_label = 1
fake_label = 0

# Define optimizers for Generator and Discriminator
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# Initialize lists to store training progress
img_list = []
G_losses = []
D_losses = []
iters = 0
prev_time = time.time()

# Training loop
for epoch in range(num_epochs):
    for i, (img, _) in enumerate(dataloader, 0):

        # Train Discriminator
        netD.zero_grad()
        real_data = img.to(device)
        b_size = real_data.size(0)
        label = torch.full((b_size,), real_label,dtype=torch.float, device=device)

        output = netD(real_data).view(-1)
        errD_real = criterion(output, label)

        errD_real.backward()
        D_x = output.mean().item()

        noise = torch.randn(b_size, nz, 1, 1, device=device)

        fake = netG(noise)
        label.fill_(fake_label)

        output = netD(fake.detach()).view(-1)
        errD_fake = criterion(output, label)

        errD_fake.backward()
        D_G_z1 = output.mean().item()

        errD = errD_real + errD_fake

        optimizerD.step()

        # Train Generator
        netG.zero_grad()
        label.fill_(real_label)
        output = netD(fake).view(-1)

        errG = criterion(output, label)
        errG.backward()
        D_G_z2 = output.mean().item()

        optimizerG.step()

        # Output training stats
        if i % print_every == 0:
            batches_done = epoch * len(dataloader) + i
            batches_left = num_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\t ETA: %s'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), time_left))

        G_losses.append(errG.item())
        D_losses.append(errD.item())

        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1
real_batch = next(iter(dataloader))

#testing the model


# Plot the real images
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

# Plot the fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()

# Plot the fake images
plt.figure(figsize=(15,15))
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-2],(1,2,0)))

# Plot the fake images with variation
plt.figure(figsize=(15,15))
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-3],(1,2,0)))
plt.show()
plt.show()
