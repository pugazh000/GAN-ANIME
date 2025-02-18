# GAN-ANIME
Collection of PyTorch implementations of Generative Adversarial Network varieties presented in research papers.

# Anime Face Generation using GANs

This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) to generate anime-style faces. The model is inspired by [Pooya Moini's GAN projects](https://github.com/pooyamoini/GAN-projects), with some modifications to improve data preprocessing and model performance.

## Key Improvements
- **Filtered Dataset**: Implemented a filtering mechanism to ignore or delete unreadable images before training.
- **Optimized Training Pipeline**: Adjusted learning rates, batch sizes, and network architecture for better convergence.
- **Performance Tracking**: Logs losses and visualizes image generations over training epochs.

## Dataset
The dataset consists of anime faces stored in a compressed `.tar.gz` format. It is extracted and preprocessed before training.
I downloaded the dataset from [here](https://sourceforge.net/projects/animegan.mirror/files/data/anime-faces.tar.gz/download)
## Model Architecture
### Generator
- Uses transposed convolutional layers with batch normalization and ReLU activations.
- Outputs a `64x64` RGB image with `Tanh` activation.

### Discriminator
- Convolutional layers with LeakyReLU activations and batch normalization.
- Outputs a probability score indicating real or fake images.

## Setup & Usage
### 1. Install Dependencies
```bash
pip install torch torchvision numpy matplotlib pillow
```

### 2. Load Dataset & Preprocessing
Extract the dataset and filter out corrupted images:
```python
from google.colab import drive
import tarfile

drive.mount('/content/drive')
# this is the location where my file was present
tar = tarfile.open('/content/drive/MyDrive/GAN/anime-faces.tar.gz', "r:gz")
tar.extractall()
tar.close()
```

### 3. Train the GAN
```python
GAN-Anime.py
```
Training will start and output losses for the generator and discriminator, along with sample images at different epochs.

## Results
Sample generated images during training:

![Generated Anime Faces](https://github.com/user-attachments/assets/9674eda0-2e1f-476b-a6b5-386540f2e863)

## Handling Unreadable Images  

During dataset loading, some images might be corrupted, incomplete, or hidden system files (such as `.DS_Store` or `._filename` on macOS). These files can cause errors during training. To ensure only valid images are used, we implemented a **custom dataset loader** that automatically filters out unreadable files.  

### 🛠 How It Works  
- A function `is_valid_image(filepath)` attempts to open each image using **PIL (Pillow)** and verifies its integrity.  
- If an image is corrupted or hidden, it is ignored.  
- The `FilteredImageFolder` class extends `torchvision.datasets.ImageFolder` and applies this validation before loading data.  

This method helps maintain **data quality**, prevents crashes, and ensures smooth training. 🚀  


## Future Improvements
- Implementing conditional GANs for more control over generated outputs.
- Exploring different architectures like StyleGAN for better quality images.

## Acknowledgments
This project is inspired by [Pooya Moini's GAN projects](https://github.com/pooyamoini/GAN-projects). Additional tweaks were made to improve data preprocessing and training stability.

