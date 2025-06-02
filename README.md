# Deep Convolutional GAN (DCGAN) for Artistic Image Generation

This repository contains an implementation of a **Deep Convolutional Generative Adversarial Network (DCGAN)** trained on artwork images from the WikiArt dataset. It supports training, checkpointing, and evaluation using **Inception Score** and **Frechet Inception Distance (FID)**.

## Model Architecture

- **Generator**: 
  - Starts with a fully connected layer expanding a latent vector into a 4D feature map
  - Followed by a series of 2D upsampling + convolutional blocks
  - Outputs RGB images of size **128×128**

- **Discriminator**:
  - Series of downsampling convolutional layers
  - Outputs a scalar prediction for real/fake classification

- **Latent Dim**: Configurable (default = 100)

- **Loss Function**: Binary Cross-Entropy with Logits (BCEWithLogitsLoss)

- **Optimizer**: Adam (`lr=0.0002`, `betas=(0.5, 0.999)`)

## Dataset

- **WikiArt (Resized to 128×128)**  
- Preprocessed and saved as a `.pt` tensor file for efficient loading
