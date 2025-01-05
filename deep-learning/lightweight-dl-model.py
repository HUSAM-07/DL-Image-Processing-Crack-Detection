import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from datetime import datetime
from torchmetrics import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
import logging
import cv2
from torchvision.transforms import functional as TF
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(
    filename='training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Modified training function without tensorboard
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, patience=5):
    # Initialize metrics
    ssim = StructuralSimilarityIndexMeasure().to(device)
    psnr = PeakSignalNoiseRatio().to(device)

    # Initialize lists to store metrics for plotting
    train_losses = []
    val_losses = []
    train_ssims = []
    val_ssims = []
    train_psnrs = []
    val_psnrs = []

    best_val_loss = float('inf')
    patience_counter = 0

    logging.info(f"Starting training with {len(train_loader.dataset)} training samples")
    logging.info(f"Using device: {device}")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_ssim_val = 0.0
        train_psnr_val = 0.0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # Calculate metrics
            batch_ssim = ssim(outputs, targets)
            batch_psnr = psnr(outputs, targets)

            train_loss += loss.item()
            train_ssim_val += batch_ssim.item()
            train_psnr_val += batch_psnr.item()

            # Log batch progress
            if batch_idx % 10 == 0:
                logging.info(f'Epoch {epoch+1}/{num_epochs} - Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}')

        # Calculate average training metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_train_ssim = train_ssim_val / len(train_loader)
        avg_train_psnr = train_psnr_val / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_ssim_val = 0.0
        val_psnr_val = 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)

                batch_ssim = ssim(outputs, targets)
                batch_psnr = psnr(outputs, targets)

                val_loss += loss.item()
                val_ssim_val += batch_ssim.item()
                val_psnr_val += batch_psnr.item()

        # Calculate average validation metrics
        avg_val_loss = val_loss / len(val_loader)
        avg_val_ssim = val_ssim_val / len(val_loader)
        avg_val_psnr = val_psnr_val / len(val_loader)

        # Store metrics for plotting
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_ssims.append(avg_train_ssim)
        val_ssims.append(avg_val_ssim)
        train_psnrs.append(avg_train_psnr)
        val_psnrs.append(avg_val_psnr)

        # Log epoch metrics
        logging.info(f'''Epoch {epoch+1}/{num_epochs}:
            Train Loss: {avg_train_loss:.4f}, SSIM: {avg_train_ssim:.4f}, PSNR: {avg_train_psnr:.4f}
            Val Loss: {avg_val_loss:.4f}, SSIM: {avg_val_ssim:.4f}, PSNR: {avg_val_psnr:.4f}
        ''')

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_ssim': avg_train_ssim,
                'val_ssim': avg_val_ssim,
                'train_psnr': avg_train_psnr,
                'val_psnr': avg_val_psnr,
            }, 'best_model.pth')
            logging.info(f'Saved new best model at epoch {epoch+1}')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f'Early stopping triggered at epoch {epoch+1}')
                break

    # Plot training metrics
    plot_metrics(train_losses, val_losses, 'Loss', 'loss.png')
    plot_metrics(train_ssims, val_ssims, 'SSIM', 'ssim.png')
    plot_metrics(train_psnrs, val_psnrs, 'PSNR', 'psnr.png')

    return model

def plot_metrics(train_metric, val_metric, metric_name, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_metric, label=f'Train {metric_name}')
    plt.plot(val_metric, label=f'Validation {metric_name}')
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.title(f'Training and Validation {metric_name}')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

class EnhancedCrackDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_pairs = []
        
        # Collect image pairs using the correct naming pattern
        for filename in os.listdir(data_dir):
            if filename.endswith('first_frame.png'):  # Initial frame
                number = filename.split('_')[0]  # Get the number prefix
                first_frame = filename
                last_frame = f"{number}_last_frame.png"
                if os.path.exists(os.path.join(data_dir, last_frame)):
                    self.image_pairs.append((first_frame, last_frame))
    
    def extract_features(self, image):
        # Convert PIL Image to numpy array
        img_np = np.array(image)
        
        # Edge detection
        edges = cv2.Canny(img_np, threshold1=50, threshold2=150)
        
        # Hough transform for line detection
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                               threshold=50, 
                               minLineLength=30, 
                               maxLineGap=10)
        
        # Create feature maps
        edge_map = torch.FloatTensor(edges) / 255.0
        
        # Create line map
        line_map = np.zeros_like(img_np)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_map, (x1, y1), (x2, y2), 255, 2)
        
        line_map = torch.FloatTensor(line_map) / 255.0
        
        return edge_map, line_map
    
    def __getitem__(self, idx):
        input_name, target_name = self.image_pairs[idx]
        
        # Load images
        input_img = Image.open(os.path.join(self.data_dir, input_name)).convert('L')
        target_img = Image.open(os.path.join(self.data_dir, target_name)).convert('L')
        
        # Extract features
        edge_map, line_map = self.extract_features(input_img)
        
        # Apply transforms
        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)
            edge_map = TF.resize(edge_map.unsqueeze(0), input_img.shape[1:])
            line_map = TF.resize(line_map.unsqueeze(0), input_img.shape[1:])
        
        # Concatenate input image with feature maps
        enhanced_input = torch.cat([input_img, edge_map, line_map], dim=0)
        
        return enhanced_input, target_img
    
    def __len__(self):
        return len(self.image_pairs)

class EnhancedCrackModel(nn.Module):
    def __init__(self):
        super(EnhancedCrackModel, self).__init__()
        
        # Modified encoder to handle 3 input channels (image + edge map + line map)
        self.enc1 = self._conv_block(3, 64)  # Changed input channels from 1 to 3
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)
        
        # Decoder remains the same
        self.dec4 = self._upconv_block(512, 256)
        self.dec3 = self._upconv_block(512, 128)
        self.dec2 = self._upconv_block(256, 64)
        self.dec1 = self._upconv_block(128, 32)
        
        self.final = nn.Conv2d(32, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

