import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np

# Custom Dataset class
class CrackPropagationDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_pairs = []

        # Collect all image pairs
        for filename in os.listdir(data_dir):
            if filename.endswith('first_frame.png'):
                number = filename.split('_')[0]
                first_frame = filename
                last_frame = f"{number}_last_frame.png"
                if os.path.exists(os.path.join(data_dir, last_frame)):
                    self.image_pairs.append((first_frame, last_frame))

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        first_frame, last_frame = self.image_pairs[idx]

        # Load images
        first_img = Image.open(os.path.join(self.data_dir, first_frame)).convert('L')
        last_img = Image.open(os.path.join(self.data_dir, last_frame)).convert('L')

        if self.transform:
            first_img = self.transform(first_img)
            last_img = self.transform(last_img)

        return first_img, last_img

# U-Net style model
class CrackPropagationModel(nn.Module):
    def __init__(self):
        super(CrackPropagationModel, self).__init__()

        # Encoder
        self.enc1 = self._conv_block(1, 64)
        self.enc2 = self._conv_block(64, 128)
        self.enc3 = self._conv_block(128, 256)
        self.enc4 = self._conv_block(256, 512)

        # Decoder
        self.dec4 = self._upconv_block(512, 256)
        self.dec3 = self._upconv_block(512, 128)
        self.dec2 = self._upconv_block(256, 64)
        self.dec1 = self._upconv_block(128, 32)

        self.final = nn.Conv2d(32, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def _upconv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Decoder with skip connections
        d4 = self.dec4(e4)
        d4 = torch.cat([d4, e3], dim=1)
        d3 = self.dec3(d4)
        d3 = torch.cat([d3, e2], dim=1)
        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d1 = self.dec1(d2)

        out = self.sigmoid(self.final(d1))
        return out

# Training function
def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}')

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Create dataset and dataloader with the new path
    dataset = CrackPropagationDataset(
        'C:\\Users\\UNIHU\\Documents\\GitHub\\Crack_Defect_Interaction\\generated_frames', 
        transform=transform
    )
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Initialize model, loss, and optimizer
    model = CrackPropagationModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, num_epochs=50, device=device)

    # Save the model
    torch.save(model.state_dict(), 'crack_propagation_model.pth')

if __name__ == "__main__":
    main()
