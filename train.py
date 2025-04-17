import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

# Load FashionMNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

# Define the Generator Model
class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(z_dim, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 28*28)
        self.fc4.apply(self.init_weights)
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.02)

    def forward(self, z):
        x = torch.relu(self.fc1(z))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))  # Output in range [-1, 1]
        return x.view(-1, 1, 28, 28)

# Define the Discriminator Model
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(28*28, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 1)
        self.fc4.apply(self.init_weights)
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.02)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input image
        x = F.leaky_relu(self.fc1(x), 0.2)  # Use F.leaky_relu instead of torch.leaky_relu
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = torch.sigmoid(self.fc4(x))  # Output probability of real/fake
        return x

# Hyperparameters
z_dim = 100  # Latent space dimension
lr = 0.0002  # Learning rate
epochs = 50  # Number of epochs

# Initialize models
generator = Generator(z_dim)
discriminator = Discriminator()

# Loss function
criterion = nn.BCELoss()

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# Train the GAN
for epoch in range(epochs):
    for i, (imgs, _) in enumerate(trainloader):
        # Real images
        real_imgs = imgs
        real_labels = torch.ones(imgs.size(0), 1)
        
        # Fake images
        z = torch.randn(imgs.size(0), z_dim)
        fake_imgs = generator(z)
        fake_labels = torch.zeros(imgs.size(0), 1)
        
        # Train Discriminator
        optimizer_D.zero_grad()

        # Loss for real images
        real_loss = criterion(discriminator(real_imgs), real_labels)
        # Loss for fake images
        fake_loss = criterion(discriminator(fake_imgs.detach()), fake_labels)
        
        # Total discriminator loss
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D.step()
        
        # Train Generator
        optimizer_G.zero_grad()
        
        # Generator wants discriminator to classify fake images as real
        g_loss = criterion(discriminator(fake_imgs), real_labels)
        g_loss.backward()
        optimizer_G.step()
        
    print(f'Epoch [{epoch+1}/{epochs}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}')

    # Show generated images every 10 epochs
    if (epoch+1) % 10 == 0:
        with torch.no_grad():
            z = torch.randn(16, z_dim)
            generated_images = generator(z)
            generated_images = generated_images / 2 + 0.5  # Denormalize

            # Plot generated images
            fig, axes = plt.subplots(4, 4, figsize=(6, 6))
            for i in range(4):
                for j in range(4):
                    axes[i, j].imshow(generated_images[i*4 + j].cpu().numpy().squeeze(), cmap='gray')
                    axes[i, j].axis('off')
            plt.show()