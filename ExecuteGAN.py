import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from GANModel import WaveGANGenerator, WaveGANDiscriminator
import soundfile as sf
import os

# Hyperparameters
batch_size = 2
learning_rate = 0.005
num_epochs = 100
noise_latent_dim = 100

# Initialize the WaveGAN model
generator = WaveGANGenerator()
discriminator = WaveGANDiscriminator()

# Load pre-trained weights
generator.load_state_dict(torch.load("generator.pth"))
discriminator.load_state_dict(torch.load("discriminator.pth"))

# Loss function and optimizer
adversarial_loss = nn.BCEWithLogitsLoss()
optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

new_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# Training loop on new dataset
for epoch in range(num_epochs):
    for i, real_data in enumerate(new_dataloader):
        # Adversarial ground truths
        valid = torch.ones((batch_size, 1))
        fake = torch.zeros((batch_size, 1))

        # Train Generator
        optimizer_G.zero_grad()
        z = torch.randn((batch_size, noise_latent_dim))
        generated_data = generator(z)
        g_loss = adversarial_loss(discriminator(generated_data), valid)
        g_loss.backward()
        optimizer_G.step()

        # Train Discriminator
        optimizer_D.zero_grad()
        real_data = real_data.cuda()
        d_loss_real = adversarial_loss(discriminator(real_data), valid)
        d_loss_fake = adversarial_loss(discriminator(generated_data.detach()), fake)
        d_loss = (d_loss_real + d_loss_fake) / 2
        d_loss.backward()
        optimizer_D.step()

        # Print training progress
        if i % 100 == 0:
            print(
                f"[Epoch {epoch}/{num_epochs}] [Batch {i}/{len(new_dataloader)}] "
                f"[D loss: {d_loss.item()}] [G loss: {g_loss.item()}]"
            )

# Save the fine-tuned models
torch.save(generator.state_dict(), "fine_tuned_generator.pth")
torch.save(discriminator.state_dict(), "fine_tuned_discriminator.pth")

# Generate a sound sample
z_sample = torch.randn((1, noise_latent_dim))
generated_sample = generator(z_sample).detach().cpu().numpy().squeeze()
sf.write("generated_sample.wav", generated_sample, 16000)