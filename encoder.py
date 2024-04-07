import torch
from torch import nn
import torch.optim as optim
from dataset import encoding_dataloader
import pandas as pd
class ImageAutoencoder(nn.Module):
    def __init__(self, input_output_size):
        super(ImageAutoencoder, self).__init__()

        self.input_output_size = input_output_size

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(self.input_output_size, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, self.input_output_size, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # to ensure the output is between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    


# Instantiate the autoencoder and the optimizer
autoencoder = ImageAutoencoder(input_output_size=3)
optimizer = optim.Adam(autoencoder.parameters())

# Specify the loss function
criterion = nn.MSELoss()

# Specify the number of epochs to train for
num_epochs = 100

# Specify the device to train on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
autoencoder = autoencoder.to(device)
dataframe = pd.read_csv("encoding.csv")
print("Training...")
# Training loop
for epoch in range(num_epochs):
    for batch in encoding_dataloader(dataframe):
        # Move the batch to the device we're using. 
        batch = batch.to(device).float()

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = autoencoder(batch)

        # Compute the loss
        loss = criterion(outputs, batch)

        # Backward pass
        loss.backward()

        # Update the weights
        optimizer.step()

    # Print loss for this epoch
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")