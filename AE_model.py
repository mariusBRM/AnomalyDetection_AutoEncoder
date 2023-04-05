import torch
import torch.nn as nn

class DeepAutoencoder(nn.Module):
    def __init__(self, dim_input, intermediate, latent_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim_input, intermediate),
            nn.BatchNorm1d(intermediate),
            nn.LeakyReLU(),
            nn.Linear(intermediate, latent_size),
            nn.BatchNorm1d(latent_size),
            nn.LeakyReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, intermediate),
            nn.BatchNorm1d(intermediate),
            nn.LeakyReLU(),
            nn.ReLU(),
            nn.Linear(intermediate, dim_input),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def feature_extration(self, x):
        return self.encoder(x)
    
class SimpleAutoencoder(nn.Module):
    def __init__(self, dim_input, latent_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(dim_input, latent_size),
            nn.BatchNorm1d(latent_size),
            nn.LeakyReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, dim_input),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def feature_extration(self, x):
        return self.encoder(x)