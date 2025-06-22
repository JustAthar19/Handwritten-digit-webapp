import torch
from train_digit_generator import Generator, latent_dim, num_classes
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_images(digit: int, n_samples: int = 5):
    model = Generator().to(device)
    model.load_state_dict(torch.load("digit_generator.pth", map_location=device))
    model.eval()

    noise = torch.randn(n_samples, latent_dim).to(device)
    labels = torch.full((n_samples,), digit, dtype=torch.long).to(device)

    with torch.no_grad():
        gen_imgs = model(noise, labels).cpu()

    images = gen_imgs * 0.5 + 0.5  # Denormalize to [0,1]
    return images.numpy()
