import os
import sys
sys.path.append(os.getcwd() + "/..")

import pandas as pd
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
import torch
import numpy as np
from PIL import Image
from src.lfxai.models.pretext import Identity, Mask, RandomNoise
from tqdm import tqdm

# Parameters
freeze_epoch = 90
prototypes = 128
pretext_dict = {
        "reconstruction": Identity(),
        "denoising": RandomNoise(noise_level=0.3),
        "inpainting": Mask(mask_proportion=0.2),
    }
runs = 5

# Load MNIST
data_dir = Path.cwd() / "data/mnist"
train_transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=train_transform)
train_subset = Subset(train_dataset, indices=list(range(50000)))
train_loader = DataLoader(train_subset, batch_size=50000, shuffle=False)

# Helper function to calculate mean squared error
def mse(image1, image2):
    return ((image1 - image2) ** 2).mean(dim=(1, 2, 3))

train_images, train_labels = next(iter(train_loader))

# Main processing loop
for run in range(runs):
    for task, task_fn in pretext_dict.items():  
        load_dir = Path.cwd() / f"results/fashion_mnist/predictive_performance/PAE_{task}_{prototypes}/prototypes/PAE_{task}_{prototypes}_run{run}/epoch-{freeze_epoch}"
        
        results = []
        
        for nr_proto in tqdm(range(prototypes)):
            img_path = load_dir / f"prototype-img-original{nr_proto}.png"
                        
            # Load the prototype image
            prototype_image = Image.open(img_path).convert('L')
            prototype_tensor = transforms.ToTensor()(prototype_image).unsqueeze(0)
            
            # Calculate MSE with all training images
            current_mse = mse(prototype_tensor, train_images)
            
            # Find the label of the image with the lowest MSE
            min_mse_idx = current_mse.argmin().item()
            best_label = train_labels[min_mse_idx].item()
            min_mse = current_mse[min_mse_idx].item()

            results.append({
                "prototype_number": nr_proto,
                "best_label": best_label,
                "mse": min_mse
            })
        
        # Save results to CSV
        results_df = pd.DataFrame(results)
        results_csv_path = load_dir / f"comparison_results_run{run}.csv"
        results_df.to_csv(results_csv_path, index=False)
        print(f"Saved results to {results_csv_path}")
print("Processing complete.")