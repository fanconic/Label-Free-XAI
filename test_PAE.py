import torch
from pathlib import Path
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import logging

logging.getLogger().setLevel(logging.INFO)

from src.lfxai.models.images import (
    ProtoAutoEncoderMnist,
    ProtoDecoderMnist,
    ProtoEncoderMnist,
)
from src.lfxai.models.pretext import Identity, RandomNoise, Mask
from src.lfxai.models.local_analysis import LocalAnalysis
import os
import random
import numpy as np


def set_seed(seed: int):
    """Set all random seeds
    Args:
        seed (int): integer for reproducible experiments
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


set_seed(42)

# Select torch device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# pert = Identity()
# pert = RandomNoise(noise_level=0.3)
pert = Mask(mask_proportion=0.2)

model_name = "ProtoAE_inpainting_2"

# Get Prototype Autoencoder model
latent_dim = 32
n_prototypes = 128
encoder = ProtoEncoderMnist(encoded_space_dim=latent_dim)
decoder = ProtoDecoderMnist(encoded_space_dim=latent_dim)
protoautoencoder = ProtoAutoEncoderMnist(
    encoder,
    decoder,
    prototype_shape=(n_prototypes, latent_dim, 1, 1),
    input_pert=pert,
    name=model_name,
    metric="l2",
    prototype_activation_function="log",
)
protoautoencoder.to(device)


# Initialize seed and device
random_seed = 42
torch.random.manual_seed(random_seed)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
W = 28  # Image width = height
pert_percentages = [5, 10, 20, 50, 80, 100]

# Load MNIST
data_dir = Path.cwd() / "data/mnist"
train_dataset = MNIST(data_dir, train=True, download=True)
test_dataset = MNIST(data_dir, train=False, download=True)
train_transform = transforms.Compose([transforms.ToTensor()])
test_transform = transforms.Compose([transforms.ToTensor()])
train_dataset.transform = train_transform
test_dataset.transform = test_transform

train_subset = Subset(train_dataset, indices=list(range(5000)))
val_subset = Subset(train_dataset, indices=list(range(5000, 5000 + 1000)))
test_subset = Subset(test_dataset, indices=list(range(10)))

train_loader = DataLoader(train_subset, batch_size=50)
train_push_loader = DataLoader(train_subset, batch_size=50, shuffle=False)
val_loader = DataLoader(val_subset, batch_size=50, shuffle=False)
test_loader = DataLoader(test_subset, batch_size=1, shuffle=False)


# Train the denoising autoencoder
epochs = 100
start_push_epoch = 70
push_epoch_frequency = 10
freeze_epoch = 90
save_dir = Path.cwd() / "results/mnist/consistency_features"
if not save_dir.exists():
    os.makedirs(save_dir)
protoautoencoder.fit(
    device,
    train_loader,
    train_push_loader,
    val_loader,
    save_dir,
    epochs,
    patience=epochs,
    lr=1e-3,
    start_push_epoch=start_push_epoch,
    push_epoch_frequency=push_epoch_frequency,
    freeze_epoch=freeze_epoch,
)

# Export model to ONNX to inspect it
protoautoencoder.eval()
protoautoencoder.load_state_dict(
    torch.load(save_dir / (protoautoencoder.name + ".pt")), strict=False
)

# Visluaise the
load_model_dir = "results/mnist/consistency_features"
load_model_name = f"{model_name}.pt"
prototypes_dir = f"results/prototypes/{model_name}"
prototype_img_save_dir = (
    f"results/prototypes/{model_name}/epoch-{freeze_epoch}-visualize"
)
loc_analysis = LocalAnalysis(
    protoautoencoder,
    prototypes_dir,
    epoch_number=epochs - 10,
    image_save_directory=prototype_img_save_dir,
)

torch_input = torch.randn(1, 1, 28, 28)
onnx_program = torch.onnx.dynamo_export(protoautoencoder, torch_input)
onnx_program.save("PAE_2.onnx")


for i, (input_img, _) in enumerate(test_loader):
    loc_analysis.visualization(
        pert(input_img),
        f"test_img_{i}",
        input_img,
        show_images=True,
        max_prototypes=5,
    )
