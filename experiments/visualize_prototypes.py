import argparse
import logging
import os
from pathlib import Path
import random
from typing import List
import matplotlib.pyplot as plt

import sys

sys.path.append(os.getcwd() + "/..")

import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from src.lfxai.models.images import (
    AutoEncoderMnist,
    DecoderMnist,
    EncoderMnist,
    ProtoEncoderMnist,
    ProtoDecoderMnist,
    ProtoAutoEncoderMnist,
)

from src.lfxai.models.pretext import Identity, Mask, RandomNoise


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


def visualize_predictive_performance(
    dataset: str,
    dim_latent: int = 32,
) -> None:
    """Visualize the prototypes of reconstruction, denoising, and inpainting
        Save teh pdf plot, to use in latex

    Args:
        dataset (str): name of the dataset ("mnist" or "fashion_mnist")
        dim_latent (int): size of the latent dimension
    """

    # Load MNIST
    data_dir = Path.cwd() / "data/mnist"
    if dataset.lower() == "fashion_mnist":
        test_dataset = torchvision.datasets.FashionMNIST(
            data_dir, train=False, download=True
        )
    elif dataset.lower() == "mnist":
        test_dataset = torchvision.datasets.MNIST(data_dir, train=False, download=True)
    else:
        raise ValueError("Dataset does not exist")

    test_transform = transforms.Compose([transforms.ToTensor()])
    test_dataset.transform = test_transform

    test_subset = Subset(test_dataset, indices=list(range(10)))
    test_loader = DataLoader(test_subset, batch_size=10, shuffle=False)

    perturbations = {
        "reconstruction": Identity(),
        "denoising": RandomNoise(noise_level=0.3),
        "inpainting": Mask(mask_proportion=0.2),
    }

    n_prototypes = 256
    i = 0
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    for pert_name, pert in perturbations.items():
        ae_model_name = f"AE_{pert_name}"
        logging.info(f"Pretext task: {pert_name}")

        # Initialize normal encoder, decoder and autoencoder wrapper
        encoder = EncoderMnist(encoded_space_dim=dim_latent)
        decoder = DecoderMnist(encoded_space_dim=dim_latent)
        autoencoder = AutoEncoderMnist(
            encoder, decoder, dim_latent, pert, name=ae_model_name + f"_run{i}"
        )
        encoder.to(device)
        decoder.to(device)

        # Train the denoising autoencoder
        load_dir = (
            Path.cwd() / f"results/{dataset}/predictive_performance/{ae_model_name}"
        )

        # Testing
        autoencoder.eval()
        autoencoder.load_state_dict(
            torch.load(load_dir / (autoencoder.name + ".pt")), strict=False
        )

        pae_model_name = f"PAE_{pert_name}_{n_prototypes}"

        # Initialize normal encoder, decoder and autoencoder wrapper
        protoncoder = ProtoEncoderMnist(encoded_space_dim=dim_latent).to(device)
        protodecoder = ProtoDecoderMnist(encoded_space_dim=dim_latent).to(device)
        protoautoencoder = ProtoAutoEncoderMnist(
            protoncoder,
            protodecoder,
            prototype_shape=(n_prototypes, dim_latent, 1, 1),
            input_pert=pert,
            name=pae_model_name + f"_run{i}",
            metric="l2",
            prototype_activation_function="log",
        )
        protoautoencoder.to(device)

        load_dir = (
            Path.cwd() / f"results/{dataset}/predictive_performance/{pae_model_name}"
        )
        protoautoencoder.eval()
        protoautoencoder.load_state_dict(
            torch.load(load_dir / (protoautoencoder.name + ".pt")), strict=False
        )

        for target_img, _ in test_loader:
            input_img = pert(target_img)
            prediction_ae = autoencoder(input_img)
            prediction_pae, distances = protoautoencoder(input_img)

            for j, (inp, tgt, rec_ae, rec_pae) in enumerate(
                zip(input_img, target_img, prediction_ae, prediction_pae)
            ):
                fig, axs = plt.subplots(
                    1, 4, figsize=(12, 3)
                )  # Adjust the figsize as needed

                # Plot input image
                axs[0].imshow(inp.squeeze().detach().numpy(), cmap="gray")
                axs[0].set_title("Input")

                # Plot target image
                axs[1].imshow(tgt.squeeze().detach().numpy(), cmap="gray")
                axs[1].set_title("Target")

                # Plot reconstructed images
                axs[2].imshow(rec_ae.squeeze().detach().numpy(), cmap="gray")
                axs[2].set_title("Reconstructed AE")

                axs[3].imshow(rec_pae.squeeze().detach().numpy(), cmap="gray")
                axs[3].set_title("Reconstructed PAE")

                for ax in axs:
                    ax.axis("off")

                plt.tight_layout()
                plt.savefig(
                    f"results/{dataset}/predictive_performance/{pert_name}_example_{j + 1}.pdf",
                    bbox_inches="tight",
                )
                plt.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="disvae")
    parser.add_argument("--dataset", type=str, default="mnist")
    args = parser.parse_args()

    if args.name == "visualize_predictive_performance":
        visualize_predictive_performance(
            dataset=args.dataset,
            dim_latent=32,
        )
    else:
        raise ValueError("Invalid experiment name")
