import argparse
import logging
import os
from pathlib import Path
import random
from typing import List

import sys
from matplotlib import pyplot as plt


sys.path.append(os.getcwd() + "/..")

import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import seaborn as sns

from src.lfxai.explanations.features import proto_attribute
from src.lfxai.utils.feature_attribution import generate_masks
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


def predictive_performance_and_ablation(
    runs: int = 1,
    random_seeds: List[int] = [42],
    batch_size: int = 50,
    dim_latent: int = 32,
    n_epochs: int = 100,
    start_push_epoch: int = 70,
    push_epoch_frequency: int = 10,
    freeze_epoch: int = 90,
) -> None:
    """In this function we perform experiments I and IV, where we evaluate the
        predictive performance of the PAE against a normal AE on the tasks of
        reconstruction, denoising, and inpainting. Additionally, we check if adding
        Multiple prototypes gives us better results

    Args:
        runs (int): number of runs, over which it is averaged
        seeds (List[int]): list of unique random seeds over every run
        batch_size (int): size of batches
        dim_latent (int): size of the latent dimension
        n_epochs (int): number of epochs
        start_push_epoch (int): epoch at which the PAE starts pushing
        push_epoch_frequency (int): epoch frequency when PAE pushes
        freeze_epoch (int): after this epoch, there are no more pushes, only decoder is trained
    """

    results_dict = {
        "AE_reconstruction": [],
        "AE_denoising": [],
        "AE_inpainting": [],
        "PAE_reconstruction_128": [],
        "PAE_reconstruction_256": [],
        "PAE_reconstruction_512": [],
        "PAE_denoising_128": [],
        "PAE_denoising_256": [],
        "PAE_denoising_512": [],
        "PAE_inpainting_128": [],
        "PAE_inpainting_256": [],
        "PAE_inpainting_512": [],
    }

    W = 28

    # Load MNIST
    data_dir = Path.cwd() / "data/mnist"
    train_dataset = torchvision.datasets.FashionMNIST(
        data_dir, train=True, download=True
    )
    test_dataset = torchvision.datasets.FashionMNIST(
        data_dir, train=False, download=True
    )
    train_transform = transforms.Compose([transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset.transform = train_transform
    test_dataset.transform = test_transform

    train_subset = Subset(train_dataset, indices=list(range(5000)))
    val_subset = Subset(train_dataset, indices=list(range(5000, 5000 + 1000)))
    test_subset = Subset(test_dataset, indices=list(range(1000)))

    train_loader = DataLoader(train_subset, batch_size=batch_size)
    train_push_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    perturbations = {
        "reconstruction": Identity(),
        "denoising": RandomNoise(noise_level=0.3),
        "inpainting": Mask(mask_proportion=0.2),
    }
    ns_prototypes = [128, 256, 512]

    for i in range(runs):
        logging.info(f"Predictive Performance run {i}")
        set_seed(random_seeds[i])
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

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
            save_dir = (
                Path.cwd()
                / f"results/fashion_mnist/predictive_performance/{ae_model_name}"
            )
            if not save_dir.exists():
                print("making dir")
                os.makedirs(save_dir)
            logging.info("fitting AE")
            autoencoder.fit(
                device, train_loader, test_loader, save_dir, n_epochs, patience=20
            )

            # Testing
            autoencoder.eval()
            autoencoder.load_state_dict(
                torch.load(save_dir / (autoencoder.name + ".pt")), strict=False
            )
            ae_loss = autoencoder.test_epoch(device, test_loader)
            results_dict[ae_model_name].append(ae_loss)

            for n_prototypes in ns_prototypes:
                pae_model_name = f"PAE_{pert_name}_{n_prototypes}"

                # Initialize normal encoder, decoder and autoencoder wrapper
                protoncoder = ProtoEncoderMnist(encoded_space_dim=dim_latent).to(device)
                protodecoder = ProtoDecoderMnist(encoded_space_dim=dim_latent).to(
                    device
                )
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

                save_dir = (
                    Path.cwd()
                    / f"results/fashion_mnist/predictive_performance/{pae_model_name}"
                )
                if not save_dir.exists():
                    os.makedirs(save_dir)
                logging.info(f"fitting PAE with {n_prototypes} prototypes")
                protoautoencoder.fit(
                    device,
                    train_loader,
                    train_push_loader,
                    val_loader,
                    save_dir,
                    n_epochs,
                    patience=n_epochs,
                    lr=1e-3,
                    start_push_epoch=start_push_epoch,
                    push_epoch_frequency=push_epoch_frequency,
                    freeze_epoch=freeze_epoch,
                )
                protoautoencoder.eval()
                protoautoencoder.load_state_dict(
                    torch.load(save_dir / (protoautoencoder.name + ".pt")), strict=False
                )
                pae_loss = protoautoencoder.test_epoch(device, test_loader)
                results_dict[pae_model_name].append(pae_loss)

    # Print results and save df
    for experiment_name, result_list in results_dict.items():
        logging.info(
            f"{experiment_name}: {np.array(result_list).mean():.4f} +- {np.array(result_list).std():.4f} "
        )

    df = pd.DataFrame.from_dict(results_dict)
    df.to_csv("results/fashion_mnist/predictive_performance/results.csv")


def proto_consistency_feature_importance(
    random_seed: int = 1,
    batch_size: int = 200,
    dim_latent: int = 32,
) -> None:
    # Initialize seed and device
    set_seed(random_seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    W = 28  # Image width = height
    pert_percentages = [5, 10, 20, 50, 80, 100]
    # Here it is equivalent to perturbing the input feature from the prototype

    # Load (test) MNIST
    data_dir = Path.cwd() / "data/mnist"
    test_dataset = torchvision.datasets.FashionMNIST(
        data_dir, train=False, download=True
    )
    test_transform = transforms.Compose([transforms.ToTensor()])
    test_dataset.transform = test_transform
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    # Initialize encoder, decoder and autoencoder wrapper
    pert = RandomNoise(noise_level=0.3)
    protoncoder = ProtoEncoderMnist(encoded_space_dim=dim_latent).to(device)
    protodecoder = ProtoDecoderMnist(encoded_space_dim=dim_latent).to(device)

    n_prototypes = 128
    pae_model_name = f"PAE_denoising_{n_prototypes}"
    run = 4

    # Initialize normal encoder, decoder and autoencoder wrapper
    protoncoder = ProtoEncoderMnist(encoded_space_dim=dim_latent).to(device)
    protodecoder = ProtoDecoderMnist(encoded_space_dim=dim_latent).to(device)
    protoautoencoder = ProtoAutoEncoderMnist(
        protoncoder,
        protodecoder,
        prototype_shape=(n_prototypes, dim_latent, 1, 1),
        input_pert=pert,
        name=pae_model_name + f"_run{run}",
        metric="l2",
        prototype_activation_function="log",
    )
    protoautoencoder.to(device)

    load_dir = (
        Path.cwd() / f"results/fashion_mnist/predictive_performance/{pae_model_name}"
    )
    protoautoencoder.eval()
    protoautoencoder.load_state_dict(
        torch.load(load_dir / (protoautoencoder.name + ".pt")), strict=False
    )

    attr_methods = {
        "PAE": protoautoencoder,
        "Random": None,
    }
    results_data = []
    for method_name in attr_methods:
        logging.info(f"Computing feature importance with {method_name}")
        results_data.append([method_name, 0, 0])
        attr_method = attr_methods[method_name]
        if attr_method is not None:
            attr = proto_attribute(
                protoautoencoder, test_loader, device, pert, top_k_weights=10
            )
        else:
            np.random.seed(random_seed)
            attr = np.random.randn(len(test_dataset), 1, W, W)

        for pert_percentage in pert_percentages:
            logging.info(
                f"Perturbing {pert_percentage}% of the features with {method_name}"
            )
            mask_size = int(pert_percentage * W**2 / 100)
            masks = generate_masks(attr, mask_size)
            for batch_id, (images, _) in enumerate(test_loader):
                images = pert(images)
                mask = masks[
                    batch_id * batch_size : batch_id * batch_size + len(images)
                ].to(device)
                images = images.to(device)
                original_reps = protoautoencoder.get_representations(images)
                images = mask * images
                pert_reps = protoautoencoder.get_representations(images).flatten(1)
                rep_shift = torch.mean(
                    torch.sum((original_reps - pert_reps) ** 2, dim=-1)
                ).item()
                results_data.append([method_name, pert_percentage, rep_shift])

    logging.info("Saving the plot")
    results_df = pd.DataFrame(
        results_data, columns=["Method", "% Perturbed Pixels", "Representation Shift"]
    )
    sns.set(font_scale=1.3)
    sns.set_style("white")
    sns.set_palette("colorblind")
    sns.lineplot(
        data=results_df, x="% Perturbed Pixels", y="Representation Shift", hue="Method"
    )
    plt.tight_layout()
    save_dir = Path.cwd() / "results/fashion_mnist/consistency_features"
    if not save_dir.exists():
        os.makedirs(save_dir)
    plt.savefig(save_dir / "proto_fashion_mnist_consistency_features.pdf")
    plt.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="disvae")
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=300)
    parser.add_argument("--random_seed", type=int, default=1)
    args = parser.parse_args()

    if args.name == "predictive_performance_and_ablation":
        predictive_performance_and_ablation(
            runs=5,
            random_seeds=[11, 19, 42, 69, 110],
            batch_size=50,
            n_epochs=100,
            dim_latent=32,
            start_push_epoch=70,
            freeze_epoch=90,
            push_epoch_frequency=10,
        )
    elif args.name == "proto_consistency_feature_importance":
        proto_consistency_feature_importance()
    else:
        raise ValueError("Invalid experiment name")
