import argparse
import logging
import os
from pathlib import Path
import random
from typing import List

import sys


sys.path.append(os.getcwd() + "/..")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchvision
from captum.attr import GradientShap, IntegratedGradients, Saliency, Lime
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, RandomSampler, Subset
from torchvision import transforms

from src.lfxai.explanations.features import (
    proto_attribute,
)
from src.lfxai.models.images import AutoEncoderMnist
    

from src.lfxai.models.proto_network import (
    ProtoClassifierMnist,
    ProtoResNet18Enc,
    ProtoResNet18Dec,
    ProtoAutoEncoderMnist,
    ResNet18Dec,
    ResNet18Enc
)
from src.lfxai.models.pretext import Identity, Mask, RandomNoise
from src.lfxai.utils.feature_attribution import generate_masks
from src.lfxai.utils.metrics import (
    proto_similarity_rates,
)
from src.lfxai.utils.visualize import (
    correlation_latex_table,
)
from src.lfxai.models.local_analysis import LocalAnalysis


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
        "PAE_reconstruction_128": [],
        "PAE_reconstruction_256": [],
        "PAE_reconstruction_512": [],
        #"AE_denoising": [],
        #"AE_inpainting": [],
        #"PAE_denoising_128": [],
        #"PAE_denoising_256": [],
        #"PAE_denoising_512": [],
        #"PAE_inpainting_128": [],
        #"PAE_inpainting_256": [],
        #"PAE_inpainting_512": [],
    }

    W = 32
    data_dir = Path.cwd() / "data/mnist"
    train_dataset = torchvision.datasets.CIFAR10(data_dir, train=True, download=True)
    push_dataset = torchvision.datasets.CIFAR10(data_dir, train=True, download=True)
    test_dataset = torchvision.datasets.CIFAR10(data_dir, train=False, download=True)
    train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    push_transform = transforms.Compose([transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset.transform = train_transform
    push_dataset.transform = push_transform
    test_dataset.transform = test_transform
    mse_loss = torch.nn.MSELoss()
    
    
    train_indices = list(range(int(len(train_dataset)*0.8)))
    val_indices = list(range(int(len(train_dataset)*0.8), len(train_dataset)))
    test_indices = list(range(len(test_dataset)))
    train_subset = Subset(train_dataset, indices=train_indices)
    push_subset = Subset(push_dataset, indices=train_indices)
    val_subset = Subset(train_dataset, indices=val_indices)
    test_subset = Subset(test_dataset, indices=test_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size)
    train_push_loader = DataLoader(push_subset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    test_loader_plots = DataLoader(test_subset, batch_size=1, shuffle=False)
    perturbations = {
        "reconstruction": Identity(),
        #"denoising": RandomNoise(noise_level=0.3),
        #"inpainting": Mask(mask_proportion=0.2),
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
            encoder = ResNet18Enc(dim_latent=dim_latent)
            decoder = ResNet18Dec(dim_latent=dim_latent)
            autoencoder = AutoEncoderMnist(
                encoder, decoder, dim_latent, pert, name=ae_model_name + f"_run{i}"
            )
            encoder.to(device)
            decoder.to(device)

            # Train the denoising autoencoder
            save_dir = (
                Path.cwd() / f"results/cifar10/predictive_performance/{ae_model_name}"
            )
            if not save_dir.exists():
                print("making dir")
                os.makedirs(save_dir)
            if not Path(os.path.join(save_dir, ae_model_name + f"_run{i}.pt")).exists():
                logging.info("fitting AE")
                autoencoder.fit(
                    device, train_loader, test_loader, save_dir, n_epochs, patience=20
                )
            else:
                print(f"{save_dir} already exists, skip fitting model")

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
                protoncoder = ProtoResNet18Enc(dim_latent=dim_latent).to(device)
                protodecoder = ProtoResNet18Dec(dim_latent=dim_latent).to(
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
                    up_dim=4,
                    img_size=32
                )
                protoautoencoder.to(device)

                save_dir = (
                    Path.cwd()
                    / f"results/cifar10/predictive_performance/{pae_model_name}"
                )
                if not save_dir.exists():
                    os.makedirs(save_dir)
                if not Path(os.path.join(save_dir, pae_model_name + f"_run{i}.pt")).exists():
                    
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
                
                if i == 0:
                    logging.info("Saving plots")
                    freeze_epoch = 90
                    prototypes_dir = save_dir / f"prototypes/{protoautoencoder.name}"
                    prototype_img_save_dir = (
                        save_dir / f"images/epoch-{freeze_epoch}-visualize"
                    )

                    ii = 0
                    print(prototypes_dir)
                    print(prototype_img_save_dir)
                    print(save_dir / f"images/reconstruction_test_img_{ii}")
                    for input_img, _ in test_loader_plots:
                        output_img_ae = autoencoder(input_img)
                        output_img, _ = protoautoencoder(input_img)
                        mse = mse_loss(input_img.to(device), output_img).mean()
                        if mse < 0.1: 
                            ii += 1
                            loc_analysis = LocalAnalysis(
                                protoautoencoder,
                                prototypes_dir,
                                epoch_number=freeze_epoch,
                                image_save_directory=prototype_img_save_dir,
                                img_size=32
                            )

                            loc_analysis.visualization(
                                input_img,
                                save_dir / f"images/reconstruction_test_img_{ii}",
                                input_img,
                                show_images=False,
                                max_prototypes=5,
                            )

                        if ii == 9:
                            break

    # Print results and save df
    for experiment_name, result_list in results_dict.items():
        logging.info(
            f"{experiment_name}: {np.array(result_list).mean():.4f} +- {np.array(result_list).std():.4f} "
        )

    df = pd.DataFrame.from_dict(results_dict)
    df.to_csv("results/cifar10/predictive_performance/results.csv")


def proto_consistency_feature_importance(
    random_seed: int = 1,
    batch_size: int = 200,
    dim_latent: int = 32,
) -> None:
    # Initialize seed and device
    set_seed(random_seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    W = 32  # Image width = height
    pert_percentages = [5, 10, 20, 50, 80, 100]
    # Here it is equivalent to perturbing the input feature from the prototype

    # Load (test) MNIST
    data_dir = Path.cwd() / "data/mnist"
    test_dataset = torchvision.datasets.CIFAR10(data_dir, train=False, download=True)
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_dataset.transform = test_transform
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False
    )

    # Initialize encoder, decoder and autoencoder wrapper
    pert = Identity() #RandomNoise(noise_level=0.3)
    n_prototypes = 256
    pae_model_name = f"PAE_reconstruction_{n_prototypes}"
    runs = 5

    
    results_data = []
    for run in range(runs):
        
        # Initialize normal encoder, decoder and autoencoder wrapper
        protoncoder = ProtoResNet18Enc(dim_latent=dim_latent).to(device)
        protodecoder = ProtoResNet18Dec(dim_latent=dim_latent).to(device)
        protoautoencoder = ProtoAutoEncoderMnist(
            protoncoder,
            protodecoder,
            prototype_shape=(n_prototypes, dim_latent, 1, 1),
            input_pert=pert,
            name=pae_model_name + f"_run{run}",
            metric="l2",
            prototype_activation_function="log",
            up_dim=4,
        )
        protoautoencoder.to(device)

        load_dir = Path.cwd() / f"results/cifar10/predictive_performance/{pae_model_name}"
        protoautoencoder.eval()
        protoautoencoder.load_state_dict(
            torch.load(load_dir / (protoautoencoder.name + ".pt")), strict=False
        )

        attr_methods = {
            "PAE": protoautoencoder,
            "Random": None,
        }
        
        logging.info(f"Logging run {run}")
        for method_name in attr_methods:
            logging.info(f"Computing feature importance with {method_name}")
            results_data.append([method_name, 0, 0])
            attr_method = attr_methods[method_name]
            if attr_method is not None:
                attr = proto_attribute(
                    protoautoencoder, test_loader, device, pert, top_k_weights=128
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
                    pert_reps = protoautoencoder.get_representations(images)
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

    save_dir = Path.cwd() / "results/cifar10/consistency_features"
    if not save_dir.exists():
        os.makedirs(save_dir)

    plt.tight_layout()
    plt.savefig(save_dir / "proto_cifar10_consistency_features.pdf")
    plt.close()
    results_df.to_csv(save_dir / "consistency_features_df.csv", index=False)


def proto_consistency_examples(
    random_seed: int = 42,
    batch_size: int = 200,
    dim_latent: int = 32,
) -> None:
    # Initialize seed and device
    set_seed(random_seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load MNIST
    data_dir = Path.cwd() / "data/mnist"
    test_dataset = torchvision.datasets.CIFAR10(data_dir, train=False, download=True)
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_dataset.transform = test_transform
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False
    )

    # Initialize encoder, decoder and autoencoder wrapper
    pert = RandomNoise()

    n_prototypes = 128
    pae_model_name = f"PAE_reconstruction_{n_prototypes}"
    runs = 5
    results_list = []
    
    for run in range(runs):
        print(f"evaluating run {run}")
        # Initialize normal encoder, decoder and autoencoder wrapper
        protoncoder = ProtoResNet18Enc(dim_latent=dim_latent).to(device)
        protodecoder = ProtoResNet18Dec(dim_latent=dim_latent).to(device)
        protoautoencoder = ProtoAutoEncoderMnist(
            protoncoder,
            protodecoder,
            prototype_shape=(n_prototypes, dim_latent, 1, 1),
            input_pert=pert,
            name=pae_model_name + f"_run{run}",
            metric="l2",
            prototype_activation_function="log",
            up_dim=4,
        )
        protoautoencoder.to(device)

        load_dir = Path.cwd() / f"results/cifar10/predictive_performance/{pae_model_name}"
        protoautoencoder.eval()
        protoautoencoder.load_state_dict(
            torch.load(load_dir / (protoautoencoder.name + ".pt")), strict=False
        )

        labels_subtest = torch.cat([label for _, label in test_loader])

        n_top_list = [1, 2, 5, 10, 20, 30, 40, 50, 80, 100, 128]
        
        logging.info(f"Now fitting explainer")
        attribution = protoautoencoder.get_prototype_importances(test_loader)

        sim_most, sim_least = proto_similarity_rates(
            attribution, 
            labels_subtest, 
            n_top_list, 
            task="reconstruction", 
            n_prototypes=n_prototypes, 
            run=run, dataset="cifar10"
        )
        results_list += [
            ["Most Important", frac, sim] for frac, sim in zip(n_top_list, sim_most)
        ]
        results_list += [
            ["Least Important", frac, sim]
            for frac, sim in zip(n_top_list, sim_least)
        ]
        
    results_df = pd.DataFrame(
        results_list,
        columns=[
            "Type of Examples",
            "Prototypes removed",
            "Similarity Rate",
        ],
    )

    save_dir = Path.cwd() / "results/cifar10/consistency_examples"
    if not save_dir.exists():
        os.makedirs(save_dir)

    logging.info(f"Saving results in {save_dir}")
    results_df.to_csv(save_dir / "metrics.csv")
    sns.lineplot(
        data=results_df,
        x="Prototypes removed",
        y="Similarity Rate",
        hue="Type of Examples",
        palette="colorblind",
    )
    plt.savefig(save_dir / "similarity_rates.pdf")


def proto_pretext_task_sensitivity(
    random_seed: int = 1,
    batch_size: int = 200,
    n_runs: int = 5,
    dim_latent: int = 32,
    subtrain_size: int = 1000,
    n_plots: int = 10,
    n_prototypes: int = 128,
) -> None:
    """Pretext experiment, adapted to the PAE model

    Args:
        random_seed (int): set random seed
        batch_size (int): size of the batches
        n_runs: number of runs (model has been previously trained - only testing)
        dim_latent (int): dimension size of the latent representation
        subtrain_size (int): size of the training subpart
        n_plots: number of qualitative plots generated
        n_prototypes: numbers of prototypes

    Returns:
        None
    """
    # Initialize seed and device
    set_seed(random_seed[0])
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ce_loss = torch.nn.CrossEntropyLoss()
    mse_loss = torch.nn.MSELoss()

    # Load MNIST
    W = 32
    data_dir = Path.cwd() / "data/mnist"
    train_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True)
    push_dataset = torchvision.datasets.MNIST(data_dir, train=True, download=True)
    test_dataset = torchvision.datasets.MNIST(data_dir, train=False, download=True)
    train_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    push_transform = transforms.Compose([transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset.transform = train_transform
    push_dataset.transform = push_transform
    test_dataset.transform = test_transform
    
    
    train_indices = list(range(int(len(train_dataset)*0.8)))
    val_indices = list(range(int(len(train_dataset)*0.8), len(train_dataset)))
    test_indices = list(range(len(test_dataset)))
    train_subset = Subset(train_dataset, indices=train_indices)
    push_subset = Subset(push_dataset, indices=train_indices)
    val_subset = Subset(train_dataset, indices=val_indices)
    test_subset = Subset(test_dataset, indices=test_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size)
    train_push_loader = DataLoader(push_subset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)

    X_train = train_dataset.data
    X_train = X_train.unsqueeze(1).float()
    X_test = test_dataset.data
    X_test = X_test.unsqueeze(1).float()
    idx_subtrain = [
        torch.nonzero(train_dataset.targets == (n % 10))[n // 10].item()
        for n in range(subtrain_size)
    ]

    # Create saving directory
    save_dir = Path.cwd() / "results/mnist/pretext"
    if not save_dir.exists():
        logging.info(f"Creating saving directory {save_dir}")
        os.makedirs(save_dir)

    # Define the computed metrics and create a csv file with appropriate headers
    pretext_dict = {
        "reconstruction": Identity(),
        "denoising": RandomNoise(noise_level=0.3),
        "inpainting": Mask(mask_proportion=0.2),
    }
    headers = [pretext.title() for pretext in pretext_dict.keys()] + [
        "Classification"
    ]  # Name of each task
    n_tasks = len(pretext_dict) + 1
    feature_pearson = np.zeros((n_runs, n_tasks, n_tasks))
    feature_spearman = np.zeros((n_runs, n_tasks, n_tasks))
    example_pearson = np.zeros((n_runs, n_tasks, n_tasks))
    example_spearman = np.zeros((n_runs, n_tasks, n_tasks))

    for run in range(n_runs):
        set_seed(random_seed[run])
        feature_importance = []
        example_importance = []
        # Perform the experiment with several autoencoders trained on different pretext tasks.
        for pretext_name, pretext in pretext_dict.items():
            # load autoencoder for the pretext task
            pae_model_name = f"PAE_{pretext_name}_{n_prototypes}"
            load_dir = (
                Path.cwd() / f"results/cifar/predictive_performance/{pae_model_name}"
            )
            protoncoder = ProtoResNet18Enc(encoded_space_dim=dim_latent).to(device)
            protodecoder = ProtoResNet18Dec(encoded_space_dim=dim_latent).to(device)
            protoautoencoder = ProtoAutoEncoderMnist(
                protoncoder,
                protodecoder,
                prototype_shape=(n_prototypes, dim_latent, 1, 1),
                input_pert=pretext,
                name=pae_model_name + f"_run{run}",
                metric="l2",
                prototype_activation_function="log",
                up_dim=4,
            )
            protoautoencoder.to(device)
            logging.info(f"Now loading {protoautoencoder.name}")
            protoautoencoder.load_state_dict(
                torch.load(load_dir / (protoautoencoder.name + ".pt")), strict=False
            )

            # Compute feature importance
            logging.info("Computing feature importance")
            baseline_image = torch.zeros((1, 1, 32, 32), device=device)
            feature_importance.append(
                # np.abs(
                np.expand_dims(
                    proto_attribute(
                        protoautoencoder,
                        test_loader,
                        device,
                        pretext,
                        top_k_weights=n_prototypes,
                    ),
                    0,
                )
                # )
            )
            # Compute example importance
            logging.info("Computing example importance")
            example_importance.append(
                np.expand_dims(
                    protoautoencoder.get_prototype_importances(test_loader), 0
                )
            )

            if run == 5:
                logging.info("Saving plots")
                freeze_epoch = 90
                prototypes_dir = load_dir / f"prototypes/{protoautoencoder.name}"
                prototype_img_save_dir = (
                    save_dir / f"{pretext_name}/epoch-{freeze_epoch}-visualize"
                )

                ii = 0
                sample_loader = DataLoader(test_subset, batch_size=1, shuffle=False)
                for img, _ in sample_loader:
                    input_img = pretext(torch.Tensor(img))
                    output_img, _ = protoautoencoder(input_img)
                    mse = mse_loss(img.to(device), output_img).mean()
                    if mse < 0.02:
                        ii += 1
                        loc_analysis = LocalAnalysis(
                            protoautoencoder,
                            prototypes_dir,
                            epoch_number=freeze_epoch,
                            image_save_directory=prototype_img_save_dir,
                        )

                        loc_analysis.visualization(
                            input_img,
                            save_dir / f"{pretext_name}/{pretext_name}_test_img_{ii}",
                            torch.Tensor(img),
                            show_images=False,
                            max_prototypes=5,
                        )

                        if ii == 9:
                            break

        # Create and fit a MNIST classifier
        classifier_name = f"classifier_{n_prototypes}"
        protoncoder = ProtoEncoderMnist(encoded_space_dim=dim_latent).to(device)
        protoclassifier = ProtoClassifierMnist(
            protoncoder,
            prototype_shape=(n_prototypes, dim_latent, 1, 1),
            name=classifier_name + f"_run{run}",
            metric="l2",
            prototype_activation_function="log",
            loss_f=ce_loss,
            n_classes=10,
        )
        logging.info(f"Now fitting {classifier_name}")
        if run == 4:
            protoclassifier.fit(
                device,
                train_loader,
                train_push_loader,
                val_loader,
                save_dir,
                100,
                patience=100,
                lr=1e-3,
                start_push_epoch=70,
                push_epoch_frequency=10,
                freeze_epoch=90,
            )
        protoclassifier.load_state_dict(
            torch.load(save_dir / (protoclassifier.name + ".pt")), strict=False
        )

        # Compute feature importance for the classifier
        logging.info("Computing feature importance")
        feature_importance.append(
            # np.abs(
            np.expand_dims(
                proto_attribute(
                    protoclassifier,
                    test_loader,
                    device,
                    pretext,
                    top_k_weights=n_prototypes,
                ),
                0,
            )
            # )
        )
        # Compute example importance for the classifier
        logging.info("Computing example importance")
        example_importance.append(
            np.expand_dims(protoautoencoder.get_prototype_importances(test_loader), 0)
        )

        # Compute correlation between the saliency of different pretext tasks
        feature_importance = np.concatenate(feature_importance)
        feature_pearson[run] = np.corrcoef(feature_importance.reshape((n_tasks, -1)))
        feature_spearman[run] = spearmanr(
            feature_importance.reshape((n_tasks, -1)), axis=1
        )[0]
        example_importance = np.concatenate(example_importance)
        example_pearson[run] = np.corrcoef(example_importance.reshape((n_tasks, -1)))
        example_spearman[run] = spearmanr(
            example_importance.reshape((n_tasks, -1)), axis=1
        )[0]
        logging.info(
            f"Run {run} complete \n Feature Pearson \n {np.round(feature_pearson[run], decimals=2)}"
            f"\n Feature Spearman \n {np.round(feature_spearman[run], decimals=2)}"
            f"\n Example Pearson \n {np.round(example_pearson[run], decimals=2)}"
            f"\n Example Spearman \n {np.round(example_spearman[run], decimals=2)}"
        )

    # Compute the avg and std for each metric
    feature_pearson_avg = np.round(np.mean(feature_pearson, axis=0), decimals=2)
    feature_pearson_std = np.round(np.std(feature_pearson, axis=0), decimals=2)
    feature_spearman_avg = np.round(np.mean(feature_spearman, axis=0), decimals=2)
    feature_spearman_std = np.round(np.std(feature_spearman, axis=0), decimals=2)
    example_pearson_avg = np.round(np.mean(example_pearson, axis=0), decimals=2)
    example_pearson_std = np.round(np.std(example_pearson, axis=0), decimals=2)
    example_spearman_avg = np.round(np.mean(example_spearman, axis=0), decimals=2)
    example_spearman_std = np.round(np.std(example_spearman, axis=0), decimals=2)

    # Format the metrics in Latex tables
    with open(save_dir / "tables.tex", "w") as f:
        for corr_avg, corr_std in zip(
            [
                feature_pearson_avg,
                feature_spearman_avg,
                example_pearson_avg,
                example_spearman_avg,
            ],
            [
                feature_pearson_std,
                feature_spearman_std,
                example_pearson_std,
                example_spearman_std,
            ],
        ):
            f.write(correlation_latex_table(corr_avg, corr_std, headers))
            f.write("\n")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="disvae")
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--random_seed", type=int, default=1)
    args = parser.parse_args()
    if args.name == "predictive_performance_and_ablation":
        predictive_performance_and_ablation(
            runs=5,
            random_seeds=[11, 19, 42, 69, 110],
            batch_size=512,
            n_epochs=100,
            dim_latent=128,
            start_push_epoch=70,
            freeze_epoch=90,
            push_epoch_frequency=10,
        )
    elif args.name == "proto_consistency_feature_importance":
        proto_consistency_feature_importance(dim_latent=128,)
    elif args.name == "proto_consistency_examples":
        proto_consistency_examples(dim_latent=128,)
    elif args.name == "proto_pretext_task_sensitivity":
        proto_pretext_task_sensitivity(
            n_runs=args.n_runs, batch_size=50, random_seed=[11, 19, 42, 69, 110]
        )
    else:
        raise ValueError("Invalid experiment name")
