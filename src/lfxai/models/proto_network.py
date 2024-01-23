import json
import logging
import pathlib
from typing import Tuple
import cv2

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from lfxai.models.pretext import Identity


from src.lfxai.utils.receptive_field import compute_proto_layer_rf_info_v2
from src.lfxai.models.push import push_prototypes


class ProtoEncoderMnist(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, encoded_space_dim, 3, stride=1, padding=0),
            nn.ReLU(True),
        )
        self.encoded_space_dim = encoded_space_dim

    def forward(self, x):
        x = self.encoder_cnn(x)
        return x


class ProtoDecoderMnist(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(encoded_space_dim, 16, 3, stride=1, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1),
        )

    def forward(self, x):
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x


class ProtoAutoEncoderMnist(nn.Module):
    def __init__(
        self,
        encoder: ProtoEncoderMnist,
        decoder: ProtoDecoderMnist,
        prototype_shape: Tuple[int],
        input_pert: callable,
        name: str = "model",
        prototype_activation_function: str = "linear",
        loss_f: callable = nn.MSELoss(),
        metric: str = "l2",
    ):
        """Class which defines model and forward pass.

        Parameters:
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).
        """
        super(ProtoAutoEncoderMnist, self).__init__()
        self.latent_dim = prototype_shape[1]
        self.prototype_shape = prototype_shape
        self.n_prototypes = prototype_shape[0]
        self.d_prototypes = prototype_shape[1]
        self.encoder = encoder
        self.decoder = decoder
        self.input_pert = input_pert
        self.name = name
        self.loss_f = loss_f
        self.prototype_activation_function = prototype_activation_function
        assert self.prototype_activation_function in ["linear", "log"]
        self.metric = metric
        assert self.metric in ["cosine", "l2"]
        self.checkpoints_files = []
        self.lr = None
        self.epsilon = 1e-6

        self.proto_layer_rf_info = compute_proto_layer_rf_info_v2(
            img_size=28,
            layer_filter_sizes=[3, 3, 3],
            layer_strides=[2, 2, 1],
            layer_paddings=[1, 1, 0],
            prototype_kernel_size=prototype_shape[2],
        )

        # Prototype Layer:
        self.protolayer = nn.Parameter(
            torch.rand(self.prototype_shape), requires_grad=True
        )
        self.ones = nn.Parameter(torch.ones(self.prototype_shape), requires_grad=False)

        self.up_layer = nn.Sequential(
            nn.ConvTranspose2d(self.d_prototypes, self.d_prototypes, 5),
            nn.BatchNorm2d(self.d_prototypes),
            nn.ReLU(True),
        )

        nn.Parameter(
            torch.rand((self.d_prototypes, self.d_prototypes, 5, 5)), requires_grad=True
        )

    def get_representations(self, x) -> torch.Tensor:
        """Returns the latent representation of the PAE, that is a weighted sum of the
            latent representations of training patches

        Args:
            x (torch.Tensor): input image

        returns:
            the flattened latent representation
        """
        x = self.encoder(x)
        distances = self.compute_distances(x)
        min_distances = -F.max_pool2d(
            -distances, kernel_size=(distances.size()[2], distances.size()[3])
        )
        min_distances = min_distances.view(-1, self.n_prototypes)
        prototype_activations = self.distance_2_similarity(min_distances)

        # the new latent variable is a weighted sum of the existing prototypes
        # the weights are determined by the similarity to the prototypes
        z = torch.einsum(
            "ij,jklm->iklm", F.softmax(prototype_activations, dim=1), self.protolayer
        )

        return z.flatten(1)

    def get_prototype_importance(self, x) -> torch.Tensor:
        """Returns the importance from every prototype for this prediction
            after going through a softmax activation

        Args:
            x (torch.Tensor): input image

        returns:
            the normalised contribution of every prototype to the latent dimension
        """
        x = self.encoder(x)
        distances = self.compute_distances(x)
        min_distances = -F.max_pool2d(
            -distances, kernel_size=(distances.size()[2], distances.size()[3])
        )
        min_distances = min_distances.view(-1, self.n_prototypes)
        prototype_activations = self.distance_2_similarity(min_distances)
        return F.softmax(prototype_activations, dim=1)

    def get_feature_importance(self, x, top_k_prototypes: int) -> np.ndarray:
        """Returns the upsampled prototype activation map, weighted by the distance of the prototypes.
            This is used as a proxy for the feature importance

        Args:
            x (torch.Tensor): input image
            top_k_prototypes (int): only use these prototypes to contribute to the activation map
        returns:
            the normalised feature importance through an upsampled prototype activation map
        """
        x = self.encoder(x)
        distances = self.compute_distances(x)
        min_distances = -F.max_pool2d(
            -distances, kernel_size=(distances.size()[2], distances.size()[3])
        ).view(-1, self.n_prototypes)

        # Activation pattern from the encoder
        activation_pattern = self.distance_2_similarity(distances).detach().numpy()

        # Weight (normalised) of every prototype
        prototype_activations = (
            F.softmax(self.distance_2_similarity(min_distances), dim=1).detach().numpy()
        )

        attributions = []
        for img, weights in zip(activation_pattern, prototype_activations):
            # sort and take weighted average of top K prototypes
            sorted_weights = np.argsort(weights)
            filter = (weights >= weights[sorted_weights[-top_k_prototypes]]).astype(
                float
            )
            filtered_weights = filter * weights
            feature_importance = np.einsum("i,ikl->kl", filtered_weights, img)
            feature_importance_upsampled = cv2.resize(
                feature_importance,
                dsize=(28, 28),
                interpolation=cv2.INTER_CUBIC,
            )
            attributions.append(feature_importance_upsampled.reshape(1, 1, 28, 28))

        return np.concatenate(attributions)

    def forward(self, x) -> Tuple[torch.Tensor]:
        """Forward pass of model. decoded image is based on similarities from the prototypes

        Parameters:
        -----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        if self.training:
            x = self.input_pert(x)
        x = self.encoder(x)
        distances = self.compute_distances(x)
        # global min pooling
        min_distances = -F.max_pool2d(
            -distances, kernel_size=(distances.size()[2], distances.size()[3])
        )
        min_distances = min_distances.view(-1, self.n_prototypes)
        prototype_activations = self.distance_2_similarity(min_distances)

        # the new latent variable is a weighted sum of the existing prototypes
        # the weights are determined by the similarity to the prototypes
        z = torch.einsum(
            "ij,jklm->iklm", F.softmax(prototype_activations, dim=1), self.protolayer
        )

        # upscale image
        z = self.up_layer(z)
        out = self.decoder(z)
        return out, min_distances

    def _cosine_convolution(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self.protolayer as cosine-similarity convolution filters on input x"""
        x_normalized = F.normalize(x, p=2, dim=1)
        p_normalized = F.normalize(self.protolayer, p=2, dim=1)
        xp = F.conv2d(input=x_normalized, weight=p_normalized)
        similarity = F.relu(xp)
        return -similarity

    def _l2_convolution(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self.protolayer as l2-convolution filters on input x"""
        x2 = x**2
        # (B, P, H, W)
        x2_patch_sum = F.conv2d(input=x2, weight=self.ones, padding="same")

        p2 = self.protolayer**2
        p2 = torch.sum(p2, dim=(1, 2, 3))
        p2_reshape = p2.view(-1, 1, 1)  #

        # (B, P, H, W)
        xp = F.conv2d(input=x, weight=self.protolayer, padding="same")
        intermediate_result = -2 * xp + p2_reshape
        distances = F.relu(x2_patch_sum + intermediate_result)

        return distances  # (B, P, H, W)

    def push_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        """this method is needed for the pushing operation"""
        conv_output = self.encoder(x)
        distances = self.compute_distances(conv_output)
        return conv_output, distances

    def compute_distances(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the similarity scores between the latent encoded image and the prototypes

        Parameters:
        -----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, latent_dim, height, width)
        """
        if self.metric == "cosine":
            prototype_distances = self._cosine_convolution(x)
        elif self.metric == "l2":
            prototype_distances = self._l2_convolution(x)
        return prototype_distances

    def distance_2_similarity(self, distances):
        if self.prototype_activation_function == "log":
            return torch.log((distances + 1) / (distances + self.epsilon))
        elif self.prototype_activation_function == "linear":
            return -distances
        else:
            return NotImplementedError

    def loss(
        self, y: torch.Tensor, y_hat: torch.Tensor, min_distances: torch.Tensor
    ) -> torch.Tensor:
        """Calculate the loss based on the reconstruction loss and the prototype Losses

        Parameters:
        -----------
        y : torch.Tensor
            ground truth. Shape (batch_size, latent_dim, height, width)
        y_hat : torch.Tensor
            prediction. Shape (batch_size, latent_dim, height, width)
        min_distances : torch.Tensor
            minimal distances from the prototypes. (batch_size, n_prototypes)
        """
        base_loss = self.loss_f(y, y_hat)
        return base_loss

    def train_epoch(
        self,
        device: torch.device,
        dataloader: torch.utils.data.DataLoader,
        push_dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        n_epoch: int,
        start_push_epoch: int,
        push_epoch_frequency: int,
        freeze_epoch: int,
    ) -> np.ndarray:
        self.train()
        train_loss = []
        # Last five epochs only fit the decoder
        if epoch > freeze_epoch:
            logging.info("Freeze Encoder and Prototypes")
            self.encoder.requires_grad = False
            self.protolayer.requires_grad = False
        for image_batch, _ in tqdm(dataloader, unit="batch", leave=False):
            image_batch = image_batch.to(device)
            recon_batch, min_distances = self.forward(image_batch)
            loss = self.loss(image_batch, recon_batch, min_distances)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.detach().cpu().numpy())
        if (
            epoch % push_epoch_frequency == 0
            and epoch >= start_push_epoch
            and epoch <= freeze_epoch
        ):
            # pass
            push_prototypes(
                push_dataloader,
                prototype_network=self,
                pert=self.input_pert,
                prototype_layer_stride=1,
                root_dir_for_saving_prototypes=f"{self.save_dir}/prototypes/{self.name}",
                epoch_number=epoch,
                prototype_img_filename_prefix="prototype-img",
                prototype_self_act_filename_prefix="prototype-self-act",
                proto_bound_boxes_filename_prefix="bb",
                log=logging.info,
            )
        return np.mean(train_loss)

    def test_epoch(self, device: torch.device, dataloader: torch.utils.data.DataLoader):
        self.eval()
        test_loss = []
        with torch.no_grad():
            for image_batch, _ in dataloader:
                image_batch = image_batch.to(device)
                pert_batch = self.input_pert(image_batch)
                recon_batch, min_distances = self.forward(pert_batch)
                loss = self.loss(image_batch, recon_batch, min_distances)
                test_loss.append(loss.cpu().numpy())
        return np.mean(test_loss)

    def fit(
        self,
        device: torch.device,
        train_loader: torch.utils.data.DataLoader,
        train_push_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        save_dir: pathlib.Path,
        n_epoch: int = 30,
        patience: int = 10,
        start_push_epoch: int = 15,
        push_epoch_frequency: int = 10,
        freeze_epoch: int = 14,
        checkpoint_interval: int = -1,
        lr: float = 1e-3,
    ) -> None:
        self.to(device)
        self.lr = lr
        self.save_dir = save_dir
        optim = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-05)
        waiting_epoch = 0
        best_test_loss = float("inf")
        for epoch in range(n_epoch):
            train_loss = self.train_epoch(
                device,
                train_loader,
                train_push_loader,
                optim,
                epoch,
                n_epoch,
                start_push_epoch,
                push_epoch_frequency,
                freeze_epoch,
            )
            test_loss = self.test_epoch(device, test_loader)
            logging.info(
                f"Epoch {epoch + 1}/{n_epoch} \t "
                f"Train loss {train_loss:.3g} \t Test loss {test_loss:.3g} \t "
            )
            if test_loss < best_test_loss and epoch > freeze_epoch:
                logging.info(f"Saving the model in {save_dir}")
                self.cpu()
                self.save(save_dir)
                self.to(device)
                best_test_loss = test_loss.data
                waiting_epoch = 0
            else:
                waiting_epoch += 1
                logging.info(
                    f"No improvement over the best epoch \t Patience {waiting_epoch} / {patience}"
                )

            if checkpoint_interval > 0 and epoch % checkpoint_interval == 0:
                n_checkpoint = 1 + epoch // checkpoint_interval
                logging.info(f"Saving checkpoint {n_checkpoint} in {save_dir}")
                path_to_checkpoint = (
                    save_dir / f"{self.name}_checkpoint{n_checkpoint}.pt"
                )
                torch.save(self.state_dict(), path_to_checkpoint)
                self.checkpoints_files.append(path_to_checkpoint)
            if waiting_epoch == patience:
                logging.info("Early stopping activated")
                break

    def get_prototype_importances(
        self,
        loader: torch.utils.data.DataLoader,
    ) -> np.ndarray:
        """Returns the importance of each prototype in the loader

        Args:
            loader (DataLoader): a Datalaoder with N datapoints
            pert (Callable): perturbation on the images

        Returns:
            array of prototype importances for every prediction [N, n_prototypes]
        """
        importances = []
        for img, _ in loader:
            input_img = self.input_pert(img)
            prototype_importances = self.get_prototype_importance(input_img)
            importances.append(prototype_importances.detach().numpy())

        return np.vstack(importances)

    def save(self, directory: pathlib.Path) -> None:
        """Save a model and corresponding metadata.

        Parameters:
        -----------
        directory : pathlib.Path
            Path to the directory where to save the data.
        """
        model_name = self.name
        self.save_metadata(directory)
        path_to_model = directory / (model_name + ".pt")
        torch.save(self.state_dict(), path_to_model)

    def load_metadata(self, directory: pathlib.Path) -> dict:
        """Load the metadata of a training directory.

        Parameters:
        -----------
        directory : pathlib.Path
            Path to folder where model is saved. For example './experiments/mnist'.
        """
        path_to_metadata = directory / (self.name + ".json")

        with open(path_to_metadata) as metadata_file:
            metadata = json.load(metadata_file)
        return metadata

    def save_metadata(self, directory: pathlib.Path, **kwargs) -> None:
        """Load the metadata of a training directory.

        Parameters:
        -----------
        directory: string
            Path to folder where to save model. For example './experiments/mnist'.
        kwargs:
            Additional arguments to `json.dump`
        """
        path_to_metadata = directory / (self.name + ".json")
        metadata = {"latent_dim": self.latent_dim, "name": self.name}
        with open(path_to_metadata, "w") as f:
            json.dump(metadata, f, indent=4, sort_keys=True, **kwargs)


class ProtoClassifierMnist(nn.Module):
    def __init__(
        self,
        encoder: ProtoEncoderMnist,
        prototype_shape: Tuple[int],
        n_classes: int,
        name: str = "model",
        prototype_activation_function: str = "linear",
        loss_f: callable = nn.MSELoss(),
        metric: str = "l2",
    ):
        """Class which defines model and forward pass.

        Parameters:
        ----------
        img_size : tuple of ints
            Size of images. E.g. (1, 32, 32) or (3, 64, 64).
        """
        super(ProtoClassifierMnist, self).__init__()
        self.latent_dim = prototype_shape[1]
        self.prototype_shape = prototype_shape
        self.n_prototypes = prototype_shape[0]
        self.d_prototypes = prototype_shape[1]
        self.n_classes = n_classes
        self.encoder = encoder
        self.name = name
        self.loss_f = loss_f
        self.prototype_activation_function = prototype_activation_function
        assert self.prototype_activation_function in ["linear", "log"]
        self.metric = metric
        assert self.metric in ["cosine", "l2"]
        self.checkpoints_files = []
        self.lr = None
        self.epsilon = 1e-6

        self.proto_layer_rf_info = compute_proto_layer_rf_info_v2(
            img_size=28,
            layer_filter_sizes=[3, 3, 3],
            layer_strides=[2, 2, 1],
            layer_paddings=[1, 1, 0],
            prototype_kernel_size=prototype_shape[2],
        )

        # Prototype Layer:
        self.protolayer = nn.Parameter(
            torch.rand(self.prototype_shape), requires_grad=True
        )
        self.ones = nn.Parameter(torch.ones(self.prototype_shape), requires_grad=False)

        self.up_layer = nn.Sequential(
            nn.ConvTranspose2d(self.d_prototypes, self.d_prototypes, 5),
            nn.BatchNorm2d(self.d_prototypes),
            nn.ReLU(True),
        )

        nn.Parameter(
            torch.rand((self.d_prototypes, self.d_prototypes, 5, 5)), requires_grad=True
        )

        self.last_layer = nn.Linear(
            self.n_prototypes, self.n_classes, bias=False
        )  # do not use bias

    def get_representations(self, x) -> torch.Tensor:
        """Returns the latent representation of the PAE, that is a weighted sum of the
            latent representations of training patches

        Args:
            x (torch.Tensor): input image

        returns:
            the flattened latent representation
        """
        if self.training:
            x = self.input_pert(x)
        x = self.encoder(x)
        distances = self.compute_distances(x)
        min_distances = -F.max_pool2d(
            -distances, kernel_size=(distances.size()[2], distances.size()[3])
        )
        min_distances = min_distances.view(-1, self.n_prototypes)
        prototype_activations = self.distance_2_similarity(min_distances)
        return prototype_activations

    def get_prototype_importance(self, x) -> torch.Tensor:
        """In the classifier case, the latent vector is the same as the feature importances

        Args:
            x (torch.Tensor): input image

        returns:
            the normalised contribution of every prototype to the latent dimension
        """
        return self.get_representations(x)

    def get_feature_importance(self, x, top_k_prototypes: int) -> np.ndarray:
        """Returns the upsampled prototype activation map, weighted by the distance of the prototypes.
            This is used as a proxy for the feature importance

        Args:
            x (torch.Tensor): input image
            top_k_prototypes (int): only use these prototypes to contribute to the activation map
        returns:
            the normalised feature importance through an upsampled prototype activation map
        """
        x = self.encoder(x)
        distances = self.compute_distances(x)
        min_distances = -F.max_pool2d(
            -distances, kernel_size=(distances.size()[2], distances.size()[3])
        ).view(-1, self.n_prototypes)

        # Activation pattern from the encoder
        activation_pattern = self.distance_2_similarity(distances).detach().numpy()

        # Weight (normalised) of every prototype
        prototype_activations = (
            F.softmax(self.distance_2_similarity(min_distances), dim=1).detach().numpy()
        )

        attributions = []
        for img, weights in zip(activation_pattern, prototype_activations):
            # sort and take weighted average of top K prototypes
            sorted_weights = np.argsort(weights)
            filter = (weights >= weights[sorted_weights[-top_k_prototypes]]).astype(
                float
            )
            filtered_weights = filter * weights
            feature_importance = np.einsum("i,ikl->kl", filtered_weights, img)
            feature_importance_upsampled = cv2.resize(
                feature_importance,
                dsize=(28, 28),
                interpolation=cv2.INTER_CUBIC,
            )
            attributions.append(feature_importance_upsampled.reshape(1, 1, 28, 28))

        return np.concatenate(attributions)

    def forward(self, x) -> Tuple[torch.Tensor]:
        """Forward pass of model. output classification is based on similarities from the prototypes

        Parameters:
        -----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, n_chan, height, width)
        """
        x = self.encoder(x)
        distances = self.compute_distances(x)
        min_distances = -F.max_pool2d(
            -distances, kernel_size=(distances.size()[2], distances.size()[3])
        )
        min_distances = min_distances.view(-1, self.n_prototypes)
        prototype_activations = self.distance_2_similarity(min_distances)
        out = self.last_layer(prototype_activations)
        return out, min_distances

    def _cosine_convolution(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self.protolayer as cosine-similarity convolution filters on input x"""
        x_normalized = F.normalize(x, p=2, dim=1)
        p_normalized = F.normalize(self.protolayer, p=2, dim=1)
        xp = F.conv2d(input=x_normalized, weight=p_normalized)
        similarity = F.relu(xp)
        return -similarity

    def _l2_convolution(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self.protolayer as l2-convolution filters on input x"""
        x2 = x**2
        # (B, P, H, W)
        x2_patch_sum = F.conv2d(input=x2, weight=self.ones, padding="same")

        p2 = self.protolayer**2
        p2 = torch.sum(p2, dim=(1, 2, 3))
        p2_reshape = p2.view(-1, 1, 1)  #

        # (B, P, H, W)
        xp = F.conv2d(input=x, weight=self.protolayer, padding="same")
        intermediate_result = -2 * xp + p2_reshape
        distances = F.relu(x2_patch_sum + intermediate_result)

        return distances  # (B, P, H, W)

    def push_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        """this method is needed for the pushing operation"""
        conv_output = self.encoder(x)
        distances = self.compute_distances(conv_output)
        return conv_output, distances

    def compute_distances(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the similarity scores between the latent encoded image and the prototypes

        Parameters:
        -----------
        x : torch.Tensor
            Batch of data. Shape (batch_size, latent_dim, height, width)
        """
        if self.metric == "cosine":
            prototype_distances = self._cosine_convolution(x)
        elif self.metric == "l2":
            prototype_distances = self._l2_convolution(x)
        return prototype_distances

    def distance_2_similarity(self, distances):
        if self.prototype_activation_function == "log":
            return torch.log((distances + 1) / (distances + self.epsilon))
        elif self.prototype_activation_function == "linear":
            return -distances
        else:
            return NotImplementedError

    def proto_loss(
        self,
        y: torch.Tensor,
        y_hat: torch.Tensor,
        min_distances: torch.Tensor,
        class_specific: bool = True,
    ) -> Tuple[torch.Tensor]:
        """Calculate the various losses of the prototypes

        Parameters:
        -----------
            y (torch.Tensor): Groundtruth class
            y_hat (torch.Tensor): Logit prediction
            min_distances (torch.Tensor): prototype latent variable
            class_specific (boolean): ensure that every prototype is assigned a class

        Returns:
            the various loss values
        """
        min_distance, _ = torch.min(min_distances, dim=1)
        cluster_cost = torch.mean(min_distance)
        l1 = self.last_layer.weight.norm(p=1)
        return cluster_cost, l1

    def loss(
        self, y: torch.Tensor, y_hat: torch.Tensor, min_distances: torch.Tensor
    ) -> torch.Tensor:
        """Calculate the loss based on the reconstruction loss and the prototype Losses

        Parameters:
        -----------
        y : torch.Tensor
            ground truth. Shape (batch_size, latent_dim, height, width)
        y_hat : torch.Tensor
            prediction. Shape (batch_size, latent_dim, height, width)
        min_distances : torch.Tensor
            minimal distances from the prototypes. (batch_size, n_prototypes)
        """
        lambda_0 = 1
        lambda_1 = 0  # 0.8
        lambda_2 = 1e-4
        base_loss = self.loss_f(y_hat, y)
        cluster_loss, l1 = self.proto_loss(y, y_hat, min_distances)
        return lambda_0 * base_loss + lambda_1 * cluster_loss + lambda_2 * l1

    def train_epoch(
        self,
        device: torch.device,
        dataloader: torch.utils.data.DataLoader,
        push_dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        n_epoch: int,
        start_push_epoch: int,
        push_epoch_frequency: int,
        freeze_epoch: int,
    ) -> np.ndarray:
        self.train()
        train_loss = []
        # Last five epochs only fit the decoder
        if epoch > freeze_epoch:
            logging.info("Freeze Encoder and Prototypes")
            self.encoder.requires_grad = False
            self.protolayer.requires_grad = False
        for image_batch, y in tqdm(dataloader, unit="batch", leave=False):
            image_batch = image_batch.to(device)
            out, min_distances = self.forward(image_batch)
            loss = self.loss(torch.Tensor(y).long(), out, min_distances)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.detach().cpu().numpy())
        if (
            epoch % push_epoch_frequency == 0
            and epoch >= start_push_epoch
            and epoch <= freeze_epoch
        ):
            # pass
            push_prototypes(
                push_dataloader,
                prototype_network=self,
                pert=Identity(),
                prototype_layer_stride=1,
                root_dir_for_saving_prototypes=f"{self.save_dir}/prototypes/{self.name}",
                epoch_number=epoch,
                prototype_img_filename_prefix="prototype-img",
                prototype_self_act_filename_prefix="prototype-self-act",
                proto_bound_boxes_filename_prefix="bb",
                log=logging.info,
            )
        return np.mean(train_loss)

    def test_epoch(self, device: torch.device, dataloader: torch.utils.data.DataLoader):
        self.eval()
        test_loss = []
        with torch.no_grad():
            for image_batch, y in dataloader:
                image_batch = image_batch.to(device)
                out, min_distances = self.forward(image_batch)
                loss = self.loss(torch.Tensor(y).long(), out, min_distances)
                test_loss.append(loss.cpu().numpy())
        return np.mean(test_loss)

    def fit(
        self,
        device: torch.device,
        train_loader: torch.utils.data.DataLoader,
        train_push_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        save_dir: pathlib.Path,
        n_epoch: int = 30,
        patience: int = 10,
        start_push_epoch: int = 15,
        push_epoch_frequency: int = 10,
        freeze_epoch: int = 14,
        checkpoint_interval: int = -1,
        lr: float = 1e-3,
    ) -> None:
        self.to(device)
        self.lr = lr
        self.save_dir = save_dir
        optim = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-05)
        waiting_epoch = 0
        best_test_loss = float("inf")
        for epoch in range(n_epoch):
            train_loss = self.train_epoch(
                device,
                train_loader,
                train_push_loader,
                optim,
                epoch,
                n_epoch,
                start_push_epoch,
                push_epoch_frequency,
                freeze_epoch,
            )
            test_loss = self.test_epoch(device, test_loader)
            logging.info(
                f"Epoch {epoch + 1}/{n_epoch} \t "
                f"Train loss {train_loss:.3g} \t Test loss {test_loss:.3g} \t "
            )
            if test_loss < best_test_loss and epoch > freeze_epoch:
                logging.info(f"Saving the model in {save_dir}")
                self.cpu()
                self.save(save_dir)
                self.to(device)
                best_test_loss = test_loss.data
                waiting_epoch = 0
            else:
                waiting_epoch += 1
                logging.info(
                    f"No improvement over the best epoch \t Patience {waiting_epoch} / {patience}"
                )

            if checkpoint_interval > 0 and epoch % checkpoint_interval == 0:
                n_checkpoint = 1 + epoch // checkpoint_interval
                logging.info(f"Saving checkpoint {n_checkpoint} in {save_dir}")
                path_to_checkpoint = (
                    save_dir / f"{self.name}_checkpoint{n_checkpoint}.pt"
                )
                torch.save(self.state_dict(), path_to_checkpoint)
                self.checkpoints_files.append(path_to_checkpoint)
            if waiting_epoch == patience:
                logging.info("Early stopping activated")
                break

    def get_prototype_importances(
        self,
        loader: torch.utils.data.DataLoader,
    ) -> np.ndarray:
        """Returns the importance of each prototype in the loader

        Args:
            loader (DataLoader): a Datalaoder with N datapoints
            pert (Callable): perturbation on the images

        Returns:
            array of prototype importances for every prediction [N, n_prototypes]
        """
        importances = []
        for img, _ in loader:
            input_img = self.input_pert(img)
            prototype_importances = self.get_prototype_importance(input_img)
            importances.append(prototype_importances.detach().numpy())

        return np.vstack(importances)

    def save(self, directory: pathlib.Path) -> None:
        """Save a model and corresponding metadata.

        Parameters:
        -----------
        directory : pathlib.Path
            Path to the directory where to save the data.
        """
        model_name = self.name
        self.save_metadata(directory)
        path_to_model = directory / (model_name + ".pt")
        torch.save(self.state_dict(), path_to_model)

    def load_metadata(self, directory: pathlib.Path) -> dict:
        """Load the metadata of a training directory.

        Parameters:
        -----------
        directory : pathlib.Path
            Path to folder where model is saved. For example './experiments/mnist'.
        """
        path_to_metadata = directory / (self.name + ".json")

        with open(path_to_metadata) as metadata_file:
            metadata = json.load(metadata_file)
        return metadata

    def save_metadata(self, directory: pathlib.Path, **kwargs) -> None:
        """Load the metadata of a training directory.

        Parameters:
        -----------
        directory: string
            Path to folder where to save model. For example './experiments/mnist'.
        kwargs:
            Additional arguments to `json.dump`
        """
        path_to_metadata = directory / (self.name + ".json")
        metadata = {"latent_dim": self.latent_dim, "name": self.name}
        with open(path_to_metadata, "w") as f:
            json.dump(metadata, f, indent=4, sort_keys=True, **kwargs)
