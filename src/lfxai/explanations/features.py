import numpy as np
import torch
from captum.attr import Attribution, Saliency
from torch.nn import Module
import torch.nn.functional as F
import cv2


class AuxiliaryFunction(Module):
    def __init__(self, black_box: Module, base_features: torch.Tensor) -> None:
        super().__init__()
        self.black_box = black_box
        self.base_features = base_features
        self.prediction = black_box(base_features)

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        if len(self.prediction) == len(input_features):
            return torch.sum(
                self.prediction * self.black_box(input_features), dim=-1
            )  # This is the juicy part: g_x(x_tilde)
        elif len(input_features) % len(self.prediction) == 0:
            n_repeat = int(len(input_features) / len(self.prediction))
            return torch.sum(
                self.prediction.repeat(n_repeat, 1) * self.black_box(input_features),
                dim=-1,
            )
        else:
            raise ValueError(
                "The internal batch size should be a multiple of input_features.shape[0]"
            )


def attribute_individual_dim(
    encoder: callable,
    dim_latent: int,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    attr_method: Attribution,
    baseline: torch.Tensor,
) -> np.ndarray:
    attributions = []
    latents = []
    for input_batch, _ in data_loader:
        input_batch = input_batch.to(device)
        attributions_batch = []
        latents.append(encoder(input_batch).detach().cpu().numpy())
        for dim in range(dim_latent):
            attribution = (
                attr_method.attribute(input_batch, baseline, target=dim)
                .detach()
                .cpu()
                .numpy()
            )
            attributions_batch.append(attribution)
        attributions.append(np.concatenate(attributions_batch, axis=1))
    latents = np.concatenate(latents)
    attributions = np.concatenate(attributions)
    attributions = np.abs(np.expand_dims(latents, (2, 3)) * attributions)
    return attributions


def attribute_auxiliary(
    encoder: Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    attr_method: Attribution,
    baseline=None,
) -> np.ndarray:
    attributions = []
    for inputs, _ in data_loader:
        inputs = inputs.to(device)
        auxiliary_encoder = AuxiliaryFunction(encoder, inputs)
        attr_method.forward_func = auxiliary_encoder
        if isinstance(attr_method, Saliency):
            attributions.append(attr_method.attribute(inputs).detach().cpu().numpy())
        else:
            if isinstance(baseline, torch.Tensor):
                attributions.append(
                    attr_method.attribute(inputs, baseline).detach().cpu().numpy()
                )
            elif isinstance(baseline, Module):
                baseline_inputs = baseline(inputs)
                attributions.append(
                    attr_method.attribute(inputs, baseline_inputs)
                    .detach()
                    .cpu()
                    .numpy()
                )
            else:
                attributions.append(
                    attr_method.attribute(inputs).detach().cpu().numpy()
                )
    return np.concatenate(attributions)


def proto_attribute(
    ppnet: Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    pert=None,
    top_k_weights=10,
) -> np.ndarray:
    attributions = []
    for inputs, _ in data_loader:
        inputs = pert(inputs).to(device)

        # Forward the image variable through the network
        _, distances = ppnet.push_forward(inputs)
        activation_pattern = ppnet.distance_2_similarity(distances)
        activation_pattern = activation_pattern.detach().numpy()

        _, min_distances = ppnet(inputs)
        prototype_activations = (
            F.softmax(ppnet.distance_2_similarity(min_distances), dim=1)
            .detach()
            .numpy()
        )

        for img, weights in zip(activation_pattern, prototype_activations):
            # sort and take weighted average of top K prototypes
            sorted_weights = np.argsort(weights)
            filter = (weights >= weights[sorted_weights[-top_k_weights]]).astype(float)
            filtered_weights = filter * weights
            feature_importance = np.einsum("i,ikl->kl", filtered_weights, img)
            feature_importance_upsampled = cv2.resize(
                feature_importance,
                dsize=(28, 28),
                interpolation=cv2.INTER_CUBIC,
            )

            """feature_importance_upsampled_scaled = (
                feature_importance_upsampled - feature_importance_upsampled.min()
            ) / (
                feature_importance_upsampled.max() - feature_importance_upsampled.min()
            )"""
            attributions.append(feature_importance_upsampled.reshape(1, 1, 28, 28))

    return np.concatenate(attributions)
