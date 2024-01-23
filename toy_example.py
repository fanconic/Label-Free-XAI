import torch
from pathlib import Path
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torch.nn import MSELoss
from captum.attr import IntegratedGradients

from src.lfxai.models.images import (
    AutoEncoderMnist,
    EncoderMnist,
    DecoderMnist,
    ProtoAutoEncoderMnist,
)
from src.lfxai.models.pretext import Identity
from src.lfxai.explanations.features import attribute_auxiliary
from src.lfxai.explanations.examples import SimplEx

# Select torch device
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load data
data_dir = Path.cwd() / "data/mnist"
train_dataset = MNIST(data_dir, train=True, download=True)
test_dataset = MNIST(data_dir, train=False, download=True)
train_dataset.transform = transforms.Compose([transforms.ToTensor()])
test_dataset.transform = transforms.Compose([transforms.ToTensor()])
train_loader = DataLoader(train_dataset, batch_size=100)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

# Get a model
encoder = EncoderMnist(encoded_space_dim=10)
decoder = DecoderMnist(encoded_space_dim=10)
model = AutoEncoderMnist(encoder, decoder, latent_dim=10, input_pert=Identity())
model.to(device)

# Get Prototype Autoencoder model
encoder = EncoderMnist(encoded_space_dim=10)
decoder = DecoderMnist(encoded_space_dim=10)
model_proto = ProtoAutoEncoderMnist(
    encoder,
    decoder,
    latent_dim=10,
    n_prototypes=100,
    input_pert=Identity(),
    name="ProtoAE",
)
model_proto.to(device)


# Get label-free feature importance
print("Get label-free feature importance")
baseline = torch.zeros((1, 1, 28, 28)).to(device)  # black image as baseline
attr_method = IntegratedGradients(model)
feature_importance = attribute_auxiliary(
    encoder, test_loader, device, attr_method, baseline
)

# Get label-free example importance
print("Get label-free example importance")
train_subset = Subset(
    train_dataset, indices=list(range(200))
)  # Limit the number of training examples
train_subloader = DataLoader(train_subset, batch_size=20)
attr_method = SimplEx(model, loss_f=MSELoss())
example_importance = attr_method.attribute_loader(device, train_subloader, test_loader)
