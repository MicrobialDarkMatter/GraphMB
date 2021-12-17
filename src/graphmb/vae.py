import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np
from torch.utils.data import Dataset

from tqdm import tqdm

# based on https://github.com/Jackson-Kang/Pytorch-VAE-tutorial


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, activation):
        super(Encoder, self).__init__()

        # TODO generalize n layers
        self.FC_input = nn.Linear(input_dim, hidden_dim[0])
        self.FC_input2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.FC_mean = nn.Linear(hidden_dim[1], latent_dim)
        self.FC_var = nn.Linear(hidden_dim[1], latent_dim)

        # self.LeakyReLU =nn.LeakyReLU(0.2)
        self.activation = activation

        self.training = True

    def forward(self, x):
        # TODO generalize n layers
        h_ = self.activation(self.FC_input(x))
        h_ = self.activation(self.FC_input2(h_))
        mean = self.FC_mean(h_)
        log_var = self.FC_var(h_)  # encoder produces mean and log of variance
        #             (i.e., parateters of simple tractable normal distribution "q"

        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, activation):
        super(Decoder, self).__init__()
        # TODO generalize n layers
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim[1])
        self.FC_hidden2 = nn.Linear(hidden_dim[1], hidden_dim[0])
        self.FC_output = nn.Linear(hidden_dim[0], output_dim)

        self.activation = activation

    def forward(self, x):
        # TODO generalize n layers
        h = self.activation(self.FC_hidden(x))
        h = self.activation(self.FC_hidden2(h))

        x_hat = torch.sigmoid(self.FC_output(h))
        return x_hat


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, activation, device):
        super(VAE, self).__init__()
        self.Encoder = Encoder(
            input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, activation=activation
        )
        self.Decoder = Decoder(
            latent_dim=latent_dim, hidden_dim=hidden_dim, output_dim=input_dim, activation=activation
        )
        self.device = device

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)  # sampling epsilon
        z = mean + var * epsilon  # reparameterization trick
        return z

    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))  # takes exponential function (log var -> var)
        x_hat = self.Decoder(z)

        return x_hat, mean, log_var


def loss_function(x, x_hat, mean, log_var):
    breakpoint()
    ab_reproduction_loss = nn.functional.binary_cross_entropy(x_hat[:, -1], x[:, -1], reduction="sum")
    comp_reproduction_loss = nn.functional.mse_loss(x_hat[:, :-1], x[:, :-1], reduction="mean")
    # reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return ab_reproduction_loss * 0.85 + comp_reproduction_loss * 0.15  # + KLD


class ContigDataset(Dataset):
    def __init__(self, contigids, kmers, abundances):
        self.features = torch.cat((kmers, abundances), dim=1)
        self.contigids = contigids
        assert len(contigids) == len(self.features)

    def __len__(self):
        return len(self.contigids)

    def __getitem__(self, idx):
        return self.features[idx]


def train_vae(x_dim, train_loader, model, batch_size, lr, epochs, device):
    print("Start training VAE...")
    model.train()
    optimizer = Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, x in enumerate(train_loader):
            # x = x.view(batch_size, x_dim)
            x = x.to(device)

            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)

            overall_loss += loss.item()

            loss.backward()
            optimizer.step()

        print("\tEpoch", epoch + 1, "complete!", "\tAverage Loss: ", overall_loss / (batch_idx * batch_size))

    print("Finish!!")
    return mean


def run_vae(outdir, contigids, kmers, abundance, logfile, cuda, batchsteps, batchsize, nepochs, nhidden, nlatent, lr):

    input_dim = kmers.shape[1] + abundance.shape[1]
    if cuda is False:
        cuda = "cpu"
    model = VAE(
        input_dim=input_dim,
        hidden_dim=nhidden,
        latent_dim=nlatent,
        activation=nn.ReLU(),
        device=cuda,
    )
    train_dataset = ContigDataset(contigids, kmers, abundance)
    kwargs = {"num_workers": 1, "pin_memory": True}
    train_loader = DataLoader(dataset=train_dataset, batch_size=batchsize, shuffle=True, **kwargs)
    train_embs = train_vae(input_dim, train_loader, model, batchsize, lr, nepochs, cuda)
