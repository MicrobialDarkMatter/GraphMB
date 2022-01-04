import os
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np
from torch.utils.data import Dataset

from tqdm import tqdm

# based on https://github.com/Jackson-Kang/Pytorch-VAE-tutorial and VAMB


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, activation):
        super(Encoder, self).__init__()

        # TODO generalize n layers
        self.fc1 = nn.Linear(input_dim, hidden_dim[0])
        self.fc1_norm = nn.BatchNorm1d(hidden_dim[0])
        self.fc2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.fc2_norm = nn.BatchNorm1d(hidden_dim[1])
        self.fc_mean = nn.Linear(hidden_dim[1], latent_dim)
        self.fc_var = nn.Linear(hidden_dim[1], latent_dim)

        # self.LeakyReLU =nn.LeakyReLU(0.2)
        self.activation = activation
        self.softplus = nn.Softplus()

    def forward(self, x):
        # TODO generalize n layers
        h = self.fc1_norm(self.activation(self.fc1(x)))
        h = self.fc2_norm(self.activation(self.fc2(h)))
        mean = self.fc_mean(h)
        log_var = self.softplus(self.fc_var(h))

        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, activation):
        super(Decoder, self).__init__()
        # TODO generalize n layers
        self.fc1 = nn.Linear(latent_dim, hidden_dim[1])
        self.fc1_norm = nn.BatchNorm1d(hidden_dim[1])
        self.fc2 = nn.Linear(hidden_dim[1], hidden_dim[0])
        self.fc2_norm = nn.BatchNorm1d(hidden_dim[0])
        self.fc_output = nn.Linear(hidden_dim[0], output_dim)

        self.activation = activation

    def forward(self, x):
        # TODO generalize n layers
        h = self.fc1_norm(self.activation(self.fc1(x)))
        h = self.fc2_norm(self.activation(self.fc2(h)))

        x_hat = torch.sigmoid(self.fc_output(h))
        # x_hat = self.fc_output(h)
        return x_hat


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, activation, device):
        super(VAE, self).__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
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
        # z = self.reparameterization(mean, log_var)  # takes exponential function (log var -> var)
        x_hat = self.Decoder(z)

        return x_hat, mean, log_var


def loss_function(x, x_hat, mean, log_var, nlatent, alpha, beta, nsamples=1, ntnf=136):
    # breakpoint()
    ab_reproduction_loss = nn.functional.cross_entropy(x_hat[:, -1], x[:, -1], reduction="sum")
    comp_reproduction_loss = nn.functional.mse_loss(x_hat[:, :-1], x[:, :-1], reduction="mean")
    # reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return ab_reproduction_loss * 0.5 + comp_reproduction_loss * 0.5  # + KLD


def calc_loss(x, x_hat, mu, logsigma, nlatent, alpha, beta, nsamples=1, ntnf=136):
    # from VAMB

    tnf_in = x.narrow(1, 0, ntnf)
    depths_in = x.narrow(1, ntnf, nsamples)
    tnf_out = x_hat.narrow(1, 0, ntnf)
    depths_out = x_hat.narrow(1, ntnf, nsamples)

    # If multiple samples, use cross entropy, else use SSE for abundance
    if nsamples > 1:
        # breakpoint()
        depths_out = torch.nn.functional.softmax(depths_out, dim=1)
        # Add 1e-9 to depths_out to avoid numerical instability.
        depth_loss = -((depths_out + 1e-9).log() * depths_in).sum(dim=1).mean()
        depth_weight = (1 - alpha) / math.log(nsamples)
    else:
        depth_loss = nn.functional.mse_loss(depths_out, depths_in, reduction="mean")
        depth_weight = 1 - alpha
    tnf_loss = nn.functional.mse_loss(tnf_out, tnf_in, reduction="none").sum(dim=1).mean()
    kld = -0.5 * (1 + logsigma - mu.pow(2) - logsigma.exp()).sum(dim=1).mean()
    tnf_weight = alpha / x[:, :-1].shape[0]
    kld_weight = 1 / (nlatent * beta)
    loss = depth_loss * depth_weight + tnf_loss * tnf_weight + kld * kld_weight

    return loss, depth_loss, tnf_loss, kld


class ContigDataset(Dataset):
    def __init__(self, contigids, kmers, abundances):
        self.features = torch.cat((kmers, abundances), dim=1)
        self.contigids = contigids
        assert len(contigids) == len(self.features)

    def __len__(self):
        return len(self.contigids)

    def __getitem__(self, idx):
        return self.features[idx]


def train_vae(logger, train_loader, model, lr, epochs, device, alpha=0.5, beta=200, nsamples=1):
    logger.info("Start training VAE...")
    model.train()
    optimizer = Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, x in enumerate(train_loader):
            # x = x.view(batch_size, x_dim)
            x = x.to(device)
            # x.requires_grad = True
            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            # loss = loss_function(x, x_hat, mean, log_var)
            loss, dloss, tloss, kloss = calc_loss(
                x,
                x_hat,
                mean,
                log_var,
                model.latent_dim,
                alpha=alpha,
                beta=beta,
                nsamples=nsamples,
                ntnf=x.shape[1] - nsamples,
            )

            loss.backward()
            optimizer.step()
            overall_loss += loss.item()

        logger.info(f"\tEpoch {epoch + 1} complete! \tAverage Loss: {overall_loss / len(train_loader)}")

    logger.info("Finish!!")
    return model


def run_vae(
    outdir,
    contigids,
    kmers,
    abundance,
    logger,
    device,
    batchsteps,
    batchsize,
    nepochs,
    nhidden,
    nlatent,
    lr,
):
    # use same strategy as VAMB
    if abundance.shape[1] > 1:
        alpha = 0.15
    else:
        alpha = 0.5
    beta = 200
    logger.info(f"using alpha {alpha}, {abundance.shape[1]} samples")
    input_dim = kmers.shape[1] + abundance.shape[1]
    model = VAE(
        input_dim=input_dim,
        hidden_dim=nhidden,
        latent_dim=nlatent,
        activation=nn.LeakyReLU(),
        device=device,
    )
    model.to(device)
    train_dataset = ContigDataset(contigids, kmers, abundance)
    kwargs = {"num_workers": 1, "pin_memory": True}
    train_loader = DataLoader(dataset=train_dataset, batch_size=batchsize, shuffle=True, **kwargs)
    final_model = train_vae(
        logger, train_loader, model, lr, nepochs, device, alpha=alpha, beta=beta, nsamples=abundance.shape[1]
    )
    # run again to get train_embs
    # breakpoint()
    logger.info(f"encoding {train_dataset.features.shape}")
    x_hat, mean, log_var = final_model(train_dataset.features.to(device))
    train_embs = mean.detach().cpu().numpy()
    with open(os.path.join(outdir, "embs.tsv"), "w") as embsfile:
        for (contig, emb) in zip(contigids, train_embs):
            embsfile.write(contig + "\t" + "\t".join(map(str, emb)) + "\n")
    return mean
