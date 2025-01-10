r"""Score modules"""

import math
from typing import Callable, Union

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from .nn import UNet


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding.to(dtype=timesteps.dtype, device=timesteps.device)


class ScoreUNet(nn.Module):
    r"""Creates a U-Net score network.

    Arguments:
        channels: The number of channels.
        forcing: The number of forcing channels.
        embedding: The number of time embedding features.
    """

    def __init__(self, channels, embedding_dim, forcing_dim=0, **kwargs):
        super().__init__()

        self.map_forcing = (
            torch.nn.Linear(forcing_dim, embedding_dim) if forcing_dim > 0 else None
        )

        self.embedding_dim = embedding_dim
        self.noise_features = 32
        self.unet = UNet(channels, channels, embedding_dim, **kwargs)
        self.map_layer0 = torch.nn.Linear(self.noise_features, embedding_dim)
        self.map_layer1 = torch.nn.Linear(embedding_dim, embedding_dim)

    def forward(self, x, t, forcing=None):
        assert (forcing is None) or (self.map_forcing is not None)
        t = t.reshape(-1)
        emb = timestep_embedding(t, self.noise_features)
        emb = torch.nn.functional.silu(self.map_layer0(emb))
        emb = self.map_layer1(emb)
        if self.map_forcing is not None:
            emb = emb + self.map_forcing(forcing)
        emb = torch.nn.functional.silu(emb)

        out = self.unet(x, emb)
        return out.reshape(x.shape)


class GaussianScore(nn.Module):
    r"""Creates a score module for Gaussian inverse problems.

    .. math:: p(y | x) = N(y | A(x), Σ)

    Note:
        This module returns :math:`-\sigma(t) s(x(t), t | y)`.
    """

    def __init__(
        self,
        y: Tensor,
        A: Callable[[Tensor], Tensor],
        std: Union[float, Tensor],
        sde,
        gamma: Union[float, Tensor] = 1e-2,
        detach: bool = False,
    ):
        super().__init__()

        self.register_buffer("y", y)
        self.register_buffer("std", torch.as_tensor(std))
        self.register_buffer("gamma", torch.as_tensor(gamma))

        self.A = A
        self.sde = sde
        self.detach = detach

    def forward(self, x: Tensor, t: Tensor, c: Tensor = None) -> Tensor:
        mu, sigma = self.sde.mu(t), self.sde.sigma(t)

        if self.detach:
            eps = self.sde.eps(x, t, c)

        with torch.enable_grad():
            x = x.detach().requires_grad_(True)

            if not self.detach:
                eps = self.sde.eps(x, t, c)

            x_ = (x - sigma * eps) / mu

            err = self.y - self.A(x_)
            var = self.std**2 + self.gamma * (sigma / mu) ** 2

            log_p = -(err**2 / var).sum() / 2

        (s,) = torch.autograd.grad(log_p, x)

        return eps - sigma * s
