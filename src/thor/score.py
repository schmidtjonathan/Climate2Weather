from typing import Tuple

import torch
from tqdm.auto import trange


class AbstractScoreFunction:
    def __init__(self, unet, noise_process, unet_kwargs=None):
        self.unet = unet
        self.noise_process = noise_process
        self.unet_kwargs = unet_kwargs if unet_kwargs is not None else {}

        self.likelihood = None

        self.unet.eval()

    @property
    def is_conditioned(self):
        return self.likelihood is not None

    def net_forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.unet(x, t)

    def __call__(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if not self.is_conditioned:
            return self.score_fn(x, t)

        J, (epsilon, sigma_t) = torch.func.jacrev(
            self.likelihood,
            argnums=0,
            has_aux=True,
            chunk_size=1,
        )(x, t)

        return epsilon - sigma_t * J

    def score_fn(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    def condition_on(self, *, A, y, std, gamma=1e-2, exact_grad=True):
        if self.likelihood is not None:
            print("Warning: Overwriting old conditioning")

        def log_p(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            mu, sigma = self.noise_process.mu(t), self.noise_process.sigma(t)

            with torch.set_grad_enabled(exact_grad):
                eps_pred = self.noise_process.pred_eps(self.score_fn, x, t)
            x0_pred = (x - sigma * eps_pred) / mu
            err = y - A(x0_pred)
            var = std**2 + gamma * (sigma / mu) ** 2
            ret = -(err**2 / var).sum() / 2
            return ret, (eps_pred, sigma)

        self.likelihood = log_p
        return self


class DefaultScoreFunction(AbstractScoreFunction):
    def __init__(self, unet, markov_order, **kwargs):
        super().__init__(unet=unet, **kwargs)
        self.markov_order = markov_order

    def unfold(self, x):
        w = 2 * self.markov_order + 1
        x = x.unfold(0, w, 1)
        x = x.movedim(-1, 1)
        x = x.flatten(1, 2)

        return x

    def fold(self, x):
        k = self.markov_order
        w = 2 * k + 1
        x = x.unflatten(1, (w, -1))

        return torch.cat(
            (
                x[0, :k],
                x[:, k],
                x[-1, -k:],
            ),
            dim=0,
        )

    def score_fn(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.unfold(x)
        x = self.net_forward(x, t)
        return self.fold(x)


class BatchedScoreFunction(AbstractScoreFunction):
    """
    Batched version of the `DefaultScoreFunction`.
    This is useful when the generated data point is too large to fit into GPU memory.
    The convolution-like operation is performed while managing between CPU and GPU memory.
    """

    def __init__(self, unet, markov_order, batch_size=16, device=None, **kwargs):
        super().__init__(unet=unet, **kwargs)

        self.markov_order = markov_order
        self.batch_size = batch_size
        self.device = device if device is not None else torch.device("cuda")
        print(f">>> Initialized batched score function to use device: {self.device}")

    def _window_score(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        is_first: bool,
        is_last: bool,
    ) -> torch.Tensor:
        k = self.markov_order
        w = 2 * k + 1

        x = self.net_forward(x, t)  #   [B, w*C, H, W]
        x = x.unflatten(1, (w, -1))  #                   [B, w, C, H, W]

        if is_first and is_last:
            return torch.cat(
                (
                    x[0, :k],  #                           [k, C, H, W]
                    x[:, k],  #                            [B, C, H, W]
                    x[-1, -k:],  #                         [k, C, H, W]
                ),
                dim=0,
            )

        else:
            if is_first:
                return torch.cat((x[0, :k], x[:, k]), dim=0)

            elif is_last:
                return torch.cat((x[:, k], x[-1, -k:]), dim=0)
            else:
                return x[:, k]

    def _batch_noise(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        # Prepare noise tensor: split into windows
        k = self.markov_order
        w = 2 * k + 1

        # x.shape:                                     [L, C, H, W]
        x = x.unfold(0, w, 1)  #                       [L-w+1, C, H, W, w]
        x = x.movedim(-1, 1).flatten(1, 2)  #          [L-w+1, w*C, H, W]

        return x.split(
            self.batch_size, 0
        )  # len: ceil((L-w+1) / B), each: [B, w*C, H, W]

    def score_fn(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        batches = self._batch_noise(x)
        num_batches = len(batches)

        scores = []
        for batch_index in trange(
            num_batches, leave=False, desc="computing score...", disable=True
        ):
            # Move batch to GPU
            # GPU AREA =====================================================================
            current_window = batches[batch_index].to(self.device)
            t = t.to(self.device)

            current_window = self._window_score(
                current_window,
                t,
                is_first=batch_index == 0,
                is_last=batch_index == num_batches - 1,
            ).to(torch.device("cpu"))
            # ==============================================================================
            # Back to CPU
            t = t.to(torch.device("cpu"))

            scores.append(current_window)

        return torch.cat(scores, dim=0)
