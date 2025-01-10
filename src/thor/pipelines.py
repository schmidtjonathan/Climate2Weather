import math
import time

import torch
from tqdm.auto import tqdm


class SDAPipeline:
    def __init__(self, eta=1e-3):
        # eta for numerical stability
        self.eta = eta

    def alpha(self, t):
        return torch.cos(math.acos(math.sqrt(self.eta)) * t) ** 2

    def mu(self, t):
        return self.alpha(t)

    def sigma(self, t):
        return (1 - self.alpha(t) ** 2 + self.eta**2).sqrt()

    def forward(self, x, t):
        eps = torch.randn_like(x)
        xt = self.mu(t) * x + self.sigma(t) * eps
        return xt, eps

    def loss(self, net, x, forcing=None):
        # Sample noise levels / times
        t = torch.rand(x.shape[0], 1, 1, 1, dtype=x.dtype, device=x.device)
        # sample noise image = forward markov process
        xt, eps = self.forward(x, t)
        # Predict noise part from noised image
        eps_pred = net(xt, t, forcing=forcing)
        # Return squared error
        return (eps_pred - eps) ** 2

    def pred_eps(self, score_fn, x, t):
        eps_pred = score_fn(x, t)
        return eps_pred

    def _sample_step(self, score_fn, x, t, dt, proc_x0=None):
        eps_pred = self.pred_eps(score_fn, x, t)
        pred_x0 = (x - self.sigma(t) * eps_pred) / self.mu(t)
        if proc_x0 is not None:
            pred_x0 = proc_x0(pred_x0)
        return self.mu(t - dt) * pred_x0 + self.sigma(t - dt) * eps_pred

    def sample(
        self,
        score_fn,
        noise,
        steps: int = 64,
        corrections: int = 0,
        tau: float = 1.0,
        proc_x0=None,
        device=None,
        show_progressbar=True,
    ):
        if device is None:
            device = torch.device("cpu")

        shape = noise.shape
        x = noise.to(device=device)
        dims = tuple(range(-len(shape), 0))

        time_steps = torch.linspace(1, 0, steps + 1).to(dtype=x.dtype, device=device)
        dt = 1 / steps

        total_start_time = time.time()
        z = None
        if corrections > 0:
            z = torch.empty_like(x)
        with torch.no_grad():
            for t in tqdm(
                time_steps[:-1], desc="Sampling", disable=not show_progressbar
            ):
                # Predictor
                x = self._sample_step(score_fn, x, t, dt, proc_x0=proc_x0)

                # Corrector
                for ic in range(corrections):
                    z.normal_()
                    eps = score_fn(x, t - dt)
                    delta = tau / eps.square().mean(dim=dims, keepdim=True)
                    x = x - (delta * eps + torch.sqrt(2 * delta) * z) * self.sigma(
                        t - dt
                    )
                    del eps

                if torch.isnan(x).any():
                    raise ValueError("NaN detected in sample")

        total_time = time.time() - total_start_time
        print(
            f"Total sampling time: {total_time:.2f} s  = {total_time / 60:.3f} min = {total_time / 3600:.4f} h",
        )
        return x.reshape(noise.shape)
