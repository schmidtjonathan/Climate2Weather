import copy

import torch


class StandardEMA:
    """
    Class for tracking exponential moving average of the weights during training.
    """

    @torch.no_grad()
    def __init__(self, net, rates=[0.9999]):
        self.net = net
        self.rates = rates
        self.emas = [copy.deepcopy(net) for r in rates]

    @torch.no_grad()
    def reset(self):
        for ema in self.emas:
            for p_net, p_ema in zip(self.net.parameters(), ema.parameters()):
                p_ema.copy_(p_net)

    @torch.no_grad()
    def update(self, **kwargs):
        for rate, ema in zip(self.rates, self.emas):
            for p_net, p_ema in zip(self.net.parameters(), ema.parameters()):
                p_ema.detach().mul_(rate).add_(p_net, alpha=1 - rate)

    @torch.no_grad()
    def get(self):
        for ema in self.emas:
            for p_net, p_ema in zip(self.net.buffers(), ema.buffers()):
                p_ema.copy_(p_net)
        return [(ema, f"-{rate:.6f}") for rate, ema in zip(self.rates, self.emas)]

    def state_dict(self):
        return dict(rates=self.rates, emas=[ema.state_dict() for ema in self.emas])

    def load_state_dict(self, state):
        self.rates = state["rates"]
        for ema, s_ema in zip(self.emas, state["emas"]):
            ema.load_state_dict(s_ema)


# ----------------------------------------------------------------------------
