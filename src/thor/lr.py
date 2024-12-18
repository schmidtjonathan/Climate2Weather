# https://github.com/NVlabs/edm2/blob/4bf8162f601bcc09472ce8a32dd0cbe8889dc8fc/training/training_loop.py#L47

import numpy as np


def edm2_learning_rate_schedule(
    cur_ndata, batch_size, ref_lr, ref_batches, rampup_Mdata
):
    lr = ref_lr
    if ref_batches > 0:
        lr /= np.sqrt(max(cur_ndata / (ref_batches * batch_size), 1))
    if rampup_Mdata > 0:
        lr *= min(cur_ndata / (rampup_Mdata * 1e6), 1)
    return lr


def linear_learning_rate_schedule(cur_ndata, total_ndata, ref_lr):
    frac_done = cur_ndata / total_ndata
    return ref_lr * (1 - frac_done)
