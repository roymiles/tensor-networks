import numpy as np


def sine2D(w, h):
    # Create a constant sine ting
    c = np.zeros((w, h), dtype=np.float32)

    ns = np.arange(w)
    one_cycle = 2 * np.pi * ns / w
    for k in range(h):
        t_k = k * one_cycle
        c[:, k] = np.cos(t_k)

    return c
