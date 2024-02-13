import numpy as np
from matplotlib import pyplot as plt


def show_convs(conv: np.ndarray):
    assert conv.ndim == 4
    fig = plt.figure(figsize=(10, 7))
    vmin = np.min(conv)
    vmax = np.max(conv)
    fig.suptitle(f"vmin={vmin:.2f}, vmax={vmax:.2f}")
    nrows = conv.shape[0]
    ncols = conv.shape[1]
    for outs in range(nrows):
        for ins in range(ncols):
            n = outs * ncols + ins + 1
            fig.add_subplot(nrows, ncols, n)
            plt.imshow(conv[outs][ins], vmin=vmin, vmax=vmax, cmap="gray")
            plt.axis("off")