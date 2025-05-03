import numpy as np
import torch
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines


def plot_margins(
    unlearned_margins,
    all_full_margins,
    all_oracle_margins,
    inds_to_plot=None,
):

    if inds_to_plot is None:
        inds_to_plot = np.random.choice(unlearned_margins.shape[0],
                                        20,
                                        replace=False)
        N = 20
    else:
        N = len(inds_to_plot)

    n_cols = 5
    n_rows = N // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 2))

    plt.legend()
    for indE, E in enumerate(inds_to_plot):
        ax = axes[indE // n_cols, indE % n_cols]
        ax.set_xlabel("Margins")
        ax.hist(all_full_margins[:, E],
                bins=100,
                alpha=0.5,
                color="r",
                label="full")
        ax.hist(all_oracle_margins[:, E],
                bins=100,
                alpha=0.5,
                color="k",
                label="oracle")
        ax.axvline(unlearned_margins[E], color="g", label="unlearned")
    # Manually create handles for the legend
    handle1 = mlines.Line2D([], [], color="k", label="oracle")
    handle2 = mlines.Line2D([], [], color="g", label="unlearned")
    handle3 = mlines.Line2D([], [], color="r", label="full")

    # Creating one legend for the whole figure using the handles
    fig.legend(handles=[handle1, handle2, handle3], loc="upper center", ncol=3)
    fig.tight_layout()

    return fig
