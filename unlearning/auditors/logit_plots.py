import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from tqdm import tqdm


def to_cuda(x):
    if isinstance(x, (list, tuple)):
        return [to_cuda(xi) for xi in x]
    return x.cuda()


def compute_logits(model, loader):
    tr = model.training
    model.eval()
    logits = []
    with torch.no_grad():
        for x, _, index in tqdm(loader, total=len(loader)):
            x = to_cuda(x)
            logits.append(model(x, index).cpu())
    model.train(tr)
    return torch.cat(logits)


def plot_logits(
        unlearned_logits,
        full_logit_paths,
        oracle_logit_paths,
        n_full_logits_to_plot=5,
        n_oracle_logits_to_plot=20,
        inds_to_plot=np.arange(50),
        reorder_classes=False,
):
    """
    reorder_classes: bool - if True, reorder the classes by the average oracle logit value
    """

    oracle_logits = []
    for path in oracle_logit_paths:
        oracle_logits.append(torch.load(path))
    oracle_logits = torch.cat(oracle_logits, dim=1).numpy()
    oracle_logits = oracle_logits[:n_oracle_logits_to_plot]

    full_logits = []
    for path in full_logit_paths:
        full_logits.append(torch.load(path))
    full_logits = torch.cat(full_logits, dim=1).numpy()
    full_logits = full_logits[:n_full_logits_to_plot]

    n_cols = 5
    n_rows = len(inds_to_plot) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 2))
    x_str = "CIFAR10 classes" if not reorder_classes else "CIFAR10 classes (reordered)"
    n_classes = full_logits.shape[2]
    ordered_inds = np.arange(n_classes)

    for indE, E in enumerate(inds_to_plot):
        if reorder_classes:
            ordered_inds = np.argsort(full_logits[0][E])[::-1]

        ax = axes[indE // n_cols, indE % n_cols]
        if indE >= len(inds_to_plot) - n_cols:
            ax.set_xlabel(x_str)
        if indE % n_cols == 0:
            ax.set_ylabel("Logit value")

        sns.violinplot(
            data=oracle_logits[:, E, ordered_inds],
            color="k",
            alpha=0.2,
            inner=None,
            label="oracle dist",
            ax=ax,
        )

        sns.violinplot(
            data=full_logits[:, E, ordered_inds],
            color="r",
            alpha=0.2,
            inner=None,
            label="full dist",
            ax=ax,
        )

        ax.scatter(
            x=np.arange(n_classes),
            y=full_logits[0][E][ordered_inds],
            color="r",
            linewidth=1,
            label="before unlearning",
            alpha=0.25,
        )
        ax.scatter(
            x=np.arange(n_classes),
            y=unlearned_logits[E][ordered_inds],
            color="g",
            linewidth=2,
            label="unlearned",
            alpha=0.75,
        )
        ax.plot(
            np.arange(n_classes),
            full_logits[0][E][ordered_inds],
            color="r",
            linewidth=1,
            alpha=0.1,
            linestyle="--",
        )
        ax.plot(
            np.arange(n_classes),
            unlearned_logits[E][ordered_inds],
            color="g",
            linewidth=2,
            alpha=0.1,
            linestyle="-.",
        )

    # Manually create handles for the legend
    handle1 = mlines.Line2D([], [], color="k", label="oracle")
    handle2 = mlines.Line2D([], [], color="g", label="unlearned")
    handle3 = mlines.Line2D([], [], color="r", label="full")

    # Creating one legend for the whole figure using the handles
    fig.legend(handles=[handle1, handle2, handle3], loc="upper center", ncol=3)
    fig.tight_layout()
    return fig
