import itertools
import math
from typing import Union

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score


def _draw_cluster(x, y, hue, alpha, ax=None):
    if ax is None:
        ax = plt.gca()
    scatter = ax.scatter(
        x,
        y,
        marker="o",
        c=hue,
        alpha=alpha,
        s=25,
        edgecolor="k",
    )
    return scatter


def draw_clusters(x, y, ax, alpha=None, hue=None, labels=None, legend_loc="best"):
    scatter = _draw_cluster(x, y, hue, alpha=alpha, ax=ax)
    if hue is not None:
        ax.legend(handles=scatter.legend_elements()[0], labels=set(hue), loc=legend_loc)
    if labels:
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])


def draw_clusters_grid(data, hue, labels, legend_loc, ax, alpha=None):
    feature_combinations = list(itertools.combinations(range(data.shape[1]), 2))
    grid_size_combinations = [list(range(ax.shape[0])), list(range(ax.shape[1]))]
    figure_combinations = list(itertools.product(*grid_size_combinations))
    if len(feature_combinations) > len(figure_combinations):
        raise ValueError("The grid size is smaller than the number of features.")
    # Plot
    for (comb_1, comb_2), (ax_1, ax_2) in zip(feature_combinations, figure_combinations):
        scatter = _draw_cluster(
            data[:, comb_1], data[:, comb_2], hue, ax=ax[ax_1, ax_2], alpha=alpha
        )
        if hue is not None:
            ax[ax_1, ax_2].legend(
                handles=scatter.legend_elements()[0], labels=set(hue), loc=legend_loc
            )
        if labels:
            ax[ax_1, ax_2].set_xlabel(f"{labels[0]}_{comb_1}")
            ax[ax_1, ax_2].set_ylabel(f"{labels[1]}_{comb_2}")


def draw_ksscore(data, ks, labels, ax, random_state=None):
    """Function that takes an array (or tuple of arrays) and calculates the silhouette scores for the different ks.
    Plots the silhouette scores against the ks and saves the image as a pdf.

    Args:
        data_list (array-like of shape (n_samples, n_features) or tuple): Input samples.
        ks (list): List containing all choices for k.
    """
    sscore = []
    for k in ks:
        kkm = KMeans(
            n_clusters=k, random_state=random_state, init="k-means++", max_iter=300, tol=0.0001
        )
        cluster_labels = kkm.fit_predict(data)
        sscore.append(silhouette_score(data, cluster_labels))
    ax.plot(ks, sscore)

    ax.axhline(y=max(sscore), color="red", linestyle="--")
    # Labels ----------------------
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    print(f"Max score: {max(sscore)}")


def draw_plot(
    data,
    plot_type: str = "scatter",
    hue=None,
    alpha=None,
    labels: list[str, str] = None,
    figsize: tuple[int, int] = None,
    grid_size=(1, 1),
    legend_loc: str = "best",
    dpi: int = None,
    title: str = None,
    shareaxes: bool = False,
    ks: Union[int, list[int]] = None,
    random_state: int = None,
):
    _, axes = plt.subplots(figsize=figsize, dpi=dpi, nrows=grid_size[0], ncols=grid_size[1])
    if plot_type == "scatter":
        draw_clusters(
            data[:, 0], data[:, 1], axes, alpha=alpha, hue=hue, labels=labels, legend_loc=legend_loc
        )
    if plot_type == "grid":
        # Create indexes
        draw_clusters_grid(
            data, alpha=alpha, hue=hue, labels=labels, legend_loc=legend_loc, ax=axes
        )
    if plot_type == "ksscore":
        draw_ksscore(data, ks, labels=labels, ax=axes, random_state=random_state)
    if plot_type == "silhouette":
        draw_silhouette(data, ks, labels=labels, ax=axes, random_state=random_state)

    if shareaxes:
        lim_range = (math.floor(np.min(data)), math.ceil(np.max(data)))
        plt.setp(axes, xlim=lim_range, ylim=lim_range)
    if title:
        # plt.suptitle(
        #     "Silhouette analysis for KMeans clustering on Facebook data with n_clusters = %d"
        #     fontsize=14,
        #     fontweight="bold",
        # )
        plt.suptitle(title)
    plt.show()


def draw_silhouette(data, number_k, labels, ax, random_state=None):
    """Creates Silhouette plot for the fiven dataset (datasets).

    Credit: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html

    Args:
        data (array-like of shape (n_samples, n_features)): Input samples.
        k_clusters (int): Number of clusters.
    """
    # Generate clusters ----------------------
    kkm = KMeans(
        n_clusters=number_k, random_state=random_state, init="k-means++", max_iter=300, tol=0.0001
    )
    cluster_labels = kkm.fit_predict(data)
    sscore = silhouette_score(data, cluster_labels)
    sscore_values = silhouette_samples(data, cluster_labels)

    # Gap between silhouette plots ----------------------
    gap = 0.01
    y_lower = len(data) * gap

    # The silhouette coefficient can range from -1, 1 but here most lie within [-0.1, 1]
    ax.set_xlim([-0.1, 1])
    # The (k_clusters+1)*len(data)*gap is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax.set_ylim([0, len(data) + (number_k + 1) * len(data) * gap])

    # Plot Cluster ----------------------
    for i in range(number_k):
        # Order the sscores
        ith_sscore_values = sscore_values[cluster_labels == i]
        ith_sscore_values.sort()
        cluster_score = sum(ith_sscore_values)

        # Get the size of each cluster
        size_cluster_i = ith_sscore_values.shape[0]
        ith_sscore = cluster_score / size_cluster_i
        y_upper = y_lower + size_cluster_i
        print("Cluster:%d with %d entries. Score: %f" % (i, size_cluster_i, round(ith_sscore, 2)))

        # Create different colors for the clusters
        color = cm.nipy_spectral(float(i) / number_k)
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_sscore_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + len(data) * gap  # 10 for the 0 samples

    # Labels ----------------------
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.axvline(x=sscore, color="red", linestyle="--")
    ax.set_yticks([])  # Clear the yaxis labels / ticks


def main():
    return ()


if __name__ == "__main__":
    main()
