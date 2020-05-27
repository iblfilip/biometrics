import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_CMC(sums, n_predictions, n_ranks):
    """
        Plot Cumulative Match Curve for all subpredictions and combined prediction
        :param sums: dictionary containing sums for each rank for all classifiers
        :param n_predictions: number of predictions
        :param n_ranks: number of ranks
    """

    colors = ['#F1C40F', '#3498DB', '#2ECC71', '#F39C12', '#C70039']
    fig, ax = plt.subplots()
    for i, (name, data) in enumerate(sums.items()):
        acc_list = []
        rank_list = []
        overall_acc = 0
        for comp_1, comp_2 in data.items():
            overall_acc += (comp_2 / n_ranks)
            rank_list.append(comp_1 + 1)
            acc_list.append(overall_acc)
        plt.plot(rank_list, acc_list, '-o', color=colors[i], linewidth=1, markersize=4, label=name)
    plt.xlabel('Rank', fontsize=10)
    plt.ylabel('Identification rate', fontsize=10)
    plt.title('Cumulative Match Curve')
    plt.ylim(ymin=0)
    plt.xlim(xmin=0, xmax=1.01 * n_predictions)
    ax.legend(loc='lower right')
    plt.show()


def plot_EER(score_dict):
    """
        Plot Equal Error Rate
        :param score_dict: dictionary with distances for each classification
    """

    points_x = []
    points_y = []
    max_value = 1
    fig, ax = plt.subplots()
    for step in np.arange(0, max_value, 0.01):
        false_accept_rate = 0
        false_reject_rate = 0
        n_impostors = 0
        n_clients = 0
        for label, score in score_dict.items():
            is_match = label.startswith('match')
            if is_match:
                n_clients += 1
            else:
                n_impostors += 1
            if score <= step and not is_match:
                false_accept_rate += 1
            elif score > step and is_match:
                false_reject_rate += 1
        FAR = false_accept_rate / 22
        FRR = false_reject_rate / 22
        points_x.append(FAR)
        points_y.append(FRR)

    plt.plot(points_x, points_y, '-s', markersize=4, linewidth=1, color='#C70039', label='Combined method')

    plt.plot([0, 0.8], [0, 0.8], 'k-', label='EER')
    plt.xlim(0, 0.5)
    plt.ylim(0, 0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('False Accept Rate', fontsize=10)
    plt.ylabel('False Reject Rate', fontsize=10)
    plt.title('Equal Error Rate')
    ax.legend(loc='lower right')
    plt.show()


def plot_ICH(distance_matrix, labels_matrix, labels_test):
    """
        Calculate and plot intra/inter class histogram.
        :param distance_matrix: matrix with pairwise similarities for all test/train images
        :param labels_matrix: matrix with labels for distance_matrix
        :param labels_test: array with labels for test
    """
    imp_dist = []
    cli_dist = []

    # create arrays with client and impostors distances
    for (y_axis, x_axis), value in np.ndenumerate(distance_matrix):
        if labels_test[y_axis] == labels_matrix[y_axis, x_axis]:
            cli_dist.append(value)
        else:
            imp_dist.append(value)

    series_client = pd.Series(cli_dist)
    series_impostor = pd.Series(imp_dist)

    series_client.hist(bins=100, range=(0, 1), density=True, color='#0000FF', label='Intra-class')
    ax = series_impostor.hist(bins=100, range=(0, 1), density=True, color='#FF0000', label='Inter-class')
    ax.set_title('Inter/Intra Class Variation')
    ax.set_xlabel('Distance')
    ax.set_ylabel('Probability density')
    ax.legend()
    plt.show()
