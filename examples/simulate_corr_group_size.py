"""Validate performance of Smoca, Smoca-subset, and Woc with simulation data.

By: Robert Vogel


"""
import sys
import os
import json


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from moca import Smoca, Woc, stats
from moca import simulate as sim

N_REPS = 10

M_CLASSIFIERS = 10
AUC = np.linspace(0.55, 0.75, M_CLASSIFIERS)
N_SAMPLES = 1000
N_POSITIVE = 300

CORR_GROUPS = np.arange(2, M_CLASSIFIERS+1)
CORR_VALUE = 0.7

FONTSIZE = 15
FIGSIZE = (4, 3.75)
AX_POSITION = (0.2, 0.2, 0.75, 0.65)
MAT_AX_POSITION = (0.1, 0.175, 0.65, 0.55)

def sem(data, axis=0):
    return np.std(data, axis=0) / np.sqrt(data.shape[0])


def construct_result_dirname():
    count = 1
    dirname = f"sim_{count:03d}"

    while os.path.exists(dirname):
        count += 1
        dirname = f"sim_{count:03d}"

    return dirname

def print_pars(dirname):
    output = {
                "m_classifiers": M_CLASSIFIERS,
                "auc" : AUC.tolist(),
                "n_samples" : N_SAMPLES,
                "n_positive" : N_POSITIVE,
                "corr_group_sizes" : CORR_GROUPS.tolist(),
                "corr_value" : CORR_VALUE
            }

    with open(os.path.join(dirname, "pars.json"), "w") as fid:
        json.dump(output, fid, indent = 4)

def plot_auc(cl_auc_dict, dirname):
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for cl, rep_vals in cl_auc_dict.items():
        ax.errorbar(CORR_GROUPS, 
                    np.mean(rep_vals, 0),
                    yerr = sem(rep_vals, axis=0),
                    fmt = "-o",
                    ms = 7,
                    linewidth=1,
                    capsize=2.5,
                    capthick = 1,
                    mfc="none",
                    label = cl.name)

    ax.legend(loc = 0)

    ax.set_xlabel("|G|", fontsize=FONTSIZE)
    ax.set_ylabel("AUC", fontsize=FONTSIZE)
    ax.set_position(AX_POSITION)
    ax.set_title(f"Cond. Correlation = {CORR_VALUE}", fontsize=FONTSIZE)

    fig.savefig(os.path.join(dirname, "auc.pdf"))

def plot_weights(wstats, corr_groups, corr_value, dirname):
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for cl_means, cl_sems in zip(wstats["mean"], wstats["sem"]):
        ax.errorbar(corr_groups, 
                    cl_means,
                    yerr=cl_sems,
                    fmt= "-o",
                    linewidth = 1,
                    ms = 7,
                    mfc="none",
                    elinewidth = 1,
                    capsize=2.5,
                    capthick=1,
                    color="black")
                    

    ax.set_xlabel("|G|", fontsize=FONTSIZE)
    ax.set_ylabel("Weights", fontsize=FONTSIZE)
    ax.set_position(AX_POSITION)
    ax.set_title(f"Cond. Correlation = {corr_value}", fontsize=FONTSIZE)

    fig.savefig(os.path.join(dirname, "weights.pdf"))


def plot_avg_corr_matrix(corr_matrix, auc, dirname, group_size):
    fig, ax = plt.subplots(figsize=FIGSIZE)

    matax = ax.matshow(corr_matrix, 
                        cmap = cm.coolwarm,
                        vmin = -1, 
                        vmax = 1)
    fig.colorbar(matax)

    ax.set_xticks([i for i in range(auc.size)])
    ax.set_xticklabels([f"{auc_i:0.2f}" for auc_i in auc], rotation=90)

    ax.set_yticks([i for i in range(auc.size)])
    ax.set_yticklabels([f"{auc_i:0.2f}" for auc_i in auc])

    ax.set_xlabel("B.C. AUC", fontsize=FONTSIZE)
    ax.set_ylabel("B.C. AUC", fontsize=FONTSIZE)
    ax.set_title(r"Mean $(C_0 + C_1) / 2$", fontsize=FONTSIZE)
    ax.set_position(MAT_AX_POSITION)

    fig.savefig(os.path.join(dirname, f"avg_corr_matrix_{group_size}.pdf"))


def main():

    cl_auc = { 
                Smoca(subset_select = "greedy") : None,
                Smoca() : None,
                Woc() : None
            }

    # stats for greedy ensemble subset selection
    gsubset_select_weights = { 
            "mean" : np.zeros(shape=(M_CLASSIFIERS, CORR_GROUPS.size)),
            "sem" : np.zeros(shape=(M_CLASSIFIERS, CORR_GROUPS.size))
            }

    # initialize the data matrices
    for cl in cl_auc:
        cl_auc[cl] = np.zeros(shape=(N_REPS, CORR_GROUPS.size))
                

    dirname = construct_result_dirname()
    os.mkdir(dirname)

    # perform N_REP simulation experiments for set
    # of conditionally dependent classifiers

    for i, gr_size in enumerate(CORR_GROUPS):

#         ens_train = sim.EnsembleRankPredictions(
#                           auc = AUC,
#                           group_sizes = (M_CLASSIFIERS-gr_size, gr_size),
#                           group_corrs = (0, CORR_VALUE))
# 
#         ens_test = sim.EnsembleRankPredictions(
#                           auc = AUC,
#                           group_sizes = (M_CLASSIFIERS-gr_size, gr_size),
#                           group_corrs = (0, CORR_VALUE))
        ens_train = sim.EnsembleRankPredictions(
                          auc = AUC,
                          group_sizes = (gr_size,),
                          group_corrs = CORR_VALUE)

        ens_test = sim.EnsembleRankPredictions(
                          auc = AUC,
                          group_sizes = (gr_size, ),
                          group_corrs = CORR_VALUE)

        corr_matrix = 0

        # subset select weight stats
        tmp_weights = np.zeros(shape=(N_REPS, M_CLASSIFIERS))

        for n in range(N_REPS):

            train_data, train_labels = ens_train.sample(N_SAMPLES, N_POSITIVE)

            corr_matrix += (np.corrcoef(train_data[:, train_labels == 0]) + 
                            np.corrcoef(train_data[:, train_labels == 1]))
            
            test_data, test_labels = ens_test.sample(N_SAMPLES, N_POSITIVE)
             
            for cl in cl_auc:
                cl.subset_select_par = None
                cl.fit(train_data, train_labels)

                if "greedy" in cl.name:
                    tmp_weights[n, :] = cl.weights

                _, _, tmp_auc = stats.roc(cl.get_scores(test_data), 
                                            test_labels)
                cl_auc[cl][n, i] = tmp_auc

        plot_avg_corr_matrix(corr_matrix / (2 * N_REPS), 
                                AUC,
                                dirname,
                                gr_size)
        gsubset_select_weights["mean"][:, i] = np.mean(tmp_weights, axis=0)
        gsubset_select_weights["sem"][:, i] = sem(tmp_weights, axis=0)

    plot_auc(cl_auc, dirname)
    plot_weights(gsubset_select_weights, CORR_GROUPS, CORR_VALUE, dirname)
    print_pars(dirname)


if __name__ == "__main__":
    main()

