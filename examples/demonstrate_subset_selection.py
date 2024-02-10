"""Demonstrate ensemble selection by simulation

By: Robert Vogel


Peform k fold cross-validation and greedy ensemble 
selection plot the mean and S.E.M. of SNR and AUC metrics.
Overlay on the plot the threshold SNR used for determining
the optimal number of base classifiers,

threshold = max(SNR) - S.E.M.

Using the same seed, run the Smoca greedy selection 
classifier, and plot the optimal number of base classifiers.
In the SNR plot, the optimal number of base classifiers 
should be the smallest number, m, such that

SNR_m + S.E.M._m >= threshold

Simulation parameters and other plots are printed to the
destination directory.
"""

_epilog = """
Example:
    Simulate 15 conditionally dependent base classifiers
    with AUC values randomly distributed on interval
    [0.55, 0.65).  Print results to `sim_results` directory.

    python demostrate_subset_selection.py \\
            --auc 0.55 0.65 \\
            --m_classifiers 15 \\
           sim_results


    Perform similar simulation of conditionally *independent*
    base classifiers.

    python demostrate_subset_selection.py \\
            --auc 0.55 0.65 \\
            --cond_ind \\
            --m_classifiers 15 \\
           sim_results
"""

import os
import sys
import secrets
import argparse
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


from moca.classifiers import Smoca
from moca import simulate, stats
from moca import cross_validate as cv


FONTSIZE = 15
FIGSIZE = (4, 3.75)
AX_POSITION = (0.2, 0.2, 0.75, 0.65)
MAT_AX_POSITION = (0.1, 0.175, 0.65, 0.55)


def sem(data, axis=0):
    return np.std(data, axis=0) / np.sqrt(data.shape[0])


def args_to_json(args, savename):

    with open(savename, "w") as fid:
        json.dump(args.__dict__, fid, indent = 4)


def plot_errorbars(x, y, xerr=None, yerr=None, 
        label=None, xlabel=None, ylabel=None):

    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.errorbar(x, y, 
                xerr=xerr,
                yerr=yerr,
                fmt="-o",
                ms = 7,
                color = "black",
                linewidth=1,
                capsize=2.5,
                capthick=1,
                mfc="none",
                label=label)

    ax.set_xlabel(xlabel, fontsize=FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE)
    ax.set_position(AX_POSITION)

    return fig, ax


def _parse_args(args):
    parser = argparse.ArgumentParser(description = __doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_epilog)
    
    parser.add_argument("--auc", 
            dest = "auc",
            type = float,
            nargs = 2,
            default = (0.7, 0.95))
    parser.add_argument("--cond_ind",
            dest = "cond_ind",
            action="store_true",
            help=("Are base classifier predictions"
                  " conditionally independent (default False)"))
    parser.add_argument("--seed", 
            dest = "seed",
            type = int,
            default = None)
    parser.add_argument("--m_classifiers",
            dest = "m_classifiers",
            type = int,
            default = None)
    parser.add_argument("--cv_kfolds",
            dest = "cv_kfolds",
            type = int,
            default = 10)
    parser.add_argument("--n_samples",
            dest = "n_samples",
            type = int,
            default = 1000)
    parser.add_argument("--n_pos",
            dest = "n_pos",
            type = int,
            default = 300)
    parser.add_argument("dest",
            type=str,
            help="directory to print plots and files")
    return parser.parse_args(args)


def plot_bc_corr_matrix(corr_matrix, auc, savename):
    fig, ax = plt.subplots(figsize=FIGSIZE)

    matax = ax.matshow(corr_matrix, 
                        cmap = cm.coolwarm,
                        vmin = -1, 
                        vmax = 1)
    fig.colorbar(matax)

    ax.set_xticks([i for i in range(len(auc))])
    ax.set_xticklabels([f"{auc_i:0.2f}" for auc_i in auc],
                       rotation=90)

    ax.set_yticks([i for i in range(len(auc))])
    ax.set_yticklabels([f"{auc_i:0.2f}" for auc_i in auc])

    ax.set_xlabel("B.C. AUC", fontsize=FONTSIZE)
    ax.set_ylabel("B.C. AUC", fontsize=FONTSIZE)
    ax.set_title(r"$C_Y$", fontsize=FONTSIZE)
    ax.set_position(MAT_AX_POSITION)

    fig.savefig(savename)


def main():
    args = _parse_args(sys.argv[1:])

    if not os.path.exists(args.dest):
        os.mkdir(args.dest)

    if args.seed is None:
        args.seed = secrets.randbits(128)

    rng = np.random.default_rng(seed=args.seed)

    args.auc = rng.uniform(low=args.auc[0],
                           high=args.auc[1],
                           size=args.m_classifiers).tolist()

    args_to_json(args, os.path.join(args.dest, "sim_pars.json"))

    args.auc = np.array(args.auc)


    corr_matrix = simulate.make_corr_matrix(args.m_classifiers,
                                independent=args.cond_ind,
                                seed=rng)

    data, labels = simulate.rank_scores(args.n_samples,
                                        args.n_pos,
                                        args.auc,
                                        corr_matrix,
                                        seed=rng)

    plot_bc_corr_matrix(corr_matrix,
                        args.auc,
                        os.path.join(args.dest,"corr_matrix.pdf"))

    scl_opt = Smoca(subset_select= "greedy")
    scl_opt.fit(data, labels, seed=args.seed)

    # test_data, test_labels = g.sample(args.n_samples, args.n_pos)
        
    subset_par = np.arange(1, args.m_classifiers + 1)
    auc = {
            "mean": np.zeros(len(subset_par)),
            "sem":np.zeros(len(subset_par))
            }

    snr = {
            "mean": np.zeros(len(subset_par)),
            "sem":np.zeros(len(subset_par))
            }


    # Generator for cross validation below
    rng = np.random.default_rng(seed=args.seed)

    for i, m in enumerate(subset_par):

        tmp_auc = np.zeros(args.cv_kfolds)
        tmp_snr = np.zeros(args.cv_kfolds)

        cv_generator = cv.stratified_kfold(data,
                                labels,
                                args.cv_kfolds, 
                                ascending=True,
                                seed=rng)

        n = 0
        for train, test in cv_generator:

            scl = Smoca(subset_select = "greedy",
                        subset_select_par = m)
            scl.fit(train["data"], train["labels"])

            s = scl.get_scores(test["data"])

            tmp_auc[n] = stats.roc(s,
                                   test["labels"])[2]
            tmp_snr[n] = stats.snr(-s, test["labels"])

            n += 1
            

        auc["mean"][i] = np.mean(tmp_auc)
        auc["sem"][i] = sem(tmp_auc)
        snr["mean"][i] = np.mean(tmp_snr)
        snr["sem"][i] = sem(tmp_snr)


    fig, ax = plot_errorbars(subset_par, auc["mean"], 
                            yerr=auc["sem"],
                            label = "Test Set AUC",
                            xlabel = r"$m$ nonzero weights",
                            ylabel = "AUC")
    ax.axvline(scl_opt.subset_select_par, 
                linestyle=":",
                linewidth=1,
                color="black",
                label=r"$m_{opt}$")
    ax.legend(loc = 0)

    fig.savefig(os.path.join(args.dest, "auc.pdf"))


    fig, ax = plot_errorbars(subset_par, snr["mean"], 
                            yerr = snr["sem"],
                            label = "Test Set SNR",
                            xlabel = r"$m$ nonzero weights",
                            ylabel = "SNR")

    # here I want to plot the threshold in which 
    # the subselect par is determined.  
    thresh = np.max(snr["mean"])
    thresh_idx = np.where(snr["mean"] == thresh)[0]
    thresh -= snr["sem"][thresh_idx]

    ax.axvline(scl_opt.subset_select_par, 
                linestyle=":",
                linewidth=1,
                color="black",
                label=r"Smoca $m_{opt}$")

    ax.axhline(thresh,
                linestyle=":",
                linewidth=1,
                color="black",
                label=r"Threshold <SNR>+S.E.M.")

    ax.legend(loc = 0)

    fig.savefig(os.path.join(args.dest, "snr.pdf"))

    fig, ax = plot_errorbars(snr["mean"], auc["mean"],
                            xerr = snr["sem"],
                            yerr = auc["sem"],
                            xlabel = "SNR",
                            ylabel = "AUC")
    fig.savefig(os.path.join(args.dest, "snr_auc.pdf"))






if __name__ == "__main__":
    main()
