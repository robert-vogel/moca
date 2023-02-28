#!/usr/bin/env python
"""Show ensemble selection by simulation


"""

_epilog = """
Examples:
    Simulate 20 base classifiers with AUC values evenly distributed
    between [0.55, 0.65].  Moreover classifiers 1-5, 6-15, and 16-20
    are conditionally dependent with conditional correlations 0.3,
    0.1, and 0.8.  Lastly, print results to `sim_results` directory.

    python -m demostrate_subset_selection.py \
            --auc 0.55 0.65 \
            --m_classifiers 20 \
            --group_sizes 5 10 5 \
            --group_corrs 0.3 0.1 0.8 -- \
           sim_results
"""

import os
import sys
import argparse
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


from moca import Smoca, simulate, stats
from moca import cross_validate as cv

SEED_MAX = 592355334


FONTSIZE = 15
FIGSIZE = (4, 3.75)
AX_POSITION = (0.2, 0.2, 0.75, 0.65)
MAT_AX_POSITION = (0.1, 0.175, 0.65, 0.55)

def sem(data, axis=0):
    return np.std(data, axis=0) / np.sqrt(data.shape[0])

def args_to_json(args, savename):

    with open(savename, "w") as fid:
        json.dump(args.__dict__, fid, indent = 4)

def _parse_args(args):
    parser = argparse.ArgumentParser(description = __doc__)
    
    parser.add_argument("--auc", 
            dest = "auc",
            type = float,
            nargs = "*",
            default = [0.55, 0.6, 0.65, 0.7, 0.75])
    parser.add_argument("--seed", 
            dest = "seed",
            type = int,
            default = None)
    parser.add_argument("--m_classifiers",
            dest = "m_classifiers",
            type = int,
            default = None)
    parser.add_argument("--cv",
            dest = "cv",
            type = int,
            default = 10)
    parser.add_argument("--group_sizes",
            dest = "group_sizes",
            type = int,
            nargs = "*",
            default = None)
    parser.add_argument("--group_corrs",
            dest = "group_corrs",
            type = float,
            nargs = "*",
            default = None)
    parser.add_argument("--n_samples",
            dest = "n_samples",
            type = int,
            default = 1000)
    parser.add_argument("--n_positives",
            dest = "n_positives",
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
    ax.set_xticklabels([f"{auc_i:0.2f}" for auc_i in auc], rotation=90)

    ax.set_yticks([i for i in range(len(auc))])
    ax.set_yticklabels([f"{auc_i:0.2f}" for auc_i in auc])

    ax.set_xlabel("B.C. AUC", fontsize=FONTSIZE)
    ax.set_ylabel("B.C. AUC", fontsize=FONTSIZE)
    ax.set_title(r"Mean $(C_0 + C_1) / 2$", fontsize=FONTSIZE)
    ax.set_position(MAT_AX_POSITION)

    fig.savefig(savename)


def main():
    args = _parse_args(sys.argv[1:])

    print(args)

    if args.seed is None:
        args.seed = np.random.choice(SEED_MAX)

    if not os.path.exists(args.dest):
        os.mkdir(args.dest)

    args_to_json(args, os.path.join(args.dest, "sim_pars.json"))

    g = simulate.EnsembleRankPredictions(auc=args.auc,
                                        m_classifiers = args.m_classifiers,
                                        group_corrs = args.group_corrs,
                                        group_sizes = args.group_sizes)

    data, labels = g.sample(args.n_samples, args.n_positives)


    scl_opt = Smoca(subset_select= "greedy", 
                    seed=args.seed)

    scl_opt.fit(data, labels)

    plot_bc_corr_matrix(0.5 * (np.corrcoef(data[:, labels == 0]) +
                                np.corrcoef(data[:, labels == 1])),
                        g.auc,
                        os.path.join(args.dest,"corr_matrix.pdf"))


    # test_data, test_labels = g.sample(args.n_samples, args.n_positives)
        
    subset_par = np.arange(1, len(g.auc) + 1)
    auc = {
            "mean": np.zeros(len(subset_par)),
            "sem":np.zeros(len(subset_par))
            }

    snr = {
            "mean": np.zeros(len(subset_par)),
            "sem":np.zeros(len(subset_par))
            }
    rng = np.random.default_rng(seed=args.seed)

    for i, m in enumerate(subset_par):

        tmp_auc = np.zeros(args.cv)
        tmp_snr = np.zeros(args.cv)

        cv_generator = cv.cv_sample_data(data,
                                labels,
                                args.cv, 
                                ascending=True,
                                seed=rng)

        for n, (train, test) in enumerate(cv_generator):

            scl = Smoca(subset_select = "greedy", subset_select_par = m)
            scl.fit(train["data"], train["labels"])

            s = scl.get_scores(test["data"])

            tmp_auc[n] = stats.roc(s,
                                   test["labels"])[2]
            tmp_snr[n] = -stats.snr(s, test["labels"])
            

        auc["mean"][i] = np.mean(tmp_auc)
        auc["sem"][i] = sem(tmp_auc)
        snr["mean"][i] = np.mean(tmp_snr)
        snr["sem"][i] = sem(tmp_snr)


    m_opt_roc = None
    best_auc = 0

    for i, auc_val in enumerate(auc["mean"]):
        if auc_val > best_auc:
            best_auc = auc_val
            m_opt_roc = i + 1

    best_snr = 0
    m_opt_snr = None
    for i, snr_val in enumerate(snr["mean"]):
        if snr_val > best_snr:
            best_snr = snr_val
            m_opt_snr = i + 1


    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.errorbar(snr["mean"],
                auc["mean"],
                xerr = snr["sem"],
                yerr = auc["sem"],
                fmt="o",
                ms = 7,
                color = "black",
                linewidth=1,
                capsize=2.5,
                capthick=1,
                mfc="none")
    ax.set_xlabel("SNR", fontsize=FONTSIZE)
    ax.set_ylabel("AUC", fontsize=FONTSIZE)

    ax.set_position(AX_POSITION)
    fig.savefig(os.path.join(args.dest, "snr_v_auc.pdf"))


    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.errorbar(subset_par,
                auc["mean"],
                yerr = auc["sem"],
                fmt="-o",
                ms = 7,
                color = "black",
                linewidth=1,
                capsize=2.5,
                capthick=1,
                mfc="none",
                label="Test Set AUC")
    ax.axhline(best_auc, 
                linestyle=":",
                color = cm.tab10(0),
                label=r"AUC$_{max}$")

    ax.axvline(m_opt_roc, 
                linestyle=":",
                color = cm.tab10(0),
                label=r"$m_{opt}$(AUC)")

    ax.axvline(scl_opt.subset_select_par, 
                linestyle=":",
                linewidth=1,
                color="black",
                label=r"$m_{opt}(SNR)$")

    ax.set_xlabel(r"$m$ nonzero weights", fontsize=FONTSIZE)
    ax.set_ylabel("AUC", fontsize=FONTSIZE)
    ax.set_position(AX_POSITION)
    ax.legend(loc=0)
    # ax.set_title(f"Cond. Correlation = {args.corr_value}", fontsize=FONTSIZE)

    fig.savefig(os.path.join(args.dest, "auc.pdf"))


if __name__ == "__main__":
    main()
