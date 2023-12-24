import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
import dvu


def set_mpl_params():
    mpl.rcParams['axes.grid'] = False
    mpl.rcParams['legend.fontsize'] = 14
    mpl.rcParams["figure.figsize"] = [8, 6]
    plt.rcParams['figure.facecolor'] = 'white'

    label_size = 14
    mpl.rcParams['xtick.labelsize'] = label_size
    mpl.rcParams['ytick.labelsize'] = label_size
    mpl.rcParams['axes.labelsize'] = label_size
    mpl.rcParams['axes.titlesize'] = label_size
    mpl.rcParams['figure.titlesize'] = label_size

    mpl.rcParams['figure.dpi'] = 250
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.linewidth'] = 2


def plot_curves(results, with_linear=True, name=None, export=False, n_iter=50, title=True):
    set_mpl_params()

    results_df = results["vals"]
    title_string = ""
    if name is not None:
        title_string += f"{name}, "
    for k, v in results.items():
        if k != "vals":
            title_string += f"{k} = {v}, "
    title_string = title_string[:-2]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    plot_types = ["mse", "bias", "variance"]

    labels = dict({
        "FIGS": "FIGS",
        "CART": "CART",
        "GB (max depth=1)": "GB-1",
        "GB (max depth=2)": "GB-2"
    })

    line_colors = dict({
        "FIGS": "black",
        "CART": "orange",
        "GB (max depth=1)": "purple",
        "GB (max depth=2)": "green"
    })

    indicators = dict({
        "FIGS": results_df["type"] == "figs",
        "CART": results_df["type"] == "cart",
        "GB (max depth=1)": (results_df["type"] == "gb") &
                            (results_df["max_depth"] == 1),
        "GB (max depth=2)": (results_df["type"] == "gb") &
                            (results_df["max_depth"] == 2)})
    alphas = dict({
        "FIGS": 1,
        "CART": 0.5,
        "GB (max depth=1)": 0.5,
        "GB (max depth=2)": 0.5})

    for idx, ax in enumerate(axes):
        plot_type = plot_types[idx]
        if with_linear:
            lin_model_val = results_df[results_df["type"] == "linear"][plot_type].values[0]
            ax.axhline(y=lin_model_val, label='OLS', ls='--', color="grey")
        if plot_type == "mse":
            subtitle_string = "MSE"
            for k, v in indicators.items():
                kwargs = dict(color=line_colors[k], ms=7, lw=3, label=labels[k], alpha=alphas[k])
                filtered_df = results_df[v]
                ax.errorbar(filtered_df["n_rules"], filtered_df[plot_type], yerr=filtered_df["mse_sd"] / np.sqrt(n_iter), **kwargs)
        else:
            subtitle_string = plot_type.capitalize()
            for k, v in indicators.items():
                kwargs = dict(color=line_colors[k], ms=7, lw=3, label=labels[k], alpha=alphas[k])
                filtered_df = results_df[v]
                ax.plot(filtered_df["n_rules"], filtered_df[plot_type], **kwargs)
        ax.set_title(subtitle_string)
        ax.set_xlabel("Number of splits")
        ax.set_xticks([0, 10, 20, 30])
        # ax.tick_params(axis='both', which='both', direction='in', length=6, width=2)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        if idx == 0:
            ax.set_ylabel("Error")
        if idx == 2:
            dvu.line_legend(ax=ax)
            # ax.legend()
            # dvu.line_legend(ax=ax)
    plt.tight_layout()
    if title:
        plt.suptitle(title_string, fontsize=16, y=1.1)
    if export:
        plt.savefig(f"plots/{name} plot.pdf")
    plt.show()


# def plot_curves(results, with_linear=True, name=None, export=False):
#     dvu.set_style()
#     mpl.rcParams['figure.dpi'] = 250
#     results_df = results["vals"]
#     title_string = ""
#     if name is not None:
#         title_string += f"{name}, "
#     for k, v in results.items():
#         if k != "vals":
#             title_string += f"{k} = {v}, "
#     title_string = title_string[:-2]
#     fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor="w")
#     # plt.figure(facecolor="w")
#     plot_types = ["mse", "bias", "variance"]
#     for idx, ax in enumerate(axes):
#         plot_type = plot_types[idx]
#         results_df_filtered = results_df[results_df["type"] == "figs"]
#         ax.plot(results_df_filtered["n_rules"], results_df_filtered[plot_type], label='FIGS')
#         results_df_filtered = results_df[results_df["type"] == "cart"]
#         ax.plot(results_df_filtered["n_rules"], results_df_filtered[plot_type], label='CART')
#         results_df_filtered = results_df[(results_df["type"] == "gb") & (results_df["max_depth"] == 1)]
#         ax.plot(results_df_filtered["n_rules"], results_df_filtered[plot_type], label='GB (max depth=1)')
#         results_df_filtered = results_df[(results_df["type"] == "gb") & (results_df["max_depth"] == 2)]
#         ax.plot(results_df_filtered["n_rules"], results_df_filtered[plot_type], label='GB (max depth=2)')
#         lin_model_val = results_df[results_df["type"] == "linear"][plot_type].values[0]
#         if with_linear:
#             ax.axhline(y=lin_model_val, label='Linear', ls='--', color="black")
#         if plot_type == "mse":
#             subtitle_string = "MSE"
#         else:
#             subtitle_string = plot_type.capitalize()
#         ax.set_title(subtitle_string)
#         ax.set_xlabel("No. of rules")
#         ax.set_ylabel("Error")
#         if idx == 2:
#             ax.legend()
#     plt.tight_layout()
#     plt.suptitle(title_string, fontsize=16, y=1.1)
#     if export:
#         plt.savefig(f"plots/{name}_plot.pdf")
#     plt.show()